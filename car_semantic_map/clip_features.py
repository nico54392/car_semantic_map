"""2D feature extractors. Anything with .image_features(rgb) -> [C,h,w] (L2-normalized)
and .text_features(list[str]) -> [K,C] can be plugged into fusion.fuse_features.

DenseCLIP here is the training-free stand-in for OpenSeg/LSeg (ClearCLIP-style last layer).
A SAM3 / OpenSeg / LSeg extractor would just be another class with the same two methods.
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

TEMPLATES = ["a photo of {}.", "a photo of {} in an off-road scene.", "a close-up photo of {}.", "{}"]


class DenseCLIP:
    def __init__(self, weights="laion400m_e32", arch="ViT-B-16", short_side=448, device=None):
        """weights: an open_clip pretrained tag (downloads on first use, e.g. 'laion400m_e32')
        or a path to a .pt file."""
        import open_clip
        torch.set_grad_enabled(False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=weights)
        self.model.eval().to(self.device)
        self.tok = open_clip.get_tokenizer(arch)
        self.short = short_side
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(3, 1, 1)

    def query_features(self, queries):
        """queries: list of prompt strings in the paper's style, e.g. "grass . lawn".
        Terms joined by ' . ' are averaged into one query (aggregation). -> [Q,512]"""
        out = []
        for q in queries:
            terms = [t.strip() for t in q.split(" . ") if t.strip()]
            e = self.model.encode_text(self.tok([t.format(p) for p in terms for t in TEMPLATES]).to(self.device))
            out.append(F.normalize(F.normalize(e.float(), dim=-1).mean(0), dim=-1))
        return torch.stack(out).cpu()

    def class_features(self, prompt_lists):
        """one list of queries per class (addition = several queries). Returns
        query embeddings [Q,512] and the class index of each query."""
        qs, owner = [], []
        for k, pl in enumerate(prompt_lists):
            qs += pl
            owner += [k] * len(pl)
        return self.query_features(qs), np.array(owner)

    def image_features(self, rgb):
        H0, W0 = rgb.shape[:2]
        s = self.short / min(H0, W0)
        H, W = int(round(H0 * s / 16)) * 16, int(round(W0 * s / 16)) * 16
        v = self.model.visual
        x = torch.from_numpy(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float().to(self.device) / 255
        x = ((x - self.mean) / self.std)[None]
        x = v.conv1(x)
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        pos = v.positional_embedding
        g0 = int((pos.shape[0] - 1) ** 0.5)
        pp = pos[1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        pp = F.interpolate(pp, (gh, gw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(gh * gw, -1)
        x = torch.cat([v.class_embedding[None, None].expand(1, 1, -1), x], 1) + torch.cat([pos[:1], pp])[None]
        x = v.ln_pre(x)
        for blk in v.transformer.resblocks[:-1]:
            x = blk(x)
        blk = v.transformer.resblocks[-1]  # q-q attention, drop residual + FFN
        a = blk.attn
        y = blk.ln_1(x)
        q, _, val = F.linear(y, a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
        nh = a.num_heads
        hd = q.shape[-1] // nh
        q = q.view(1, -1, nh, hd).transpose(1, 2)
        val = val.view(1, -1, nh, hd).transpose(1, 2)
        y = ((q @ q.transpose(-1, -2) / hd ** 0.5).softmax(-1) @ val).transpose(1, 2).reshape(1, -1, nh * hd)
        f = v.ln_post(a.out_proj(y))[:, 1:] @ v.proj
        return F.normalize(f.float(), dim=-1).reshape(gh, gw, -1).permute(2, 0, 1).cpu()


class DensePE:
    """Dense, text-aligned features from Meta's Perception Encoder (PE-Core) via open_clip.

    PE-Core is a timm Eva ViT with RoPE and an attention-pooling head (no CLS token), so the
    CLIP trick above doesn't apply as-is. Here:
      - the image is cut into overlapping tiles at the model's native size (224 for B/16,
        336 for L/14) so the RoPE grid never has to be stretched; overlaps are averaged
      - every patch token goes through the *value path* of the pooling head (as if it only
        attended to itself) -> pool MLP -> projection into the text space
      - `variant` sets what happens in the last ViT block before that:
          "plain"   normal last block
          "last_v"  last block attention replaced by its value path (keeps residual + MLP)
          "last_v_only" last block = value path only, no residual, no MLP (closest to ClearCLIP)
    Which variant works best is an experiment: tools/backbone_compare.py.
    """

    def __init__(self, model_name="PE-Core-B-16", weights="meta", variant="last_v_only",
                 short_side=448, device=None):
        import open_clip
        torch.set_grad_enabled(False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=weights)
        self.model.eval().to(self.device)
        self.tok = open_clip.get_tokenizer(model_name)
        self.variant = variant
        t = self.model.visual.trunk
        self.crop = t.patch_embed.img_size[0]
        self.patch = t.patch_embed.patch_size[0]
        self.short = max(short_side, self.crop)
        cfg = self.model.visual.preprocess_cfg if hasattr(self.model.visual, "preprocess_cfg") else {}
        mean = cfg.get("mean", (0.5, 0.5, 0.5)); std = cfg.get("std", (0.5, 0.5, 0.5))
        self.mean = torch.tensor(mean, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(3, 1, 1)

    # same text side as DenseCLIP
    query_features = DenseCLIP.query_features
    class_features = DenseCLIP.class_features

    def _tokens(self, x):
        """[B,3,crop,crop] -> dense text-space features [B, gh*gw, D]"""
        t = self.model.visual.trunk
        x = t.patch_embed(x)
        x, rope = t._pos_embed(x)
        x = t.norm_pre(x)
        blocks = list(t.blocks)
        mixed = getattr(t, "rope_mixed", False) and rope is not None
        for i, blk in enumerate(blocks[:-1]):
            x = blk(x, rope=rope[i] if mixed else rope)
        blk = blocks[-1]
        if self.variant == "plain":
            x = blk(x, rope=rope[-1] if mixed else rope)
        else:
            a = blk.attn
            y = blk.norm1(x)
            B, N, C = y.shape
            v = a.qkv(y).reshape(B, N, 3, C)[:, :, 2]          # value path = each token attends to itself
            y = a.proj(a.norm(v))
            if getattr(blk, "gamma_1", None) is not None:
                y = blk.gamma_1 * y
            if self.variant == "last_v":
                x = x + y
                m = blk.mlp(blk.norm2(x))
                x = x + (blk.gamma_2 * m if getattr(blk, "gamma_2", None) is not None else m)
            else:                                               # last_v_only
                x = y
        x = t.norm(x)
        x = x[:, getattr(t, "num_prefix_tokens", 0):]           # drop cls/register tokens
        ap = t.attn_pool                                        # pooling head, value path per token
        B, N, C = x.shape
        v = ap.kv(x).reshape(B, N, 2, C)[:, :, 1]
        z = ap.proj(v)
        if ap.mlp is not None:
            z = z + ap.mlp(ap.norm(z))
        z = t.fc_norm(z)
        z = t.head(z)
        z = self.model.visual.head(z)
        return F.normalize(z.float(), dim=-1)

    def image_features(self, rgb):
        H0, W0 = rgb.shape[:2]
        p, c = self.patch, self.crop
        s = self.short / min(H0, W0)
        H = max(c, int(round(H0 * s / p)) * p)
        W = max(c, int(round(W0 * s / p)) * p)
        x = torch.from_numpy(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float().to(self.device) / 255
        x = (x - self.mean) / self.std
        gh, gw, g = H // p, W // p, c // p
        step = max(g // 2, 1)
        ys = sorted(set(list(range(0, gh - g + 1, step)) + [gh - g]))
        xs = sorted(set(list(range(0, gw - g + 1, step)) + [gw - g]))
        tiles, where = [], []
        for ty in ys:
            for tx in xs:
                tiles.append(x[:, ty * p:ty * p + c, tx * p:tx * p + c]); where.append((ty, tx))
        acc = None; cnt = torch.zeros(gh, gw, device=self.device)
        for i in range(0, len(tiles), 16):
            f = self._tokens(torch.stack(tiles[i:i + 16]))     # [b, g*g, D]
            if acc is None:
                acc = torch.zeros(f.shape[-1], gh, gw, device=self.device)
            for (ty, tx), fi in zip(where[i:i + 16], f):
                acc[:, ty:ty + g, tx:tx + g] += fi.T.reshape(-1, g, g)
                cnt[ty:ty + g, tx:tx + g] += 1
        return F.normalize(acc / cnt, dim=0).cpu()


def make_extractor(backbone="clip", weights=None, **kw):
    """backbone: 'clip' (ViT-B/16 LAION) | 'pe-b' (PE-Core-B-16) | 'pe-l' (PE-Core-L-14-336) | 'pe-s' | 'pe-t'"""
    if backbone == "clip":
        return DenseCLIP(weights or "laion400m_e32", **kw)
    names = {"pe-t": "PE-Core-T-16-384", "pe-s": "PE-Core-S-16-384", "pe-b": "PE-Core-B-16",
             "pe-l": "PE-Core-L-14-336", "pe-g": "PE-Core-bigG-14-448"}
    return DensePE(names[backbone], weights or "meta", **kw)
