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
