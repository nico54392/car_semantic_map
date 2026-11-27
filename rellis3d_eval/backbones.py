"""2D feature backbones. Each one gives a per-pixel (patch-grid) feature map and a way
to turn features into class probabilities for the ontology.

  clip    open_clip ViT with a ClearCLIP-style last block (q-q attention, no residual,
          no FFN), sliding windows. Same stand-in we used on the single RELLIS frame.
  oracle  one-hot GT labels pooled to the patch grid. No model; it isolates projection,
          fusion and metric bugs from feature quality. Should give ~1.0 everywhere
          except at boundaries.

Add a new one (LSeg, OpenSeg, SAM3-mask + CLIP) by subclassing Backbone.
"""
from __future__ import annotations

import numpy as np

from ontology import Ontology


class Backbone:
    name = "base"
    dim = 0
    needs_gt = False

    def set_ontology(self, ont: Ontology):
        self.ont = ont

    def image_features(self, img: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
        """img (H,W,3) uint8 -> (h,w,D) float32, L2-normalised (patch grid)"""
        raise NotImplementedError

    def classify(self, feat: np.ndarray) -> np.ndarray:
        """(N,D) -> (N,K) probabilities over ontology classes"""
        raise NotImplementedError

    def text_sim(self) -> np.ndarray | None:
        """(K,K) cosine similarity of class text embeddings (for SCS / SwCM)"""
        return None


# --------------------------------------------------------------------------
class OracleBackbone(Backbone):
    name = "oracle"
    needs_gt = True

    def __init__(self, patch: int = 16, noise: float = 0.0, seed: int = 0):
        self.patch, self.noise = patch, noise
        self.rng = np.random.default_rng(seed)

    def set_ontology(self, ont):
        super().set_ontology(ont)
        self.dim = len(ont)
        self.lut = np.full(256, -1)
        for j, e in enumerate(ont.eval_id):
            if e >= 0 and self.lut[e] < 0:
                self.lut[e] = j

    def image_features(self, img, gt=None):
        H, W = gt.shape
        p = self.patch
        h, w = H // p, W // p
        cls = self.lut[gt[:h * p, :w * p]]
        oh = np.zeros((h * p, w * p, self.dim), np.float32)
        m = cls >= 0
        oh[m, cls[m]] = 1
        oh[~m] = 1.0 / self.dim                    # void / unmapped -> uniform
        f = oh.reshape(h, p, w, p, self.dim).mean((1, 3))
        if self.noise > 0:
            f = np.abs(f + self.noise * self.rng.standard_normal(f.shape).astype(np.float32))
        f /= f.sum(-1, keepdims=True)
        return f

    def classify(self, feat):
        f = np.clip(feat, 0, None)
        return f / np.maximum(f.sum(1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------
class ClipDenseBackbone(Backbone):
    name = "clip"
    TEMPLATES = ["a photo of {}.", "a photo of {} terrain.", "an image of {} outdoors.",
                 "a close-up photo of {}."]

    def __init__(self, model="ViT-B-16", pretrained="laion400m_e32", device=None,
                 short_side=448, window=224, stride=112, temperature=0.01):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.eval().to(self.device)
        self.tok = open_clip.get_tokenizer(model)
        norm = pre.transforms[-1]
        self.mean = torch.tensor(norm.mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm.std, device=self.device).view(1, 3, 1, 1)
        v = self.model.visual
        self.patch = v.patch_size[0] if isinstance(v.patch_size, tuple) else v.patch_size
        self.dim = v.proj.shape[1]
        self.short, self.win, self.stride = short_side, window, stride
        self.T = temperature
        attn = v.transformer.resblocks[0].attn
        # old open_clip: nn.MultiheadAttention with batch_first False -> seq-first layout
        self.seq_first = isinstance(attn, torch.nn.MultiheadAttention) and not attn.batch_first

    # ---- text -------------------------------------------------------------
    def _embed_terms(self, terms):
        torch = self.torch
        with torch.no_grad():
            out = []
            for t in terms:
                toks = self.tok([tpl.format(t) for tpl in self.TEMPLATES]).to(self.device)
                e = self.model.encode_text(toks).float()
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                out.append(e / e.norm())
            e = torch.stack(out).mean(0)
            return (e / e.norm()).cpu().numpy()

    def set_ontology(self, ont):
        super().set_ontology(ont)
        q, owner = [], []
        for j, c in enumerate(ont.classes):
            for query in c.queries:
                terms = [t.strip() for t in query.split(" . ") if t.strip()]
                q.append(self._embed_terms(terms))
                owner.append(j)
        self.Q = np.stack(q).astype(np.float32)          # (Nq, D)
        self.owner = np.array(owner)
        ce = np.stack([self.Q[self.owner == j].mean(0) for j in range(len(ont))])
        self.class_emb = ce / np.linalg.norm(ce, axis=1, keepdims=True)

    def text_sim(self):
        return self.class_emb @ self.class_emb.T

    def classify(self, feat):
        s = feat @ self.Q.T                               # (N, Nq) cosine
        K = len(self.ont)
        cls = np.full((len(feat), K), -np.inf, np.float32)
        for j in range(K):                                # 'addition': max over a class's queries
            cls[:, j] = s[:, self.owner == j].max(1)
        z = cls / self.T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    # ---- image ------------------------------------------------------------
    def _dense_tokens(self, x):
        """x (B,3,win,win) normalised -> (B, g, g, D) ClearCLIP-style patch embeddings"""
        torch = self.torch
        v = self.model.visual
        B = x.shape[0]
        t = v.conv1(x)
        g = t.shape[-1]
        t = t.reshape(B, t.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(t.dtype).expand(B, 1, -1)
        t = torch.cat([cls, t], 1)
        pe = v.positional_embedding.to(t.dtype)
        if pe.shape[0] != t.shape[1]:                     # interpolate if window != train size
            n0 = int(round((pe.shape[0] - 1) ** 0.5))
            grid = pe[1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
            grid = torch.nn.functional.interpolate(grid, size=(g, g), mode="bicubic", align_corners=False)
            pe = torch.cat([pe[:1], grid.permute(0, 2, 3, 1).reshape(g * g, -1)], 0)
        t = v.ln_pre(t + pe)
        blocks = v.transformer.resblocks
        if self.seq_first:
            t = t.permute(1, 0, 2)
        for blk in blocks[:-1]:
            t = blk(t)
        if self.seq_first:
            t = t.permute(1, 0, 2)
        last = blocks[-1]
        y = last.ln_1(t)
        a = last.attn
        D = y.shape[-1]
        nh = a.num_heads
        qkv = torch.nn.functional.linear(y, a.in_proj_weight, a.in_proj_bias)
        q, _, val = qkv.split(D, dim=-1)
        q = q.reshape(B, -1, nh, D // nh).transpose(1, 2)
        val = val.reshape(B, -1, nh, D // nh).transpose(1, 2)
        w = torch.softmax(q @ q.transpose(-1, -2) * (D // nh) ** -0.5, dim=-1)   # q-q attention
        o = (w @ val).transpose(1, 2).reshape(B, -1, D)
        o = a.out_proj(o)                                 # no residual, no FFN
        o = v.ln_post(o) @ v.proj
        o = o[:, 1:].reshape(B, g, g, -1)
        return o / o.norm(dim=-1, keepdim=True)

    def image_features(self, img, gt=None):
        torch = self.torch
        H, W = img.shape[:2]
        s = self.short / min(H, W)
        h, w = int(round(H * s)), int(round(W * s))
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        P, win, st = self.patch, self.win, self.stride
        hp = max(win, int(np.ceil((h - win) / st)) * st + win)
        wp = max(win, int(np.ceil((w - win) / st)) * st + win)
        x = torch.nn.functional.pad(x, (0, wp - w, 0, hp - h), mode="replicate")
        acc = torch.zeros(hp // P, wp // P, self.dim, device=self.device)
        cnt = torch.zeros(hp // P, wp // P, 1, device=self.device)
        crops, pos = [], []
        for y0 in range(0, hp - win + 1, st):
            for x0 in range(0, wp - win + 1, st):
                crops.append(x[0, :, y0:y0 + win, x0:x0 + win]); pos.append((y0 // P, x0 // P))
        g = win // P
        with torch.no_grad():
            for i in range(0, len(crops), 16):
                out = self._dense_tokens(torch.stack(crops[i:i + 16]))
                for o, (py, px) in zip(out, pos[i:i + 16]):
                    acc[py:py + g, px:px + g] += o
                    cnt[py:py + g, px:px + g] += 1
        f = (acc / cnt)[: int(np.ceil(h / P)), : int(np.ceil(w / P))]
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)


def make_backbone(name: str, **kw) -> Backbone:
    if name == "oracle":
        return OracleBackbone(noise=kw.get("oracle_noise", 0.0))
    if name == "clip":
        return ClipDenseBackbone(model=kw.get("clip_model", "ViT-B-16"),
                                 pretrained=kw.get("clip_pretrained", "laion400m_e32"),
                                 temperature=kw.get("temperature", 0.01))
    raise ValueError(f"unknown backbone {name}")
