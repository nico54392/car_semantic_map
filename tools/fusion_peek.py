#!/usr/bin/env python3
"""
Look inside SAM3's fusion encoder on one image (run on the 4090, in nico_sam3; only imports sam3, edits nothing).

python tools/fusion_peek.py --image ~/frames/a.jpg --prompts "grass" "dirt path" "tree" --out out/fusion_peek
  [--checkpoint /path/sam3.pt]   (default: whatever build_sam3_image_model() finds / downloads)

Per prompt it records what goes into, through and out of the fusion encoder
(sam3/model/encoder.py: TransformerEncoderFusion, 6 layers, each = self-attn on image tokens,
 cross-attn image->text, feed-forward) and scores every map against SAM3's own final mask (AUROC):

  cos_in      cosine(image token BEFORE fusion, mean text vector)    <- "are the inputs comparable?"
  cos_L{i}    same, after layer i                                     <- does plain cosine get better inside?
  move_L{i}   how far each image token moved in layer i (norm of the change)
  attn_L{i}:<word>   cross-attention weight each image token gives that word in layer i (mean over 8 heads)
  semantic    SAM3's own per-pixel semantic_seg logit (the decoder side, for reference)

AUROC: 0.5 = coin flip, 1.0 = ranks every mask pixel above every non-mask pixel.
Writes: shapes.txt, results.csv, <prompt>_grid.png
"""
import argparse, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image


def auroc(score, mask):
    s, m = score.ravel(), mask.ravel().astype(bool)
    if m.all() or not m.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0                        # ranks (ties ignored, fine for heatmaps)
    return float((r[m].sum() - m.sum() * (m.sum() + 1) / 2) / (m.sum() * (~m).sum()))


def shp(x):
    return "x".join(map(str, x.shape)) if torch.is_tensor(x) else type(x).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 5], help="layers to draw attention for")
    ap.add_argument("--out", default="out/fusion_peek")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    kw = dict(checkpoint_path=a.checkpoint, load_from_HF=False) if a.checkpoint else {}
    model = build_sam3_image_model(**kw)
    proc = Sam3Processor(model)
    enc = model.transformer.encoder
    lb = model.backbone.language_backbone
    tok, ctx = lb.tokenizer, lb.context_length

    # ---------------- hooks ----------------
    cap = {}

    def pre_enc(mod, args, kwargs):                            # inputs to the fusion encoder
        cap["img_in"] = kwargs["src"][0].detach().clone()      # (HW, B, 256), seq first
        cap["txt"] = kwargs["prompt"].detach().clone()         # (T, B, 256)
        cap["txt_pad"] = kwargs["prompt_key_padding_mask"].detach().clone()  # (B, T) True = padding
        cap["hw"] = tuple(kwargs["feat_sizes"][0])
        cap["layers"], cap["attn"] = [], []
    enc.register_forward_pre_hook(pre_enc, with_kwargs=True)
    enc.register_forward_hook(lambda m, i, o: cap.__setitem__("img_out", o["memory"].detach().clone()))

    for layer in enc.layers:
        layer.register_forward_hook(lambda m, i, o: cap["layers"].append(o.detach().clone()))
        mha = layer.cross_attn_image
        orig = mha.forward

        def fwd(*args, _orig=orig, **kwargs):                  # ask for per-head attention weights
            kwargs["need_weights"] = True; kwargs["average_attn_weights"] = False
            res = _orig(*args, **kwargs)
            if res[1] is not None:
                cap["attn"].append(res[1].detach().float().cpu())   # (B, heads, HW, T)
            return res
        mha.forward = fwd

    orig_fg = model.forward_grounding
    def fg(*args, **kwargs):
        o = orig_fg(*args, **kwargs); cap["grounding"] = o; return o
    model.forward_grounding = fg

    # ---------------- run ----------------
    img = Image.open(a.image).convert("RGB")
    W, H = img.size
    state = proc.set_image(img)
    lines = ["== backbone_out (image side, computed once, prompt-independent)"]
    for k, v in state["backbone_out"].items():
        if torch.is_tensor(v):
            lines.append(f"  {k:28s} {shp(v)}")
        elif isinstance(v, (list, tuple)):
            lines += [f"  {k}[{i}]{'':{24 - len(k)}s} {shp(t)}" for i, t in enumerate(v)]

    rows = []
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for p in a.prompts:
        state = proc.set_text_prompt(prompt=p, state=state)
        if "img_in" not in cap:
            raise SystemExit("fusion encoder hook never fired; tell Claude the sam3 version")
        h, w = cap["hw"]
        valid = ~cap["txt_pad"][0].cpu()                        # real tokens
        ids = tok([p], context_length=ctx)[0]                  # same tokenization SAM3 used
        words = []
        for t_i in range(len(valid)):
            try:
                words.append(tok.decode([int(ids[t_i])]).strip() or f"t{t_i}")
            except Exception:
                words.append(f"t{t_i}")
        txt = cap["txt"][:, 0].float().cpu()                   # (T, 256)
        tmean = F.normalize(txt[valid].mean(0), dim=0)

        mask = (state["masks"].any(0)[0].cpu().numpy() if len(state["masks"]) else np.zeros((H, W), bool))
        def up(m2d):                                           # (h, w) -> (H, W)
            return F.interpolate(torch.as_tensor(m2d)[None, None].float(), (H, W), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
        def tokens_hw(t):                                      # any (HW,B,C) or (B,HW,C) -> (HW, C)
            t = t.float().cpu()
            return t[:, 0] if t.shape[0] == h * w else t[0]

        maps = {}
        x_in = tokens_hw(cap["img_in"])
        maps["cos_in"] = (F.normalize(x_in, dim=1) @ tmean).reshape(h, w).numpy()
        prev = x_in
        for i, t in enumerate(cap["layers"]):
            x = tokens_hw(t)
            maps[f"cos_L{i}"] = (F.normalize(x, dim=1) @ tmean).reshape(h, w).numpy()
            maps[f"move_L{i}"] = (x - prev).norm(dim=1).reshape(h, w).numpy()
            prev = x
        for i in a.layers:
            if i < len(cap["attn"]):
                at = cap["attn"][i][0].mean(0)                 # (HW, T), mean over heads
                for t_i in np.nonzero(valid.numpy())[0]:
                    maps[f"attn_L{i}:{words[t_i]}"] = at[:, t_i].reshape(h, w).numpy()
        sem = cap.get("grounding", {}).get("semantic_seg")
        if sem is not None:
            maps["semantic"] = sem[0, 0].float().cpu().numpy()

        for k, m2d in maps.items():
            big = up(m2d) if m2d.shape != (H, W) else m2d
            rows.append(dict(prompt=p, map=k, auroc=round(auroc(big, mask), 3), mask_px=int(mask.sum())))

        if p == a.prompts[0]:
            lines += ["", f"== fusion encoder, prompt '{p}'",
                      f"  image tokens in    {shp(cap['img_in'])}   ({h}x{w} grid, seq first)",
                      f"  text tokens        {shp(cap['txt'])}   ({int(valid.sum())} real: {[words[i] for i in np.nonzero(valid.numpy())[0]]})",
                      f"  per layer output   {shp(cap['layers'][0])}  x {len(cap['layers'])} layers",
                      f"  cross-attn weights {'x'.join(map(str, cap['attn'][0].shape)) if cap['attn'] else 'not returned'}"
                      f"   (batch, heads, image tokens, text tokens)",
                      f"  image tokens out   {shp(cap['img_out'])}",
                      f"  attention share on padding slots, per layer: "
                      f"{[round(float(x[0].mean(0)[:, ~valid].sum(1).mean()), 3) for x in cap['attn']]}",
                      f"  text after fusion  unchanged: the image reads the text, the text never reads the image"]

        # figure
        show = ["cos_in", f"cos_L{len(cap['layers']) - 1}", f"move_L0", f"move_L{len(cap['layers']) - 1}"] + \
               [k for k in maps if k.startswith("attn_")][:8] + (["semantic"] if "semantic" in maps else [])
        n = 2 + len(show); cols = 4; rws = (n + cols - 1) // cols
        fig, ax = plt.subplots(rws, cols, figsize=(4 * cols, 3.2 * rws))
        ax = ax.ravel()
        ax[0].imshow(img); ax[0].set_title("image")
        ax[1].imshow(img); ax[1].imshow(mask, alpha=.5, cmap="Reds"); ax[1].set_title(f"SAM3 mask '{p}'")
        au = {r["map"]: r["auroc"] for r in rows if r["prompt"] == p}
        for j, k in enumerate(show):
            ax[2 + j].imshow(maps[k], cmap="viridis"); ax[2 + j].set_title(f"{k}  AUROC {au[k]}", fontsize=9)
        for x in ax: x.axis("off")
        fig.tight_layout(); fig.savefig(out / f"{p.replace(' ', '_')}_grid.png", dpi=110); plt.close(fig)

    (out / "shapes.txt").write_text("\n".join(lines) + "\n")
    with open(out / "results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("\n".join(lines))
    print("\nAUROC vs SAM3's own mask (0.5 = chance):")
    for p in a.prompts:
        r = {x["map"]: x["auroc"] for x in rows if x["prompt"] == p}
        best_attn = max((v for k, v in r.items() if k.startswith("attn_") and v == v), default=float("nan"))
        last = max(int(k[5:]) for k in r if k.startswith("cos_L"))
        print(f"  {p:14s} cos_in {r.get('cos_in')}  cos_last {r.get(f'cos_L{last}')}"
              f"  best attn {best_attn}  semantic {r.get('semantic', '-')}")
    print("saved", out)


if __name__ == "__main__":
    if torch.cuda.is_available():                 # SAM3 expects bf16 autocast around every call
        with torch.autocast("cuda", dtype=torch.bfloat16):
            main()
    else:
        main()
