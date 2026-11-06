#!/usr/bin/env python3
"""
PE dot-product test.

Question: are SAM3's backbone patch features directly comparable to its text
features? If yes, a 3D map of those features can be queried per point with a
plain dot product (Path A) and we don't need a second model.

What it does, per image:
  1. runs SAM3's backbone once (Jaime's processor, `set_image`)
  2. finds every image feature map in backbone_out and every text feature
     from forward_text, prints names + shapes
  3. for each (image map, text feature) pair with matching channels:
     cosine similarity per patch -> heatmap
  4. runs the full SAM3 pipeline for the same prompts -> reference masks
  5. scores each heatmap against the SAM3 mask (best-threshold IoU, AUROC)

Outputs in --out: one PNG grid per image, results.csv, shapes.txt

Usage:
  export JAIME_SERVER=~/go2_ws/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server
  python tools/pe_probe.py --images some_frames/ --prompts "grass" "dirt . path" "tree" --out out/pe_probe
"""
import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_semantic_map.jaime import add_jaime_server_to_path


# ---------- helpers ----------

def walk_tensors(obj, prefix=""):
    """Yield (name, tensor) for every tensor inside nested dicts/lists."""
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_tensors(v, f"{prefix}[{i}]")


def image_feature_maps(backbone_out):
    """4D tensors shaped like [B, C, H, W] with a real spatial grid. Skips positional encodings."""
    maps = {}
    for name, t in walk_tensors(backbone_out):
        if t.ndim == 4 and t.shape[0] == 1 and t.shape[2] >= 8 and t.shape[3] >= 8 and "pos" not in name.lower():
            maps[name] = t[0].float()  # [C, H, W]
    return maps


def pooled_text_vectors(text_out):
    """
    One vector per text key by averaging over the prompt's real tokens.
    SAM3 pads text to a fixed length; we don't know which way the mask is
    encoded, so pick the reading that gives the fewer valid tokens (a short
    prompt has way fewer real tokens than padding).
    """
    mask = text_out.get("language_mask")
    vecs = {}
    for k, v in text_out.items():
        if not torch.is_tensor(v) or v.ndim != 3:
            continue
        seq = v[:, 0, :].float()  # [seq, dim]
        valid = None
        if mask is not None and mask.ndim == 2 and mask.shape[1] == seq.shape[0]:
            m = mask[0].bool()
            valid = m if m.sum() <= (~m).sum() else ~m
            if valid.sum() == 0:
                valid = None
        vecs[k] = seq[valid].mean(0) if valid is not None else seq.mean(0)
    return vecs


def heatmap(feat_chw, text_vec, size_hw):
    f = F.normalize(feat_chw, dim=0)
    t = F.normalize(text_vec.to(f.device), dim=0)
    sim = torch.einsum("chw,c->hw", f, t)[None, None]
    sim = F.interpolate(sim, size=size_hw, mode="bilinear", align_corners=False)[0, 0]
    return sim.cpu().numpy()


def best_iou(scores, mask):
    """IoU between mask and the heatmap thresholded at its best percentile."""
    if mask.sum() == 0:
        return float("nan"), float("nan")
    best, best_t = 0.0, None
    for q in np.linspace(50, 99, 25):
        t = np.percentile(scores, q)
        pred = scores >= t
        inter = (pred & mask).sum()
        union = (pred | mask).sum()
        iou = inter / union if union else 0.0
        if iou > best:
            best, best_t = iou, q
    return best, best_t


def auroc(scores, mask):
    """Chance that a random mask pixel scores higher than a random non-mask pixel. 0.5 = useless."""
    pos, neg = scores[mask], scores[~mask]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    pos = rng.choice(pos, min(len(pos), 20000), replace=False)
    neg = rng.choice(neg, min(len(neg), 20000), replace=False)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def sam3_mask(result, size_hw, score_thr):
    """Union of SAM3 instance masks above the score threshold."""
    masks, scores = result["masks"], result["scores"]
    out = np.zeros(size_hw, dtype=bool)
    if masks.numel() == 0:
        return out
    m = masks[:, 0] if masks.ndim == 4 else masks
    m = m.bool().cpu().numpy()
    s = scores.float().cpu().numpy()
    for i in range(m.shape[0]):
        if s[i] >= score_thr:
            out |= m[i]
    return out


def save_grid(path, image, prompts, ref_masks, maps_by_pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 1 + len(maps_by_pair)
    fig, axes = plt.subplots(len(prompts), cols, figsize=(3.2 * cols, 2.6 * len(prompts)), squeeze=False)
    for r, p in enumerate(prompts):
        ax = axes[r][0]
        ax.imshow(image)
        ax.imshow(np.ma.masked_where(~ref_masks[r], ref_masks[r]), alpha=0.5, cmap="autumn")
        ax.set_title(f"SAM3 mask: {p}", fontsize=8)
        for c, (pair, hms) in enumerate(maps_by_pair.items(), start=1):
            axes[r][c].imshow(hms[r], cmap="inferno")
            axes[r][c].set_title(pair, fontsize=6)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of .png/.jpg frames")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", default="out/pe_probe")
    ap.add_argument("--jaime-server", default=None, help="defaults to $JAIME_SERVER")
    ap.add_argument("--mask-score", type=float, default=0.3, help="SAM3 instance score to count in the reference mask")
    ap.add_argument("--max-images", type=int, default=20)
    args = ap.parse_args()

    add_jaime_server_to_path(args.jaime_server)
    from sam3.model_builder import build_sam3_image_model
    from inference.sam3_multi_prompt import Sam3MultiPromptProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    print("Loading SAM3...")
    model = build_sam3_image_model().to(device)
    proc = Sam3MultiPromptProcessor(model=model, device=device, confidence_threshold=0.1)

    with torch.inference_mode(), amp:
        proc.set_text_prompts(args.prompts)
    text_vecs = [pooled_text_vectors(t) for t in proc._cached_text_outputs]

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    paths = paths[: args.max_images]
    if not paths:
        sys.exit(f"No images in {args.images}")

    rows, shapes_written = [], False
    for ip in paths:
        img = Image.open(ip).convert("RGB")
        hw = (img.height, img.width)
        with torch.inference_mode(), amp:
            state = proc.set_image(img, state={})
            fmaps = image_feature_maps(state["backbone_out"])
            results = proc.segment_image_with_text_prompts(img)

        if not shapes_written:
            with open(out / "shapes.txt", "w") as f:
                f.write("backbone_out tensors:\n")
                for n, t in walk_tensors(state["backbone_out"]):
                    f.write(f"  {n}: {tuple(t.shape)} {t.dtype}\n")
                f.write("\ntext tensors (first prompt):\n")
                for n, t in walk_tensors(proc._cached_text_outputs[0]):
                    f.write(f"  {n}: {tuple(t.shape)} {t.dtype}\n")
                f.write("\npooled text vectors:\n")
                for k, v in text_vecs[0].items():
                    f.write(f"  {k}: {tuple(v.shape)}\n")
            print((out / "shapes.txt").read_text())
            shapes_written = True

        ref = [sam3_mask(r, hw, args.mask_score) for r in results]

        maps_by_pair = {}
        for fname, feat in fmaps.items():
            for tkey in text_vecs[0]:
                if text_vecs[0][tkey].shape[0] != feat.shape[0]:
                    continue
                pair = f"{fname} · {tkey}"
                hms = [heatmap(feat, tv[tkey], hw) for tv in text_vecs]
                maps_by_pair[pair] = hms
                for p, hm, m in zip(args.prompts, hms, ref):
                    iou, q = best_iou(hm, m)
                    rows.append({"image": ip.name, "prompt": p, "pair": pair,
                                 "mask_px": int(m.sum()), "best_iou": round(iou, 3),
                                 "best_percentile": q, "auroc": round(auroc(hm, m), 3)})

        if not maps_by_pair:
            print(f"[{ip.name}] no image map shares a channel size with any text vector. "
                  "See shapes.txt: the comparison then needs a projection layer.")
        save_grid(out / f"{ip.stem}_grid.png", np.asarray(img), args.prompts, ref, maps_by_pair)
        print(f"[{ip.name}] {len(maps_by_pair)} feature/text pairs tested")

    if rows:
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        # quick summary per pair: AUROC near 0.5 = noise, above ~0.8 = usable for rough queries
        print("\nMean AUROC / best IoU per pair (prompts with a non-empty SAM3 mask):")
        for pair in sorted({r["pair"] for r in rows}):
            rs = [r for r in rows if r["pair"] == pair and r["mask_px"] > 0]
            if rs:
                print(f"  {pair}: AUROC {np.nanmean([r['auroc'] for r in rs]):.3f}, "
                      f"IoU {np.nanmean([r['best_iou'] for r in rs]):.3f}  (n={len(rs)})")
    print(f"\nDone. Look at {out}/")


if __name__ == "__main__":
    main()
