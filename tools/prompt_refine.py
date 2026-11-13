#!/usr/bin/env python3
"""
Prompt-set refinement for terrain segmentation.

Goal: a prompt list that (a) covers as much of the ground as possible and
(b) has as little overlap between prompts as possible. This measures both on
real masks SAM3 produces, not on word similarity (that's Jaime's complementary
UMAP/cosine tool on the text side).

Give it a GENEROUS candidate pool. It runs SAM3 with every prompt on every
image, then reports:

  1. Redundancy matrix  - global IoU between each pair of prompt masks.
                          High IoU = the two prompts claim the same pixels.
  2. Per-prompt stats   - total area, and how much of it is UNIQUE (claimed by
                          that prompt alone). Low unique % = a drop candidate.
  3. Greedy shortlist   - the minimal ordered subset that covers the terrain
                          the full pool finds. This is your non-redundant list.
  4. Coverage curve     - cumulative coverage as each prompt is added.
  5. Per-image overlays - *_assignment.png (each pixel -> its best prompt) and
                          *_uncovered.png (pixels NO prompt claims = possibly
                          missed ground -> a concept you should add a prompt for).

"Coverage" is measured against the union of all candidate prompts (the ground
the pool finds), so sky/irrelevant pixels no prompt ever claims don't count
against you - except they show up in the uncovered overlay for you to eyeball.

Usage:
  export JAIME_SERVER=~/nico_ws/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server
  python tools/prompt_refine.py --images ~/frames --out out/prompt_refine \
    --target 0.97 \
    --prompts "asphalt . paved road" "dirt road . trail" "grass" "sand" \
              "gravel . loose stones" "mud . wet soil" "rubble . debris" \
              "rocks . boulders" "bush . brush" "water . puddle"
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

PALETTE = np.array([  # tab20-ish, distinct
    [31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],
    [140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207],
    [174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
    [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229],
], dtype=np.uint8)


def score_and_mask(results, prompt_idx, img_hw, score_thr):
    """Return (score_map float32 [H,W], union bool mask [H,W]) for one prompt."""
    r = results[prompt_idx]
    masks, scores = r["masks"], r["scores"]
    H, W = img_hw
    smap = np.zeros((H, W), dtype=np.float32)
    if masks.numel() == 0:
        return smap, smap > 0
    m = masks[:, 0] if masks.ndim == 4 else masks
    s = scores.float().cpu().numpy()
    if tuple(m.shape[-2:]) != (H, W):
        m = F.interpolate(m.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
    m = m.bool().cpu().numpy()
    for i in range(m.shape[0]):
        if s[i] >= score_thr:
            smap = np.maximum(smap, m[i].astype(np.float32) * s[i])
    return smap, smap > 0


def save_overlay(path, image_rgb, rgb_overlay, alpha=0.55):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = image_rgb.astype(np.float32)
    blend = (1 - alpha) * base + alpha * rgb_overlay.astype(np.float32)
    plt.figure(figsize=(image_rgb.shape[1] / 120, image_rgb.shape[0] / 120), dpi=120)
    plt.imshow(blend.astype(np.uint8)); plt.axis("off")
    plt.tight_layout(pad=0); plt.savefig(path, bbox_inches="tight", pad_inches=0); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", default="out/prompt_refine")
    ap.add_argument("--jaime-server", default=None)
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--mask-score", type=float, default=0.3)
    ap.add_argument("--target", type=float, default=0.97, help="coverage target for the greedy shortlist")
    ap.add_argument("--redundant-iou", type=float, default=0.5, help="flag prompt pairs above this IoU")
    ap.add_argument("--max-images", type=int, default=30)
    args = ap.parse_args()

    add_jaime_server_to_path(args.jaime_server)
    from sam3.model_builder import build_sam3_image_model
    from inference.sam3_multi_prompt import Sam3MultiPromptProcessor

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    P = args.prompts

    print("Loading SAM3...")
    model = build_sam3_image_model().to(device)
    proc = Sam3MultiPromptProcessor(model=model, resolution=args.resolution, device=device, confidence_threshold=0.1)
    with torch.inference_mode(), amp:
        proc.set_text_prompts(P)

    paths = sorted(q for q in Path(args.images).iterdir()
                   if q.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.max_images]
    if not paths:
        sys.exit(f"No images in {args.images}")

    flat_masks = []  # each: bool [P, Himg*Wimg]
    for ip in paths:
        img = Image.open(ip).convert("RGB")
        rgb = np.asarray(img)
        hw = (img.height, img.width)
        with torch.inference_mode(), amp:
            res = proc.segment_image_with_text_prompts(img)
        smaps, masks = [], []
        for i in range(len(P)):
            sm, mk = score_and_mask(res, i, hw, args.mask_score)
            smaps.append(sm); masks.append(mk)
        stack = np.stack(masks)                       # [P,H,W] bool
        scores = np.stack(smaps)                       # [P,H,W] float
        flat_masks.append(stack.reshape(len(P), -1))

        # assignment overlay: each covered pixel -> colour of its best-scoring prompt
        best = scores.argmax(0)
        any_cov = stack.any(0)
        amap = np.zeros_like(rgb)
        for i in range(len(P)):
            amap[(best == i) & any_cov] = PALETTE[i % len(PALETTE)]
        save_overlay(out / f"{ip.stem}_assignment.png", rgb, amap)

        # uncovered overlay: pixels no prompt claims, in red
        unc = np.zeros_like(rgb); unc[~any_cov] = [220, 20, 60]
        save_overlay(out / f"{ip.stem}_uncovered.png", rgb, unc, alpha=0.5)
        print(f"[{ip.name}] coverage {100*any_cov.mean():.1f}% of pixels")

    B = np.concatenate(flat_masks, axis=1)             # [P, total_px] bool
    TP = B.shape[1]
    area = B.sum(1)
    universe = B.any(0); U = int(universe.sum())
    counts = B.sum(0)                                   # prompts per pixel

    # 1. redundancy matrix (IoU)
    inter = (B.astype(np.int64) @ B.T.astype(np.int64))  # [P,P] intersection counts
    iou = np.zeros((len(P), len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            uni = area[i] + area[j] - inter[i, j]
            iou[i, j] = inter[i, j] / uni if uni else 0.0
    with open(out / "redundancy_iou.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prompt"] + P)
        for i in range(len(P)):
            w.writerow([P[i]] + [f"{iou[i, j]:.3f}" for j in range(len(P))])

    # 2. per-prompt stats
    stats = []
    for i in range(len(P)):
        uniq = int((B[i] & (counts == 1)).sum())
        tot = int(area[i])
        stats.append({"prompt": P[i], "total_px": tot, "unique_px": uniq,
                      "unique_frac": round(uniq / tot, 3) if tot else 0.0,
                      "cover_of_universe": round(tot / U, 3) if U else 0.0})
    with open(out / "prompt_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)

    # 3+4. greedy shortlist to cover the universe
    covered = np.zeros(TP, dtype=bool)
    remaining = set(range(len(P)))
    order = []
    while remaining:
        gains = {i: int((B[i] & ~covered).sum()) for i in remaining}
        i = max(gains, key=gains.get)
        if gains[i] == 0:
            break
        covered |= B[i]
        remaining.discard(i)
        order.append((P[i], gains[i], covered.sum() / U))
    with open(out / "greedy_order.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "prompt", "new_px", "cumulative_coverage"])
        for k, (pr, g, cov) in enumerate(order, 1):
            w.writerow([k, pr, g, f"{cov:.3f}"])

    # ---- printed summary ----
    print(f"\n=== Coverage ===  universe = {U} px ({100*U/TP:.1f}% of all pixels the pool claims)")
    print("Greedy shortlist (add until target), coverage of that universe:")
    short = []
    for k, (pr, g, cov) in enumerate(order, 1):
        mark = ""
        if not short or short[-1][1] < args.target:
            short.append((pr, cov))
            if cov >= args.target:
                mark = "  <- hits target"
        print(f"  {k:>2}. +{g:>8} px  cum {cov:5.1%}  {pr}{mark}")
    keep = [pr for pr, cov in short]
    print(f"\nShortlist for >= {args.target:.0%} coverage ({len(keep)} prompts):")
    print("  " + "  ".join(f'\"{p}\"' for p in keep))

    print("\n=== Redundant pairs (mask IoU >= {:.2f}) ===".format(args.redundant_iou))
    pairs = [(P[i], P[j], iou[i, j]) for i in range(len(P)) for j in range(i + 1, len(P))
             if iou[i, j] >= args.redundant_iou]
    for a, b, v in sorted(pairs, key=lambda x: -x[2]):
        print(f"  {v:.2f}  {a}   <->   {b}")
    if not pairs:
        print("  none - prompts are well separated")

    print("\n=== Drop candidates (little unique coverage) ===")
    for s in sorted(stats, key=lambda x: x["unique_frac"]):
        if s["total_px"] and s["unique_frac"] < 0.2:
            print(f"  unique {s['unique_frac']:.0%} of its {s['total_px']} px  -> {s['prompt']}")

    print(f"\nFiles in {out}/ :")
    print("  redundancy_iou.csv, prompt_stats.csv, greedy_order.csv")
    print("  *_assignment.png (who claims what), *_uncovered.png (red = missed -> maybe add a prompt)")


if __name__ == "__main__":
    main()
