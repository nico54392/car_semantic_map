#!/usr/bin/env python3
"""
Detail sweep for SAM3 on terrain.

Question: does terrain segmentation rely on fine image detail, or is the coarse
semantic information enough? (Terrain is "stuff", so we expect the latter.)

NOTE ON RESOLUTION: SAM3 bakes its input resolution into the model's rotary
position tables at build time (default 1008). Passing a smaller `resolution` to
the processor breaks with an assert in `reshape_for_broadcast` - you'd have to
rebuild the model to actually run at a lower resolution. So this does NOT change
the model's resolution (and therefore does NOT measure a speedup). Instead it
keeps the model at its native size and feeds it a DETAIL-DEGRADED image:
downsample by a factor then upsample back, which removes fine detail while
keeping the token grid the model expects.

Per image, per factor f (f<1 = coarser):
  degraded = upsample(downsample(image, f), original_size)  ->  run SAM3
  union mask per prompt, compared (IoU) to the full-detail (f=1.0) result.

Read it as:
  IoU stays ~1.0 as f shrinks -> terrain doesn't need fine detail; a rebuilt
                                 low-res model would likely be safe (and faster).
  IoU drops                   -> that terrain does rely on detail.

To chase the actual speedup later, first check whether the model can be built at
a smaller size:
  grep -rn "image_size\\|img_size\\|resolution\\|freqs_cis" \\
      ~/sam3/sam3/model_builder.py ~/sam3/sam3/configs/ 2>/dev/null | head

Usage:
  export JAIME_SERVER=~/nico_ws/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server
  python tools/resolution_sweep.py --images ~/frames \
    --prompts "grass" "dirt road . trail" "rubble . debris" "rocks . boulders" \
    --factors 0.25 0.5 1.0 --out out/detail_sweep
"""
import argparse
import csv
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_semantic_map.jaime import add_jaime_server_to_path


def degrade(img, f):
    """Downsample by factor f then back to original size — strips fine detail, keeps dimensions."""
    if f >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def union_mask(results, prompt_idx, img_hw, score_thr):
    r = results[prompt_idx]
    masks, scores = r["masks"], r["scores"]
    H, W = img_hw
    out = np.zeros((H, W), dtype=bool)
    if masks.numel() == 0:
        return out
    m = masks[:, 0] if masks.ndim == 4 else masks
    s = scores.float().cpu().numpy()
    if tuple(m.shape[-2:]) != (H, W):
        m = F.interpolate(m.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
    m = m.bool().cpu().numpy()
    for i in range(m.shape[0]):
        if s[i] >= score_thr:
            out |= m[i]
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--factors", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--out", default="out/detail_sweep")
    ap.add_argument("--jaime-server", default=None)
    ap.add_argument("--mask-score", type=float, default=0.3)
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

    factors = sorted(set(args.factors))
    ref_f = max(factors)  # the most detail = reference

    paths = sorted(q for q in Path(args.images).iterdir()
                   if q.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.max_images]
    if not paths:
        sys.exit(f"No images in {args.images}")

    # warmup
    with torch.inference_mode(), amp:
        proc.segment_image_with_text_prompts(Image.open(paths[0]).convert("RGB"))
    if device == "cuda":
        torch.cuda.synchronize()

    rows, lat = [], {f: [] for f in factors}
    for ip in paths:
        img = Image.open(ip).convert("RGB")
        hw = (img.height, img.width)
        masks_by_f = {}
        for f in factors:
            di = degrade(img, f)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), amp:
                res = proc.segment_image_with_text_prompts(di)
            if device == "cuda":
                torch.cuda.synchronize()
            lat[f].append((time.perf_counter() - t0) * 1000.0)
            masks_by_f[f] = [union_mask(res, i, hw, args.mask_score) for i in range(len(args.prompts))]

        ref = masks_by_f[ref_f]
        for i, prompt in enumerate(args.prompts):
            for f in factors:
                m = masks_by_f[f][i]
                rows.append({"image": ip.name, "prompt": prompt, "detail_factor": f,
                             "mask_px": int(m.sum()),
                             "iou_vs_full": 1.0 if f == ref_f else round(iou(m, ref[i]), 3)})
        print(f"[{ip.name}] done")

    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ref_nonempty = {(r["image"], r["prompt"]) for r in rows
                    if r["detail_factor"] == ref_f and r["mask_px"] > 0}
    print(f"\nReference = full detail (factor {ref_f}).  prompts with non-empty ref mask: {len(ref_nonempty)}")
    print("  factor | mean IoU vs full detail   (1.0 = unchanged)")
    for f in factors:
        ius = [r["iou_vs_full"] for r in rows
               if r["detail_factor"] == f and (r["image"], r["prompt"]) in ref_nonempty]
        miou = np.nanmean(ius) if ius else float("nan")
        print(f"  {f:>5} | {miou:.3f}")

    print("\nPer-prompt, how detail-sensitive (mean IoU vs full at the smallest factor):")
    small = min(factors)
    per = {}
    for r in rows:
        if r["detail_factor"] == small and (r["image"], r["prompt"]) in ref_nonempty:
            per.setdefault(r["prompt"], []).append(r["iou_vs_full"])
    for p, v in sorted(per.items(), key=lambda x: np.nanmean(x[1])):
        print(f"  {np.nanmean(v):.3f}  {p}")
    print(f"\nDetail in {out}/results.csv")
    print("High IoU at small factors = terrain robust to losing detail (a low-res rebuild would likely be safe).")


if __name__ == "__main__":
    main()
