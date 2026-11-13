#!/usr/bin/env python3
"""
Resolution sweep for SAM3 on terrain.

Question: how much does input resolution matter for terrain masks? Terrain is
semantically coarse, so the fine (high-res) FPN level may be overkill. Lowering
the input resolution starves that level and speeds inference. This measures the
trade-off directly.

Per image, per resolution:
  - run SAM3, time it (GPU-synced)
  - build a union mask per prompt (instances above --mask-score)
Reference = the highest resolution in the list. For each lower resolution we
report IoU(lower-res mask, reference mask) per prompt, plus mean latency.

Read the summary as:
  IoU near 1.0 at low res  -> terrain needs no fine detail; run low res, it's faster
  IoU drops at low res     -> that prompt/terrain does rely on resolution

Usage:
  export JAIME_SERVER=~/nico_ws/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server
  python tools/resolution_sweep.py --images ~/frames \
    --prompts "grass" "dirt road . trail" "rubble . debris" "rocks . boulders" \
    --resolutions 512 768 1008 --out out/res_sweep
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


def union_mask(results, prompt_idx, img_hw, score_thr):
    """Union of a prompt's instance masks above score_thr, as a bool array at image size."""
    r = results[prompt_idx]
    masks, scores = r["masks"], r["scores"]
    H, W = img_hw
    out = np.zeros((H, W), dtype=bool)
    if masks.numel() == 0:
        return out
    m = masks[:, 0] if masks.ndim == 4 else masks          # [N, h, w]
    s = scores.float().cpu().numpy()
    if tuple(m.shape[-2:]) != (H, W):                      # masks may come back at model res
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
    ap.add_argument("--resolutions", nargs="+", type=int, default=[512, 768, 1008])
    ap.add_argument("--out", default="out/res_sweep")
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

    resolutions = sorted(set(args.resolutions))
    ref_res = max(resolutions)

    # one processor per resolution (resolution is fixed at construction time)
    procs = {}
    for R in resolutions:
        p = Sam3MultiPromptProcessor(model=model, resolution=R, device=device, confidence_threshold=0.1)
        with torch.inference_mode(), amp:
            p.set_text_prompts(args.prompts)
        procs[R] = p

    paths = sorted(q for q in Path(args.images).iterdir()
                   if q.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.max_images]
    if not paths:
        sys.exit(f"No images in {args.images}")

    # warmup: the first GPU call at each resolution is slow and would skew timing
    warm = Image.open(paths[0]).convert("RGB")
    for R in resolutions:
        with torch.inference_mode(), amp:
            procs[R].segment_image_with_text_prompts(warm)
    if device == "cuda":
        torch.cuda.synchronize()

    rows = []
    lat = {R: [] for R in resolutions}
    for ip in paths:
        img = Image.open(ip).convert("RGB")
        hw = (img.height, img.width)
        masks_by_res = {}
        for R in resolutions:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), amp:
                res = procs[R].segment_image_with_text_prompts(img)
            if device == "cuda":
                torch.cuda.synchronize()
            lat[R].append((time.perf_counter() - t0) * 1000.0)  # ms
            masks_by_res[R] = [union_mask(res, i, hw, args.mask_score) for i in range(len(args.prompts))]

        ref = masks_by_res[ref_res]
        for i, prompt in enumerate(args.prompts):
            for R in resolutions:
                m = masks_by_res[R][i]
                rows.append({
                    "image": ip.name, "prompt": prompt, "resolution": R,
                    "mask_px": int(m.sum()),
                    "iou_vs_ref": 1.0 if R == ref_res else round(iou(m, ref[i]), 3),
                })
        print(f"[{ip.name}] " + "  ".join(f"{R}:{lat[R][-1]:.0f}ms" for R in resolutions))

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: average only over (image, prompt) pairs where the reference mask was non-empty
    ref_nonempty = {(r["image"], r["prompt"]) for r in rows
                    if r["resolution"] == ref_res and r["mask_px"] > 0}
    print(f"\nReference resolution: {ref_res}   (prompts with a non-empty ref mask: {len(ref_nonempty)})")
    print("  res  | mean latency | mean IoU vs ref | speedup")
    base_lat = np.mean(lat[ref_res])
    for R in resolutions:
        ius = [r["iou_vs_ref"] for r in rows
               if r["resolution"] == R and (r["image"], r["prompt"]) in ref_nonempty]
        mlat = np.mean(lat[R])
        miou = np.nanmean(ius) if ius else float("nan")
        print(f"  {R:>4} | {mlat:8.0f} ms | {miou:>13.3f} | {base_lat / mlat:4.1f}x")

    print(f"\nPer-prompt detail: {out}/results.csv")
    print("If IoU stays ~0.9+ at 512, terrain doesn't need the fine level -> run 512 for the speedup.")


if __name__ == "__main__":
    main()
