#!/usr/bin/env python3
"""
2D vs 3D vs both, on the same RELLIS LiDAR points and the same 19 classes.

  2D = per-pixel features (CLIP / PE-B) projected onto the points the camera sees   (backbone_compare.py --save_dir)
  3D = OpenScene's 3D model on the point cloud alone                                (openscene3d.py --classes rellis)

python tools/ensemble_eval.py data/rellis_sample \
    --p2d out/bb_probs/pe-b/prob.npy --idx out/bb_probs/pe-b/idx.npy \
    --p3d out/os3d_rellis19/prob.npy --out out/ensemble

Ensembles tried (all per point):
  mean    average of the two class distributions
  maxconf OpenScene's rule: take whichever model is more confident at that point
  w=..    weighted average, sweep of the 2D weight (0 = 3D only, 1 = 2D only)
Plain numpy, runs in any env.
"""
import argparse, json
from pathlib import Path
import numpy as np

RELLIS = {1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
          10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
          27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}


def score(g, p, names):
    ious = {}
    for c in np.unique(g):
        ious[names[c]] = float(((g == c) & (p == c)).sum() / max(((g == c) | (p == c)).sum(), 1))
    return float((g == p).mean()), float(np.mean(list(ious.values()))), ious


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rellis")
    ap.add_argument("--p2d", required=True); ap.add_argument("--idx", required=True)
    ap.add_argument("--p3d", required=True)
    ap.add_argument("--out", default="out/ensemble")
    a = ap.parse_args()

    d = Path(a.rellis)
    xyz = np.fromfile(d / "000104.bin", np.float32).reshape(-1, 4)[:, :3]
    gt = (np.fromfile(d / "000104.label", np.uint32) & 0xFFFF).astype(np.int64)
    gt = gt[np.linalg.norm(xyz, axis=1) > 0.5]                 # same filter as both producers

    c2 = json.load(open(Path(a.p2d).parent / "classes.json"))
    c3 = json.load(open(Path(a.p3d).parent / "classes.json"))
    names = [RELLIS[i] for i in sorted(RELLIS)]
    assert c2 == names and c3 == names, f"class lists differ:\n2D {c2}\n3D {c3}\n(run openscene3d.py with --classes rellis)"
    idx = np.load(a.idx)
    P2 = np.load(a.p2d).astype(np.float32)
    P3 = np.load(a.p3d).astype(np.float32)
    assert len(P3) == len(gt), f"3D probs for {len(P3)} points, scan has {len(gt)}"
    P3 = P3[idx]                                               # only points the camera sees
    name2k = {n: k for k, n in enumerate(names)}
    g = np.array([name2k.get(RELLIS.get(int(v), ""), -1) for v in gt[idx]])
    keep = (g >= 0) & (g != name2k["sky"])
    g, P2, P3 = g[keep], P2[keep], P3[keep]
    print(f"{len(g)} points seen by the camera with a label")

    rows = {}
    rows["2D only"] = P2.argmax(1)
    rows["3D only"] = P3.argmax(1)
    rows["mean"] = (P2 + P3).argmax(1)
    rows["maxconf"] = np.where(P2.max(1) >= P3.max(1), P2.argmax(1), P3.argmax(1))
    for w in (0.25, 0.75):
        rows[f"w2D={w}"] = (w * P2 + (1 - w) * P3).argmax(1)

    res = {}
    print(f"\n{'method':10s}  acc    mIoU   " + "  ".join(f"{names[c][:8]:>8s}" for c in np.unique(g)))
    for m, p in rows.items():
        acc, miou, ious = score(g, p, names)
        res[m] = dict(acc=acc, miou=miou, iou=ious)
        print(f"{m:10s}  {acc:.3f}  {miou:.3f}  " + "  ".join(f"{ious[names[c]]:8.2f}" for c in np.unique(g)))
    pick2 = (P2.max(1) >= P3.max(1)).mean()
    print(f"\nmaxconf used the 2D model on {pick2:.0%} of points")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "ensemble.json", "w"), indent=2)
    print("saved", out / "ensemble.json")


if __name__ == "__main__":
    main()
