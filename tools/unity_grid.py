#!/usr/bin/env python3
"""
Unity scene (GT labels) -> the control grid (car_semantic_map/control_grid.py), same format the RELLIS
tools write. GT labels give one-hot probabilities, so trav_std is 0 except where classes mix in a cell.

python tools/unity_grid.py ~/nico_ws/scenes/flat_mix                 # -> <scene>/grid/gt_grid0.5.npz + preview png
python tools/unity_grid.py ~/nico_ws/scenes/flat_mix --res 0.1 --trav mud=0.05
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.scene_io import load_scene          # noqa: E402
from car_semantic_map.evaluate import load_ontology      # noqa: E402
from car_semantic_map import control_grid as cg          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--unknown", type=float, default=0.5)
    ap.add_argument("--ontology", default=str(ROOT / "config" / "ontology.json"))
    ap.add_argument("--trav", nargs="*", default=[], metavar="CLASS=VALUE")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sc = load_scene(a.scene, max_views=1)
    P, L = sc.points, sc.point_labels.astype(int)
    ids = sorted(c for c in sc.classes if c != 0)
    names = [sc.classes[c] for c in ids]
    col = {c: k for k, c in enumerate(ids)}
    keep = np.isin(L, ids)                                  # void points carry no class
    P, L = P[keep], np.array([col[c] for c in L[keep]])
    onehot = np.eye(len(ids))[L]
    over = {k.strip(): float(v) for k, v in (kv.split("=") for kv in a.trav)}
    g = cg.from_points(P, onehot, names, load_ontology(Path(a.ontology)), res=a.res, unknown=a.unknown,
                       gt_labels=L, trav_override=over)
    out = Path(a.out) if a.out else Path(a.scene) / "grid" / f"gt_grid{a.res:g}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    cg.save(out, g)
    X, Y = g["seen"].shape
    print(f"grid {X}x{Y} cells of {a.res} m, origin ({g['origin'][0]:.2f}, {g['origin'][1]:.2f}), "
          f"{g['seen'].mean():.0%} seen, classes {names}")
    print(f"trav_mean {np.nanmin(g['trav_mean']):.2f}..{np.nanmax(g['trav_mean']):.2f}, "
          f"step max {np.nanmax(g['step']):.2f} m, slope max {np.nanmax(g['slope']):.1f} deg")
    print("saved", out)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ext = [-(g["origin"][1] + Y * a.res), -g["origin"][1], g["origin"][0], g["origin"][0] + X * a.res]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        for axx, k, cm, v in [(ax[0], "trav_mean", "RdYlGn", (0, 1)), (ax[1], "mu_mean", "viridis", (0, 1)),
                              (ax[2], "slope", "magma", (0, max(5, np.nanmax(g["slope"]))))]:
            im = axx.imshow(np.fliplr(g[k]), origin="lower", extent=ext, cmap=cm, vmin=v[0], vmax=v[1])
            axx.set_title(k); plt.colorbar(im, ax=axx, fraction=0.046)
        fig.suptitle("control grid (top-down, Unity-style view: right = -y, up = x)")
        fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)
        print("preview", out.with_suffix(".png"))
    except Exception as e:
        print("(no preview:", e, ")")


if __name__ == "__main__":
    main()
