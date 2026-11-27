#!/usr/bin/env python3
"""
Map -> Nav2 map (map.pgm + map.yaml, map_server "scale" mode) so Nav2 sees terrain COST, not just free/occupied.
Input is either a Unity scene folder (GT labels) or any control-grid npz (car_semantic_map/control_grid.py:
Unity GT, RELLIS oracle, SAM3, PE-B... all the same format). Nav2 doesn't know which.

python tools/nav2_map.py ~/nico_ws/scenes/flat_mix                         # scene -> grid at 0.1 m -> <scene>/nav2/
python tools/nav2_map.py ~/nico_ws/scenes/flat_mix/grid/gt_grid0.5.npz     # any grid npz -> <npz dir>/nav2/
python tools/nav2_map.py <grid.npz> --risk 1 --max_step 0.15 --max_slope 25 # cautious + geometry walls

Planning traversability = trav_mean - risk*trav_std (unseen -> --unknown; step/slope over the limits -> wall).
How it reaches Nav2:
  pixel = 255 * trav  (white = easy, black = wall)
  map_server (mode: scale, negate: 0): occupancy = 1 - pixel/255 -> 0..100; >= 0.95 (trav < 0.05) = wall
  static layer (trinary_costmap: false): cost = occupancy/100 * 254
  SmacPlanner2D: step cost 1 + 4*cost/252 = oracle_plan's 1 + 4*(1-trav)
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map import control_grid as cg          # noqa: E402


def write_nav2(t, origin, res, out):
    img = np.flipud(t.T)                          # t[ix, iy] -> image rows = y (top = max y), cols = x
    pix = np.round(img * 255).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "map.pgm", "wb") as f:
        f.write(f"P5\n{pix.shape[1]} {pix.shape[0]}\n255\n".encode()); f.write(pix.tobytes())
    (out / "map.yaml").write_text(
        f"image: map.pgm\nmode: scale\nresolution: {res}\norigin: [{origin[0]:.3f}, {origin[1]:.3f}, 0.0]\n"
        f"negate: 0\noccupied_thresh: 0.95\nfree_thresh: 0.0\n")
    return pix.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Unity scene folder, or a control-grid .npz")
    ap.add_argument("--res", type=float, default=0.1, help="grid resolution when building from a scene folder")
    ap.add_argument("--risk", type=float, default=0.0, help="cautious planning: subtract risk*trav_std")
    ap.add_argument("--max_step", type=float, default=None, help="m, bigger height step in a cell = wall")
    ap.add_argument("--max_slope", type=float, default=None, help="deg, steeper = wall")
    ap.add_argument("--unknown", type=float, default=0.5)
    ap.add_argument("--ontology", default=str(ROOT / "config" / "ontology.json"))
    ap.add_argument("--trav", nargs="*", default=[], metavar="CLASS=VALUE", help="scene input only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    if src.suffix == ".npz":
        g = cg.load(src)
        out = Path(a.out) if a.out else src.parent / "nav2"
    else:
        from car_semantic_map.scene_io import load_scene
        from car_semantic_map.evaluate import load_ontology
        sc = load_scene(str(src), max_views=1)
        L = sc.point_labels.astype(int)
        ids = sorted(c for c in sc.classes if c != 0)
        keep = np.isin(L, ids); col = {c: k for k, c in enumerate(ids)}
        Lk = np.array([col[c] for c in L[keep]])
        over = {k.strip(): float(v) for k, v in (kv.split("=") for kv in a.trav)}
        g = cg.from_points(sc.points[keep], np.eye(len(ids))[Lk], [sc.classes[c] for c in ids],
                           load_ontology(Path(a.ontology)), res=a.res, unknown=a.unknown, trav_override=over)
        if not g["seen"].all():                   # holes between map points at fine cells: copy nearest neighbour
            from scipy import ndimage
            idx = ndimage.distance_transform_edt(~g["seen"], return_distances=False, return_indices=True)
            for k in ("trav_mean", "trav_std", "step", "slope"):
                g[k] = g[k][tuple(idx)]
            g["seen"] = np.ones_like(g["seen"])
        out = Path(a.out) if a.out else src / "nav2"

    t = cg.cost_trav(g, risk=a.risk, max_step=a.max_step, max_slope=a.max_slope, unknown=a.unknown)
    h, w = write_nav2(t, g["origin"], g["res"], out)
    print(f"classes: {g['class_names'].tolist()}")
    print(f"map {w}x{h} px at {g['res']} m, origin ({g['origin'][0]:.2f}, {g['origin'][1]:.2f}), "
          f"{(t < 0.05).mean():.1%} walls, {(~g['seen']).mean():.0%} unseen (-> {a.unknown}), risk {a.risk}")
    print("saved", out / "map.yaml")


if __name__ == "__main__":
    main()
