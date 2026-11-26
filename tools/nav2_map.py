#!/usr/bin/env python3
"""
Scene -> Nav2 map: the same traversability grid as oracle_plan.py, saved as map.pgm + map.yaml
for nav2_map_server in "scale" mode, so Nav2 sees terrain COST, not just free/occupied.

python tools/nav2_map.py ~/nico_ws/scenes/flat_mix                      # -> <scene>/nav2/map.{pgm,yaml}
python tools/nav2_map.py ~/nico_ws/scenes/flat_mix --trav mud=0.05     # same overrides as oracle_plan

How the cost gets through Nav2:
  pixel = 255 * traversability  (white = easy, black = wall)
  map_server (mode: scale, negate: 0): occupancy = 1 - pixel/255 -> 0..100
     occupancy >= occupied_thresh (0.95, i.e. trav < 0.05) -> wall, same rule as oracle_plan
  static layer (trinary_costmap: false): cost = occupancy/100 * 254
  SmacPlanner2D step cost = 1 + cost_travel_multiplier * cost/252  (multiplier 4 = oracle_plan's 1 + 4*(1-trav))
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from car_semantic_map.scene_io import load_scene          # noqa: E402
from car_semantic_map.evaluate import load_ontology      # noqa: E402
from oracle_plan import grid                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--cell", type=float, default=0.1, help="map resolution, m (Nav2 likes finer than our 0.25)")
    ap.add_argument("--unknown", type=float, default=0.5)
    ap.add_argument("--ontology", default=str(ROOT / "config" / "ontology.json"))
    ap.add_argument("--trav", nargs="*", default=[], metavar="CLASS=VALUE")
    ap.add_argument("--out", default=None, help="default <scene>/nav2")
    a = ap.parse_args()

    sc = load_scene(a.scene, max_views=1)
    P, L = sc.points, sc.point_labels
    onto = load_ontology(Path(a.ontology))
    tv = {}
    for cid, name in sc.classes.items():
        t = onto.get(name, {}).get("traversability") if cid != 0 else None
        tv[cid] = a.unknown if t is None else float(t)
    for kv in a.trav:
        k, v = kv.split("=")
        for c in [c for c, n in sc.classes.items() if n == k.strip()]:
            tv[c] = float(v)

    trav_pt = np.array([tv.get(int(c), a.unknown) for c in L], float)
    lo = P[:, :2].min(0); shape = np.ceil((P[:, :2].max(0) - lo) / a.cell).astype(int) + 1
    tg = grid(P, trav_pt, a.cell, lo, shape)
    if np.isnan(tg).any():                       # holes between map points at fine cells: fill from neighbours
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(np.isnan(tg), return_distances=False, return_indices=True)
        tg = tg[tuple(idx)]
    t = np.clip(tg, 0, 1)                         # t[i, j]: i = x index, j = y index

    img = np.flipud(t.T)                          # image rows = y (top = max y), cols = x
    pix = np.round(img * 255).astype(np.uint8)
    out = Path(a.out) if a.out else Path(a.scene) / "nav2"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "map.pgm", "wb") as f:
        f.write(f"P5\n{pix.shape[1]} {pix.shape[0]}\n255\n".encode()); f.write(pix.tobytes())
    (out / "map.yaml").write_text(
        f"image: map.pgm\nmode: scale\nresolution: {a.cell}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\n"
        f"negate: 0\noccupied_thresh: 0.95\nfree_thresh: 0.0\n")
    used = {sc.classes[c]: tv[c] for c in np.unique(L).astype(int) if c in sc.classes}
    print(f"traversability: {used}")
    print(f"map {pix.shape[1]}x{pix.shape[0]} px at {a.cell} m, origin ({lo[0]:.2f}, {lo[1]:.2f}), "
          f"{(t < 0.05).mean():.1%} walls")
    print("saved", out / "map.yaml")


if __name__ == "__main__":
    main()
