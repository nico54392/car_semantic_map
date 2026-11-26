#!/usr/bin/env python3
"""
Path planning on the PERFECT map: cost map and goal straight from the Unity labels, no AI.
Use it to test planning + driving on their own. If the robot drives these paths fine, any later
problem with the model-built map is perception, not the planner or the follower.

python tools/oracle_plan.py <scene> --start 10 -10 --goal 30 -40     # goal by coordinates
python tools/oracle_plan.py <scene> --start 10 -10 --to dirt         # goal by label name ("go to the dirt")
python tools/oracle_plan.py <scene> --start 10 -10 --to bench        # an object: stops --standoff m in front of it

Cost map: every map point gets its class's traversability from config/ontology.json (by name),
averaged per --cell xy cell (same rule as feature_map.py plan / Jaime's Go2 cost map).
void (0) and classes missing from the ontology count as unknown (--unknown).
Goal by label: blobs of that class -> pick (nearest / largest) ->
  drivable class (traversability >= --min_trav): goal = nearest reachable cell ON the blob
  obstacle class (bench, rock, tree…):          goal = nearest free cell on a ring --standoff m around it
Writes <scene>/goals/oracle_<name>/path.json in the same format as goal_from_prompt.py, so
ros2_follow_path.py drives it unchanged. Plus path.csv and goal.png.
Coordinates: map frame = ROS FLU from the Unity export. Unity (x,y,z) = (-y_ros, z_ros, x_ros).
"""
import argparse, heapq, json, re, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.scene_io import load_scene
from car_semantic_map.evaluate import load_ontology


def grid(points, values, cell, lo, shape):
    ij = np.floor((points[:, :2] - lo) / cell).astype(int)
    ok = ~np.isnan(values)
    s = np.zeros(shape); n = np.zeros(shape)
    np.add.at(s, (ij[ok, 0], ij[ok, 1]), values[ok]); np.add.at(n, (ij[ok, 0], ij[ok, 1]), 1)
    g = np.full(shape, np.nan); g[n > 0] = s[n > 0] / n[n > 0]
    return g


def astar(t, start, goal, block_below=0.05, w=4.0):
    """8-connected A*, step cost = length * (1 + w*(1-trav)); cells below block_below are walls."""
    H, W = t.shape
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    g = {start: 0.0}; parent = {}
    h = lambda c: np.hypot(c[0] - goal[0], c[1] - goal[1])
    pq = [(h(start), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]; path.append(cur)
            return path[::-1], g[goal]
        for di, dj, Lm in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or t[nb] < block_below:
                continue
            ng = g[cur] + Lm * (1 + w * (1 - t[nb]))
            if ng < g.get(nb, np.inf):
                g[nb] = ng; parent[nb] = cur
                heapq.heappush(pq, (ng + h(nb), nb))
    return None, np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--start", type=float, nargs=2, required=True, help="robot start x y (map frame, ROS FLU)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--goal", type=float, nargs=2, help="goal x y (map frame)")
    g.add_argument("--to", help="class name from the scene's labels, e.g. dirt, bench")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--pick", choices=["nearest", "largest"], default="nearest")
    ap.add_argument("--standoff", type=float, default=1.0)
    ap.add_argument("--min_trav", type=float, default=0.3)
    ap.add_argument("--unknown", type=float, default=0.5, help="traversability for void / unknown classes")
    ap.add_argument("--ontology", default=str(ROOT / "config" / "ontology.json"))
    a = ap.parse_args()

    sc = load_scene(a.scene, max_views=1)
    P, L = sc.points, sc.point_labels
    onto = load_ontology(Path(a.ontology))
    tv = {}
    for cid, name in sc.classes.items():
        t = onto.get(name, {}).get("traversability") if cid != 0 else None
        if t is None and cid != 0 and name not in onto:
            print(f"  class '{name}' not in the ontology -> traversability {a.unknown}")
        tv[cid] = a.unknown if t is None else float(t)
    present = {sc.classes.get(int(c), f"id{c}"): int(n) for c, n in zip(*np.unique(L, return_counts=True))}
    print(f"{len(P)} map points, labels: {present}")
    print("traversability:", {sc.classes[c]: v for c, v in tv.items() if sc.classes[c] in present})

    trav_pt = np.array([tv.get(int(c), a.unknown) for c in L], float)
    lo = P[:, :2].min(0); shape = np.ceil((P[:, :2].max(0) - lo) / a.cell).astype(int) + 1
    tg = grid(P, trav_pt, a.cell, lo, shape)
    zg = grid(P, P[:, 2].astype(float), a.cell, lo, shape)
    t = np.where(np.isnan(tg), a.unknown, tg)
    print(f"cost map {shape[0]}x{shape[1]} cells of {a.cell} m, x {lo[0]:.1f}..{lo[0] + shape[0] * a.cell:.1f}, "
          f"y {lo[1]:.1f}..{lo[1] + shape[1] * a.cell:.1f}, {(t < 0.05).sum()} wall cells")

    cell_of = lambda xy: tuple(np.clip(np.floor((np.asarray(xy) - lo) / a.cell).astype(int), 0, shape - 1))
    s_xy = np.array(a.start); start = cell_of(s_xy)
    if t[start] < 0.05:
        sys.exit(f"start {tuple(s_xy)} is inside an obstacle cell, pick another")

    if a.goal:
        name = f"xy_{a.goal[0]:g}_{a.goal[1]:g}"
        cands = [cell_of(a.goal)]
    else:
        name = a.to.strip().lower()
        ids = [c for c, n in sc.classes.items() if n == name]
        if not ids:
            sys.exit(f"'{name}' is not a class in this scene. Classes: {list(present)}")
        mask_pt = np.isin(L, ids)
        if not mask_pt.any():
            sys.exit(f"no map points labelled '{name}' in this scene")
        occ = np.zeros(shape, bool); ij = np.floor((P[mask_pt, :2] - lo) / a.cell).astype(int)
        occ[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        blobs = [k + 1 for k in range(n) if sizes[k] >= 2]
        cent = {k: np.argwhere(lab == k).mean(0) for k in blobs}
        dist = {k: np.linalg.norm(cent[k] - np.array(start)) for k in blobs}
        k = min(blobs, key=dist.get) if a.pick == "nearest" else max(blobs, key=lambda k: sizes[k - 1])
        blob = lab == k
        drivable = tv[ids[0]] >= a.min_trav
        if drivable:
            ring = blob & (t >= a.min_trav)                     # drive onto it
        else:
            d = ndimage.distance_transform_edt(~blob) * a.cell
            ring = (d > 0) & (d <= a.standoff + a.cell / 2) & (t >= a.min_trav)
        cand = np.argwhere(ring)
        if not len(cand):
            sys.exit("no free goal cell; try --standoff / --min_trav")
        order = np.argsort(np.linalg.norm(cand - np.array(start), axis=1))
        cands = [tuple(cand[i]) for i in order[:30]]
        print(f"'{name}': {len(blobs)} blob(s), picked #{k} ({a.pick}, {int(sizes[k - 1])} cells), "
              f"{'drivable -> goal on it' if drivable else f'obstacle -> goal {a.standoff} m in front'}")

    path = None
    for goal in cands:
        path, cost = astar(t, start, goal)
        if path:
            break
    if not path:
        sys.exit("A* found no route (goal walled off or outside the map)")

    cells = np.array(path); xy = lo + (cells + 0.5) * a.cell
    z = np.array([zg[tuple(c)] for c in cells]); z = np.where(np.isnan(z), np.nanmean(zg), z)
    ros = np.c_[xy, z]; unity = np.c_[-ros[:, 1], ros[:, 2], ros[:, 0]]
    length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    slug = "oracle_" + re.sub(r"[^a-z0-9]+", "_", name)[:40]
    out = Path(a.scene) / "goals" / slug; out.mkdir(parents=True, exist_ok=True)
    res = dict(prompt=f"oracle: {name}", query=name, source="oracle (Unity labels, no model)", cell=a.cell,
               start_ros=list(map(float, s_xy)), goal_ros=list(map(float, ros[-1])), goal_unity=list(map(float, unity[-1])),
               path_cost=float(cost), path_length_m=length,
               mean_trav_on_path=float(np.mean([t[tuple(c)] for c in cells])),
               frames="ros = map frame (FLU, x fwd, y left, z up) from the Unity export; unity = (-y, z, x)",
               waypoints_ros=ros.round(3).tolist(), waypoints_unity=unity.round(3).tolist())
    json.dump(res, open(out / "path.json", "w"), indent=1)
    np.savetxt(out / "path.csv", ros, fmt="%.3f", delimiter=",", header="x,y,z (ROS map frame)", comments="")
    print(f"path: {len(path)} cells, {length:.1f} m, mean trav {res['mean_trav_on_path']:.2f}")
    print(f"goal ROS ({ros[-1][0]:.2f}, {ros[-1][1]:.2f}) | Unity ({unity[-1][0]:.2f}, {unity[-1][1]:.2f}, {unity[-1][2]:.2f})")
    print("saved", out / "path.json")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        col = {c: np.array(onto.get(nm, {}).get("color", rng.integers(40, 255, 3))) / 255 for c, nm in sc.classes.items()}
        col[0] = np.array([.6, .6, .6])
        # top-down in Unity-style view: right = -y_ros, up = x_ros
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sub = np.random.default_rng(1).choice(len(P), min(len(P), 120000), replace=False)
        ax[0].scatter(-P[sub, 1], P[sub, 0], c=np.stack([col.get(int(c), col[0]) for c in L[sub]]), s=1, marker="s")
        ax[0].set_title("ground-truth labels (Unity)")
        ext = [-(lo[1] + shape[1] * a.cell), -lo[1], lo[0], lo[0] + shape[0] * a.cell]
        im = ax[1].imshow(np.fliplr(t), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax[1], fraction=0.046, label="traversability")
        ax[1].set_title(f"cost map + path to '{name}' ({length:.1f} m)")
        for axx in ax:
            axx.plot(-xy[:, 1], xy[:, 0], "b-", lw=2)
            axx.plot(-s_xy[1], s_xy[0], "wo", ms=9, mec="k"); axx.plot(-xy[-1, 1], xy[-1, 0], "r*", ms=15, mec="k")
            axx.set_aspect("equal"); axx.set_xlabel("-y (m)"); axx.set_ylabel("x (m)")
        fig.tight_layout(); fig.savefig(out / "goal.png", dpi=110)
        print("saved", out / "goal.png")
    except Exception as e:
        print("(no picture:", e, ")")


if __name__ == "__main__":
    main()
