#!/usr/bin/env python3
"""Perception -> control hand-off: voxel map (.npz from `rellis.py buildmap`) -> 2D grid in the
car_semantic_map/control_grid.py format (stored [iy, ix] with axes="yx"; control_grid.load transposes it).

Grid layers (all HxW, row = y, col = x, cell (0,0) at `origin`):
  trav_mean, trav_std   expected traversability under the class probabilities, and its spread
  mu_mean, mu_std       same for friction
  probs                 (H,W,K) class probabilities, names in `class_names`
  height                min z of the cell (ground estimate)
  step                  tallest point within --clearance of the ground minus ground (canopy ignored)
  step_rigid            same, counting only rigid classes (tree, log, rock...) and unseen voxels; grass/bush height ignored
  slope                 ground slope in degrees, from the gradient of `height`
  seen                  1 if any camera saw the cell
  gt_trav               traversability from the GT labels (for scoring only)

  python to_grid.py out/map/rellis_map_*.npz --res 0.5 [--plan]
--plan runs 8-connected A* across the map (cost = 1/trav, blocked below --block) and saves a PNG.
Swap --ontology to change traversability/mu values without rebuilding the map.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os

import numpy as np


def build_grid(m, res, trav, mu, rigid, min_pts=2, clearance=1.0, unknown_rigid=True):
    xyz, probs = m["xyz"], m["probs"].astype(np.float32)
    seen = m["seen"] > 0
    lo = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - lo) / res).astype(int)
    W, H = ij.max(0) + 1
    flat = ij[:, 1] * W + ij[:, 0]
    n = H * W
    cnt = np.bincount(flat, minlength=n)
    zmin = np.full(n, np.inf); np.minimum.at(zmin, flat, xyz[:, 2])
    # step = tallest thing within `clearance` of the ground; canopy the robot drives under doesn't count
    low = xyz[:, 2] <= zmin[flat] + clearance
    zmax = np.full(n, -np.inf); np.maximum.at(zmax, flat[low], xyz[low, 2])
    # semantic step: only rigid classes (tree, log, rock...), plus unknown voxels if unknown_rigid
    # (pred: no camera saw it -> assume rigid; gt: void = nobody annotated it -> don't)
    p_rigid = probs @ rigid.astype(np.float32)
    hard = low & ((p_rigid > 0.5) | (~seen if unknown_rigid else False))
    zr = np.full(n, -np.inf); np.maximum.at(zr, flat[hard], xyz[hard, 2])
    s = seen.astype(float)
    ns = np.bincount(flat, s, n)
    K = probs.shape[1]
    P = np.zeros((n, K))
    for k in range(K):
        P[:, k] = np.bincount(flat, probs[:, k] * s, n)
    P /= np.maximum(ns, 1)[:, None]
    tm, mm = P @ trav, P @ mu
    ts = np.sqrt(np.maximum(P @ trav ** 2 - tm ** 2, 0))
    ms = np.sqrt(np.maximum(P @ mu ** 2 - mm ** 2, 0))
    ok = cnt >= min_pts
    seen_c = (ns > 0) & ok
    height = np.where(ok, zmin, np.nan).reshape(H, W)
    gy, gx = np.gradient(np.where(np.isnan(height), np.nanmedian(height), height), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # GT traversability per cell from voxel GT labels mapped through the ontology
    gt = m["gt"]
    names = list(m["ont_names"])
    cls_names = list(m["class_names"])
    t_by_eval = np.full(len(cls_names), np.nan)
    for j, nm in enumerate(names):
        if nm in cls_names:
            t_by_eval[cls_names.index(nm)] = trav[j]
    tg = t_by_eval[gt]
    v = ~np.isnan(tg)
    gt_tr = np.bincount(flat[v], tg[v], n) / np.maximum(np.bincount(flat[v], None, n), 1)
    gt_tr[np.bincount(flat[v], None, n) == 0] = np.nan
    R = lambda a: a.reshape(H, W)
    nanify = lambda a: np.where(R(seen_c), R(a), np.nan)
    return dict(origin=lo, res=res, axes="yx", trav_mean=nanify(tm), trav_std=nanify(ts), mu_mean=nanify(mm),
                mu_std=nanify(ms), probs=P.reshape(H, W, K).astype(np.float16), height=height,
                step=np.where(ok, zmax - zmin, np.nan).reshape(H, W), slope=slope,
                step_rigid=np.where(ok, np.maximum(zr - zmin, 0), np.nan).reshape(H, W),
                seen=R(seen_c).astype(np.uint8), gt_trav=np.where(R(ok), R(gt_tr), np.nan),
                class_names=np.array(names))


def astar(cost, start, goal):
    H, W = cost.shape
    g = np.full((H, W), np.inf); g[start] = 0
    prev = {}
    pq = [(0.0, start)]
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    h = lambda p: np.hypot(p[0] - goal[0], p[1] - goal[1])
    while pq:
        _, p = heapq.heappop(pq)
        if p == goal:
            break
        for dy, dx in nb:
            q = (p[0] + dy, p[1] + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and np.isfinite(cost[q]):
                ng = g[p] + np.hypot(dy, dx) * 0.5 * (cost[p] + cost[q])
                if ng < g[q]:
                    g[q] = ng; prev[q] = p
                    heapq.heappush(pq, (ng + h(q), q))
    if goal not in prev and goal != start:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_npz")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--ontology", help="override traversability/mu (same JSON as rellis.py)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--block", type=float, default=0.2, help="cells with trav below this are obstacles")
    ap.add_argument("--max-step", type=float, default=0.4, help="m; taller cells are obstacles")
    ap.add_argument("--risk", type=float, default=1.0, help="weight on trav_std (cautious planning)")
    ap.add_argument("--clearance", type=float, default=1.0, help="m above ground that counts for `step` (robot height + margin)")
    ap.add_argument("--unknown-rigid", default="auto", choices=["auto", "yes", "no"],
                    help="do unlabelled/unseen voxels block by height? auto: yes for pred, no for gt")
    ap.add_argument("--geometric", action="store_true", help="old rule: any height blocks, whatever the class")
    ap.add_argument("--source", default="pred", choices=["pred", "gt"],
                    help="pred = fused camera map; gt = LiDAR labels directly (clean map for controls)")
    ap.add_argument("--unknown-trav", type=float, default=0.5, help="traversability of cells no camera saw")
    a = ap.parse_args()
    m = np.load(a.map_npz)
    trav, mu = m["ont_trav"].astype(float), m["ont_mu"].astype(float)
    names = list(m["ont_names"])
    from ontology import Ontology, default_path
    oc = {c.name: c for c in Ontology.load(a.ontology or default_path()).classes}
    rigid = np.array([oc[n].rigid if n in oc else True for n in names])          # unknown class: assume rigid
    colors = np.array([oc[n].color if n in oc else (128, 128, 128) for n in names], np.uint8)
    lut = {n: {"traversability": c.trav, "mu": c.mu} for n, c in oc.items()}
    if a.ontology:                                     # re-value an existing map without rebuilding it
        trav = np.array([lut[n]["traversability"] if n in lut else t for n, t in zip(names, trav)])
        mu = np.array([lut[n]["mu"] if n in lut else u for n, u in zip(names, mu)])
    if a.source == "gt":
        names = list(m["ont_names"]); cls_names = list(m["class_names"])
        oh = np.zeros((len(m["gt"]), len(names)), np.float32)
        for j, nm in enumerate(names):
            if nm in cls_names:
                oh[m["gt"] == cls_names.index(nm), j] = 1
        valid = oh.sum(1) > 0
        m = {k: m[k] for k in m.files}; m["probs"] = oh; m["seen"] = valid.astype(np.int32)   # every labelled voxel counts as seen
    unk_rigid = (a.source == "pred") if a.unknown_rigid == "auto" else (a.unknown_rigid == "yes")
    G = build_grid(m, a.res, trav, mu, rigid, clearance=a.clearance, unknown_rigid=unk_rigid)
    out = a.map_npz.replace(".npz", f"_grid{a.res}{'_gt' if a.source == 'gt' else ''}.npz")
    np.savez_compressed(out, **G)
    v = ~np.isnan(G["trav_mean"]) & ~np.isnan(G["gt_trav"])
    mae = np.abs(G["trav_mean"][v] - G["gt_trav"][v]).mean() if v.any() else float("nan")
    H, W = G["seen"].shape
    print(f"grid {H}x{W} @ {a.res} m, seen cells {G['seen'].mean():.2f}, trav MAE vs GT {mae:.3f} -> {out}")
    if not a.plan:
        return
    # cautious cost. unseen cells are unknown (--unknown-trav), not walls; tall cells are always obstacles
    unk = a.unknown_trav
    t_eff = G["trav_mean"] - a.risk * G["trav_std"]
    t_eff = np.where(np.isnan(t_eff), unk, t_eff)
    t_gt = np.where(np.isnan(G["gt_trav"]), unk, G["gt_trav"])
    step_used = G["step"] if a.geometric else G["step_rigid"]
    tall = np.nan_to_num(step_used, nan=0.0) > a.max_step
    cost = np.where((t_eff < a.block) | tall, np.inf, 1.0 / np.maximum(t_eff, 1e-3))
    gcost = np.where((t_gt < a.block) | tall, np.inf, 1.0 / np.maximum(t_gt, 1e-3))
    H, W = cost.shape
    to_cell = lambda xy: (int(np.clip((xy[1] - G["origin"][1]) / a.res, 0, H - 1)),
                          int(np.clip((xy[0] - G["origin"][0]) / a.res, 0, W - 1)))
    def nearest_free(c):
        free = np.argwhere(np.isfinite(cost) & np.isfinite(gcost))
        return tuple(free[np.argmin(np.hypot(*(free - c).T))]) if len(free) else None
    if "traj" in m and len(m["traj"]) > 1:      # start/goal = where the robot actually drove
        start, goal = nearest_free(to_cell(m["traj"][0])), nearest_free(to_cell(m["traj"][-1]))
        traj_cells = np.array([to_cell(p) for p in m["traj"]])
    else:
        from scipy import ndimage
        lab, n = ndimage.label(np.isfinite(cost) & np.isfinite(gcost), structure=np.ones((3, 3)))
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        free = np.argwhere(lab == big)
        start = tuple(free[0]); goal = tuple(free[np.argmax(np.hypot(*(free - start).T))])
        traj_cells = None
    if start is None or goal is None:
        print("no free cells to plan on"); return
    if traj_cells is not None:                         # why is the driven route blocked?
        rc = np.unique(traj_cells, axis=0)
        rr, cc = rc[:, 0], rc[:, 1]
        why = {"rigid obstacle" if not a.geometric else "tall step": tall[rr, cc], "low trav (pred)": (t_eff[rr, cc] < a.block) & ~tall[rr, cc],
               "unseen": np.isnan(G["trav_mean"][rr, cc])}
        print(f"driven route: {len(rc)} cells, blocked for pred {np.mean(~np.isfinite(cost[rr, cc])):.0%} "
              "(" + ", ".join(f"{k} {v.mean():.0%}" for k, v in why.items()) + ")"
              f", median step on route {np.nanmedian(G['step'][rr, cc]):.2f} m")
    path, gpath = astar(cost, start, goal), astar(gcost, start, goal)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # class map: most likely class per cell (light grey = no camera / no label)
    P = G["probs"].astype(np.float32)
    cls_img = np.full(P.shape[:2] + (3,), 238, np.uint8)
    has = P.sum(-1) > 0
    cls_img[has] = colors[P[has].argmax(-1)]
    cls_img[np.isnan(G["height"])] = 255
    rig = np.nan_to_num(G["step_rigid"], nan=0) > a.max_step
    geo_only = (np.nan_to_num(G["step"], nan=0) > a.max_step) & ~rig
    obs = np.full(P.shape[:2] + (3,), 255, np.uint8)
    obs[~np.isnan(G["height"])] = 225
    obs[geo_only] = (255, 170, 60); obs[rig] = (200, 0, 0)
    fig, axs = plt.subplots(1, 4, figsize=(21, 6))
    panels = ((axs[0], cls_img, "classes (what the planner reads)", None),
              (axs[1], obs, f"obstacles: red = rigid > {a.max_step} m, orange = only vegetation height", None),
              (axs[2], G["trav_mean"], "traversability (from classes)", (0, 1)),
              (axs[3], G["trav_std"], "uncertainty (std)", (0, 0.5)))
    for axx, img, title, lim in panels:
        im = axx.imshow(img, origin="lower", cmap="viridis", vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        if traj_cells is not None:
            axx.plot(traj_cells[:, 1], traj_cells[:, 0], "c--", lw=1.2, label="driven")
        for pth, col, lab in ((path, "r", "plan (this map)"), (gpath, "w", "plan (LiDAR GT)")):
            if pth:
                pp = np.array(pth); axx.plot(pp[:, 1], pp[:, 0], col, lw=1.5, label=lab)
        axx.set_title(title, fontsize=9)
        if lim:
            plt.colorbar(im, ax=axx, fraction=0.04)
    present = [j for j in range(len(names)) if (P[has].argmax(-1) == j).any()] if has.any() else []
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=colors[j] / 255, label=names[j] + (" (rigid)" if rigid[j] else ""))
                           for j in present], fontsize=7, loc="upper right")
    axs[3].legend(loc="upper right", fontsize=7)
    png = out.replace(".npz", "_plan.png")
    plt.tight_layout(); plt.savefig(png, dpi=110)
    L = lambda p: sum(np.hypot(*np.subtract(p[i + 1], p[i])) for i in range(len(p) - 1)) * a.res if p else float("nan")
    print(f"A* start {tuple(map(int, start))} goal {tuple(map(int, goal))}: pred path {L(path):.1f} m, GT path {L(gpath):.1f} m -> {png}")


if __name__ == "__main__":
    main()
