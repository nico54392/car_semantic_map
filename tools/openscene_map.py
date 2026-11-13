"""
OpenScene-style embedded map -> text queries -> traversability cost map -> path.
This is "01 Map to Trajectory" with OpenScene as the semantic layer.

python tools/openscene_map.py build <scene>                 # CLIP features fused onto map.ply -> <scene>/embedded_map.npz
python tools/openscene_map.py build <scene> --from-depth    # no map.ply: build the map from the depth images (fusion_core)
python tools/openscene_map.py query <scene> "wet mud" "standing water"     # heatmaps on the map
python tools/openscene_map.py plan  <scene> --start x y --goal x y [--cell 0.2] [--classes grass mud ...]

--backbone clip | pe-b | pe-l (build only; query/plan use whatever the map was built with).
--weights: open_clip tag or .pt path, default per backbone (laion400m_e32 / meta).

Coordinates are the map's world frame (ROS FLU, meters). The map file stores one fp16
512-d CLIP embedding per point, so any new class or query is just a dot product, no re-run
over the images.
"""
import argparse, heapq, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.scene_io import load_scene
from car_semantic_map.evaluate import load_ontology

ONTOLOGY = ROOT / "config" / "ontology.json"


# ---------------------------------------------------------------- map

def build(scene_dir, weights, max_views=None, from_depth=False, voxel=0.1, backbone='clip', variant='plain'):
    from car_semantic_map.clip_features import make_extractor
    ext = make_extractor(backbone, weights, **({'variant': variant} if backbone != 'clip' else {}))
    if not from_depth:
        # OpenScene: project the given map (map.ply) into every view
        from car_semantic_map.openscene_fusion import fuse_features
        s = load_scene(scene_dir, max_views)
        feat, count, t = fuse_features(s, ext)
        pts, feat = s.points, feat.numpy()
    else:
        # no map given: back-project each view's depth and voxel-average (fusion_core)
        from car_semantic_map.fusion_core import View as FView, fuse_views
        s = load_scene(scene_dir, max_views, need_map=False)
        fviews, ts = [], []
        for v in s.views:
            t0 = time.time(); fmap = ext.image_features(v.rgb).numpy(); ts.append(time.time() - t0)
            fviews.append(FView(depth=v.depth, K=v.K, cam_to_world=v.T_wc, features=fmap, labels=v.label))
        m = fuse_views(fviews, voxel=voxel)
        nrm = np.linalg.norm(m["features"], axis=1, keepdims=True)
        pts, feat, count, t = m["points"], m["features"] / np.maximum(nrm, 1e-8), m["counts"], float(np.mean(ts))
        print(f"fusion_core: {len(pts)} voxels at {voxel} m from {len(s.views)} views")
    out = Path(scene_dir) / "embedded_map.npz"
    np.savez_compressed(out, points=pts, feat=feat.astype(np.float16), count=count,
                        backbone=backbone, variant=variant, sec_per_view=t)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


class EmbeddedMap:
    def __init__(self, scene_dir, weights, ontology=ONTOLOGY):
        d = np.load(Path(scene_dir) / "embedded_map.npz")
        backbone = str(d["backbone"]) if "backbone" in d else "clip"   # text side must match the map
        self.points, self.count = d["points"], d["count"]
        self.feat = torch.from_numpy(d["feat"].astype(np.float32))
        self.seen = self.count > 0
        self.onto = load_ontology(ontology)
        from car_semantic_map.clip_features import make_extractor
        self.enc = make_extractor(backbone, weights)
        print(f"embedded map built with {backbone}")

    def query(self, text):
        """cosine similarity of every point to a free-text query (NaN where never seen)"""
        q = self.enc.query_features([text])[0]
        s = (self.feat @ q).numpy()
        s[~self.seen] = np.nan
        return s

    def classify(self, class_names=None, temperature=100.0):
        """softmax over ontology classes. class score = best of its prompts (paper's 'addition').
        Returns names, probs (N,K)."""
        names = class_names or [n for n, c in self.onto.items() if c.get("traversability") is not None]
        Q, owner = self.enc.class_features([self.onto[n]["prompts"] for n in names])
        sims = self.feat @ Q.T
        logits = torch.stack([sims[:, torch.from_numpy(np.nonzero(owner == k)[0])].max(1).values
                              for k in range(len(names))], 1)
        p = (logits * temperature).softmax(1).numpy()
        p[~self.seen] = np.nan
        return names, p

    def properties(self, names, probs, key):
        """expected value + std of an ontology property (traversability, mu) under the class probs.
        Classes with null for that property are left out and the rest renormalized."""
        v = np.array([np.nan if self.onto[n].get(key) is None else self.onto[n][key] for n in names], float)
        ok = ~np.isnan(v)
        p = probs[:, ok]
        w = p.sum(1, keepdims=True)
        p = p / np.where(w > 0, w, 1)
        mean = (p * v[ok]).sum(1)
        std = np.sqrt((p * (v[ok] - mean[:, None]) ** 2).sum(1))
        mean[~self.seen], std[~self.seen] = np.nan, np.nan
        return mean, std


# ---------------------------------------------------------------- cost map + planner

def grid(points, values, cell, bounds=None):
    """mean of `values` per xy cell (the paper's cost-map rule). NaN = no data."""
    lo = points[:, :2].min(0) if bounds is None else bounds[0]
    hi = points[:, :2].max(0) if bounds is None else bounds[1]
    shape = np.ceil((hi - lo) / cell).astype(int) + 1
    ij = np.floor((points[:, :2] - lo) / cell).astype(int)
    ok = ~np.isnan(values)
    s = np.zeros(shape); n = np.zeros(shape)
    np.add.at(s, (ij[ok, 0], ij[ok, 1]), values[ok])
    np.add.at(n, (ij[ok, 0], ij[ok, 1]), 1)
    g = np.full(shape, np.nan)
    g[n > 0] = s[n > 0] / n[n > 0]
    return g, lo


def astar(trav, start, goal, unknown_cost=0.5, block_below=0.05, w=4.0):
    """8-connected A* on a traversability grid (1 = best). step cost = length * (1 + w*(1-trav)).
    Cells below block_below are walls; unknown cells get trav = unknown_cost."""
    t = np.where(np.isnan(trav), unknown_cost, trav)
    H, W = t.shape
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    g = {start: 0.0}; parent = {}
    h = lambda a: np.hypot(a[0] - goal[0], a[1] - goal[1])
    pq = [(h(start), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]; path.append(cur)
            return path[::-1], g[goal]
        for di, dj, L in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or t[nb] < block_below:
                continue
            ng = g[cur] + L * (1 + w * (1 - t[nb]))
            if ng < g.get(nb, np.inf):
                g[nb] = ng; parent[nb] = cur
                heapq.heappush(pq, (ng + h(nb), nb))
    return None, np.inf


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query", "plan"])
    ap.add_argument("scene")
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--backbone", default="clip", help="build only: clip | pe-b | pe-l | pe-s | pe-t")
    ap.add_argument("--variant", default="plain", help="build, PE only: plain | last_v | last_v_only")
    ap.add_argument("--weights", default=None, help="open_clip tag or .pt (default per backbone)")
    ap.add_argument("--from-depth", action="store_true", help="build: no map.ply, back-project depth (fusion_core)")
    ap.add_argument("--voxel", type=float, default=0.1, help="build --from-depth voxel size (m)")
    ap.add_argument("--ontology", default=str(ONTOLOGY))
    ap.add_argument("--max_views", type=int)
    ap.add_argument("--start", type=float, nargs=2)
    ap.add_argument("--goal", type=float, nargs=2)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--classes", nargs="*", help="restrict the ontology, e.g. --classes grass mud puddle bush")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    if a.cmd == "build":
        return build(a.scene, a.weights, a.max_views, a.from_depth, a.voxel, a.backbone, a.variant)

    m = EmbeddedMap(a.scene, a.weights, a.ontology)
    P = m.points
    X, Y = -P[:, 1], P[:, 0]                      # plot: right = -y, up = +x

    if a.cmd == "query":
        fig, ax = plt.subplots(1, len(a.texts), figsize=(5 * len(a.texts), 5), squeeze=False)
        for k, t in enumerate(a.texts):
            t0 = time.time(); s = m.query(t); dt = time.time() - t0
            sc = ax[0, k].scatter(X, Y, c=s, s=1, cmap="jet", marker="s")
            ax[0, k].set_title(f'"{t}"  ({1000 * dt:.0f} ms)'); ax[0, k].set_aspect("equal")
            plt.colorbar(sc, ax=ax[0, k], fraction=0.046)
        plt.tight_layout(); out = Path(a.scene) / "query.png"; plt.savefig(out, dpi=90)
        print("saved", out)
        return

    # plan
    names, probs = m.classify(a.classes)
    trav, trav_std = m.properties(names, probs, "traversability")
    mu, mu_std = m.properties(names, probs, "mu")
    tg, lo = grid(P, trav, a.cell)
    sg, _ = grid(P, trav_std, a.cell, (lo, P[:, :2].max(0)))
    cell_of = lambda xy: tuple(np.floor((np.array(xy) - lo) / a.cell).astype(int))
    start, goal = cell_of(a.start), cell_of(a.goal)
    path, cost = astar(tg, start, goal)
    # straight-line ("geometry only") path for reference, like the paper's red paths
    flat, _ = astar(np.where(np.isnan(tg), np.nan, 1.0), start, goal)

    res = dict(classes=names, path_cells=len(path) if path else None, path_cost=float(cost),
               mean_trav_on_path=float(np.nanmean([tg[c] for c in path])) if path else None,
               mean_trav_on_straight=float(np.nanmean([tg[c] for c in flat])) if flat else None)
    print(json.dumps(res, indent=2))
    np.savez_compressed(Path(a.scene) / "costmap.npz", trav=tg, trav_std=sg, origin=lo, cell=a.cell,
                        path=np.array(path) if path else np.zeros((0, 2)))

    ext = [lo[1], lo[1] + tg.shape[1] * a.cell, lo[0], lo[0] + tg.shape[0] * a.cell]   # y range, x range
    def show(axh, img, title, cmap, vmin=None, vmax=None):
        im = axh.imshow(img, origin="lower", extent=[-ext[1], -ext[0], ext[2], ext[3]], cmap=cmap,
                        vmin=vmin, vmax=vmax, aspect="equal")
        axh.set_title(title, fontsize=10); plt.colorbar(im, ax=axh, fraction=0.046)
    # imshow wants rows = x (up), cols = y; flip columns so right = -y
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.4))
    top = names[int(np.nanargmax(np.nanmean(probs, 0)))]
    lab = np.array(names)[np.nan_to_num(probs, nan=0).argmax(1)]
    for k, n in enumerate(names):
        mk = (lab == n) & m.seen
        if mk.any():
            ax[0].scatter(X[mk], Y[mk], s=1, marker="s", label=n, color=np.array(m.onto[n]["color"]) / 255)
    ax[0].legend(markerscale=8, fontsize=8, loc="upper right"); ax[0].set_aspect("equal"); ax[0].set_title("argmax class")
    show(ax[1], tg[:, ::-1], f"traversability (cell {a.cell} m)", "RdYlGn", 0, 1)
    show(ax[2], sg[:, ::-1], "traversability uncertainty (std)", "magma")
    show(ax[3], tg[:, ::-1], "A* path (green) vs geometry-only (red)", "RdYlGn", 0, 1)
    to_xy = lambda cells: (-(lo[1] + (np.array(cells)[:, 1] + 0.5) * a.cell), lo[0] + (np.array(cells)[:, 0] + 0.5) * a.cell)
    if flat: ax[3].plot(*to_xy(flat), "r-", lw=2)
    if path: ax[3].plot(*to_xy(path), "g-", lw=2.5)
    for axh in ax: axh.set_xlabel("-y [m]"); axh.set_ylabel("x [m]")
    plt.tight_layout(); out = Path(a.scene) / "plan.png"; plt.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
