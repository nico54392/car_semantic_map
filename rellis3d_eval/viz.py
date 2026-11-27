#!/usr/bin/env python3
"""Pictures of the data and the map.

  python viz.py out/rellis_map/rellis_map_00000_1001-1297.npz --root /data/rellis/Rellis-3D

writes next to the map (or --out):
  frames.png     a few frames: RGB | human image labels | LiDAR labels projected | LiDAR depth
  map_views.png  the voxel map from above and at an angle, coloured by GT / prediction / photo colour
  map3d.html     interactive 3D map (rotate, zoom, switch GT / pred / RGB), driven route on top.
                 needs `pip install plotly`; the file is self-contained, open it in any browser
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import rellis_io as rio
from ontology import Ontology

HERE = os.path.dirname(os.path.abspath(__file__))


def palette(ont):
    pal = np.full((len(rio.EVAL_CLASSES), 3), 40, np.uint8)
    for c in ont.classes:
        if c.eval_id is not None:
            pal[c.eval_id] = c.color
    return pal


def voxel_rgb(xyz, root, seq, frames, T, max_views=20, max_range=40.0):
    """mean photo colour per voxel from the camera views that see it"""
    calib = rio.load_calib(root, seq)
    idx = rio.index_sequence(root, seq)
    acc = np.zeros((len(xyz), 3)); cnt = np.zeros(len(xyz))
    use = frames[:: max(1, len(frames) // max_views)]
    for k in use:
        fr = idx[int(k)]
        img = rio.read_image(fr.image)
        calib.height, calib.width = img.shape[:2]
        p = (np.linalg.inv(T[int(k)]) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
        uv, z, v = rio.project(p, calib)
        v &= np.linalg.norm(p, axis=1) < max_range
        own = rio.read_scan(fr.scan)
        uvo, zo, vo = rio.project(own, calib)
        vis = rio.visibility_mask(np.r_[uv, uvo], np.r_[z, zo], np.r_[v, vo], calib)[: len(p)]
        u = uv[vis].astype(int)
        acc[vis] += img[u[:, 1], u[:, 0]]; cnt[vis] += 1
    rgb = np.full((len(xyz), 3), 128, np.uint8)
    m = cnt > 0
    rgb[m] = (acc[m] / cnt[m, None]).astype(np.uint8)
    return rgb, m


def fig_frames(path, root, seq, frames, pal, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = rio.index_sequence(root, seq)
    calib = rio.load_calib(root, seq)
    picks = [int(f) for f in np.array(frames)[np.linspace(0, len(frames) - 1, n).astype(int)]]
    fig, axs = plt.subplots(n, 4, figsize=(20, 3.3 * n))
    axs = np.atleast_2d(axs)
    for r, k in enumerate(picks):
        fr = idx[k]
        img = rio.read_image(fr.image)
        H, W = img.shape[:2]; calib.height, calib.width = H, W
        axs[r, 0].imshow(img); axs[r, 0].set_ylabel(f"frame {k}")
        if fr.image_label:
            lab = rio.read_image_label(fr.image_label)
            axs[r, 1].imshow((0.45 * img + 0.55 * pal[lab]).astype(np.uint8))
        else:
            axs[r, 1].imshow(img); axs[r, 1].text(20, 60, "no image label", color="w")
        p = rio.read_scan(fr.scan); l = rio.read_scan_label(fr.scan_label)
        uv, z, v = rio.project(p, calib)
        v = rio.visibility_mask(uv, z, v, calib)
        for ax, col, cmap in ((axs[r, 2], pal[l[v]] / 255, None), (axs[r, 3], z[v], "turbo")):
            ax.imshow((img * 0.6).astype(np.uint8))
            ax.scatter(uv[v, 0], uv[v, 1], s=1.5, c=col, cmap=cmap, vmin=0 if cmap else None,
                       vmax=30 if cmap else None)
    for c, t in enumerate(["camera", "human image labels", "LiDAR labels projected", "LiDAR depth (0-30 m)"]):
        axs[0, c].set_title(t)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()


def fig_map(path, xyz, cols, titles, traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cols)
    fig = plt.figure(figsize=(6.5 * n, 11))
    order = np.argsort(xyz[:, 2])
    for i, (c, t) in enumerate(zip(cols, titles)):
        ax = fig.add_subplot(2, n, i + 1)
        ax.scatter(xyz[order, 0], xyz[order, 1], c=c[order] / 255, s=0.4, linewidths=0)
        ax.plot(traj[:, 0], traj[:, 1], "c-", lw=2)
        ax.set_aspect("equal"); ax.set_title(f"{t} (top)"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax3 = fig.add_subplot(2, n, n + i + 1, projection="3d")
        ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c / 255, s=0.3, linewidths=0)
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], "c-", lw=2)
        rng = np.ptp(xyz[:, :2], 0).max() / 2; mid = xyz.mean(0)
        ax3.set_xlim(mid[0] - rng, mid[0] + rng); ax3.set_ylim(mid[1] - rng, mid[1] + rng)
        ax3.set_zlim(mid[2] - rng / 4, mid[2] + rng / 4); ax3.view_init(35, -60)
        ax3.set_title(f"{t} (3D)")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def html_map(path, xyz, cols, titles, traj, info):
    import plotly.graph_objects as go
    tr = []
    for c, t in zip(cols, titles):
        tr.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=t,
                               marker=dict(size=1.6, color=[f"rgb({a},{b},{d})" for a, b, d in c]),
                               visible=(t == titles[0]), hoverinfo="skip"))
    tr.append(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2] + 0.3, mode="lines",
                           name="driven route", line=dict(color="cyan", width=6)))
    btn = [dict(label=t, method="update",
                args=[{"visible": [j == i for j in range(len(titles))] + [True]}]) for i, t in enumerate(titles)]
    fig = go.Figure(tr)
    fig.update_layout(title=info, scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]",
                                             zaxis_title="z [m]", bgcolor="rgb(20,20,25)"),
                      updatemenus=[dict(buttons=btn, x=0.02, y=0.98, xanchor="left")],
                      paper_bgcolor="rgb(245,245,245)", margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_npz")
    ap.add_argument("--root", help="RELLIS root, needed for frames.png and photo colours")
    ap.add_argument("--out")
    ap.add_argument("--ontology", default=os.path.join(HERE, "ontology_rellis.json"))
    ap.add_argument("--n-points", type=int, default=150000, help="points drawn (subsampled)")
    ap.add_argument("--seen-only", action="store_true", help="only voxels a camera saw")
    a = ap.parse_args()
    out = a.out or os.path.dirname(os.path.abspath(a.map_npz))
    os.makedirs(out, exist_ok=True)
    m = np.load(a.map_npz)
    ont = Ontology.load(a.ontology)
    pal = palette(ont)
    xyz, gt, seen = m["xyz"], m["gt"], m["seen"] > 0
    probs = m["probs"].astype(np.float32)
    pred = np.where(seen, np.maximum(ont.eval_id[probs.argmax(1)], 0), 0)
    traj = m["traj"] if "traj" in m.files else xyz[:1]
    keep = np.where(seen)[0] if a.seen_only else np.arange(len(xyz))
    rng = np.random.default_rng(0)
    if len(keep) > a.n_points:
        keep = np.sort(rng.choice(keep, a.n_points, replace=False))
    P = xyz[keep]
    pred_col = pal[pred[keep]].copy()
    pred_col[~seen[keep]] = 70                        # unseen = grey
    cols, titles = [pal[gt[keep]], pred_col], ["GT (LiDAR labels)", "prediction (grey = no camera)"]
    have_frames = a.root and "frame_idx" in m.files
    if have_frames:
        seq, fidx = str(m["seq"]), m["frame_idx"]
        T = rio.read_poses(a.root, seq, str(m["pose_frame"]) if "pose_frame" in m.files else "lidar")
        rgb, has = voxel_rgb(P, a.root, seq, fidx, T)
        cols.append(rgb); titles.append("photo colour")
        fig_frames(os.path.join(out, "frames.png"), a.root, seq, fidx, pal)
        print("->", os.path.join(out, "frames.png"))
    elif a.root:
        print("map has no frame list (built with an older rellis.py) - skipping frames.png / photo colours")
    fig_map(os.path.join(out, "map_views.png"), P, cols, titles, traj)
    print("->", os.path.join(out, "map_views.png"))
    try:
        info = f"{os.path.basename(a.map_npz)}: {len(xyz):,} voxels, {seen.mean():.0%} seen by a camera, {len(P):,} drawn"
        html_map(os.path.join(out, "map3d.html"), P, cols, titles, traj, info)
        print("->", os.path.join(out, "map3d.html"), "(open in a browser)")
    except ImportError:
        print("pip install plotly for the interactive map3d.html")
    # legend for the class colours
    names = [rio.EVAL_CLASSES[e] for e in np.unique(np.r_[gt[keep], pred[keep]]) if e]
    print("colours:", ", ".join(f"{n} {tuple(int(v) for v in pal[rio.EVAL_CLASSES.index(n)])}" for n in names))


if __name__ == "__main__":
    main()
