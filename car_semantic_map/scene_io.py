"""Read the scene format written by unity/SceneCapture.cs (or make_synthetic.py).

scene/
  scene.json                 classes {id: name} + frame conventions
  map.ply                    ascii: x y z label  (world = ROS FLU, meters)
  views/000000_rgb.png
  views/000000_depth.bin     float32 HxW z-depth in meters, 0 = no hit
  views/000000_label.png     class id in R channel
  views/000000.json          width height fx fy cx cy T_world_cam (row-major, OpenCV cam -> world)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class View:
    name: str
    rgb: np.ndarray        # HxWx3 uint8
    depth: np.ndarray      # HxW float32, meters
    label: np.ndarray      # HxW int
    K: np.ndarray          # 3x3
    T_wc: np.ndarray       # 4x4 camera -> world


@dataclass
class Scene:
    root: Path
    classes: dict          # id -> name
    points: np.ndarray     # Nx3
    point_labels: np.ndarray
    views: list


def read_ply_ascii(path):
    with open(path) as f:
        n = 0
        while True:
            line = f.readline()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, max_rows=n, ndmin=2) if n else np.zeros((0, 4))
    return data[:, :3].astype(np.float32), data[:, 3].astype(np.int64)


def load_view(stem: Path) -> View:
    meta = json.loads(stem.with_suffix(".json").read_text())
    W, H = meta["width"], meta["height"]
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], dtype=np.float64)
    T = np.array(meta["T_world_cam"], dtype=np.float64).reshape(4, 4)
    rgb = np.array(Image.open(f"{stem}_rgb.png").convert("RGB"))
    depth = np.fromfile(f"{stem}_depth.bin", dtype="<f4").reshape(H, W)
    lab = np.array(Image.open(f"{stem}_label.png"))
    lab = (lab[..., 0] if lab.ndim == 3 else lab).astype(np.int64)
    return View(stem.name, rgb, depth, lab, K, T)


def load_scene(root, max_views=None, need_map=True) -> Scene:
    root = Path(root)
    meta = json.loads((root / "scene.json").read_text())
    classes = {int(k): v for k, v in meta["classes"].items()}
    if need_map or (root / "map.ply").exists():
        pts, lab = read_ply_ascii(root / "map.ply")
    else:
        pts, lab = np.zeros((0, 3), np.float32), np.zeros(0, np.int64)
    stems = sorted(p.with_suffix("") for p in (root / "views").glob("*.json"))
    if max_views:
        stems = stems[:max_views]
    return Scene(root, classes, pts, lab, [load_view(s) for s in stems])


def project(points, view: View, depth_tol_abs=0.15, depth_tol_rel=0.03, border=2):
    """OpenScene's PointCloudToImageMapper, with an absolute+relative depth tolerance
    instead of their 25%-of-depth one (too loose outdoors: 5 m at 20 m range).

    returns u, v (float pixel coords), visible mask
    """
    H, W = view.depth.shape
    T_cw = np.linalg.inv(view.T_wc)
    pc = points @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = view.K[0, 0] * pc[:, 0] / z + view.K[0, 2]
        v = view.K[1, 1] * pc[:, 1] / z + view.K[1, 2]
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inside = (z > 0.05) & (ui >= border) & (ui < W - border) & (vi >= border) & (vi < H - border)
    vis = inside.copy()
    d = view.depth[vi[inside], ui[inside]]
    vis[inside] = (d > 0) & (np.abs(d - z[inside]) <= depth_tol_abs + depth_tol_rel * d)
    return u, v, vis


def check_conventions(scene: Scene, verbose=True):
    """Project the labeled map into every view and compare against the label images.
    If poses/intrinsics/axis conventions are right this is ~95-100%. If it's low, the
    export is broken -- fix that before comparing any methods."""
    agree, total, vis_frac = 0, 0, []
    for view in scene.views:
        u, v, vis = project(scene.points, view)
        ui, vi = np.round(u[vis]).astype(int), np.round(v[vis]).astype(int)
        img_lab = view.label[vi, ui]
        agree += int((img_lab == scene.point_labels[vis]).sum())
        total += int(vis.sum())
        vis_frac.append(vis.mean())
    rate = agree / max(total, 1)
    if verbose:
        print(f"convention check: {rate:.1%} of {total} point-pixel pairs agree on label "
              f"(avg {np.mean(vis_frac):.1%} of map points visible per view)")
        if rate < 0.9 or np.mean(vis_frac) < 0.005:
            print("  !! looks broken. typical causes: pose not converted to ROS/OpenCV, y axis not flipped,\n"
                  "     world->cam saved instead of cam->world, horizontal fov used for fy, label png flipped")
        else:
            print("  OK")
    return rate
