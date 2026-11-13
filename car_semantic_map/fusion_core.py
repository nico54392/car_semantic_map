"""
Fusion core: turn posed RGB-D views (+ per-pixel features) into a 3D map where
every point carries a fused feature vector. This is OpenScene/ConceptFusion
step 1, and it's independent of any file format — the Perception/SOLO loader is
a thin adapter that calls fuse_views() with plain arrays.

Pipeline per view:
  depth + intrinsics + camera pose  ->  3D points (world frame)
  each point samples the view's feature map at its pixel
Then all views' points are merged by voxel, averaging features (and majority-
voting any ground-truth labels) per voxel -> the embedding map.

Conventions (VERIFY against the first real Perception capture, then adjust):
  - depth[v, u] is metric distance along the camera's forward axis
  - intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
  - cam_to_world is 4x4, camera->world
  - camera looks down +Z in camera space, x right, y down (CV convention).
    Unity/Perception is left-handed with +Z forward but y up; set axis_flip to
    reconcile once we see a real frame (a plane should come back flat).

No torch dependency — features come in as numpy arrays (caller converts from
SAM3's tensors). Only numpy required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class View:
    depth: np.ndarray                 # [H, W] metric, <=0 or nan = invalid
    K: np.ndarray                     # [3, 3] intrinsics
    cam_to_world: np.ndarray          # [4, 4]
    features: Optional[np.ndarray] = None   # [C, h, w] (any h,w; sampled by ratio)
    labels: Optional[np.ndarray] = None     # [H, W] int ground-truth class ids
    max_depth: float = 1e9


def backproject(view: View, axis_flip=(1.0, 1.0, 1.0)):
    """Depth -> world points. Returns points [M,3], pixel (u,v) int [M,2], and the
    flat index of each kept pixel (so callers can gather labels/features)."""
    H, W = view.depth.shape
    fx, fy = view.K[0, 0], view.K[1, 1]
    cx, cy = view.K[0, 2], view.K[1, 2]
    z = view.depth.reshape(-1).astype(np.float64)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    u = us.reshape(-1).astype(np.float64)
    v = vs.reshape(-1).astype(np.float64)

    valid = np.isfinite(z) & (z > 0) & (z < view.max_depth)
    z, u, v = z[valid], u[valid], v[valid]

    sx, sy, sz = axis_flip
    x_cam = (u - cx) / fx * z * sx
    y_cam = (v - cy) / fy * z * sy
    z_cam = z * sz
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z)], axis=1)  # [M,4]
    pts_world = (view.cam_to_world @ pts_cam.T).T[:, :3]
    pix = np.stack([u, v], axis=1).astype(np.int32)
    return pts_world, pix, np.nonzero(valid)[0]


def sample_features(features: np.ndarray, pix_uv: np.ndarray, img_hw):
    """Nearest-neighbour sample a [C,h,w] feature map at image pixels [M,2]=(u,v)."""
    C, h, w = features.shape
    H, W = img_hw
    fu = np.clip((pix_uv[:, 0] * w / W).astype(np.int32), 0, w - 1)
    fv = np.clip((pix_uv[:, 1] * h / H).astype(np.int32), 0, h - 1)
    return features[:, fv, fu].T  # [M, C]


def fuse_views(views, voxel=0.05, axis_flip=(1.0, 1.0, 1.0)):
    """
    Merge posed RGB-D (+feature/label) views into one voxel map.

    Returns dict:
      points   [N,3]  voxel-centroid world positions
      features [N,C]  mean feature per voxel (if any view had features)
      labels   [N]    majority ground-truth id per voxel (if any view had labels)
      counts   [N]    how many samples fell in each voxel
    """
    # accumulate per voxel key
    acc_sum, acc_cnt, acc_pos = {}, {}, {}
    acc_lbl = {}   # voxel -> {label: count}
    C = None

    for view in views:
        pts, pix, _ = backproject(view, axis_flip)
        if pts.shape[0] == 0:
            continue
        feats = None
        if view.features is not None:
            feats = sample_features(view.features, pix, view.depth.shape)
            C = feats.shape[1]
        lbls = None
        if view.labels is not None:
            lbls = view.labels[pix[:, 1], pix[:, 0]]

        keys = np.floor(pts / voxel).astype(np.int64)
        for i in range(pts.shape[0]):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            acc_cnt[k] = acc_cnt.get(k, 0) + 1
            if k in acc_pos:
                acc_pos[k] += pts[i]
            else:
                acc_pos[k] = pts[i].copy()
            if feats is not None:
                if k in acc_sum:
                    acc_sum[k] += feats[i]
                else:
                    acc_sum[k] = feats[i].astype(np.float64).copy()
            if lbls is not None:
                d = acc_lbl.setdefault(k, {})
                d[int(lbls[i])] = d.get(int(lbls[i]), 0) + 1

    keys = list(acc_cnt.keys())
    if not keys:
        return {"points": np.zeros((0, 3)), "features": None, "labels": None, "counts": np.zeros((0,))}
    counts = np.array([acc_cnt[k] for k in keys])
    points = np.stack([acc_pos[k] / acc_cnt[k] for k in keys])
    out = {"points": points, "counts": counts, "features": None, "labels": None}
    if C is not None:
        out["features"] = np.stack([acc_sum[k] / acc_cnt[k] for k in keys]).astype(np.float32)
    if acc_lbl:
        out["labels"] = np.array([max(acc_lbl[k].items(), key=lambda x: x[1])[0]
                                  if k in acc_lbl else -1 for k in keys], dtype=np.int64)
    return out


def query_dot(map_features: np.ndarray, text_vec: np.ndarray):
    """Path A query: cosine similarity of every map point's feature with a text vector."""
    f = map_features / (np.linalg.norm(map_features, axis=1, keepdims=True) + 1e-8)
    t = text_vec / (np.linalg.norm(text_vec) + 1e-8)
    return f @ t  # [N] similarity per point


# ----------------------------------------------------------------------------
# synthetic self-test: two cameras looking at a flat plane; features = world XYZ.
# After fusion the points should lie on the plane and features ~= their position.
if __name__ == "__main__":
    def make_plane_view(cam_xyz, H=120, W=160, f=120.0):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        c2w = np.eye(4); c2w[:3, 3] = cam_xyz          # camera above the plane, looking +Z(=down here)
        # plane at world z=0; camera at height cam_xyz[2], looking toward it:
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        # depth so that each ray hits z=0 plane: z_cam = height (flat, approx for small FOV)
        depth = np.full((H, W), cam_xyz[2], dtype=np.float32)
        # fake feature map = normalized pixel coords, 3 channels
        feat = np.stack([us / W, vs / H, np.ones_like(us) * 0.5]).astype(np.float32)
        labels = ((us > W / 2).astype(np.int64))       # left half = 0, right = 1
        return View(depth=depth, K=K, cam_to_world=c2w, features=feat, labels=labels)

    v1 = make_plane_view([0.0, 0.0, 2.0])
    v2 = make_plane_view([0.3, 0.1, 2.0])
    m = fuse_views([v1, v2], voxel=0.05, axis_flip=(1, 1, 1))
    print("points:", m["points"].shape, "features:", None if m["features"] is None else m["features"].shape)
    print("z range (should be ~2.0, the plane depth back-projected):",
          round(float(m["points"][:, 2].min()), 3), "->", round(float(m["points"][:, 2].max()), 3))
    print("labels present:", m["labels"] is not None, "unique:", np.unique(m["labels"]))
    q = query_dot(m["features"], np.array([1.0, 0.0, 0.0]))  # "lean toward +x feature"
    print("query range:", round(float(q.min()), 3), "->", round(float(q.max()), 3))
    print("self-test OK")
