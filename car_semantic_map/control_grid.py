"""
The perception -> control hand-off grid ("contract", proposed 2026-11-26 with the RELLIS tools).
Every map source (Unity GT, RELLIS oracle, SAM3, PE-B, memory...) writes this; Nav2 / planners read only this.

npz keys
  origin (2,)  x, y of cell [0, 0]'s corner, map frame (ROS FLU)     res  cell size, m
  axes         "xy": arrays are indexed [ix, iy]  (ix along x, iy along y)
  probs (X,Y,K) class probabilities per cell       class_names (K,)
  trav_mean, trav_std   expected traversability under probs, and its spread
  mu_mean, mu_std       same for friction (classes without a mu are left out; nan if none)
  height (min z), step (max - min z), slope (deg, from mean z)
  seen  bool, any data in the cell           gt_trav  for scoring only (nan if unknown)
"""
import numpy as np

KEYS = ["origin", "res", "axes", "probs", "class_names", "trav_mean", "trav_std", "mu_mean", "mu_std",
        "height", "step", "slope", "seen", "gt_trav"]


def from_points(P, point_probs, class_names, onto, res=0.5, unknown=0.5, gt_labels=None, trav_override=None):
    """P (N,3) points, point_probs (N,K) per-point class probabilities (one-hot for GT labels).
    onto: {name: {"traversability":…, "mu":…}}. Returns the contract dict."""
    trav_override = trav_override or {}
    K = len(class_names)
    tv = np.array([trav_override.get(n, onto.get(n, {}).get("traversability")) for n in class_names], object)
    tv = np.array([unknown if t is None else float(t) for t in tv])
    mu = np.array([np.nan if onto.get(n, {}).get("mu") is None else float(onto[n]["mu"]) for n in class_names])
    lo = P[:, :2].min(0)
    shape = tuple(np.ceil((P[:, :2].max(0) - lo) / res).astype(int) + 1)
    ij = np.floor((P[:, :2] - lo) / res).astype(int)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    n = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
    seen = n > 0
    probs = np.zeros(shape + (K,))
    for k in range(K):
        probs[..., k] = np.bincount(flat, point_probs[:, k], minlength=n.size).reshape(shape)
    probs[seen] /= np.maximum(probs[seen].sum(-1, keepdims=True), 1e-9)

    def moments(vals):
        ok = ~np.isnan(vals)
        if not ok.any():
            return np.full(shape, np.nan), np.full(shape, np.nan)
        p = probs[..., ok]; w = p.sum(-1)
        m = np.where(w > 0, (p * vals[ok]).sum(-1) / np.maximum(w, 1e-9), np.nan)
        v = np.where(w > 0, (p * (vals[ok] - m[..., None]) ** 2).sum(-1) / np.maximum(w, 1e-9), np.nan)
        return m, np.sqrt(v)

    trav_mean, trav_std = moments(tv)
    trav_mean[~seen] = np.nan; trav_std[~seen] = np.nan
    mu_mean, mu_std = moments(mu)

    z = P[:, 2]
    zmin = np.full(n.size, np.inf); np.minimum.at(zmin, flat, z)
    zmax = np.full(n.size, -np.inf); np.maximum.at(zmax, flat, z)
    zsum = np.bincount(flat, z, minlength=n.size)
    height = np.where(seen, zmin.reshape(shape), np.nan)
    step = np.where(seen, (zmax - zmin).reshape(shape), np.nan)
    zm = np.where(seen, zsum.reshape(shape) / np.maximum(n, 1), np.nan)
    zf = np.where(np.isnan(zm), np.nanmean(zm), zm)
    gx, gy = np.gradient(zf, res)
    slope = np.where(seen, np.degrees(np.arctan(np.hypot(gx, gy))), np.nan)

    gt = np.full(shape, np.nan)
    if gt_labels is not None:
        gt_p = np.zeros(shape + (K,))
        for k in range(K):
            gt_p[..., k] = np.bincount(flat, (gt_labels == k).astype(float), minlength=n.size).reshape(shape)
        w = gt_p.sum(-1)
        gt = np.where(w > 0, (gt_p * tv).sum(-1) / np.maximum(w, 1e-9), np.nan)

    return dict(origin=lo.astype(float), res=float(res), axes="xy", probs=probs.astype(np.float32),
                class_names=np.array(class_names), trav_mean=trav_mean, trav_std=trav_std,
                mu_mean=mu_mean, mu_std=mu_std, height=height, step=step, slope=slope, seen=seen, gt_trav=gt)


def save(path, g):
    np.savez_compressed(path, **g)


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    d["res"] = float(d["res"]); d["origin"] = np.asarray(d["origin"], float)
    d["axes"] = str(d.get("axes", "xy"))
    if d["axes"] == "yx":                 # [iy, ix] layout: transpose the 2D/3D arrays to [ix, iy]
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 2 and k != "class_names":
                d[k] = np.swapaxes(v, 0, 1)
        d["axes"] = "xy"
    return d


def cost_trav(g, risk=0.0, max_step=None, max_slope=None, unknown=0.5):
    """traversability used for planning: trav_mean - risk*trav_std, unseen -> unknown,
    too big a step or slope -> 0 (wall)."""
    t = np.where(np.isnan(g["trav_mean"]), unknown, g["trav_mean"] - risk * np.nan_to_num(g["trav_std"]))
    if max_step is not None:
        t = np.where(np.nan_to_num(g["step"]) > max_step, 0.0, t)
    if max_slope is not None:
        t = np.where(np.nan_to_num(g["slope"]) >= max_slope, 0.0, t)
    return np.clip(t, 0, 1)
