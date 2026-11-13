"""
Fake a Unity export so the harness can be tested before Unity is set up.

- Heightfield terrain (Unity coords: x right, y up, z forward), classes painted as blobs
- Textures cropped from a real photo + its label image (default: the RELLIS-3D sample frame)
- Cameras/rays/intrinsics/pose conversion copy SceneCapture.cs line for line, so if the
  harness's convention check passes here, the C# math is consistent with the reader.

python make_synthetic.py <out_dir> --photo frame.jpg --photo_labels frame.png [--views 12]
"""
import argparse, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image

CLASSES = {0: "void", 1: "grass", 2: "mud", 3: "puddle", 4: "bush"}
RELLIS_ID = {1: 3, 2: 33, 3: 31, 4: 19}          # our class -> RELLIS label id to crop texture from
SIZE, RES = 40.0, 0.1                            # terrain extent (m), heightfield cell (m)


def smooth_noise(rng, n, sigma):
    f = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), sigma)
    return (f - f.mean()) / f.std()


def crop_textures(photo, labels, out_px=384):
    img = np.array(Image.open(photo).convert("RGB"))
    lab = np.array(Image.open(labels))
    tex = {}
    for c, rid in RELLIS_ID.items():
        m = (lab == rid).astype(np.uint8)
        d = cv2.distanceTransform(m, cv2.DIST_C, 3)      # chessboard dist -> biggest square
        y, x = np.unravel_index(d.argmax(), d.shape)
        r = int(d.max())
        patch = img[y - r + 1:y + r, x - r + 1:x + r]
        tex[c] = cv2.resize(patch, (out_px, out_px), interpolation=cv2.INTER_AREA).astype(np.float32)
        print(f"texture {CLASSES[c]}: {2 * r - 1}px square from the photo")
    return tex


def build_terrain(rng):
    n = int(SIZE / RES)
    fields = np.stack([smooth_noise(rng, n, 25) + b for b in (0.35, 0.0, -0.05, 0.05)])
    cls = fields.argmax(0) + 1                                    # 1..4
    h = 0.5 * smooth_noise(rng, n, 40)                            # gentle hills, ~0.5 m std
    h = h + np.where(cls == 4, 0.6, 0.0) + np.where(cls == 3, -0.08, 0.0)
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 1.5)
    return h, cls                                                  # indexed [iz, ix]


def sample(grid, x, z, nearest=False):
    n = grid.shape[0]
    fx, fz = np.clip(x / RES - 0.5, 0, n - 1.001), np.clip(z / RES - 0.5, 0, n - 1.001)
    if nearest:
        return grid[np.round(fz).astype(int), np.round(fx).astype(int)]
    x0, z0 = fx.astype(int), fz.astype(int)
    ax, az = fx - x0, fz - z0
    return (grid[z0, x0] * (1 - ax) * (1 - az) + grid[z0, x0 + 1] * ax * (1 - az)
            + grid[z0 + 1, x0] * (1 - ax) * az + grid[z0 + 1, x0 + 1] * ax * az)


# ---- these mirror SceneCapture.cs ----
def unity_cam_axes(pitch_deg, yaw_deg):
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    f = np.array([np.sin(y) * np.cos(p), -np.sin(p), np.cos(y) * np.cos(p)])
    r = np.array([np.cos(y), 0.0, -np.sin(y)])
    up = np.cross(f, r)                      # Unity's Vector3.Cross, numerically the same
    return r, up, f


def U2R(v):
    return np.array([v[2], -v[0], v[1]])


def view_json(W, H, fov_v, r, up, f, pos):
    fy = H * 0.5 / np.tan(np.radians(fov_v) / 2)
    fx = fy
    c0, c1, c2, t = U2R(r), U2R(-up), U2R(f), U2R(pos)
    T = [c0[0], c1[0], c2[0], t[0], c0[1], c1[1], c2[1], t[1], c0[2], c1[2], c2[2], t[2], 0, 0, 0, 1]
    return dict(width=W, height=H, fx=fx, fy=fy, cx=(W - 1) / 2, cy=(H - 1) / 2, T_world_cam=[float(x) for x in T])


def viewport_rays(W, H, fov_v, r, up, f):
    tv = np.tan(np.radians(fov_v) / 2)
    th = tv * W / H
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ndc_x = 2 * (u + 0.5) / W - 1
    ndc_y = 2 * (1 - (v + 0.5) / H) - 1
    d = f[None, None] + ndc_x[..., None] * th * r + ndc_y[..., None] * tv * up
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ---- renderer ----
def render(h, cls, tex, pos, r, up, f, W, H, fov_v, tile=3.0, tmax=45.0, step=0.04):
    d = viewport_rays(W, H, fov_v, r, up, f).reshape(-1, 3)
    n = len(d)
    t_hit = np.full(n, np.nan)
    active = np.ones(n, bool)
    t_prev = np.zeros(n)
    t = 0.05
    while t < tmax and active.any():
        idx = np.nonzero(active)[0]
        p = pos + d[idx] * t
        inb = (p[:, 0] > 0) & (p[:, 0] < SIZE) & (p[:, 2] > 0) & (p[:, 2] < SIZE)
        below = inb & (p[:, 1] <= sample(h, p[:, 0], p[:, 2]))
        lo, hi = t_prev[idx[below]].copy(), np.full(below.sum(), t)
        for _ in range(10):                                   # bisection refine
            mid = (lo + hi) / 2
            pm = pos + d[idx[below]] * mid[:, None]
            under = pm[:, 1] <= sample(h, pm[:, 0], pm[:, 2])
            hi, lo = np.where(under, mid, hi), np.where(under, lo, mid)
        t_hit[idx[below]] = hi
        active[idx[below]] = False
        t_prev[idx] = t
        t += step * (1 + t / 8)                               # coarser steps far away
    hit = ~np.isnan(t_hit)
    P = pos + d * np.nan_to_num(t_hit)[:, None]
    depth = np.where(hit, (P - pos) @ f, 0.0).astype(np.float32)
    lab = np.where(hit, sample(cls, P[:, 0], P[:, 2], nearest=True), 0)

    # color: tiled texture * lambert, sky gradient otherwise
    rgb = np.zeros((n, 3), np.float32)
    sky = np.array([170, 195, 230], np.float32)
    rgb[~hit] = sky * (0.85 + 0.15 * d[~hit, 1:2].clip(0, 1))
    for c, tx in tex.items():
        m = lab[hit] == c
        if not m.any():
            continue
        s_ = tx.shape[0]
        mx = (P[hit, 0][m] % tile) / tile * s_ - 0.5
        mz = (P[hit, 2][m] % tile) / tile * s_ - 0.5
        x0, z0 = np.floor(mx).astype(int), np.floor(mz).astype(int)
        ax, az = (mx - x0)[:, None], (mz - z0)[:, None]
        x0, x1, z0, z1 = x0 % s_, (x0 + 1) % s_, z0 % s_, (z0 + 1) % s_     # wrap = seamless-ish tiling
        rgb[np.nonzero(hit)[0][m]] = (tx[z0, x0] * (1 - ax) * (1 - az) + tx[z0, x1] * ax * (1 - az)
                                      + tx[z1, x0] * (1 - ax) * az + tx[z1, x1] * ax * az)
    e = 0.1
    gx = (sample(h, P[hit, 0] + e, P[hit, 2]) - sample(h, P[hit, 0] - e, P[hit, 2])) / (2 * e)
    gz = (sample(h, P[hit, 0], P[hit, 2] + e) - sample(h, P[hit, 0], P[hit, 2] - e)) / (2 * e)
    nrm = np.stack([-gx, np.ones_like(gx), -gz], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    shade = 0.55 + 0.45 * (nrm @ np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])).clip(0, 1)
    rgb[hit] *= shade[:, None]
    return rgb.clip(0, 255).astype(np.uint8).reshape(H, W, 3), depth.reshape(H, W), lab.reshape(H, W).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--photo_labels", required=True)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--W", type=int, default=480)
    ap.add_argument("--H", type=int, default=360)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "views").mkdir(parents=True, exist_ok=True)

    tex = crop_textures(a.photo, a.photo_labels)
    h, cls = build_terrain(rng)

    # map: same as ExportMap's straight-down rays at 0.2 m
    g = np.arange(0.1, SIZE, 0.2)
    X, Z = np.meshgrid(g, g)
    Y = sample(h, X.ravel(), Z.ravel())
    L = sample(cls, X.ravel(), Z.ravel(), nearest=True)
    pts = np.stack([Z.ravel(), -X.ravel(), Y], -1)              # U2R
    with open(out / "map.ply", "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty int label\nend_header\n")
        np.savetxt(fh, np.column_stack([pts, L]), fmt="%.4f %.4f %.4f %d")
    json.dump(dict(classes={str(k): v for k, v in CLASSES.items()}, world_frame="ROS FLU", camera_frame="OpenCV",
                   source="make_synthetic.py"), open(out / "scene.json", "w"), indent=2)

    for k in range(a.views):
        x, z = rng.uniform(6, SIZE - 6, 2)
        pos = np.array([x, sample(h, np.array([x]), np.array([z]))[0] + 0.6, z])
        pitch, yaw = rng.uniform(15, 30), rng.uniform(0, 360)
        r, up, f = unity_cam_axes(pitch, yaw)
        rgb, depth, lab = render(h, cls, tex, pos, r, up, f, a.W, a.H, a.fov)
        stem = out / "views" / f"{k:06d}"
        Image.fromarray(rgb).save(f"{stem}_rgb.png")
        depth.astype("<f4").tofile(f"{stem}_depth.bin")
        Image.fromarray(np.stack([lab] * 3 + [np.full_like(lab, 255)], -1)).save(f"{stem}_label.png")
        json.dump(view_json(a.W, a.H, a.fov, r, up, f, pos), open(f"{stem}.json", "w"), indent=1)
        print(f"view {k}: {(depth > 0).mean():.0%} ground pixels")


if __name__ == "__main__":
    main()
