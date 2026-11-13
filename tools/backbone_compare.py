#!/usr/bin/env python3
"""
Which dense backbone gives the best text-queryable features? Runs each one on the RELLIS-3D
sample frame (zero-shot, Jaime's RELLIS prompts from config/ontology.json) and scores
2D pixels and 3D LiDAR points against the ground truth.

python tools/backbone_compare.py data/rellis_sample out/backbones \
    --backbones clip pe-b:plain pe-b:last_v pe-b:last_v_only pe-l:last_v_only

backbone spec = name[:variant]. Names: clip, pe-t, pe-s, pe-b, pe-l, pe-g. Variants (PE only):
plain, last_v, last_v_only (see DensePE in car_semantic_map/clip_features.py).
One frame is a smoke signal, not a benchmark: use it to pick a variant, then check on more data.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2, yaml
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.clip_features import make_extractor
from car_semantic_map.evaluate import load_ontology

RELLIS = {1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
          10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
          27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}
FRAME = "frame000104-1581624663_149"


def scores(gt, pr):
    ious = []
    for c in np.unique(gt):
        if c == 0:
            continue
        g, p = gt == c, pr == c
        ious.append((RELLIS[int(c)], (g & p).sum() / max((g | p).sum(), 1)))
    v = gt != 0
    return float((gt[v] == pr[v]).mean()), float(np.mean([i for _, i in ious])), dict(ious)


def lidar_to_pixels(data, IH, IW):
    pts = np.fromfile(data / "000104.bin", np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(data / "000104.label", np.uint32) & 0xFFFF
    keep = np.linalg.norm(pts, axis=1) > 0.5
    pts, lab = pts[keep], lab[keep]
    fx, fy, cx, cy = np.loadtxt(data / "camera_info.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0])
    d = yaml.safe_load(open(data / "transforms.yaml"))["os1_cloud_node-pylon_camera_node"]
    T = np.eye(4); T[:3, :3] = Rotation.from_quat([d["q"][k] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [d["t"][k] for k in "xyz"]; RT = np.linalg.inv(T)
    pc = (RT[:3, :3] @ pts.T + RT[:3, 3:]).T
    front = pc[:, 2] > 0.5
    uv = np.full((len(pts), 2), -1.0)
    uv[front] = cv2.projectPoints(pts[front], cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3], K, dist)[0][:, 0]
    inside = front & (uv[:, 0] >= 10) & (uv[:, 0] < IW - 10) & (uv[:, 1] >= 10) & (uv[:, 1] < IH - 10)
    CELL = 12
    zb = np.full((IH // CELL + 1, IW // CELL + 1), np.inf)
    ci = (uv[inside] // CELL).astype(int)
    np.minimum.at(zb, (ci[:, 1], ci[:, 0]), pc[inside, 2])
    vis = inside.copy()
    vis[inside] = pc[inside, 2] <= zb[ci[:, 1], ci[:, 0]] * 1.05 + 0.3
    return uv[vis], lab[vis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--backbones", nargs="+", default=["clip", "pe-b:plain", "pe-b:last_v", "pe-b:last_v_only"])
    ap.add_argument("--short_side", type=int, default=448)
    ap.add_argument("--clip_weights", default=None, help="local .pt for clip (default: download laion400m_e32)")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    onto = load_ontology(ROOT / "config" / "ontology.json")
    ids = list(RELLIS)
    prompts = [onto.get(RELLIS[i], {}).get("prompts", [RELLIS[i]]) for i in ids]
    img = cv2.cvtColor(cv2.imread(str(data / f"{FRAME}.jpg")), cv2.COLOR_BGR2RGB)
    IH, IW = img.shape[:2]
    gt = np.array(Image.open(data / f"{FRAME}.png"))
    EH, EW = 300, 480
    gt_s = cv2.resize(gt, (EW, EH), interpolation=cv2.INTER_NEAREST)
    uv, lab3 = lidar_to_pixels(data, IH, IW)

    rows, preds = [], {}
    for spec in a.backbones:
        name, _, var = spec.partition(":")
        kw = {"short_side": a.short_side}
        if var:
            kw["variant"] = var
        ext = make_extractor(name, a.clip_weights if name == "clip" else None, **kw)
        Q, owner = ext.class_features(prompts)
        ext.image_features(img)                                 # warm-up (cuda init)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time(); fm = ext.image_features(img); dt = time.time() - t0
        sim = torch.einsum("chw,qc->qhw", fm, Q)
        cls = torch.stack([sim[torch.from_numpy(np.nonzero(owner == k)[0])].max(0).values for k in range(len(ids))])
        up = F.interpolate(cls[None], (EH, EW), mode="bilinear", align_corners=False)[0]
        p2 = np.array(ids)[up.argmax(0).numpy()]
        acc2, miou2, iou2 = scores(gt_s, p2)
        g = torch.from_numpy(np.stack([(uv[:, 0] + .5) / IW * 2 - 1, (uv[:, 1] + .5) / IH * 2 - 1], -1)).float()[None, None]
        pc = F.grid_sample(cls[None], g, align_corners=False)[0, :, 0]
        p3 = np.array(ids)[pc.argmax(0).numpy()]
        acc3, miou3, iou3 = scores(lab3, p3)
        rows.append(dict(backbone=spec, acc2d=acc2, miou2d=miou2, acc3d=acc3, miou3d=miou3, sec=dt,
                         grid=f"{fm.shape[1]}x{fm.shape[2]}", iou3d=iou3))
        preds[spec] = p2
        print(f"{spec:22s} 2D acc {acc2:.3f} mIoU {miou2:.3f} | 3D acc {acc3:.3f} mIoU {miou3:.3f} | {dt:.2f}s grid {fm.shape[1]}x{fm.shape[2]}")
        del ext; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(out / "backbones.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "iou3d"]); w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "iou3d"})
    json.dump(rows, open(out / "backbones.json", "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {c: np.array(onto.get(n, {}).get("color", [120, 120, 120])) / 255 for c, n in RELLIS.items()}
    col[0] = np.zeros(3)
    paint = lambda l: np.stack([np.vectorize(lambda c: col.get(int(c), np.zeros(3))[k])(l) for k in range(3)], -1)
    n = len(preds) + 2
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    ax[0].imshow(cv2.resize(img, (EW, EH))); ax[0].set_title("RELLIS frame 104")
    ax[1].imshow(paint(gt_s)); ax[1].set_title("ground truth")
    for k, (spec, p) in enumerate(preds.items()):
        r = rows[k]
        ax[k + 2].imshow(paint(p)); ax[k + 2].set_title(f"{spec}\n2D mIoU {r['miou2d']:.2f} · 3D acc {r['acc3d']:.2f}", fontsize=9)
    for x in ax:
        x.axis("off")
    plt.tight_layout(); plt.savefig(out / "backbones.png", dpi=90)
    print("saved", out / "backbones.png", out / "backbones.csv")


if __name__ == "__main__":
    main()
