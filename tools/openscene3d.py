#!/usr/bin/env python3
"""
OpenScene 3D-only ("distill") on any point cloud: no images, no fused features.

Uses the official OpenScene code + their pretrained MinkowskiNet (nuScenes = outdoor LiDAR,
default) to give every point a 768-d CLIP-space feature (OpenSeg / CLIP ViT-L/14@336 text side).
Their run/eval.sh needs the fused 2D features even in distill mode (165 GB for nuScenes); this doesn't.

Run in the env with MinkowskiEngine + OpenScene requirements (nico_minkowski):

  python tools/openscene3d.py --openscene ~/nico_ws/src/openscene \
      --rellis data/rellis_sample --out out/os3d_rellis                     # RELLIS scan, scored vs GT
  python tools/openscene3d.py --openscene ~/nico_ws/src/openscene \
      --ply <scene>/map.ply --classes grass dirt mud gravel rock bench --out out/os3d_unity
  python tools/openscene3d.py --openscene ~/nico_ws/src/openscene \
      --pth <openscene>/data/nuscenes_3d/val/<scene>.pth --labelset nuscenes --out out/os3d_nusc

  add --query "standing water" "somewhere to sit" for text heatmaps

Inputs
  --rellis DIR   RELLIS-3D sample (000104.bin + 000104.label), GT scored with the RELLIS classes
  --ply FILE     ascii x y z [label] (our Unity map.ply / scene format; labels named via scene.json)
  --pth FILE     OpenScene processed file (locs, colors, labels)
Classes
  default: ontology.json classes present in the GT (or --classes), prompts from the ontology
  --labelset nuscenes   the 16 nuScenes classes the model was evaluated on
Outputs in --out: feat.npy (fp16 per-point features), pred.npy, pred.ply (colored), bev.png,
  query_<text>.png / .ply, metrics.json (if GT)
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELLIS = {1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
          10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
          27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}
NUSC16 = ["barrier", "bicycle", "bus", "car", "construction vehicle", "motorcycle", "person", "traffic cone",
          "trailer", "truck", "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation"]
MODELS = {"nuscenes": "https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar",
          "scannet": "https://cvg-data.inf.ethz.ch/openscene/models/scannet_openseg.pth.tar",
          "matterport": "https://cvg-data.inf.ethz.ch/openscene/models/matterport_openseg.pth.tar"}


def load_points(a):
    """-> xyz (N,3) float, gt (N,) int or None, id->name dict or None"""
    if a.rellis:
        d = Path(a.rellis)
        xyz = np.fromfile(d / "000104.bin", np.float32).reshape(-1, 4)[:, :3]
        gt = (np.fromfile(d / "000104.label", np.uint32) & 0xFFFF).astype(np.int64)
        keep = np.linalg.norm(xyz, axis=1) > 0.5          # drop the zero returns
        return xyz[keep], gt[keep], {**RELLIS, 0: "void"}
    if a.ply:
        p = Path(a.ply)
        with open(p) as f:
            n = 0
            while True:
                line = f.readline()
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                if line.strip() == "end_header":
                    break
            data = np.loadtxt(f, max_rows=n, ndmin=2)
        names = None
        sj = p.parent / "scene.json"
        if sj.exists():
            names = {int(k): v for k, v in json.load(open(sj))["classes"].items()}
        gt = data[:, 3].astype(np.int64) if data.shape[1] > 3 else None
        return data[:, :3].astype(np.float32), gt, names
    import torch
    locs, _, labels = torch.load(a.pth)
    labels = np.asarray(labels).astype(np.int64)
    return np.asarray(locs, np.float32), labels, None


def text_features(classes_prompts, template="a {} in a scene"):
    """OpenSeg's text side = CLIP ViT-L/14@336px. ' . ' joined terms are averaged (Jaime-style prompts)."""
    import torch, clip
    model, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False)
    out = []
    with torch.no_grad():
        for prompts in classes_prompts:
            terms = [t.strip() for p in prompts for t in p.split(" . ") if t.strip()]
            e = model.encode_text(clip.tokenize([template.format(t) for t in terms]).cuda()).float()
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            out.append(m / m.norm())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openscene", required=True, help="path to the cloned OpenScene repo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rellis"); src.add_argument("--ply"); src.add_argument("--pth")
    ap.add_argument("--model", default="nuscenes", help="nuscenes | scannet | matterport | path/to/.pth.tar")
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size the model was trained with (0.05)")
    ap.add_argument("--labelset", default=None, help="nuscenes = their 16 classes")
    ap.add_argument("--classes", nargs="*", help="ontology class names to classify into")
    ap.add_argument("--ontology", default=str(ROOT / "config" / "ontology.json"))
    ap.add_argument("--query", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.openscene).expanduser().resolve()))
    # OpenScene's code uses collections.Sequence / .Iterable, removed in Python 3.10
    import collections, collections.abc
    collections.Sequence, collections.Iterable = collections.abc.Sequence, collections.abc.Iterable
    import torch
    from torch.utils import model_zoo
    from MinkowskiEngine import SparseTensor
    from models.disnet import DisNet
    from dataset.voxelizer import Voxelizer

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xyz, gt, id2name = load_points(a)
    print(f"{len(xyz)} points, extent {np.ptp(xyz, 0).round(1)} m")

    # --- 3D model (same construction as OpenScene's evaluate.py) ---
    cfg = argparse.Namespace(feature_2d_extractor="openseg", arch_3d="MinkUNet18A")
    model = DisNet(cfg=cfg).cuda().eval()
    path = MODELS.get(a.model, a.model)
    ck = model_zoo.load_url(path, progress=True) if path.startswith("http") else torch.load(path, map_location="cuda")
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd, strict=True)

    # --- voxelize (no augmentation), color = (1,1,1) like their input_color=False ---
    vox = Voxelizer(voxel_size=a.voxel, clip_bound=None, use_augmentation=False)
    dummy = np.zeros_like(xyz)
    locs, _, _, inds_rec, _ = vox.voxelize(xyz, dummy, np.zeros(len(xyz), np.int64), return_ind=True)
    coords = torch.cat([torch.zeros(len(locs), 1, dtype=torch.int), torch.from_numpy(locs).int()], 1)
    feats = torch.ones(len(locs), 3)
    t0 = time.time()
    with torch.no_grad():
        f = model(SparseTensor(feats.cuda(), coords.cuda()))
        f = f.F if hasattr(f, "F") else f              # their MinkUNet already returns a dense tensor
        f = f[torch.from_numpy(inds_rec).long().cuda()]
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
    torch.cuda.synchronize()
    print(f"3D features {tuple(f.shape)} in {time.time() - t0:.2f}s ({len(locs)} voxels at {a.voxel} m)")
    np.save(out / "feat.npy", f.half().cpu().numpy())

    # --- classes ---
    onto = {c["name"]: c for c in json.load(open(a.ontology))["classes"]}
    if a.labelset == "nuscenes":
        names = NUSC16; prompts = [[n] for n in names]
    else:
        if a.classes:
            names = a.classes
        elif gt is not None and id2name:
            names = [id2name[int(c)] for c in np.unique(gt) if int(c) in id2name and id2name[int(c)] not in ("void", "sky")]
        else:
            names = [n for n, c in onto.items() if c.get("traversability") is not None]
        prompts = [onto.get(n, {}).get("prompts", [n]) for n in names]
    T = text_features(prompts)
    logits = f.float() @ T.T
    pred_idx = logits.argmax(1).cpu().numpy()
    pred = np.array(names)[pred_idx]
    np.save(out / "pred.npy", pred_idx)
    print("classes:", names)

    # --- score vs GT (by name) ---
    metrics = {}
    if gt is not None and id2name:
        gname = np.array([id2name.get(int(c), "void") for c in gt])
        valid = ~np.isin(gname, ["void", "sky"])
        g, p = gname[valid], pred[valid]
        ious = {}
        for n in np.unique(g):
            ious[n] = float(((g == n) & (p == n)).sum() / max(((g == n) | (p == n)).sum(), 1))
        metrics = dict(acc=float((g == p).mean()), miou=float(np.mean(list(ious.values()))), iou=ious,
                       n_points=int(valid.sum()))
        print(f"vs GT: acc {metrics['acc']:.3f}  mIoU {metrics['miou']:.3f}")
        for n, v in sorted(ious.items(), key=lambda x: -x[1]):
            print(f"   {n:12s} {v:.3f}  ({(g == n).sum()} pts)")
        json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    # --- outputs: colored ply + BEV + queries ---
    rng = np.random.default_rng(0)
    cols = {n: np.array(onto.get(n, {}).get("color", rng.integers(40, 255, 3))) for n in names}
    rgb = np.stack([cols[n] for n in pred]).astype(np.uint8)

    def write_ply(path, pts, colors):
        with open(path, "w") as fh:
            fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                     f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            np.savetxt(fh, np.c_[pts, colors], fmt="%.3f %.3f %.3f %d %d %d")
    write_ply(out / "pred.ply", xyz, rgb)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("prediction", rgb / 255.0)]
    if gt is not None and id2name:
        gc = np.stack([cols.get(id2name.get(int(c), ""), np.array([200, 200, 200])) for c in gt]) / 255.0
        panels.insert(0, ("ground truth", gc))
    fig, ax = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    for x_, (title, c) in zip(np.atleast_1d(ax), panels):
        x_.scatter(-xyz[:, 1], xyz[:, 0], c=c, s=0.5)
        x_.set_title(title + (f"  (acc {metrics['acc']:.2f}, mIoU {metrics['miou']:.2f})" if title == "prediction" and metrics else ""))
        x_.set_aspect("equal"); x_.set_xlabel("-y [m]"); x_.set_ylabel("x [m]")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=cols[n] / 255.0) for n in names], names, loc="lower center",
               ncol=min(len(names), 8))
    plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(out / "bev.png", dpi=90); plt.close()

    for q in a.query:
        s = (f.float() @ text_features([[q]])[0]).cpu().numpy()
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower())[:40]
        fig, x_ = plt.subplots(figsize=(7, 7))
        sc = x_.scatter(-xyz[:, 1], xyz[:, 0], c=s, s=0.5, cmap="jet")
        plt.colorbar(sc); x_.set_title(f'"{q}"'); x_.set_aspect("equal")
        plt.savefig(out / f"query_{slug}.png", dpi=90); plt.close()
        cm = (plt.get_cmap("jet")((s - s.min()) / (np.ptp(s) + 1e-8))[:, :3] * 255).astype(np.uint8)
        write_ply(out / f"query_{slug}.ply", xyz, cm)
    print("saved", out)


if __name__ == "__main__":
    main()
