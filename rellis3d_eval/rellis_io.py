"""RELLIS-3D loading: images, labels, Ouster scans, calibration, poses.

Expected layout (what the official downloads unpack to):

    <root>/
      00000/ ... 00004/
        pylon_camera_node/                  frameXXXXXX-<sec>_<msec>.jpg
        pylon_camera_node_label_id/         frameXXXXXX-<sec>_<msec>.png
        os1_cloud_node_kitti_bin/           XXXXXX.bin   (float32 x,y,z,intensity)
        os1_cloud_node_semantickitti_label_id/  XXXXXX.label (uint32, low 16 bits = class)
        poses.txt                           one 3x4 row-major pose per scan
        calib.txt                           (KITTI style, only used for --pose-frame kitti)
      camera_info.txt                       fx fy cx cy        (Camera Intrinsic download)
      transforms.yaml                       os1 -> pylon        (Basler-to-Ouster download)
      test.lst / pt_test.lst ...            split files (optional)

camera_info.txt / transforms.yaml can also sit inside each sequence folder;
the per-sequence copy wins.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
# Raw RELLIS ids -> names (from the official ontology / rellis.yaml)
RAW_NAMES = {
    0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky",
    8: "vehicle", 9: "object", 10: "asphalt", 12: "building", 15: "log",
    17: "person", 18: "fence", 19: "bush", 23: "concrete", 27: "barrier",
    31: "puddle", 33: "mud", 34: "rubble",
}

# Image benchmark: 19 train ids, 0 = void (ignored). Same mapping as the
# official Evaluate_img.ipynb (dirt->void, 29/30->grass, 32->water).
EVAL_CLASSES = ["void", "grass", "tree", "pole", "water", "sky", "vehicle", "object",
                "asphalt", "building", "log", "person", "fence", "bush", "concrete",
                "barrier", "puddle", "mud", "rubble"]
RAW_TO_EVAL = {0: 0, 1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 12: 9,
               15: 10, 17: 11, 18: 12, 19: 13, 23: 14, 27: 15, 29: 1, 30: 1,
               31: 16, 32: 4, 33: 17, 34: 18}
IGNORE = 0

_LUT = np.zeros(65536, dtype=np.uint8)
for _k, _v in RAW_TO_EVAL.items():
    _LUT[_k] = _v


def raw_to_eval(raw: np.ndarray) -> np.ndarray:
    return _LUT[raw.astype(np.int64) & 0xFFFF]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
# Distortion of the Basler camera, from the official lidar2img.ipynb.
DEFAULT_DIST = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0])


@dataclass
class Calib:
    K: np.ndarray                 # 3x3
    dist: np.ndarray              # (5,)
    T_cam_lidar: np.ndarray       # 4x4, Ouster frame -> camera (OpenCV) frame
    width: int = 1920
    height: int = 1200


def load_intrinsics(path: str) -> np.ndarray:
    fx, fy, cx, cy = np.loadtxt(path)[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def load_T_cam_lidar(path: str, key: str = "os1_cloud_node-pylon_camera_node") -> np.ndarray:
    """transforms.yaml stores the camera pose in the Ouster frame; invert it."""
    with open(path) as f:
        d = yaml.safe_load(f)[key]
    q, t = d["q"], d["t"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return np.linalg.inv(T)


def _find(root: str, seq: str | None, name: str) -> str | None:
    cands = ([os.path.join(root, seq, name)] if seq else []) + [os.path.join(root, name)]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_calib(root: str, seq: str | None = None) -> Calib:
    ci = _find(root, seq, "camera_info.txt")
    tf = _find(root, seq, "transforms.yaml")
    if ci is None or tf is None:
        raise FileNotFoundError(
            f"Need camera_info.txt and transforms.yaml in {root} or {root}/{seq} "
            "(RELLIS 'Camera Intrinsic' and 'Basler Camera to Ouster LiDAR' downloads)")
    return Calib(K=load_intrinsics(ci), dist=DEFAULT_DIST.copy(), T_cam_lidar=load_T_cam_lidar(tf))


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
_FRAME_RE = re.compile(r"frame(\d+)-")


@dataclass
class Frame:
    seq: str
    idx: int
    image: str | None = None
    image_label: str | None = None
    scan: str | None = None
    scan_label: str | None = None
    extra: dict = field(default_factory=dict)


def frame_index(path: str) -> int:
    b = os.path.basename(path)
    m = _FRAME_RE.search(b)
    return int(m.group(1)) if m else int(os.path.splitext(b)[0].lstrip("vel"))


def index_sequence(root: str, seq: str) -> dict[int, Frame]:
    """Pair image, image label, scan and scan label by frame number."""
    d = os.path.join(root, seq)
    frames: dict[int, Frame] = {}

    def put(pattern, attr):
        for p in sorted(glob.glob(os.path.join(d, pattern))):
            i = frame_index(p)
            frames.setdefault(i, Frame(seq, i))
            setattr(frames[i], attr, p)

    put("pylon_camera_node/*.jpg", "image")
    put("pylon_camera_node/*.png", "image")
    put("pylon_camera_node_label_id/*.png", "image_label")
    put("os1_cloud_node_kitti_bin/*.bin", "scan")
    put("os1_cloud_node_semantickitti_label_id/*.label", "scan_label")
    return frames


def list_sequences(root: str) -> list[str]:
    return sorted(s for s in os.listdir(root) if re.fullmatch(r"\d{5}", s)
                  and os.path.isdir(os.path.join(root, s)))


def read_split(root: str, lst: str) -> list[tuple[str, int]]:
    """Read an official .lst file -> [(seq, frame_idx)]. First token per line is used."""
    p = lst if os.path.isabs(lst) else os.path.join(root, lst)
    out = []
    for line in open(p):
        tok = line.strip().split()
        if not tok:
            continue
        rel = tok[0].replace("\\", "/")
        seq = next((s for s in rel.split("/") if re.fullmatch(r"\d{5}", s)), None)
        out.append((seq, frame_index(rel)))
    return out


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def read_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_image_label(path: str) -> np.ndarray:
    return raw_to_eval(np.array(Image.open(path)))


def read_scan(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)


def read_scan_label(path: str) -> np.ndarray:
    return raw_to_eval(np.fromfile(path, dtype=np.uint32))


def read_poses(root: str, seq: str, frame: str = "lidar") -> np.ndarray:
    """(N,4,4) world_T_lidar. frame='kitti' applies Tr from calib.txt like SemanticKITTI."""
    P = np.loadtxt(os.path.join(root, seq, "poses.txt")).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :] = P
    if frame == "kitti":
        Tr = None
        for line in open(os.path.join(root, seq, "calib.txt")):
            k, *v = line.replace(":", " ").split()
            if k == "Tr":
                Tr = np.eye(4)
                Tr[:3, :] = np.array(v, float).reshape(3, 4)
        if Tr is None:
            raise ValueError("calib.txt has no Tr line")
        T = np.linalg.inv(Tr) @ T @ Tr
    return T


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
def project(points_lidar: np.ndarray, calib: Calib, min_depth: float = 0.5):
    """Ouster points -> pixel coords (with distortion). Returns uv (N,2), depth (N,), valid (N,)."""
    import cv2
    ph = np.c_[points_lidar, np.ones(len(points_lidar))]
    pc = (calib.T_cam_lidar @ ph.T).T[:, :3]
    z = pc[:, 2]
    valid = z > min_depth
    # stay well inside the FOV before applying the distortion polynomial,
    # which folds back on itself for points far outside the image
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = pc[:, 0] / z, pc[:, 1] / z
    lim_x = 1.15 * max(calib.K[0, 2], calib.width - calib.K[0, 2]) / calib.K[0, 0]
    lim_y = 1.15 * max(calib.K[1, 2], calib.height - calib.K[1, 2]) / calib.K[1, 1]
    valid &= (np.abs(xn) < lim_x) & (np.abs(yn) < lim_y)
    uv = np.full((len(pc), 2), -1.0)
    if valid.any():
        pts, _ = cv2.projectPoints(pc[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                   calib.K, calib.dist)
        uv[valid] = pts.reshape(-1, 2)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < calib.width) & (uv[:, 1] >= 0) & (uv[:, 1] < calib.height)
    return uv, z, valid


def visibility_mask(uv: np.ndarray, z: np.ndarray, valid: np.ndarray, calib: Calib,
                    cell: int = 8, rel_tol: float = 0.05, abs_tol: float = 0.3) -> np.ndarray:
    """Drop points hidden behind nearer points (camera and LiDAR sit ~20 cm apart).

    Keeps a point if its depth is within tolerance of the nearest LiDAR depth
    in its (cell x cell) pixel block.
    """
    vis = valid.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return vis
    gx = (uv[idx, 0] // cell).astype(int)
    gy = (uv[idx, 1] // cell).astype(int)
    W = calib.width // cell + 1
    key = gy * W + gx
    zmin = np.full(key.max() + 1, np.inf)
    np.minimum.at(zmin, key, z[idx])
    near = zmin[key]
    vis[idx] = z[idx] <= near + np.maximum(abs_tol, rel_tol * near)
    return vis
