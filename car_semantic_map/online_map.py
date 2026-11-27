"""
Online embedding map: posed RGB-D keyframes in, one 3D map of text-queryable features out, built
incrementally (keyframe by keyframe) so it can run live on a robot. No ROS in here:
tools/semantic_mapper.py wraps it in a node, and the same class works on recorded data.

Per keyframe:
  rgb -> feature extractor (PE-B by default, clip_features.make_extractor) -> [C,h,w] patch features
  depth + K + T_world_cam -> 3D points (every `stride`-th pixel, within max_depth)
  each point takes the feature of the patch it falls in -> voxel hash (voxel m): running sum of
  features, colour and position, plus a count.

Conventions (same as SceneCapture.cs / scene_io.py, and what a RealSense gives after alignment):
  depth[v,u] = metric z-depth along the camera axis, 0/NaN = no data
  K = [[fx,0,cx],[0,fy,cy],[0,0,1]], camera = OpenCV (x right, y down, z forward)
  T_world_cam = 4x4 camera -> world (world = map frame, ROS FLU)

Query: a prompt competes against the ontology classes (same rule as goal_from_prompt.target_probability):
P(prompt wins) per voxel = the "intensity" the ROS node publishes. Saved maps use feature_map.py's
embedded_map.npz format (points, feat, count, backbone), so prompt_drive.py --map ai reads them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class OnlineEmbeddingMap:
    def __init__(self, extractor=None, backbone="pe-b", voxel=0.1, max_depth=5.0, min_depth=0.2,
                 stride=4, ontology=ROOT / "config" / "ontology.json", temperature=100.0, synonym=0.85):
        self.backbone, self.voxel = backbone, voxel
        self.max_depth, self.min_depth, self.stride = max_depth, min_depth, stride
        self.T, self.synonym = temperature, synonym
        self.ex = extractor
        self.onto = {c["name"]: c for c in json.load(open(ontology))["classes"]}
        self.index = {}                       # voxel key -> row
        self.fsum = None                      # (N, C) float32
        self.csum = np.zeros((0, 3), np.float32)
        self.psum = np.zeros((0, 3), np.float64)
        self.count = np.zeros(0, np.int32)
        self.lsum = None                      # optional GT label votes (N, L) for evaluation
        self.n = 0
        self.keyframes = []                   # T_world_cam of every inserted keyframe
        self._class_cache = {}

    # ------------------------------------------------------------------ building
    def _grow(self, need, C):
        cap = 0 if self.fsum is None else len(self.fsum)
        if need <= cap:
            return
        new = max(need, int(cap * 1.5) + 1024)
        f = np.zeros((new, C), np.float32); c = np.zeros((new, 3), np.float32)
        p = np.zeros((new, 3), np.float64); k = np.zeros(new, np.int32)
        if cap:
            f[:cap], c[:cap], p[:cap], k[:cap] = self.fsum, self.csum, self.psum, self.count
        self.fsum, self.csum, self.psum, self.count = f, c, p, k
        if self.lsum is not None:
            l2 = np.zeros((new, self.lsum.shape[1]), np.float32); l2[:cap] = self.lsum; self.lsum = l2

    def backproject(self, depth, K, T_world_cam):
        H, W = depth.shape
        s = self.stride
        vs, us = np.mgrid[s // 2:H:s, s // 2:W:s]
        z = depth[vs, us].astype(np.float64)
        ok = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        u, v, z = us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok]
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        pc = np.stack([x, y, z, np.ones_like(z)], 1)
        pw = (np.asarray(T_world_cam) @ pc.T).T[:, :3]
        return pw, u.astype(int), v.astype(int)

    def insert(self, rgb, depth, K, T_world_cam, features=None, labels=None, n_labels=None):
        """Add one keyframe. rgb HxWx3 uint8, depth HxW metres (same size as rgb, i.e. aligned).
        features: optional precomputed [C,h,w] (else the extractor runs on rgb).
        labels: optional HxW int GT ids (Unity) -> per-voxel votes for scoring. Returns #points added."""
        if features is None:
            f = self.ex.image_features(rgb)
            features = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
        C, h, w = features.shape
        H, W = depth.shape
        pw, u, v = self.backproject(depth, K, T_world_cam)
        if len(pw) == 0:
            return 0
        fu = np.clip((u * w) // W, 0, w - 1); fv = np.clip((v * h) // H, 0, h - 1)
        feat = features[:, fv, fu].T.astype(np.float32)
        col = rgb[v, u].astype(np.float32) if rgb is not None else np.zeros((len(u), 3), np.float32)
        key = np.floor(pw / self.voxel).astype(np.int64)
        key = (key[:, 0] + 2 ** 20) * 2 ** 42 + (key[:, 1] + 2 ** 20) * 2 ** 21 + (key[:, 2] + 2 ** 20)
        uk, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)
        rows = np.empty(len(uk), np.int64)
        for i, k in enumerate(uk.tolist()):                 # map voxel keys -> rows (new voxels appended)
            r = self.index.get(k)
            if r is None:
                r = self.index[k] = self.n; self.n += 1
            rows[i] = r
        if labels is not None and self.lsum is None:
            self.lsum = np.zeros((0 if self.fsum is None else len(self.fsum), n_labels or 256), np.float32)
        self._grow(self.n, C)
        r = rows[inv]
        np.add.at(self.fsum, r, feat)
        np.add.at(self.csum, r, col)
        np.add.at(self.psum, r, pw)
        np.add.at(self.count, r, 1)
        if labels is not None:
            np.add.at(self.lsum, (r, labels[v, u].astype(int)), 1)
        self.keyframes.append(np.asarray(T_world_cam, float))
        self._class_cache.clear()
        return len(pw)

    # ------------------------------------------------------------------ reading
    @property
    def points(self):
        return (self.psum[:self.n] / np.maximum(self.count[:self.n, None], 1)).astype(np.float32)

    @property
    def colors(self):
        return (self.csum[:self.n] / np.maximum(self.count[:self.n, None], 1)).clip(0, 255).astype(np.uint8)

    @property
    def features(self):
        f = self.fsum[:self.n]
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)

    def gt_labels(self):
        return None if self.lsum is None else self.lsum[:self.n].argmax(1)

    # ------------------------------------------------------------------ queries
    def _text(self, prompts):
        q = self.ex.query_features(prompts)
        return q.numpy() if hasattr(q, "numpy") else np.asarray(q)

    def _classes(self, exclude=()):
        key = tuple(sorted(exclude))
        if key not in self._class_cache:
            names = [n for n in self.onto if n not in exclude]
            Q, own = self.ex.class_features([self.onto[n].get("prompts", [n]) for n in names])
            self._class_cache[key] = (names, Q.numpy() if hasattr(Q, "numpy") else np.asarray(Q), np.asarray(own))
        return self._class_cache[key]

    def score(self, prompt, mode="prob"):
        """per-voxel score for a prompt. mode 'prob': P(prompt beats every ontology class), 0..1
        (classes that are near-synonyms of the prompt don't compete). 'cosine': raw similarity."""
        F = self.features
        if len(F) == 0:
            return np.zeros(0, np.float32), []
        prompt = prompt.strip().lower()
        tq = self._text(self.onto[prompt].get("prompts", [prompt]) if prompt in self.onto else [prompt])
        t = (F @ tq.T).max(1)
        if mode == "cosine":
            return t.astype(np.float32), []
        names, Q, own = self._classes()
        csim = np.array([(Q[own == k] @ tq.T).max() for k in range(len(names))])
        drop = {names[k] for k in np.nonzero(csim >= self.synonym)[0]} | ({prompt} if prompt in self.onto else set())
        if len(drop) >= len(names):
            drop = set()
        names2, Q2, own2 = self._classes(exclude=drop)
        sims = F @ Q2.T
        neg = np.stack([sims[:, own2 == k].max(1) for k in range(len(names2))], 1)
        logits = np.concatenate([t[:, None], neg], 1) * self.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p = p[:, 0] / p.sum(1)
        return p.astype(np.float32), sorted(drop - {prompt})

    # ------------------------------------------------------------------ io
    def save(self, out_dir):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        extra = {} if self.lsum is None else {"gt_labels": self.gt_labels()}
        np.savez_compressed(out / "embedded_map.npz", points=self.points, feat=self.features.astype(np.float16),
                            count=self.count[:self.n].copy(), colors=self.colors, backbone=self.backbone,
                            variant="plain", voxel=self.voxel, keyframes=np.array(self.keyframes), **extra)
        return out / "embedded_map.npz"
