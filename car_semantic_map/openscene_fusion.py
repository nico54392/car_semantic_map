"""Lift 2D stuff onto the map points.

fuse_features: OpenScene 2D fusion (average per-pixel embeddings over all views that see a point)
fuse_labels:   majority vote of per-view 2D label maps (e.g. plain SAM3 masks -> class ids)
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from .scene_io import project


def fuse_features(scene, extractor, log=True):
    N = len(scene.points)
    feat_sum, count = None, np.zeros(N, np.int32)
    t_img = []
    for view in scene.views:
        t0 = time.time()
        fmap = extractor.image_features(view.rgb)            # [C,h,w]
        t_img.append(time.time() - t0)
        u, v, vis = project(scene.points, view)
        if not vis.any():
            continue
        H, W = view.depth.shape
        # pixel centers -> grid_sample coords (align_corners=False)
        g = np.stack([(u[vis] + 0.5) / W * 2 - 1, (v[vis] + 0.5) / H * 2 - 1], -1)
        samp = F.grid_sample(fmap[None], torch.from_numpy(g).float()[None, None], align_corners=False)[0, :, 0].T
        if feat_sum is None:
            feat_sum = torch.zeros(N, samp.shape[1])
        feat_sum[torch.from_numpy(np.nonzero(vis)[0])] += samp
        count[vis] += 1
    feat = F.normalize(feat_sum, dim=-1)
    feat[torch.from_numpy(count == 0)] = 0
    if log:
        print(f"fused {len(scene.views)} views, {np.mean(t_img):.2f}s/view features, "
              f"{(count > 0).mean():.1%} of points seen (median {np.median(count[count > 0]):.0f} views/point)")
    return feat, count, float(np.mean(t_img))


def classify(feat, query_emb, query_owner, class_ids, count=None):
    """class score = best of its queries (the paper's 'addition'), then argmax (paper eq. 6)"""
    q = feat @ query_emb.T                                   # [N,Q]
    K = len(class_ids)
    logits = torch.full((q.shape[0], K), -1e9)
    for k in range(K):
        logits[:, k] = q[:, torch.from_numpy(np.nonzero(query_owner == k)[0])].max(1).values
    pred = np.asarray(class_ids)[logits.argmax(1).numpy()]
    prob = (logits * 100).softmax(-1).numpy()
    if count is not None:
        pred[count == 0] = -1
    return pred, prob


def fuse_labels(scene, label_maps, n_classes=256):
    """label_maps: {view_name: HxW int array}. Majority vote per point, -1 if never seen."""
    votes = np.zeros((len(scene.points), n_classes), np.int32)
    for view in scene.views:
        if view.name not in label_maps:
            continue
        lm = label_maps[view.name]
        u, v, vis = project(scene.points, view)
        idx = np.nonzero(vis)[0]
        lab = lm[np.round(v[vis]).astype(int), np.round(u[vis]).astype(int)]
        np.add.at(votes, (idx, lab), 1)
    pred = votes.argmax(1)
    pred[votes.sum(1) == 0] = -1
    return pred
