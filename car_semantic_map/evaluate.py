"""Score per-point predictions against the map's GT labels.

Metrics follow Bravo Algaba et al. (Language-Guided Semantic Terrain Segmentation):
Acc, mIoU, FWIoU, mean precision/recall/F1, SCS and the semantic-weighted confusion matrix
(SwCM). Computed per map point instead of per pixel. Plus the stuff that matters for
control: obstacle F1, traversability cost-map error and friction error.
"""
import json
import numpy as np


def load_ontology(path):
    d = json.load(open(path))
    return {c["name"]: c for c in d["classes"]}


def class_similarity(ontology, names, encoder):
    """cosine sim between class text embeddings (mean of their queries), clipped to [0,1]
    like the paper assumes. encoder: anything with .query_features(list[str])."""
    import torch.nn.functional as F
    embs = []
    for n in names:
        prompts = ontology.get(n, {}).get("prompts", [n])
        embs.append(F.normalize(encoder.query_features(prompts).mean(0), dim=-1))
    E = np.stack([e.numpy() for e in embs])
    return np.clip(E @ E.T, 0, 1)


def confusion(gt, pred, ids):
    idx = {c: i for i, c in enumerate(ids)}
    C = np.zeros((len(ids), len(ids)), np.int64)
    m = np.isin(gt, ids) & np.isin(pred, ids)
    np.add.at(C, (np.vectorize(idx.get)(gt[m]), np.vectorize(idx.get)(pred[m])), 1)
    return C


def costmap(points, values, cell=0.5):
    """paper's 2D cost map: mean traversability of the points in each xy cell"""
    ok = ~np.isnan(values)
    ij = np.floor(points[ok, :2] / cell).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    s = np.bincount(inv, values[ok]); n = np.bincount(inv)
    return {tuple(k): v for k, v in zip(keys, s / n)}


def evaluate(gt, pred, classes, ontology=None, sim=None, points=None, prob=None, class_ids=None, cell=0.5):
    """gt, pred: (N,) scene class ids, pred -1 = unseen. classes: {id: name}.
    sim: (K,K) class similarity in the order of sorted non-void ids."""
    ids = [c for c in sorted(classes) if c != 0]
    names = [classes[c] for c in ids]
    valid = gt != 0
    seen = valid & (pred >= 0)
    g, p = gt[seen], pred[seen]
    C = confusion(g, p, ids)
    present = C.sum(1) > 0
    tp = np.diag(C).astype(float)
    fp, fn = C.sum(0) - tp, C.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    nk = C.sum(1)
    out = dict(
        coverage=float(seen.sum() / max(valid.sum(), 1)),
        acc=float(tp.sum() / max(C.sum(), 1)),
        miou=float(iou[present].mean()) if present.any() else 0.0,
        fwiou=float((nk * iou).sum() / max(nk.sum(), 1)),
        precision=float(prec[present].mean()) if present.any() else 0.0,
        recall=float(rec[present].mean()) if present.any() else 0.0,
        f1=float(f1[present].mean()) if present.any() else 0.0,
        iou={n: float(iou[i]) for i, n in enumerate(names) if present[i]},
        confusion=C.tolist(), class_names=names)

    if sim is not None:
        out["scs"] = float((C * sim).sum() / max(C.sum(), 1))           # eq. 16
        out["swcm"] = (C * (1 - sim)).tolist()                          # eq. 18

    if ontology:
        def prop(key, labels):
            lut = {c: (ontology.get(n) or {}).get(key) for c, n in classes.items()}
            return np.array([np.nan if lut.get(int(c)) is None else lut[int(c)] for c in labels], float)
        tg, tpred = prop("traversability", gt), prop("traversability", pred)
        both = seen & ~np.isnan(tg) & ~np.isnan(tpred)
        if both.any():
            ob_g, ob_p = tg[both] == 0, tpred[both] == 0
            t_ = (ob_g & ob_p).sum()
            # undefined when the scene has no obstacle classes and none were predicted
            out["obstacle_f1"] = float(2 * t_ / (ob_g.sum() + ob_p.sum())) if (ob_g.sum() + ob_p.sum()) else None
            out["trav_mae"] = float(np.abs(tg[both] - tpred[both]).mean())
        if points is not None:
            cg = costmap(points[valid], tg[valid], cell)
            cp = costmap(points[seen], tpred[seen], cell)
            common = [k for k in cp if k in cg]
            if common:
                out["costmap_mae"] = float(np.mean([abs(cg[k] - cp[k]) for k in common]))
                out["costmap_cells"] = len(common)
        mg, mp = prop("mu", gt), prop("mu", pred)
        m = seen & ~np.isnan(mg) & ~np.isnan(mp)
        out["mu_mae"] = float(np.abs(mg[m] - mp[m]).mean()) if m.any() else None
        if prob is not None and class_ids is not None:
            mvec = prop("mu", np.asarray(class_ids))
            t = ~np.isnan(mvec)
            em = (prob[:, t] * mvec[t]).sum(1) / np.maximum(prob[:, t].sum(1), 1e-6)
            m2 = seen & ~np.isnan(mg)
            out["mu_mae_expected"] = float(np.abs(em[m2] - mg[m2]).mean()) if m2.any() else None
    return out


def compare_table(results):
    cols = [("coverage", "cov"), ("acc", "Acc"), ("miou", "mIoU"), ("fwiou", "FWIoU"), ("f1", "F1"),
            ("scs", "SCS"), ("obstacle_f1", "obst F1"), ("costmap_mae", "costmap MAE"), ("mu_mae", "μ MAE"),
            ("sec_per_view", "s/view")]
    cols = [c for c in cols if any(c[0] in r for r in results.values())]
    names = sorted({k for r in results.values() for k in r["iou"]})
    lines = ["| method | " + " | ".join(h for _, h in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, r in results.items():
        cells = []
        for k, _ in cols:
            v = r.get(k)
            cells.append("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Per-class IoU:", "", "| method | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for m, r in results.items():
        lines.append(f"| {m} | " + " | ".join(f"{r['iou'].get(n, float('nan')):.2f}" for n in names) + " |")
    return "\n".join(lines)
