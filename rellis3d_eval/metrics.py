"""Metrics from Jaime's paper (Acc, mIoU, FWIoU, mean P/R/F1, SCS, SwCM) plus the
control-side ones (traversability MAE, mu MAE, obstacle F1). Accumulates over frames."""
from __future__ import annotations

import numpy as np

from ontology import Ontology
from rellis_io import EVAL_CLASSES, IGNORE


class Accumulator:
    def __init__(self, ont: Ontology, text_sim: np.ndarray | None = None, obstacle_thr: float = 0.5):
        self.ont = ont
        self.E = sorted({int(e) for e in ont.eval_id if e >= 0})        # evaluated RELLIS ids
        self.row = {e: i for i, e in enumerate(self.E)}
        self.C = np.zeros((len(self.E), len(ont)), np.float64)          # GT eval class x predicted ont class
        self.text_sim = text_sim
        self.trav_gt, self.mu_gt = ont.gt_tables()
        self.thr = obstacle_thr
        self.err_t = self.err_m = self.n_ctrl = 0.0
        self.tp = self.fp = self.fn = 0.0

    def add(self, gt_eval: np.ndarray, probs: np.ndarray, valid: np.ndarray | None = None):
        """gt_eval (N,) RELLIS eval ids; probs (N,K) over ontology classes."""
        gt_eval = gt_eval.reshape(-1)
        probs = probs.reshape(len(gt_eval), -1)
        keep = gt_eval != IGNORE
        if valid is not None:
            keep &= valid.reshape(-1)
        lut = np.full(len(EVAL_CLASSES), -1)
        for e, i in self.row.items():
            lut[e] = i
        r = lut[gt_eval]
        keep &= r >= 0
        r, p = r[keep], probs[keep]
        pred = p.argmax(1)
        np.add.at(self.C, (r, pred), 1)
        # control metrics on expected values
        g = gt_eval[keep]
        t_hat, m_hat = p @ self.ont.trav, p @ self.ont.mu
        t_gt, m_gt = self.trav_gt[g], self.mu_gt[g]
        self.err_t += np.abs(t_hat - t_gt).sum()
        self.err_m += np.abs(m_hat - m_gt).sum()
        self.n_ctrl += len(g)
        ob_hat, ob_gt = t_hat < self.thr, t_gt < self.thr
        self.tp += (ob_hat & ob_gt).sum(); self.fp += (ob_hat & ~ob_gt).sum(); self.fn += (~ob_hat & ob_gt).sum()

    # ------------------------------------------------------------------
    def _collapsed(self, C):
        """(|E|, |ont|) -> (|E|, |E|+1): predicted ont classes folded onto eval ids, last col = other."""
        out = np.zeros((len(self.E), len(self.E) + 1))
        for j, e in enumerate(self.ont.eval_id):
            out[:, self.row[int(e)] if e >= 0 else -1] += C[:, j]
        return out

    def _sim_rows(self):
        """similarity between each GT eval class (its first ontology class) and each ont class"""
        if self.text_sim is None:
            return None
        first = [int(np.where(self.ont.eval_id == e)[0][0]) for e in self.E]
        return self.text_sim[first]

    def summary(self, C=None) -> dict:
        C = self.C if C is None else C
        Cc = self._collapsed(C)
        tp = np.diag(Cc[:, :-1])
        gt_n = Cc.sum(1)
        pr_n = Cc[:, :-1].sum(0)
        tot = Cc.sum()
        iou = tp / np.maximum(gt_n + pr_n - tp, 1)
        present = gt_n > 0
        prec = np.where(pr_n > 0, tp / np.maximum(pr_n, 1), 0)
        rec = np.where(gt_n > 0, tp / np.maximum(gt_n, 1), 0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
        res = {
            "n": int(tot),
            "Acc": tp.sum() / max(tot, 1),
            "mIoU": iou[present].mean() if present.any() else 0.0,
            "mIoU_all": iou.mean(),
            "FWIoU": (gt_n / max(tot, 1) * iou).sum(),
            "P": prec[present].mean() if present.any() else 0.0,
            "R": rec[present].mean() if present.any() else 0.0,
            "F1": f1[present].mean() if present.any() else 0.0,
            "per_class_IoU": {EVAL_CLASSES[e]: float(iou[i]) for i, e in enumerate(self.E) if present[i]},
            "per_class_share": {EVAL_CLASSES[e]: float(gt_n[i] / max(tot, 1)) for i, e in enumerate(self.E) if present[i]},
        }
        S = self._sim_rows()
        if S is not None:
            res["SCS"] = float((C * S).sum() / max(C.sum(), 1))
            res["SwCM_sum"] = float((C * (1 - S)).sum() / max(C.sum(), 1))   # mean semantic cost per element
        return res

    def baseline(self) -> dict:
        """Everything predicted as the most common GT class."""
        rows = self.C.sum(1)
        e = self.E[int(rows.argmax())]
        j = int(np.where(self.ont.eval_id == e)[0][0])
        B = np.zeros_like(self.C)
        B[:, j] = rows
        return self.summary(B) | {"baseline_class": EVAL_CLASSES[e]}

    def control(self) -> dict:
        n = max(self.n_ctrl, 1)
        f1 = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        return {"trav_MAE": self.err_t / n, "mu_MAE": self.err_m / n, "obstacle_F1": f1}


def fmt_row(name: str, s: dict) -> str:
    cols = ["Acc", "mIoU", "mIoU_all", "FWIoU", "P", "R", "F1", "SCS"]
    return f"{name:<22}" + "".join(f"{s.get(c, float('nan')):>9.3f}" for c in cols)


def fmt_header() -> str:
    return f"{'':<22}" + "".join(f"{c:>9}" for c in ["Acc", "mIoU", "mIoU_all", "FWIoU", "P", "R", "F1", "SCS"])
