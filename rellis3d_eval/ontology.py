"""Class ontology: prompts, traversability, mu, and the mapping onto RELLIS eval ids.

Accepts our ontology_rellis.json and is lenient enough to read the harness's
ontology.json / Jaime's class JSON (keys 'name'|'class', 'prompt'|'prompts'|'sam3_prompt').
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from rellis_io import EVAL_CLASSES


@dataclass
class OntClass:
    name: str
    queries: list[str]           # one entry per query; terms inside joined by ' . '
    eval_id: int | None          # RELLIS eval id (1..18) or None if no counterpart
    trav: float
    mu: float
    color: tuple
    rigid: bool = True


class Ontology:
    def __init__(self, classes: list[OntClass]):
        self.classes = classes
        self.names = [c.name for c in classes]
        self.trav = np.array([c.trav for c in classes], np.float32)
        self.mu = np.array([c.mu for c in classes], np.float32)
        self.rigid = np.array([c.rigid for c in classes])
        self.eval_id = np.array([c.eval_id if c.eval_id is not None else -1 for c in classes])

    def __len__(self):
        return len(self.classes)

    @classmethod
    def load(cls, path: str, only: list[str] | None = None) -> "Ontology":
        d = json.load(open(path))
        items = d["classes"] if isinstance(d, dict) and "classes" in d else d
        if isinstance(items, dict):          # {"grass": {...}, ...}
            items = [dict(v, name=k) for k, v in items.items()]
        out = []
        for it in items:
            name = it.get("name") or it.get("class")
            if only and name not in only:
                continue
            p = it.get("prompts") or it.get("prompt") or it.get("sam3_prompt") or name
            queries = p if isinstance(p, list) else [p]
            ev = it.get("eval_as", name)
            eval_id = EVAL_CLASSES.index(ev) if ev in EVAL_CLASSES and ev != "void" else None
            out.append(OntClass(name, queries, eval_id,
                                float(it.get("traversability", it.get("trav", 0.5))),
                                float(it.get("mu", it.get("friction", 0.5))),
                                tuple(it.get("color", (128, 128, 128))),
                                bool(it.get("rigid", True))))
        if not out:
            raise ValueError(f"no classes loaded from {path}")
        return cls(out)

    # per-eval-class lookup of GT traversability / mu (for MAE metrics)
    def gt_tables(self):
        t = np.full(len(EVAL_CLASSES), np.nan, np.float32)
        m = np.full(len(EVAL_CLASSES), np.nan, np.float32)
        for c in self.classes:
            if c.eval_id is not None:
                t[c.eval_id], m[c.eval_id] = c.trav, c.mu
        return t, m
