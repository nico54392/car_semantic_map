"""Class ontology: prompts, traversability, mu, and the mapping onto RELLIS eval ids.

Accepts our ontology_rellis.json and is lenient enough to read the harness's
ontology.json / Jaime's class JSON (keys 'name'|'class', 'prompt'|'prompts'|'sam3_prompt').
"""
from __future__ import annotations

import json
import os
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
    def load(cls, path: str, only: list[str] | None = None, rellis_only: bool = False) -> "Ontology":
        """rellis_only: keep only classes that exist in RELLIS (drops goal objects like bench/car,
        which would otherwise steal predictions when scoring against RELLIS labels)."""
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
            if rellis_only and eval_id is None:
                continue
            t = it.get("traversability", it.get("trav"))
            u = it.get("mu", it.get("friction"))
            # rigid = its height is an obstacle. Default: classes with traversability 0 (tree, log, rock...)
            rigid = it.get("rigid", t is not None and float(t) <= 0.0 and name != "sky")
            out.append(OntClass(name, queries, eval_id,
                                0.5 if t is None else float(t),       # sky etc.: never on the ground map
                                0.0 if u is None else float(u),
                                tuple(it.get("color", (128, 128, 128))), bool(rigid)))
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


def default_path() -> str:
    """the repo's config/ontology.json if rellis3d_eval sits in car_semantic_map, else the bundled copy"""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(here, "..", "config", "ontology.json")
    return repo if os.path.exists(repo) else os.path.join(here, "ontology_rellis.json")
