"""
Check a scene export and score methods against its ground truth.

python tools/scene_eval.py check       <scene>
python tools/scene_eval.py openscene   <scene> [--backbone clip|pe-b|pe-l] [--variant last_v_only]
python tools/scene_eval.py fuse-labels <scene> --labels <dir with 000000.png ...> --name sam3_vote
python tools/scene_eval.py compare     <scene> [--backbone clip]   (text encoder used for SCS / SwCM)

--weights is an open_clip pretrained tag (downloads once) or a path to a .pt file.

Classes, prompts, traversability and friction come from ontology.json (matched to
scene.json by class name). Every method writes <scene>/preds/<name>.npz with `pred` (N,)
scene class ids (-1 = unseen), optionally `prob` (N,K) + `class_ids` (K,).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.scene_io import load_scene, check_conventions
from car_semantic_map.evaluate import evaluate, compare_table, load_ontology, class_similarity

ONTOLOGY = ROOT / "config" / "ontology.json"


def scene_prompts(classes, onto):
    ids = [c for c in sorted(classes) if c != 0]
    missing = [classes[c] for c in ids if classes[c] not in onto]
    if missing:
        print(f"note: {missing} not in ontology.json, using the bare class name as prompt")
    return ids, [onto.get(classes[c], {}).get("prompts", [classes[c]]) for c in ids]


def cmd_check(a):
    s = load_scene(a.scene, a.max_views)
    print(f"{len(s.points)} map points, {len(s.views)} views, classes {s.classes}")
    print("GT class counts:", {s.classes.get(int(c), c): int(n) for c, n in zip(*np.unique(s.point_labels, return_counts=True))})
    check_conventions(s)


def cmd_openscene(a):
    from car_semantic_map.clip_features import make_extractor
    from car_semantic_map.openscene_fusion import fuse_features, classify
    s = load_scene(a.scene, a.max_views)
    ids, prompts = scene_prompts(s.classes, load_ontology(a.ontology))
    ext = make_extractor(a.backbone, a.weights, **({'variant': a.variant} if a.backbone != 'clip' else {}))
    Q, owner = ext.class_features(prompts)
    feat, count, t_img = fuse_features(s, ext)
    pred, prob = classify(feat, Q, owner, ids, count)
    out = Path(a.scene) / "preds"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{a.name}.npz", pred=pred, prob=prob.astype(np.float16), class_ids=np.array(ids),
                        feat=feat.numpy().astype(np.float16), count=count, sec_per_view=t_img)
    print("saved", out / f"{a.name}.npz")


def cmd_fuse_labels(a):
    from PIL import Image
    from car_semantic_map.openscene_fusion import fuse_labels
    s = load_scene(a.scene, a.max_views)
    maps = {}
    for v in s.views:
        p = Path(a.labels) / f"{v.name}.png"
        if p.exists():
            m = np.array(Image.open(p))
            maps[v.name] = (m[..., 0] if m.ndim == 3 else m).astype(np.int64)
    print(f"{len(maps)}/{len(s.views)} views have label maps")
    pred = fuse_labels(s, maps)
    out = Path(a.scene) / "preds"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{a.name}.npz", pred=pred)
    print("saved", out / f"{a.name}.npz")


def cmd_compare(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = load_scene(a.scene, max_views=1)              # only need the map
    onto = load_ontology(a.ontology)
    ids = [c for c in sorted(s.classes) if c != 0]
    names = [s.classes[c] for c in ids]

    sim = None
    cache = Path(a.scene) / f"class_similarity_{a.backbone}.json"     # SCS depends on the text encoder
    if cache.exists() and json.load(open(cache))["names"] == names:
        sim = np.array(json.load(open(cache))["sim"])
    else:
        from car_semantic_map.clip_features import make_extractor
        sim = class_similarity(onto, names, make_extractor(a.backbone, a.weights))
        json.dump({"names": names, "sim": sim.tolist(), "encoder": a.backbone}, open(cache, "w"))

    res, preds = {}, {}
    for f in sorted((Path(a.scene) / "preds").glob("*.npz")):
        d = np.load(f)
        preds[f.stem] = d["pred"]
        res[f.stem] = evaluate(s.point_labels, d["pred"], s.classes, onto, sim, s.points,
                               d["prob"].astype(np.float32) if "prob" in d else None,
                               list(d["class_ids"]) if "class_ids" in d else None)
        if "sec_per_view" in d:
            res[f.stem]["sec_per_view"] = float(d["sec_per_view"])
    # reference row: predict the most common GT class everywhere the best method saw.
    # SCS with CLIP text sims sits in a narrow band, so this shows what "doing nothing" scores.
    if preds:
        seen = np.max([p >= 0 for p in preds.values()], axis=0)
        vals, cnt = np.unique(s.point_labels[s.point_labels != 0], return_counts=True)
        base = np.where(seen, vals[cnt.argmax()], -1)
        res["baseline_majority"] = evaluate(s.point_labels, base, s.classes, onto, sim, s.points)
    table = compare_table(res)
    print(table)
    slim = {m: {k: v for k, v in r.items() if k not in ("confusion", "swcm")} for m, r in res.items()}
    (Path(a.scene) / "report.md").write_text(table + "\n\n```json\n" + json.dumps(slim, indent=2) + "\n```\n")

    # top-down maps, colors from the ontology
    col = {c: np.array(onto.get(s.classes[c], {}).get("color", [120, 120, 120])) / 255 for c in ids}
    col[0] = np.array([0.6, 0.6, 0.6]); col[-1] = np.array([1.0, 1.0, 1.0])
    panels = [("ground truth", s.point_labels)] + list(preds.items())   # baseline not drawn
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.4))
    for a_, (title, lab) in zip(np.atleast_1d(ax), panels):
        a_.scatter(-s.points[:, 1], s.points[:, 0], c=[col.get(int(c), (0, 0, 0)) for c in lab], s=1, marker="s")
        sub = "" if title == "ground truth" else f"  (Acc {res[title]['acc']:.2f}, mIoU {res[title]['miou']:.2f})"
        a_.set_title(title + sub, fontsize=10); a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=col[c]) for c in ids] + [plt.Rectangle((0, 0), 1, 1, fc="w", ec="k")],
               names + ["unseen"], loc="lower center", ncol=len(ids) + 1)
    plt.tight_layout(rect=(0, 0.07, 1, 0.94))
    plt.savefig(Path(a.scene) / "compare_topdown.png", dpi=90); plt.close()

    # paper fig. 5 style: confusion | similarity | SwCM, one row per method
    if sim is not None:
        fig, ax = plt.subplots(len(res), 3, figsize=(13, 4.2 * len(res)), squeeze=False)
        for row, (m, r) in enumerate(res.items()):
            C = np.array(r["confusion"], float); Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
            for j, (M, t) in enumerate([(Cn, f"{m}: confusion (row-norm)"), (sim, "semantic similarity"),
                                        (Cn * (1 - sim), f"SwCM (SCS {r['scs']:.3f})")]):
                im = ax[row, j].imshow(M, cmap="viridis", vmin=0, vmax=1)
                ax[row, j].set_xticks(range(len(names)), names, rotation=60, fontsize=8)
                ax[row, j].set_yticks(range(len(names)), names, fontsize=8)
                ax[row, j].set_title(t, fontsize=10)
                for (i, k), v in np.ndenumerate(M):
                    if len(names) <= 8:
                        ax[row, j].text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < 0.6 else "k")
        plt.tight_layout(); plt.savefig(Path(a.scene) / "compare_swcm.png", dpi=90); plt.close()
    print("wrote report.md, compare_topdown.png" + (", compare_swcm.png" if sim is not None else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["check", "openscene", "fuse-labels", "compare"]:
        q = sub.add_parser(name)
        q.add_argument("scene")
        q.add_argument("--max_views", type=int, default=None)
        q.add_argument("--ontology", default=str(ONTOLOGY))
        if name == "openscene":
            q.add_argument("--name", default=None, help="default: openscene_<backbone>")
        if name == "fuse-labels":
            q.add_argument("--labels", required=True)
            q.add_argument("--name", required=True)
        if name in ("openscene", "compare"):
            q.add_argument("--backbone", default="clip", help="clip | pe-b | pe-l | pe-s | pe-t")
            q.add_argument("--variant", default="last_v_only", help="PE only: plain | last_v | last_v_only")
            q.add_argument("--weights", default=None, help="open_clip tag or .pt (default per backbone)")
    a = p.parse_args()
    if a.cmd == 'openscene' and a.name is None:
        a.name = 'openscene_' + a.backbone.replace('-', '')
    {"check": cmd_check, "openscene": cmd_openscene, "fuse-labels": cmd_fuse_labels, "compare": cmd_compare}[a.cmd](a)
