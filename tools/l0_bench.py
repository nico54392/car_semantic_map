#!/usr/bin/env python3
"""
Benchmark: which per-pixel features would make the best queryable map, on terrain AND common objects?

  sam3   -> L0-eot / L0-words / L0-attn (SAM3 layer-0 stored query, see l0_query.py for the idea)
            + SAM3-semantic (full SAM3 per class = ceiling)              [env: sam3, 4090]
  pe-b   -> our current map features (PE-Core B/16, 672 px)               [env: nico_clip]
  clip   -> CLIP B/16 (old baseline)                                      [env: nico_clip]

Datasets (same images, same classes, same scoring for every model):
  rellis -> the RELLIS-3D sample frame, Jaime's 19 terrain classes and prompts (off-road)
  coco   -> COCO val2017 images with everyday objects (cat, dog, bottle, bench, chair, car…),
            all 80 COCO object classes competing on every image. Ground truth = COCO object masks.
            Pixels with no object annotated (background) are ignored: this asks "WHICH object is
            this?", not "is there an object?".

Steps (4090, data downloaded once into data/coco, ~250 MB annotations zip deleted after use):
  python tools/l0_bench.py prep-coco --per_class 3                          # any env with PIL + numpy
  conda activate sam3 && export PYTHONPATH=~/nico_ws/src/sam3 HF_HUB_OFFLINE=1
  python tools/l0_bench.py run --model sam3 --dataset rellis coco
  conda activate nico_clip && unset PYTHONPATH
  python tools/l0_bench.py run --model pe-b --dataset rellis coco
  python tools/l0_bench.py summary

Scoring: dataset-level IoU per class (intersection and union summed over all images), mIoU over the
classes present in the ground truth, pixel accuracy over labelled pixels.
"""
import argparse, json, random, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RELLIS = {1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
          10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
          27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}
RELLIS_FRAME = "frame000104-1581624663_149"
FOCUS = ["cat", "dog", "bottle", "bench", "chair", "car", "bicycle", "cup", "potted plant", "person",
         "couch", "dining table"]
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"


# ---------------------------------------------------------------- data
def prep_coco(a):
    d = Path(a.coco_dir); (d / "images").mkdir(parents=True, exist_ok=True); (d / "gt").mkdir(exist_ok=True)
    ann = d / "instances_val2017.json"
    if not ann.exists():
        z = d / "annotations.zip"
        print("downloading COCO annotations (~250 MB)…")
        urllib.request.urlretrieve(ANN_URL, z)
        with zipfile.ZipFile(z) as zf:
            ann.write_bytes(zf.read("annotations/instances_val2017.json"))
        z.unlink()
    js = json.load(open(ann))
    cats = sorted(js["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]; cid2k = {c["id"]: k + 1 for k, c in enumerate(cats)}   # 0 = ignore
    by_img = {}
    for an in js["annotations"]:
        by_img.setdefault(an["image_id"], []).append(an)
    imgs = {im["id"]: im for im in js["images"]}
    rng = random.Random(a.seed); chosen = []
    for f in a.focus:
        k = names.index(f) + 1
        cand = [i for i, anns in by_img.items() if any(cid2k[x["category_id"]] == k and not x["iscrowd"]
                                                        and x["area"] > a.min_area for x in anns) and i not in chosen]
        chosen += rng.sample(sorted(cand), min(a.per_class, len(cand)))
    print(f"{len(chosen)} images ({a.per_class} per focus class: {', '.join(a.focus)})")
    meta = []
    for i in chosen:
        im = imgs[i]; jp = d / "images" / im["file_name"]
        if not jp.exists():
            urllib.request.urlretrieve(IMG_URL.format(im["file_name"]), jp)
        gt = Image.new("L", (im["width"], im["height"]), 0); dr = ImageDraw.Draw(gt)
        for an in sorted(by_img[i], key=lambda x: -x["area"]):         # small objects painted last = on top
            if an["iscrowd"] or not isinstance(an["segmentation"], list):
                continue
            for poly in an["segmentation"]:
                if len(poly) >= 6:
                    dr.polygon(poly, fill=cid2k[an["category_id"]])
        gt.save(d / "gt" / (Path(im["file_name"]).stem + ".png"))
        meta.append(im["file_name"])
    json.dump(dict(classes=names, images=meta), open(d / "subset.json", "w"), indent=1)
    print("saved", d / "subset.json")


def load_dataset(name, a):
    """-> class names, prompt strings per class, list of (id, rgb uint8 HxWx3, gt HxW int with 0 = ignore)"""
    if name == "rellis":
        from car_semantic_map.evaluate import load_ontology
        onto = load_ontology(ROOT / "config" / "ontology.json")
        ids = list(RELLIS); names = [RELLIS[i] for i in ids]
        prompts = [onto.get(n, {}).get("prompts", [n])[0] for n in names]
        d = Path(a.rellis_dir)
        img = Image.open(d / f"{RELLIS_FRAME}.jpg").convert("RGB")             # full res in, scored at 300x480
        raw = np.array(Image.fromarray(np.array(Image.open(d / f"{RELLIS_FRAME}.png"))).resize((480, 300), Image.NEAREST))
        lut = np.zeros(256, int); lut[ids] = np.arange(1, len(ids) + 1)
        return names, prompts, [(RELLIS_FRAME, np.array(img), lut[raw])]
    d = Path(a.coco_dir); meta = json.load(open(d / "subset.json"))
    items = []
    for fn in meta["images"]:
        items.append((Path(fn).stem, np.array(Image.open(d / "images" / fn).convert("RGB")),
                      np.array(Image.open(d / "gt" / (Path(fn).stem + ".png"))).astype(int)))
    return meta["classes"], list(meta["classes"]), items


# ---------------------------------------------------------------- models -> {method: (K, h, w) scores}
class Sam3Runner:
    def __init__(self, a):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = build_sam3_image_model(**(dict(checkpoint_path=a.checkpoint, load_from_HF=False) if a.checkpoint else {}))
        self.proc = Sam3Processor(self.model)
        lb = self.model.backbone.language_backbone; self.tok, self.ctx = lb.tokenizer, lb.context_length
        mha = self.model.transformer.encoder.layers[0].cross_attn_image
        W, b = mha.in_proj_weight.detach().float().cpu(), mha.in_proj_bias.detach().float().cpu()
        self.D = W.shape[1]; self.H = mha.num_heads; self.dh = self.D // self.H
        self.Wq, self.bq, self.Wk, self.bk = W[:self.D], b[:self.D], W[self.D:2 * self.D], b[self.D:2 * self.D]
        self.cap = cap = {}
        orig = mha.forward
        def fwd(*args, **kw):
            res = orig(*args, **kw)
            cap["x"] = kw["query"][0].detach().float().cpu(); cap["t"] = kw["key"][0].detach().float().cpu()
            return res
        mha.forward = fwd
        orig_fg = self.model.forward_grounding
        def fg(*args, **kw):
            o = orig_fg(*args, **kw); cap["sem"] = o.get("semantic_seg"); return o
        self.model.forward_grounding = fg
        self.eot = {}
        self.drift = 0.0

    def __call__(self, rgb, prompts):
        state = self.proc.set_image(Image.fromarray(rgb))
        x0, keys, sems = None, [], []
        for p in prompts:
            state = self.proc.set_text_prompt(prompt=p, state=state)
            if x0 is None:
                x0 = self.cap["x"]
            else:
                self.drift = max(self.drift, float((self.cap["x"] - x0).abs().max()))
            if p not in self.eot:
                self.eot[p] = int(self.tok([p], context_length=self.ctx)[0].argmax())
            keys.append((self.cap["t"], self.eot[p]))
            sems.append(self.cap["sem"][0, 0].float().cpu() if self.cap.get("sem") is not None else None)
        n = x0.shape[0]; g = int(round(n ** 0.5))
        q = (x0 @ self.Wq.T + self.bq).view(n, self.H, self.dh)
        out = {"L0-eot": [], "L0-words": [], "L0-attn": []}
        for t, eot in keys:
            k = (t @ self.Wk.T + self.bk).view(-1, self.H, self.dh)
            lg = torch.einsum("nhd,thd->hnt", q, k) / self.dh ** 0.5
            words = list(range(1, eot)) or [eot]
            out["L0-eot"].append(lg[:, :, eot].mean(0))
            out["L0-words"].append(lg[:, :, words].mean(-1).mean(0))
            out["L0-attn"].append(lg.softmax(-1)[:, :, words + [eot]].sum(-1).mean(0))
        res = {m: torch.stack(v).view(len(prompts), g, g) for m, v in out.items()}
        if all(s is not None for s in sems):
            res["SAM3-semantic"] = torch.stack(sems)
        return res


class ClipLikeRunner:
    def __init__(self, a):
        from car_semantic_map.clip_features import make_extractor
        self.name = a.model
        self.ext = make_extractor(a.model, a.clip_weights if a.model == "clip" else None)
        self.cache = {}

    def __call__(self, rgb, prompts):
        key = tuple(prompts)
        if key not in self.cache:
            self.cache[key] = self.ext.class_features([[p] for p in prompts])
        Q, owner = self.cache[key]
        fm = self.ext.image_features(rgb)
        sim = torch.einsum("chw,qc->qhw", fm.float().cpu(), Q.float())
        cls = torch.stack([sim[torch.from_numpy(np.nonzero(owner == k)[0])].max(0).values for k in range(len(prompts))])
        return {self.name.upper(): cls}


# ---------------------------------------------------------------- run / score
def run(a):
    runner = Sam3Runner(a) if a.model == "sam3" else ClipLikeRunner(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.dataset:
        names, prompts, items = load_dataset(ds, a)
        K = len(names); acc = {}
        t0 = time.time(); examples = []
        for j, (iid, rgb, gt) in enumerate(items):
            maps = runner(rgb, prompts)
            H, W = gt.shape; lab = gt > 0
            for m, s in maps.items():
                up = F.interpolate(s[None].float(), (H, W), mode="bilinear", align_corners=False)[0]
                pred = up.argmax(0).numpy() + 1
                st = acc.setdefault(m, dict(inter=np.zeros(K + 1), union=np.zeros(K + 1), gtpx=np.zeros(K + 1),
                                            correct=0, labelled=0))
                for c in np.unique(np.concatenate([gt[lab], pred[lab]])):
                    g_, p_ = (gt == c) & lab, (pred == c) & lab
                    st["inter"][c] += (g_ & p_).sum(); st["union"][c] += (g_ | p_).sum(); st["gtpx"][c] += g_.sum()
                st["correct"] += int((pred[lab] == gt[lab]).sum()); st["labelled"] += int(lab.sum())
                if j < 4:
                    examples.append((iid, m, rgb, gt, pred))
            print(f"  {ds} {j + 1}/{len(items)} {iid}  ({time.time() - t0:.0f}s)", flush=True)
        res = {}
        for m, st in acc.items():
            present = [c for c in range(1, K + 1) if st["gtpx"][c] > 0]
            iou = {names[c - 1]: float(st["inter"][c] / max(st["union"][c], 1)) for c in present}
            res[m] = dict(acc=st["correct"] / max(st["labelled"], 1), miou=float(np.mean(list(iou.values()))),
                          iou=iou, n_images=len(items), n_classes_competing=K)
        if a.model == "sam3":
            res["_check_prompt_independence_max_diff"] = runner.drift
        json.dump(res, open(out / f"{ds}_{a.model}.json", "w"), indent=2)
        print(f"\n{ds} / {a.model}: {len(items)} images, {K} classes competing")
        for m, r in res.items():
            if not m.startswith("_"):
                print(f"  {m:15s} acc {r['acc']:.3f}  mIoU {r['miou']:.3f}")
        if a.model == "sam3":
            print(f"  check: layer-0 inputs identical across prompts (max diff {runner.drift:.1e})")
        draw(examples, names, out / f"{ds}_{a.model}_examples.png")


def draw(examples, names, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    rng = np.random.default_rng(1); pal = rng.integers(40, 255, (len(names) + 1, 3)); pal[0] = 0
    ids = sorted({e[0] for e in examples})[:4]; methods = sorted({e[1] for e in examples})
    fig, ax = plt.subplots(len(ids), 2 + len(methods), figsize=(3.2 * (2 + len(methods)), 2.6 * len(ids)), squeeze=False)
    for r, iid in enumerate(ids):
        ex = {e[1]: e for e in examples if e[0] == iid}; e0 = next(iter(ex.values()))
        ax[r, 0].imshow(e0[2]); ax[r, 0].set_title(iid, fontsize=8)
        ax[r, 1].imshow(pal[e0[3]]); ax[r, 1].set_title("ground truth", fontsize=8)
        for c, m in enumerate(methods):
            p = ex[m][4].copy(); p[e0[3] == 0] = 0                       # show only labelled pixels
            ax[r, 2 + c].imshow(pal[p]); ax[r, 2 + c].set_title(m, fontsize=8)
    for x in ax.ravel(): x.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def summary(a):
    out = Path(a.out); rows = []
    for f in sorted(out.glob("*.json")):
        ds, model = f.stem.split("_", 1)
        for m, r in json.load(open(f)).items():
            if not m.startswith("_"):
                rows.append((ds, m, r))
    for ds in sorted({r[0] for r in rows}):
        sub = sorted([r for r in rows if r[0] == ds], key=lambda r: -r[2]["miou"])
        n = sub[0][2]
        print(f"\n{ds}: {n['n_images']} images, {n['n_classes_competing']} classes competing")
        print(f"  {'method':15s}  acc    mIoU")
        for _, m, r in sub:
            print(f"  {m:15s}  {r['acc']:.3f}  {r['miou']:.3f}")
        best = sub[0][2]["iou"]
        print("  per class, best method: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                                     sorted(best.items(), key=lambda x: -x[1])[:15]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-coco")
    p.add_argument("--coco_dir", default="data/coco"); p.add_argument("--per_class", type=int, default=3)
    p.add_argument("--focus", nargs="+", default=FOCUS); p.add_argument("--min_area", type=float, default=2000)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=["sam3", "pe-b", "clip"])
    r.add_argument("--dataset", nargs="+", default=["rellis", "coco"], choices=["rellis", "coco"])
    r.add_argument("--rellis_dir", default="data/rellis_sample"); r.add_argument("--coco_dir", default="data/coco")
    r.add_argument("--checkpoint", default=None); r.add_argument("--clip_weights", default=None)
    r.add_argument("--out", default="out/l0_bench")
    s = sub.add_parser("summary"); s.add_argument("--out", default="out/l0_bench")
    a = ap.parse_args()
    if a.cmd == "prep-coco":
        prep_coco(a)
    elif a.cmd == "summary":
        summary(a)
    elif torch.cuda.is_available() and a.model == "sam3":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            run(a)
    else:
        with torch.no_grad():
            run(a)


if __name__ == "__main__":
    main()
