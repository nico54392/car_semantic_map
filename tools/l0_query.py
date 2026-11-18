#!/usr/bin/env python3
"""
Can SAM3's *first look* at the text be stored per point and queried later with any prompt?

WHY
  Inside SAM3, image patches first meet the prompt in layer 0 of the fusion encoder (cross-attention).
  Everything before that point (backbone, layer-0 self-attention, norm) never sees the text, so the
  layer-0 query  q = W_q·x  of every patch is prompt-independent: compute it once, store it per 3D point.
  A new prompt then only needs the text encoder:  k = W_k·t, score = q·k / sqrt(32), per head.
  That would be a SAM3-based, open-vocabulary, queryable-later map with no fusion/decoder re-run.
  fusion_peek showed layer-0 attention lines up with SAM3's own masks for single prompts (grass 0.84).
  This test asks the real question: with ALL classes competing, against RELLIS ground truth, how good
  is it compared with PE-B (our current map features) and with full SAM3?

PROCEDURE (one RELLIS frame, Jaime's 19 class prompts from config/ontology.json)
  1. Run full SAM3 once per class prompt. Hooks record, from layer 0's cross-attention:
     the patch inputs (queries before W_q), the prompt tokens (keys before W_k), and SAM3's own
     attention weights; plus SAM3's final per-pixel `semantic_seg` logit for that class.
  2. Checks: (a) the layer-0 patch inputs are identical for every prompt (prompt-independence),
     (b) our q·k softmax reproduces SAM3's recorded attention (so a stored-q map would be exact).
  3. Score every pixel against every class, three ways, all from stored q + class keys:
       L0-eot   : q·k of the <end_of_text> token (the token that has read the whole prompt)
       L0-words : mean q·k over the prompt's word tokens
       L0-attn  : SAM3's layer-0 attention mass on words + <end_of_text> (padding acts as a baseline)
     and the reference:
       SAM3-semantic : full SAM3 per class (6 fusion layers + decoder), = the ceiling for this route
     Per pixel the highest-scoring class wins (that competition cancels each patch's own offset).
  4. Score against RELLIS ground truth exactly like backbone_compare.py (300x480, void ignored,
     same 19 classes), so numbers are directly comparable with PE-B (2D acc 0.642 / mIoU 0.300).

READING THE RESULT
  L0 >= PE-B (~0.30 mIoU)  -> worth building: store layer-0 q per point, query any prompt later
  L0 << PE-B               -> SAM3's first look is too coarse; PE-B stays the map, SAM3 = precise check
  SAM3-semantic            -> how much is left on the table between first look and final answer
                              (also the 2D version of "soft SAM3 fusion" vs Jaime's hard vote)

Run on the 4090 (SAM3 lives there), in Jaime's env without changing it:
  conda activate sam3
  export PYTHONPATH=~/nico_ws/src/sam3 HF_HUB_OFFLINE=1
  python tools/l0_query.py data/rellis_sample --out out/l0_query > out/l0_query.log 2>&1; tail -40 out/l0_query.log
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.evaluate import load_ontology

RELLIS = {1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
          10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
          27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}
FRAME = "frame000104-1581624663_149"
EH, EW = 300, 480                                   # same evaluation size as backbone_compare.py


def scores(gt, pr):                                 # identical to backbone_compare.scores
    ious = []
    for c in np.unique(gt):
        if c == 0:
            continue
        g, p = gt == c, pr == c
        ious.append((RELLIS[int(c)], (g & p).sum() / max((g | p).sum(), 1)))
    v = gt != 0
    return float((gt[v] == pr[v]).mean()), float(np.mean([i for _, i in ious])), dict(ious)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--frame", default=FRAME)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="out/l0_query")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(**(dict(checkpoint_path=a.checkpoint, load_from_HF=False) if a.checkpoint else {}))
    proc = Sam3Processor(model)
    lb = model.backbone.language_backbone
    tok, ctx = lb.tokenizer, lb.context_length

    layer0 = model.transformer.encoder.layers[0]
    mha = layer0.cross_attn_image
    W, b = mha.in_proj_weight.detach().float(), mha.in_proj_bias.detach().float()
    D = W.shape[1]; H = mha.num_heads; dh = D // H
    Wq, bq, Wk, bk = W[:D], b[:D], W[D:2 * D], b[D:2 * D]

    cap = {}
    orig = mha.forward
    def fwd(*args, **kw):                            # record layer-0 cross-attention inputs + weights
        kw["need_weights"] = True; kw["average_attn_weights"] = False
        res = orig(*args, **kw)
        cap["x"] = kw["query"][0].detach().float().cpu()          # (HW, 256) patch inputs, before W_q
        cap["t"] = kw["key"][0].detach().float().cpu()            # (T, 256)  prompt tokens, before W_k
        cap["pad"] = kw["key_padding_mask"][0].detach().cpu() if kw.get("key_padding_mask") is not None else None
        cap["w"] = res[1][0].detach().float().cpu() if res[1] is not None else None   # (heads, HW, T)
        return res
    mha.forward = fwd
    orig_fg = model.forward_grounding
    def fg(*args, **kw):
        o = orig_fg(*args, **kw); cap["sem"] = o.get("semantic_seg"); return o
    model.forward_grounding = fg

    onto = load_ontology(ROOT / "config" / "ontology.json")
    ids = list(RELLIS)
    prompts = [onto.get(RELLIS[i], {}).get("prompts", [RELLIS[i]])[0] for i in ids]   # Jaime's " . " strings

    img = Image.open(data / f"{a.frame}.jpg").convert("RGB")
    gt = np.array(Image.fromarray(np.array(Image.open(data / f"{a.frame}.png"))).resize((EW, EH), Image.NEAREST))
    state = proc.set_image(img)

    x0, per = None, []
    for cid, p in zip(ids, prompts):
        state = proc.set_text_prompt(prompt=p, state=state)
        tid = tok([p], context_length=ctx)[0]
        eot = int(tid.argmax())                                    # CLIP-style: <end_of_text> has the top id
        per.append(dict(cid=cid, prompt=p, t=cap["t"], pad=cap["pad"], w=cap["w"], eot=eot,
                        sem=None if cap.get("sem") is None else cap["sem"][0, 0].float().cpu()))
        if x0 is None:
            x0 = cap["x"]; hw = x0.shape[0]; g = int(round(hw ** 0.5))
            drift = 0.0
        else:
            drift = max(drift, float((cap["x"] - x0).abs().max()))

    q = (x0 @ Wq.T.cpu() + bq.cpu()).view(hw, H, dh)               # stored per point in the real map
    rows, maps, check = [], {}, []
    L = {"L0-eot": [], "L0-words": [], "L0-attn": []}
    for c in per:
        k = (c["t"] @ Wk.T.cpu() + bk.cpu()).view(-1, H, dh)       # computed at query time from text only
        logit = torch.einsum("nhd,thd->hnt", q, k) / dh ** 0.5     # (heads, HW, T)
        mine = logit.softmax(-1)
        if c["w"] is not None:
            check.append(float((mine - c["w"]).abs().max()))
        words = list(range(1, c["eot"]))
        L["L0-eot"].append(logit[:, :, c["eot"]].mean(0))
        L["L0-words"].append(logit[:, :, words].mean(-1).mean(0) if words else logit[:, :, c["eot"]].mean(0))
        L["L0-attn"].append(mine[:, :, words + [c["eot"]]].sum(-1).mean(0))
    for name, lst in L.items():
        maps[name] = torch.stack(lst).view(len(ids), g, g)
    if all(c["sem"] is not None for c in per):
        maps["SAM3-semantic"] = torch.stack([c["sem"] for c in per])

    res = {}
    for name, m in maps.items():
        up = F.interpolate(m[None].float(), (EH, EW), mode="bilinear", align_corners=False)[0]
        pred = np.array(ids)[up.argmax(0).numpy()]
        acc, miou, iou = scores(gt, pred)
        res[name] = dict(acc=acc, miou=miou, iou=iou); maps[name] = pred

    present = [RELLIS[int(c)] for c in np.unique(gt) if c != 0]
    lines = [f"frame {a.frame}, {len(ids)} classes competing, scored on {len(present)} classes present in the GT",
             "",
             "CHECKS",
             f"  layer-0 patch inputs, max difference across the {len(ids)} prompts: {drift:.2e}"
             f"   ({'prompt-independent, OK' if drift < 5e-2 else 'NOT prompt-independent'})",
             f"  our q.k softmax vs SAM3's own layer-0 attention, max difference: "
             f"{max(check):.2e}" if check else "  attention weights not returned, reproduction not checked",
             "",
             f"{'method':15s}  acc    mIoU   " + "  ".join(f"{n[:8]:>8s}" for n in present)]
    for name in ["SAM3-semantic", "L0-eot", "L0-words", "L0-attn"]:
        if name in res:
            r = res[name]
            lines.append(f"{name:15s}  {r['acc']:.3f}  {r['miou']:.3f}  " + "  ".join(f"{r['iou'].get(n, 0):8.2f}" for n in present))
    lines += ["", "reference, same frame/classes/scoring (backbone_compare.py, 2026-11-13):",
              "  PE-B plain 672 px   acc 0.642  mIoU 0.300      CLIP B/16   acc 0.539  mIoU 0.201"]
    txt = "\n".join(lines); print(txt)
    (out / "results.txt").write_text(txt + "\n")
    json.dump(res, open(out / "results.json", "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        pal = {i: np.array(onto.get(RELLIS[i], {}).get("color", rng.integers(40, 255, 3))) for i in ids}
        pal[0] = np.zeros(3)
        col = lambda m: np.stack([pal.get(int(v), np.zeros(3)) for v in m.ravel()]).reshape(*m.shape, 3).astype(np.uint8)
        show = [("image", np.array(img.resize((EW, EH)))), ("ground truth", col(gt))] + \
               [(f"{n}  mIoU {res[n]['miou']:.3f}", col(maps[n])) for n in ["SAM3-semantic", "L0-eot", "L0-words", "L0-attn"] if n in res]
        fig, ax = plt.subplots(2, 3, figsize=(15, 6.5)); ax = ax.ravel()
        for j, (t, im) in enumerate(show):
            ax[j].imshow(im); ax[j].set_title(t, fontsize=10)
        for x in ax: x.axis("off")
        fig.tight_layout(); fig.savefig(out / "l0_query.png", dpi=110)
    except Exception as e:
        print("(no figure:", e, ")")
    print("saved", out)


if __name__ == "__main__":
    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            main()
    else:
        main()
