#!/usr/bin/env python3
"""RELLIS-3D evaluation of the 2D-fusion perception stack (real images, human labels).

  python rellis.py check    --root R [--seq 00000] [--frames 0:200:20]
  python rellis.py eval2d   --root R --backbone clip [--split test.lst | --seq 00000 --frames 0:2000:10]
  python rellis.py eval3d   --root R --backbone clip --seq 00000 --frames 0:2000:10 --window 5
  python rellis.py buildmap --root R --backbone clip --seq 00000 --frames 100:160:2

Every command writes <out>/<cmd>.json plus a few PNGs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np

import rellis_io as rio
from backbones import make_backbone
from metrics import Accumulator, fmt_header, fmt_row
from ontology import Ontology, default_path

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_frames(s: str | None):
    if not s:
        return None
    if ":" in s:
        a, b, *c = (int(x) if x else None for x in s.split(":"))
        return slice(a, b, c[0] if c else None)
    return [int(x) for x in s.split(",")]


def select_frames(args, need=("image", "image_label")):
    """-> list of Frame objects, from a split file or --seq/--frames"""
    seqs = [args.seq] if args.seq else rio.list_sequences(args.root)
    index = {s: rio.index_sequence(args.root, s) for s in seqs}
    if getattr(args, "split", None):
        pairs = rio.read_split(args.root, args.split)
        frames = [index[s][i] for s, i in pairs if s in index and i in index[s]]
    else:
        frames = []
        sel = parse_frames(args.frames)
        for s in seqs:
            fs = [index[s][k] for k in sorted(index[s])]
            if isinstance(sel, slice):
                # the step counts real frame numbers (start, start+step, ...), so every backbone
                # gets the same frames; frames missing what this run needs are dropped afterwards
                a0 = sel.start if sel.start is not None else (fs[0].idx if fs else 0)
                fs = [f for f in fs if f.idx >= a0 and (sel.stop is None or f.idx < sel.stop)
                      and (f.idx - a0) % (sel.step or 1) == 0]
            elif sel:
                fs = [f for f in fs if f.idx in sel]
            ok = [f for f in fs if all(getattr(f, n) for n in need)]
            if len(ok) < len(fs):
                miss = [f.idx for f in fs if f not in ok]
                print(f"note: {len(miss)} of {len(fs)} selected frames in {s} lack {need} and are skipped"
                      + (f" (e.g. {miss[:3]}); with --backbone oracle pick a start/step that lands on labelled frames"
                         if "image_label" in need else ""))
            frames += ok
    frames = [f for f in frames if all(getattr(f, n) for n in need)]
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit(f"no frames with {need} found - check --root/--seq/--frames")
    return frames, index


def palette(ont: Ontology):
    pal = np.zeros((len(rio.EVAL_CLASSES), 3), np.uint8)
    for c in ont.classes:
        if c.eval_id is not None:
            pal[c.eval_id] = c.color
    return pal


def save_triptych(path, img, gt, pred_eval, pal, scale=0.25):
    from PIL import Image
    H, W = gt.shape
    sz = (int(W * scale), int(H * scale))
    a = Image.fromarray(img).resize(sz)
    b = Image.fromarray(pal[gt]).resize(sz, Image.NEAREST)
    c = Image.fromarray(pal[pred_eval]).resize(sz, Image.NEAREST)
    out = Image.new("RGB", (sz[0], sz[1] * 3))
    for k, im in enumerate((a, b, c)):
        out.paste(im, (0, k * sz[1]))
    out.save(path)


def upsample_probs(probs_grid, H, W):
    """(h,w,K) -> (H,W,K) bilinear"""
    import torch
    t = torch.from_numpy(probs_grid).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def sample_grid(feat, uv, img_hw):
    """bilinear lookup of patch-grid features at pixel coords"""
    h, w, _ = feat.shape
    H, W = img_hw
    gx = np.clip(uv[:, 0] / W * w - 0.5, 0, w - 1)
    gy = np.clip(uv[:, 1] / H * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (gx - x0)[:, None], (gy - y0)[:, None]
    return ((1 - ay) * ((1 - ax) * feat[y0, x0] + ax * feat[y0, x1])
            + ay * ((1 - ax) * feat[y1, x0] + ax * feat[y1, x1]))


class FeatureCache:
    def __init__(self, bb, size=16):
        self.bb, self.size, self.d = bb, size, OrderedDict()
        self.t_feat, self.n_feat = 0.0, 0

    def get(self, fr):
        k = (fr.seq, fr.idx)
        if k not in self.d:
            img = rio.read_image(fr.image)
            gt = rio.read_image_label(fr.image_label) if (self.bb.needs_gt and fr.image_label) else None
            if self.bb.needs_gt and gt is None:
                return None
            t0 = time.time()
            self.d[k] = (self.bb.image_features(img, gt), img.shape[:2])
            self.t_feat += time.time() - t0; self.n_feat += 1
            if len(self.d) > self.size:
                self.d.popitem(last=False)
        self.d.move_to_end(k)
        return self.d[k]


def build(args):
    ont = Ontology.load(args.ontology, args.classes.split(",") if args.classes else None,
                        rellis_only=not args.all_classes)
    bb = make_backbone(args.backbone, oracle_noise=args.oracle_noise, temperature=args.temperature,
                       clip_model=args.clip_model, clip_pretrained=args.clip_pretrained, weights=args.weights)
    bb.set_ontology(ont)
    return ont, bb


def report(name, acc: Accumulator, extra: dict, out: str, rows=None):
    s, b, c = acc.summary(), acc.baseline(), acc.control()
    print("\n" + fmt_header())
    for label, row in (rows or []):
        print(fmt_row(label, row))
    print(fmt_row(name, s))
    print(fmt_row(f"baseline ({b['baseline_class']})", b))
    print("control: " + "  ".join(f"{k}={v:.3f}" for k, v in c.items()))
    print("per-class IoU: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                        sorted(s["per_class_IoU"].items(), key=lambda kv: -s["per_class_share"][kv[0]])))
    res = {"metrics": s, "baseline": b, "control": c, **extra}
    if rows:
        res["other_rows"] = {label: row for label, row in rows}
    os.makedirs(args_out(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"-> {out}")
    return res


def args_out(p):
    return os.path.dirname(p) or "."


# --------------------------------------------------------------------------
# check: calibration, pairing, poses
# --------------------------------------------------------------------------
def cmd_check(args):
    frames, index = select_frames(args, need=("image_label", "scan", "scan_label"))
    os.makedirs(args.out, exist_ok=True)
    agree = tot = 0
    per, shifts = {}, {}
    for n, fr in enumerate(frames):
        calib = rio.load_calib(args.root, fr.seq)
        gt_img = rio.read_image_label(fr.image_label)
        calib.height, calib.width = gt_img.shape
        pts = rio.read_scan(fr.scan)
        lab = rio.read_scan_label(fr.scan_label)
        uv, z, valid = rio.project(pts, calib)
        vis = rio.visibility_mask(uv, z, valid, calib)
        m = vis & (lab != 0)
        u, v = uv[m, 0].astype(int), uv[m, 1].astype(int)
        li, ii = lab[m], gt_img[v, u]
        k = ii != 0
        agree += (li[k] == ii[k]).sum(); tot += k.sum()
        for dx in (-24, -8, 0, 8, 24):
            for dy in (-24, -8, 0, 8, 24):
                uu = np.clip(u + dx, 0, calib.width - 1); vv = np.clip(v + dy, 0, calib.height - 1)
                jj = gt_img[vv, uu]; kk = jj != 0
                sh = shifts.setdefault((dx, dy), [0, 0])
                sh[0] += (li[kk] == jj[kk]).sum(); sh[1] += kk.sum()
        for c in np.unique(li[k]):
            a = per.setdefault(rio.EVAL_CLASSES[c], [0, 0])
            a[0] += (ii[k][li[k] == c] == c).sum(); a[1] += (li[k] == c).sum()
        if n < 3:
            from PIL import Image, ImageDraw
            pal = palette(Ontology.load(args.ontology, rellis_only=True))
            im = Image.fromarray(rio.read_image(fr.image) if fr.image else pal[gt_img])
            dr = ImageDraw.Draw(im)
            for x, y, c in zip(u[::3], v[::3], li[::3]):
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=tuple(int(q) for q in pal[c]))
            im.resize((im.width // 2, im.height // 2)).save(os.path.join(args.out, f"check_{fr.seq}_{fr.idx:06d}.png"))
        print(f"{fr.seq}/{fr.idx:06d}: {valid.sum()} pts in view, {vis.sum()} visible, "
              f"label agreement {np.mean(li[k] == ii[k]) if k.any() else float('nan'):.3f}")
    overall = agree / max(tot, 1)
    print(f"\nLiDAR label vs image label agreement: {overall:.3f} over {tot} points")
    print("  per LiDAR class: " + ", ".join(f"{c} {a / max(b, 1):.2f} ({b})" for c, (a, b) in
                                         sorted(per.items(), key=lambda kv: -kv[1][1])))
    sh = {k: a / max(b, 1) for k, (a, b) in shifts.items()}
    best = max(sh, key=sh.get)
    print("agreement vs pixel shift (dx,dy): " + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(sh.items(), key=lambda kv: -kv[1])[:5]))
    ok = sh[(0, 0)] >= sh[best] - 0.03 and overall > 0.6   # region-size bias moves the peak a few px; look at the overlay too
    verdict = "OK" if ok else f"SUSPICIOUS - best agreement at shift {best}, check calibration / frame pairing"
    print(f"calibration + pairing: {verdict}  (absolute agreement is capped by image-vs-LiDAR annotation differences)")
    res = {"label_agreement": overall, "points": int(tot), "per_class": per, "verdict": verdict,
           "shift_agreement": {str(k): v for k, v in sh.items()}}

    # pose convention
    seqs = sorted({f.seq for f in frames})
    for s in seqs:
        if not os.path.exists(os.path.join(args.root, s, "poses.txt")):
            print(f"{s}: no poses.txt, skipping pose check")
            continue
        res.setdefault("poses", {})[s] = pc = check_poses(args.root, s, index[s], step=args.pose_step)
        best = min(pc, key=lambda k: pc[k]["median_nn"])
        print(f"{s}: pose check " + "  ".join(f"[{k}] median NN {v['median_nn']:.3f} m, "
                                            f"label agree {v['label_agree']:.3f}" for k, v in pc.items())
              + f"  -> use --pose-frame {best}")
    with open(os.path.join(args.out, "check.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


def check_poses(root, seq, frames, step=5, n_pairs=10):
    from scipy.spatial import cKDTree
    keys = [k for k in sorted(frames) if frames[k].scan and frames[k].scan_label]
    out = {}
    for mode in ("lidar", "kitti"):
        try:
            T = rio.read_poses(root, seq, mode)
        except Exception as e:                      # no calib.txt / Tr
            out[mode] = {"median_nn": float("inf"), "label_agree": 0.0, "error": str(e)}
            continue
        nn, ag = [], []
        picks = keys[:: max(1, len(keys) // n_pairs)][:n_pairs]
        for i in picks:
            j = i + step
            if j not in frames or not frames[j].scan or j >= len(T):
                continue
            a, b = rio.read_scan(frames[i].scan), rio.read_scan(frames[j].scan)
            la, lb = rio.read_scan_label(frames[i].scan_label), rio.read_scan_label(frames[j].scan_label)
            Tij = np.linalg.inv(T[i]) @ T[j]
            b_in_i = (Tij @ np.c_[b, np.ones(len(b))].T).T[:, :3]
            r = np.linalg.norm(b_in_i, axis=1)
            sel = (r > 2) & (r < 30) & (lb != 0)
            d, nb = cKDTree(a).query(b_in_i[sel], k=1)
            nn.append(np.median(d))
            close = d < 0.2
            ag.append(np.mean(la[nb[close]] == lb[sel][close]) if close.any() else 0)
        out[mode] = {"median_nn": float(np.mean(nn)) if nn else float("inf"),
                     "label_agree": float(np.mean(ag)) if ag else 0.0}
    return out


# --------------------------------------------------------------------------
# eval2d
# --------------------------------------------------------------------------
def cmd_eval2d(args):
    frames, _ = select_frames(args)
    ont, bb = build(args)
    acc = Accumulator(ont, bb.text_sim())
    pal = palette(ont)
    os.makedirs(args.out, exist_ok=True)
    t_feat = 0.0
    for n, fr in enumerate(frames):
        img, gt = rio.read_image(fr.image), rio.read_image_label(fr.image_label)
        t0 = time.time()
        f = bb.image_features(img, gt if bb.needs_gt else None)
        t_feat += time.time() - t0
        h, w, D = f.shape
        pg = bb.classify(f.reshape(-1, D)).reshape(h, w, -1).astype(np.float32)
        H, W = gt.shape
        if args.eval_scale != 1:
            from PIL import Image
            H, W = int(H * args.eval_scale), int(W * args.eval_scale)
            gt = np.array(Image.fromarray(gt).resize((W, H), Image.NEAREST))
        p = upsample_probs(pg, H, W)
        acc.add(gt, p)
        if n < args.n_vis:
            pred_eval = np.maximum(ont.eval_id[p.argmax(-1)], 0)
            from PIL import Image
            img_s = np.array(Image.fromarray(img).resize((W, H)))
            save_triptych(os.path.join(args.out, f"eval2d_{fr.seq}_{fr.idx:06d}.png"),
                          img_s, gt, pred_eval, pal, scale=min(1.0, 480 / W))
        if (n + 1) % 50 == 0:
            s = acc.summary()
            print(f"[{n + 1}/{len(frames)}] Acc {s['Acc']:.3f} mIoU {s['mIoU']:.3f}  ({t_feat / (n + 1):.2f} s/frame)")
    report(f"{bb.name} 2D", acc, {"frames": len(frames), "s_per_frame_features": t_feat / len(frames),
                                 "backbone": bb.name, "ontology": args.ontology},
           os.path.join(args.out, "eval2d.json"))


# --------------------------------------------------------------------------
# eval3d: per-scan, single view vs multi-view fusion
# --------------------------------------------------------------------------
def fuse_points(pts_world, views, cache, calib_by_seq, D, max_range=40.0):
    """Average features over every view that sees each point.
    views: list of (Frame, world_T_lidar). Returns (feat_sum (N,D), count (N,)); memory is
    only spent on points that at least one view sees (~10% of a scan for this camera)."""
    hits = []                                         # (view, point idx, uv)
    for fr, Tw in views:
        if not fr.image:
            continue
        H, W = rio_image_hw(fr)
        calib = calib_by_seq(fr.seq)
        calib.height, calib.width = H, W
        p = (np.linalg.inv(Tw) @ np.c_[pts_world, np.ones(len(pts_world))].T).T[:, :3]
        uv, z, valid = rio.project(p, calib)
        valid &= np.linalg.norm(p, axis=1) < max_range
        # occlusion: z-buffer from this view's own scan plus the query points
        own = rio.read_scan(fr.scan) if fr.scan else np.zeros((0, 3))
        uvo, zo, vo = rio.project(own, calib)
        vis = rio.visibility_mask(np.r_[uv, uvo], np.r_[z, zo], np.r_[valid, vo], calib)[: len(p)]
        idx = np.where(vis)[0]
        if len(idx):
            hits.append((fr, idx, uv[idx], (H, W)))
    seen = np.unique(np.concatenate([h[1] for h in hits])) if hits else np.zeros(0, int)
    pos = np.full(len(pts_world), -1); pos[seen] = np.arange(len(seen))
    fsum = np.zeros((len(seen), D), np.float32)
    cnt = np.zeros(len(pts_world), np.int32)
    for fr, idx, uv, hw in hits:
        got = cache.get(fr)
        if got is None:
            continue
        fsum[pos[idx]] += sample_grid(got[0], uv, hw)
        cnt[idx] += 1
    return SparseFeat(pos, fsum), cnt


class SparseFeat:
    """feature sums stored only for seen points; index with a boolean mask over all points"""
    def __init__(self, pos, data):
        self.pos, self.data = pos, data

    def __getitem__(self, mask):
        return self.data[self.pos[mask]]


_HW = {}


def rio_image_hw(fr):
    """image size without decoding the whole jpg"""
    if fr.image not in _HW:
        from PIL import Image
        with Image.open(fr.image) as im:
            _HW[fr.image] = (im.height, im.width)
    return _HW[fr.image]


def cmd_eval3d(args):
    frames, index = select_frames(args, need=("image", "scan", "scan_label") + (("image_label",) if args.backbone == "oracle" else ()))
    ont, bb = build(args)
    cache = FeatureCache(bb, size=4 * args.window + 4)
    calibs = {}
    calib_by_seq = lambda s: calibs.setdefault(s, rio.load_calib(args.root, s))
    poses = {}
    acc_single, acc_fused = Accumulator(ont, bb.text_sim()), Accumulator(ont, bb.text_sim())
    cm_err = cm_n = 0.0
    cover_s = cover_f = n_pts = 0
    os.makedirs(args.out, exist_ok=True)
    for n, fr in enumerate(frames):
        pts = rio.read_scan(fr.scan)
        lab = rio.read_scan_label(fr.scan_label)
        r = np.linalg.norm(pts, axis=1)
        keep = (r > 1.0) & (r < args.max_range)
        pts, lab = pts[keep], lab[keep]
        seq_frames = index[fr.seq]
        if args.window > 0:
            if fr.seq not in poses:
                poses[fr.seq] = rio.read_poses(args.root, fr.seq, args.pose_frame)
            T = poses[fr.seq]
            Ti = T[fr.idx]
            nbrs = [j for j in range(fr.idx - args.window * args.view_step,
                                     fr.idx + args.window * args.view_step + 1, args.view_step)
                    if j in seq_frames and seq_frames[j].image and 0 <= j < len(T)
                    and (bb.needs_gt is False or seq_frames[j].image_label)]
            views = [(seq_frames[j], T[j]) for j in nbrs]
        else:
            Ti = np.eye(4)
            views = [(fr, Ti)]
        pw = (Ti @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        # single view = this scan's own image only
        fs, cs = fuse_points(pw, [(fr, Ti)], cache, calib_by_seq, bb.dim, args.max_range)
        ff, cf = fuse_points(pw, views, cache, calib_by_seq, bb.dim, args.max_range) if args.window > 0 else (fs, cs)
        for fsum, cnt, acc in ((fs, cs, acc_single), (ff, cf, acc_fused)):
            seen = cnt > 0
            if seen.any():
                f = fsum[seen] / cnt[seen, None]
                if bb.name != "oracle":
                    f /= np.linalg.norm(f, axis=1, keepdims=True)
                probs = np.zeros((len(pts), len(ont)), np.float32)
                probs[seen] = bb.classify(f)
                acc.add(lab, probs, seen)
        # cost map MAE (0.5 m cells, fused, points seen)
        seen = cf > 0
        if seen.any():
            f = ff[seen] / cf[seen, None]
            if bb.name != "oracle":
                f /= np.linalg.norm(f, axis=1, keepdims=True)
            t_hat = bb.classify(f) @ ont.trav
            t_gt = acc_fused.trav_gt[lab[seen]]
            ok = ~np.isnan(t_gt) & (lab[seen] != 0)
            cell = np.floor(pts[seen][ok, :2] / args.cell).astype(int)
            if len(cell):
                _, inv = np.unique(cell, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                k = inv.max() + 1
                a = np.bincount(inv, t_hat[ok], k) / np.bincount(inv, None, k)
                b = np.bincount(inv, t_gt[ok], k) / np.bincount(inv, None, k)
                cm_err += np.abs(a - b).sum(); cm_n += k
            if n < args.n_vis:
                save_bev(os.path.join(args.out, f"eval3d_{fr.seq}_{fr.idx:06d}.png"), pts, lab, seen,
                         np.maximum(ont.eval_id[bb.classify(f).argmax(1)], 0), palette(ont))
        cover_s += (cs > 0).sum(); cover_f += (cf > 0).sum(); n_pts += len(pts)
        if (n + 1) % 20 == 0:
            a, b = acc_single.summary(), acc_fused.summary()
            print(f"[{n + 1}/{len(frames)}] mIoU single {a['mIoU']:.3f}  fused {b['mIoU']:.3f}  "
                  f"({cache.t_feat / max(cache.n_feat, 1):.2f} s/feature frame)")
    extra = {"frames": len(frames), "window": args.window, "view_step": args.view_step,
             "pose_frame": args.pose_frame, "coverage_single": cover_s / max(n_pts, 1),
             "coverage_fused": cover_f / max(n_pts, 1), "costmap_MAE": cm_err / max(cm_n, 1),
             "backbone": bb.name}
    print(f"\ncoverage of scan points (r<{args.max_range} m): single view {extra['coverage_single']:.3f}, "
          f"fused {extra['coverage_fused']:.3f};  cost-map MAE ({args.cell} m cells) {extra['costmap_MAE']:.3f}")
    report(f"{bb.name} 3D fused(w={args.window})", acc_fused, extra, os.path.join(args.out, "eval3d.json"),
           rows=[(f"{bb.name} 3D single view", acc_single.summary())])


def save_bev(path, pts, lab, seen, pred_eval, pal, res=0.08):
    """top-down GT (left) vs prediction (right), cropped to the camera's view cone"""
    from PIL import Image
    q = pts[seen]
    lo, hi = q[:, :2].min(0) - 1, q[:, :2].max(0) + 1
    nx, ny = (np.ceil((hi - lo) / res).astype(int) + 1)
    full = np.zeros(len(pts), int); full[seen] = pred_eval
    o = np.where(seen)[0]
    o = o[np.argsort(pts[o, 2])]                 # higher points drawn last
    ix = ((pts[o, 0] - lo[0]) / res).astype(int)
    iy = ((pts[o, 1] - lo[1]) / res).astype(int)
    tiles = []
    for cls in (lab, full):
        img = np.full((nx, ny, 3), 255, np.uint8)
        img[ix, iy] = pal[cls[o]]
        tiles.append(img)
    Image.fromarray(np.concatenate([tiles[0], np.zeros((nx, 4, 3), np.uint8), tiles[1]], 1)).save(path)


# --------------------------------------------------------------------------
# buildmap: fused voxel map over a chunk of scans
# --------------------------------------------------------------------------
def cmd_buildmap(args):
    frames, index = select_frames(args, need=("image", "scan", "scan_label") + (("image_label",) if args.backbone == "oracle" else ()))
    ont, bb = build(args)
    cache = FeatureCache(bb, size=8)
    calibs = {}
    calib_by_seq = lambda s: calibs.setdefault(s, rio.load_calib(args.root, s))
    seq = frames[0].seq
    T = rio.read_poses(args.root, seq, args.pose_frame)
    # 1. accumulate GT-labelled points into voxels
    allp, alll = [], []
    for fr in frames:
        p = rio.read_scan(fr.scan); l = rio.read_scan_label(fr.scan_label)
        r = np.linalg.norm(p, axis=1)
        k = (r > 1) & (r < args.max_range)
        allp.append((T[fr.idx] @ np.c_[p[k], np.ones(k.sum())].T).T[:, :3]); alll.append(l[k])
    P, L = np.concatenate(allp), np.concatenate(alll)
    vox = np.floor(P / args.voxel).astype(np.int64)
    _, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    V = inv.max() + 1
    xyz = np.zeros((V, 3)); np.add.at(xyz, inv, P); xyz /= np.bincount(inv, None, V)[:, None]
    votes = np.zeros((V, len(rio.EVAL_CLASSES)), np.int32); np.add.at(votes, (inv, L), 1)
    votes[:, 0] = np.where(votes[:, 1:].sum(1) > 0, 0, votes[:, 0])   # void only if nothing else
    gt = votes.argmax(1).astype(np.uint8)
    print(f"{len(frames)} scans -> {len(P)} points -> {V} voxels ({args.voxel} m)")
    # 2. fuse features from every view
    fsum, cnt = np.zeros((V, bb.dim), np.float32), np.zeros(V, np.int32)
    for n, fr in enumerate(frames):
        s, c = fuse_points(xyz, [(fr, T[fr.idx])], cache, calib_by_seq, bb.dim, args.max_range)
        m = s.pos >= 0
        fsum[m] += s.data[s.pos[m]]; cnt += c
        if (n + 1) % 10 == 0:
            print(f"  fused {n + 1}/{len(frames)} views")
    seen = cnt > 0
    feat = np.zeros_like(fsum)
    feat[seen] = fsum[seen] / cnt[seen, None]
    if bb.name != "oracle":
        feat[seen] /= np.linalg.norm(feat[seen], axis=1, keepdims=True)
    probs = np.zeros((V, len(ont)), np.float32)
    probs[seen] = bb.classify(feat[seen])
    acc = Accumulator(ont, bb.text_sim()); acc.add(gt, probs, seen)
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, f"rellis_map_{seq}_{frames[0].idx}-{frames[-1].idx}.npz")
    np.savez_compressed(npz, xyz=xyz.astype(np.float32), feat=feat.astype(np.float16), gt=gt,
                        seen=cnt, class_names=np.array(rio.EVAL_CLASSES), voxel=args.voxel,
                        backbone=bb.name, probs=probs.astype(np.float16),
                        ont_names=np.array(ont.names), ont_trav=ont.trav, ont_mu=ont.mu, ont_rigid=ont.rigid,
                        traj=np.array([T[f.idx][:3, 3] for f in frames], np.float32),
                        frame_idx=np.array([f.idx for f in frames]), seq=seq, pose_frame=args.pose_frame)
    pred_eval = np.where(seen, np.maximum(ont.eval_id[probs.argmax(1)], 0), 0)
    write_ply(npz.replace(".npz", "_pred.ply"), xyz, palette(ont)[pred_eval])
    write_ply(npz.replace(".npz", "_gt.ply"), xyz, palette(ont)[gt])
    if not seen.any():
        raise SystemExit("no voxel was seen by any view: check poses, calibration, and (oracle) image labels")
    print(f"voxels seen by >=1 view: {seen.mean():.3f}, mean views/voxel {cnt[seen].mean():.1f}")
    report(f"{bb.name} map voxels", acc, {"voxels": int(V), "coverage": float(seen.mean()),
                                          "npz": npz, "frames": len(frames)},
           os.path.join(args.out, "buildmap.json"))


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        f.write(rec.tobytes())


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "eval2d", "eval3d", "buildmap"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True, help="RELLIS-3D root (folder holding 00000..00004)")
        p.add_argument("--seq"); p.add_argument("--frames", help="a:b:step or comma list")
        p.add_argument("--split", help="official .lst file (relative to root ok)")
        p.add_argument("--max-frames", type=int, default=0)
        p.add_argument("--out", default="out_rellis")
        p.add_argument("--ontology", default=default_path(), help="default: the repo's config/ontology.json")
        p.add_argument("--all-classes", action="store_true", help="keep ontology classes RELLIS doesn't have (goal objects)")
        p.add_argument("--classes", help="comma list to restrict the ontology")
        p.add_argument("--backbone", default="pe-b", help="oracle | pe-b | pe-l | pe-s | clip (from clip_features.py) | clip-standalone")
        p.add_argument("--clip-model", default="ViT-B-16"); p.add_argument("--clip-pretrained", default="laion400m_e32")
        p.add_argument("--weights", help="override the backbone weights (path or open_clip tag)")
        p.add_argument("--temperature", type=float, default=0.01)
        p.add_argument("--oracle-noise", type=float, default=0.0)
        p.add_argument("--n-vis", type=int, default=4)
        p.add_argument("--pose-frame", default="lidar", choices=["lidar", "kitti"])
        p.add_argument("--max-range", type=float, default=40.0)
        if name == "check":
            p.add_argument("--pose-step", type=int, default=5)
        if name == "eval2d":
            p.add_argument("--eval-scale", type=float, default=0.5, help="label resolution for scoring")
        if name == "eval3d":
            p.add_argument("--window", type=int, default=0, help="neighbouring views each side (0 = single view)")
            p.add_argument("--view-step", type=int, default=5, help="frame gap between fused views")
            p.add_argument("--cell", type=float, default=0.5)
        if name == "buildmap":
            p.add_argument("--voxel", type=float, default=0.1)
    args = ap.parse_args()
    {"check": cmd_check, "eval2d": cmd_eval2d, "eval3d": cmd_eval3d, "buildmap": cmd_buildmap}[args.cmd](args)


if __name__ == "__main__":
    main()
