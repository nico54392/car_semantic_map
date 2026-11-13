"""
OpenScene-style 2D->3D open-vocabulary test on a real off-road frame (RELLIS-3D sample).

Pipeline (mirrors OpenScene's "2D fusion" branch, no training):
  1. dense, text-aligned per-pixel features from CLIP (ClearCLIP-style last layer,
     standing in for OpenSeg/LSeg which we can't download here)
  2. project LiDAR points into the image (OpenScene PointCloudToImageMapper logic + occlusion check)
  3. each 3D point gets a 512-d CLIP embedding -> query with ANY text afterwards
  4. text query -> class probs -> friction / traversability lookup

Usage: python tools/rellis_frame_test.py data/rellis_sample out/rellis_frame [clip_weights]
(get the data with: bash tools/get_sample_data.sh)
"""
import sys, json, time
import numpy as np, torch, torch.nn.functional as F, yaml, cv2, open_clip
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os
DATA, OUT = sys.argv[1], sys.argv[2]
WEIGHTS = sys.argv[3] if len(sys.argv) > 3 else "laion400m_e32"
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

# ---------------- ontology + (placeholder) physics table ----------------
# RELLIS-3D ids -> prompts. mu = rough dry-ish rubber-on-surface guesses, PLACEHOLDERS to replace with your own.
CLASSES = {
    1: ("dirt", ["dirt", "bare dirt ground", "dirt trail"], 0.55),
    3: ("grass", ["grass", "grassy field", "lawn"], 0.40),
    4: ("tree", ["tree", "tree trunk", "foliage of a tree"], None),
    5: ("pole", ["pole"], None),
    6: ("water", ["water", "pond"], 0.10),
    7: ("sky", ["sky", "clouds"], None),
    8: ("vehicle", ["vehicle", "car", "truck"], None),
    9: ("object", ["object", "equipment", "box"], None),
    10: ("asphalt", ["asphalt road", "paved road"], 0.75),
    12: ("building", ["building", "house"], None),
    15: ("log", ["fallen log", "log on the ground"], None),
    17: ("person", ["person", "people"], None),
    18: ("fence", ["fence", "chain-link fence"], None),
    19: ("bush", ["bush", "shrub", "tall weeds"], None),
    23: ("concrete", ["concrete", "concrete slab"], 0.75),
    27: ("barrier", ["barrier", "concrete barrier"], None),
    31: ("puddle", ["puddle", "puddle of water on the ground"], 0.20),
    33: ("mud", ["mud", "muddy ground", "wet mud"], 0.25),
    34: ("rubble", ["rubble", "gravel", "loose rocks"], 0.50),
}
IDS = list(CLASSES)
TEMPLATES = ["a photo of {}.", "a photo of {} in an off-road scene.", "a close-up photo of {}.", "{}"]
PROPERTY_QUERIES = ["slippery ground", "soft muddy ground", "firm flat ground", "an obstacle", "standing water"]

model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=WEIGHTS)
model.eval()
tok = open_clip.get_tokenizer("ViT-B-16")

def text_emb(phrases):
    e = F.normalize(model.encode_text(tok(phrases)), dim=-1)
    return F.normalize(e.mean(0), dim=-1)

T_cls = torch.stack([text_emb([t.format(p) for p in CLASSES[i][1] for t in TEMPLATES]) for i in IDS])  # [C,512]
T_prop = torch.stack([text_emb([t.format(q) for t in TEMPLATES]) for q in PROPERTY_QUERIES])

# ---------------- dense CLIP features (ClearCLIP-style) ----------------
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def dense_clip(img_rgb, H=448, W=720):
    v = model.visual
    x = torch.from_numpy(cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float() / 255
    x = ((x - MEAN) / STD)[None]
    x = v.conv1(x); gh, gw = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    pos = v.positional_embedding
    pos_p = pos[1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    pos_p = F.interpolate(pos_p, (gh, gw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(gh * gw, -1)
    x = torch.cat([v.class_embedding[None, None].expand(1, 1, -1), x], 1) + torch.cat([pos[:1], pos_p])[None]
    x = v.ln_pre(x)
    for blk in v.transformer.resblocks[:-1]:
        x = blk(x)
    blk = v.transformer.resblocks[-1]           # last block: q-q attention, no residual, no FFN
    a = blk.attn; y = blk.ln_1(x)
    q, k, val = F.linear(y, a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
    nh = a.num_heads; hd = q.shape[-1] // nh
    q = q.view(1, -1, nh, hd).transpose(1, 2); val = val.view(1, -1, nh, hd).transpose(1, 2)
    att = (q @ q.transpose(-1, -2) / hd ** 0.5).softmax(-1)
    y = (att @ val).transpose(1, 2).reshape(1, -1, nh * hd)
    y = a.out_proj(y)
    feat = v.ln_post(y)[:, 1:] @ v.proj               # [1, gh*gw, 512]
    feat = F.normalize(feat, dim=-1).reshape(gh, gw, -1).permute(2, 0, 1)
    return feat                                         # [512, gh, gw]

img_bgr = cv2.imread(f"{DATA}/frame000104-1581624663_149.jpg")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); IH, IW = img.shape[:2]
t0 = time.time(); feat = dense_clip(img); t_feat = time.time() - t0
print(f"dense CLIP features {tuple(feat.shape)} in {t_feat:.2f}s on CPU")

# ---------------- 2D zero-shot eval ----------------
gt2d = np.array(Image.open(f"{DATA}/frame000104-1581624663_149.png"))
EH, EW = 300, 480
gt_s = cv2.resize(gt2d, (EW, EH), interpolation=cv2.INTER_NEAREST)
logit = torch.einsum("chw,kc->khw", feat, T_cls) * 100
logit_up = F.interpolate(logit[None], (EH, EW), mode="bilinear", align_corners=False)[0]
pred2d = np.array(IDS)[logit_up.argmax(0).numpy()]

def iou_table(gt, pr):
    rows = {}
    for c in np.unique(gt):
        if c == 0: continue
        g, p = gt == c, pr == c
        rows[CLASSES[c][0]] = dict(iou=float((g & p).sum() / max((g | p).sum(), 1)), share=float(g.mean()))
    valid = gt != 0
    return rows, float((gt[valid] == pr[valid]).mean())

iou2d, acc2d = iou_table(gt_s, pred2d)

# ---------------- 3D: OpenScene-style point -> pixel mapping ----------------
def load_RT(path, key):
    d = yaml.safe_load(open(path))[key]; q, t = d["q"], d["t"]
    T = np.eye(4); T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return np.linalg.inv(T)                           # lidar -> camera (as in RELLIS lidar2img.ipynb)

pts = np.fromfile(f"{DATA}/000104.bin", np.float32).reshape(-1, 4)[:, :3]
lab3d = np.fromfile(f"{DATA}/000104.label", np.uint32) & 0xFFFF
keep = np.linalg.norm(pts, axis=1) > 0.5; pts, lab3d = pts[keep], lab3d[keep]
fx, fy, cx, cy = np.loadtxt(f"{DATA}/camera_info.txt")
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); dist = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0])
RT = load_RT(f"{DATA}/transforms.yaml", "os1_cloud_node-pylon_camera_node")
pc = (RT[:3, :3] @ pts.T + RT[:3, 3:]).T
front = pc[:, 2] > 0.5
uv = np.full((len(pts), 2), -1.0)
uv[front] = cv2.projectPoints(pts[front], cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3], K, dist)[0][:, 0]
inside = front & (uv[:, 0] >= 10) & (uv[:, 0] < IW - 10) & (uv[:, 1] >= 10) & (uv[:, 1] < IH - 10)

# occlusion: with Unity you'd use the rendered depth image (exactly OpenScene's depth test).
# here we only have sparse LiDAR, so build a coarse z-buffer from the points themselves.
CELL = 12
zb = np.full((IH // CELL + 1, IW // CELL + 1), np.inf)
ci = (uv[inside] // CELL).astype(int)
np.minimum.at(zb, (ci[:, 1], ci[:, 0]), pc[inside, 2])
vis = inside.copy()
vis[inside] = pc[inside, 2] <= zb[ci[:, 1], ci[:, 0]] * 1.05 + 0.3
print(f"points: {len(pts)}, in camera FOV: {inside.sum()}, visible after occlusion test: {vis.sum()}")

# sample per-point embeddings (this is what OpenScene stores per point)
gh, gw = feat.shape[1:]
grid = torch.from_numpy(np.stack([uv[vis, 0] / IW * 2 - 1, uv[vis, 1] / IH * 2 - 1], -1)).float()[None, None]
pfeat = F.normalize(F.grid_sample(feat[None], grid, align_corners=False)[0, :, 0].T, dim=-1)  # [Nv,512]
plogit = pfeat @ T_cls.T * 100
pprob = plogit.softmax(-1).numpy()
pred3d = np.array(IDS)[pprob.argmax(1)]
gt3d = lab3d[vis]
iou3d, acc3d = iou_table(gt3d, pred3d)

# ---------------- physics: hard vs expected friction ----------------
mu_vec = np.array([np.nan if CLASSES[i][2] is None else CLASSES[i][2] for i in IDS])
terrain = ~np.isnan(mu_vec)
mu_hard = mu_vec[pprob.argmax(1)]
pt = pprob[:, terrain]; p_terrain = pt.sum(1)
mu_exp = (pt * mu_vec[terrain]).sum(1) / np.maximum(p_terrain, 1e-6)
mu_std = np.sqrt((pt * (mu_vec[terrain] - mu_exp[:, None]) ** 2).sum(1) / np.maximum(p_terrain, 1e-6))
gt_mu = np.array([CLASSES.get(int(c), (None, None, None))[2] or np.nan for c in gt3d], dtype=float)
gmask = ~np.isnan(gt_mu) & ~np.isnan(mu_hard)
mu_err_hard = float(np.abs(mu_hard[gmask] - gt_mu[gmask]).mean())
mu_err_exp = float(np.abs(mu_exp[gmask] - gt_mu[gmask]).mean())

# property queries on points (OpenScene "somewhere soft" style)
pq = (pfeat @ T_prop.T).numpy()

# ---------------- timing of a query against the stored embeddings ----------------
t0 = time.time(); _ = pfeat @ text_emb(["a photo of tall grass."])[:, None]; t_query = time.time() - t0

metrics = dict(dense_feat_seconds_cpu=round(t_feat, 2), feat_grid=[gh, gw],
               acc2d=acc2d, miou2d_present=float(np.mean([r["iou"] for r in iou2d.values()])), iou2d=iou2d,
               n_points=int(len(pts)), n_fov=int(inside.sum()), n_visible=int(vis.sum()),
               acc3d=acc3d, miou3d_present=float(np.mean([r["iou"] for r in iou3d.values()])), iou3d=iou3d,
               mu_mae_hard=mu_err_hard, mu_mae_expected=mu_err_exp, n_mu_points=int(gmask.sum()),
               new_text_query_ms=round(1000 * (t_query), 1))
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))

# ---------------- figures ----------------
rng = np.random.default_rng(0); cmap = plt.get_cmap("tab20")
col = {c: cmap(k % 20)[:3] for k, c in enumerate(IDS)}; col[0] = (0, 0, 0)
def colorize(lbl): return np.stack([np.array([col.get(int(c), (0, 0, 0))[k] for c in lbl.ravel()]).reshape(lbl.shape) for k in range(3)], -1)
small = cv2.resize(img, (EW, EH))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
ax[0].imshow(small); ax[0].set_title("RELLIS-3D frame 104")
ax[1].imshow(colorize(gt_s)); ax[1].set_title("ground truth")
ax[2].imshow(colorize(pred2d)); ax[2].set_title(f"zero-shot dense CLIP (pixel acc {acc2d:.2f})")
present = sorted(set(np.unique(gt_s)) | set(np.unique(pred2d)) - {0})
fig.legend([plt.Rectangle((0, 0), 1, 1, color=col[c]) for c in present], [CLASSES[c][0] for c in present], loc="lower center", ncol=len(present))
for a in ax: a.axis("off")
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f"{OUT}/fig1_2d_segmentation.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, len(PROPERTY_QUERIES), figsize=(4 * len(PROPERTY_QUERIES), 3.2))
for k, q in enumerate(PROPERTY_QUERIES):
    s = torch.einsum("chw,c->hw", feat, T_prop[k]).numpy()
    s = cv2.resize(s, (EW, EH), interpolation=cv2.INTER_LINEAR)
    ax[k].imshow(small); ax[k].imshow(s, cmap="jet", alpha=0.5); ax[k].set_title(f'"{q}"'); ax[k].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_property_queries.png", dpi=110); plt.close()

P = pc[vis]
fig, ax = plt.subplots(1, 4, figsize=(22, 6))
for a, (c, title) in zip(ax[:2], [(gt3d, "3D ground truth"), (pred3d, f"3D zero-shot (point acc {acc3d:.2f})")]):
    a.scatter(P[:, 0], P[:, 2], c=[col.get(int(v), (0, 0, 0)) for v in c], s=1); a.set_title(title)
sc = ax[2].scatter(P[:, 0], P[:, 2], c=np.where(p_terrain > 0.5, mu_exp, np.nan), s=1, cmap="RdYlGn", vmin=0.1, vmax=0.75)
plt.colorbar(sc, ax=ax[2]); ax[2].set_title("expected friction (terrain points)")
sc = ax[3].scatter(P[:, 0], P[:, 2], c=np.where(p_terrain > 0.5, mu_std, np.nan), s=1, cmap="magma")
plt.colorbar(sc, ax=ax[3]); ax[3].set_title("friction uncertainty (std)")
for a in ax: a.set_xlabel("x right [m]"); a.set_ylabel("z forward [m]"); a.set_aspect("equal"); a.set_ylim(0, 40); a.set_xlim(-20, 20)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_bev_points_friction.png", dpi=100); plt.close()

ov = img.copy()
for (u, v_), c in zip(uv[vis].astype(int), pred3d):
    cv2.circle(ov, (u, v_), 3, tuple(int(255 * x) for x in col[int(c)]), -1)
cv2.imwrite(f"{OUT}/fig4_points_on_image.jpg", cv2.cvtColor(cv2.resize(ov, (960, 600)), cv2.COLOR_RGB2BGR))
print("saved figures to", OUT)
