# CLIP / OpenScene pipeline on the work computer

Per-point CLIP embeddings on the map (OpenScene 2D fusion) → text queries → traversability cost map → A* path.
CLIP first to get it running, then PE-Core (`--backbone pe-b / pe-l`, section 5).

You don't need the OpenScene repo for this: its 3D network needs MinkowskiEngine, and the fusion step is
reimplemented here. Keep the clone around for reference (their `scripts/feature_fusion/` is what
`openscene_fusion.py` follows).

## 1. Get the code
```bash
cd ~/nico_ws/src/car_semantic_map && git fetch && git checkout clip-openscene && git pull
```

## 2. Env (once)
```bash
source setup_clip.sh            # clones nico_sam3 -> nico_clip, installs open_clip, downloads CLIP weights
```
It should end with `cuda True | <GPU name>`. If `nico_sam3` doesn't exist: `source setup_clip.sh <some_env_with_torch>`.

## 3. Smoke test (no Unity needed)
```bash
conda activate nico_clip
bash tools/smoke_test.sh
```
What it does:
1. Downloads the RELLIS-3D sample frame (~4 MB) into `data/rellis_sample`
2. Real-frame test → `out/rellis_frame/` (expect point acc ~0.51)
3. Builds a fake Unity export from it → `out/synth01/`
4. `check` → must say **OK** (~98%)
5. Embedded map → `embedded_map.npz`
6. Text queries → `query.png`
7. Cost map + A* → `plan.png`
8. Metrics vs ground truth → `report.md`, `compare_*.png` (expect Acc ~0.47, mIoU ~0.17, same as in the sandbox)

With a GPU, steps 5 and 8 should drop from ~1 s/view to well under that.

## 4. Bring back
- terminal output
- `out/synth01/plan.png`, `out/synth01/report.md`
- `out/rellis_frame/fig1_2d_segmentation.png`

## 5. PE-Core (Perception Encoder)
Same pipeline, swap the 2D features. Weights download on first use (B ~1.6 GB, L ~2.5 GB).
```bash
conda activate nico_clip
# which backbone / variant is best? (one real frame, Jaime's RELLIS prompts)
python tools/backbone_compare.py data/rellis_sample out/backbones \
  --backbones clip pe-b:plain pe-b:last_v pe-b:last_v_only pe-l:plain pe-l:last_v pe-l:last_v_only
```
Bring back the terminal table + `out/backbones/backbones.png`. Then use the winner:
```bash
python tools/openscene_map.py build out/synth01 --backbone pe-l --variant <best>
python tools/openscene_map.py plan  out/synth01 --start 5 -5 --goal 35 -35 --classes grass mud puddle bush dirt gravel rock
python tools/scene_eval.py openscene out/synth01 --backbone pe-l --variant <best>
python tools/scene_eval.py compare   out/synth01
```
`query`/`plan` pick the text encoder from the map file automatically, so they always match how the map was built.

How the PE features are made (`DensePE` in `car_semantic_map/clip_features.py`):
- PE-Core has RoPE + an attention-pooling head instead of a CLS token, so the CLIP trick doesn't apply
- image cut into overlapping tiles at the native size (224 / 336) so RoPE never gets stretched
- each patch token goes through the pooling head's value path → pool MLP → projection to text space
- variants for the last block: `plain` (unchanged), `last_v` (attention → value path), `last_v_only` (value path only, like ClearCLIP)

Results on the RELLIS frame (work computer, 2026-11-13):

| backbone | 2D acc | 2D mIoU | 3D acc | 3D mIoU | s/frame |
|---|---|---|---|---|---|
| clip | 0.539 | 0.201 | 0.539 | 0.175 | 0.01 |
| **pe-b:plain** | **0.626** | **0.289** | 0.506 | 0.243 | 0.03 |
| pe-b:last_v | 0.625 | 0.279 | 0.529 | 0.236 | 0.03 |
| pe-b:last_v_only | 0.339 | 0.127 | 0.321 | 0.087 | 0.03 |
| pe-l:plain | 0.536 | 0.254 | 0.456 | 0.244 | 0.10 |
| pe-l:last_v | 0.490 | 0.237 | 0.390 | 0.213 | 0.10 |
| pe-l:last_v_only | 0.319 | 0.093 | 0.192 | 0.054 | 0.10 |

→ `plain` is now the default variant.

At `--short_side 672`: pe-b 2D acc 0.642 / mIoU 0.300, 3D acc 0.520 / mIoU 0.250 (0.06 s) · pe-l 0.555 / 0.264, 3D 0.461 / 0.245 (0.24 s).
→ **PE-B, plain, 672 px is the default.** PE-L doesn't pay off.

Synthetic scene, full pipeline (`scene_eval compare`):

| method | Acc | mIoU | F1 | costmap MAE | μ MAE | bush | grass | mud | puddle |
|---|---|---|---|---|---|---|---|---|---|
| openscene_clip | 0.474 | 0.174 | 0.253 | 0.211 | 0.064 | 0.00 | 0.46 | 0.00 | 0.23 |
| **openscene_peb** | **0.505** | **0.265** | **0.380** | **0.164** | **0.050** | 0.10 | 0.52 | 0.05 | 0.40 |
| baseline_majority | 0.444 | 0.111 | 0.154 | 0.226 | 0.073 | 0 | 0.44 | 0 | 0 |

Known issue: lots of false "rock" (= obstacle) when `rock` is in `--classes` but not in the scene.
Note the synthetic scene has weak textures (mud is a 35 px crop), so judge backbones on the real frame first.

## Tools
| Command | What |
|---|---|
| `tools/scene_eval.py check <scene>` | Projects the labeled map into the label images. Run on every new export. ~98% + OK = conventions right |
| `tools/openscene_map.py build <scene>` | CLIP features fused onto `map.ply` → `embedded_map.npz` |
| `tools/openscene_map.py build <scene> --from-depth` | No `map.ply`: back-project depth with `fusion_core.fuse_views` (voxel average) |
| `tools/openscene_map.py query <scene> "mud" ...` | Heatmap per text query |
| `tools/openscene_map.py plan <scene> --start x y --goal x y` | Class probs → traversability/μ → cost map → A* (`--cell`, `--classes`) |
| `tools/scene_eval.py openscene / fuse-labels / compare` | Score methods against Unity GT with Jaime's metrics (+ obstacle F1, cost-map MAE, μ MAE) |
| `tools/make_synthetic.py` | Fake Unity export for testing |
| `tools/rellis_frame_test.py` | Single real frame, 2D + 3D zero-shot test (CLIP only, original test) |
| `tools/backbone_compare.py` | Same frame, several backbones / variants side by side |

`--weights` everywhere = open_clip pretrained tag (default `laion400m_e32`) or a `.pt` path.

## Scene format
What `unity/SceneCapture.cs` writes (and `make_synthetic.py` fakes):
```
scene/
  scene.json              classes {id: name} + conventions
  map.ply                 ascii x y z label (ROS FLU world, m)
  views/000000_rgb.png
  views/000000_depth.bin  float32 HxW z-depth (m), 0 = no hit
  views/000000_label.png  class id in R
  views/000000.json       fx fy cx cy, T_world_cam (4x4 row-major, OpenCV camera -> world)
```
If we go with Unity Perception / SOLO instead, the SOLO loader just has to produce the same thing (or call
`fusion_core` directly, which already matches these conventions with `axis_flip=(1,1,1)`).

## Code
- `car_semantic_map/clip_features.py`: `DenseCLIP`, `DensePE`, `make_extractor('clip' | 'pe-b' | 'pe-l' ...)`. GPU if available
- `car_semantic_map/openscene_fusion.py`: project map points into views (depth-checked), average features; label majority vote
- `car_semantic_map/scene_io.py`: scene loader, projection, `check_conventions`
- `car_semantic_map/evaluate.py`: metrics, cost map
- `config/ontology.json`: Jaime's class JSON layout + his RELLIS prompts. **Traversability / μ are placeholders**
