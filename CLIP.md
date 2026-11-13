# CLIP / OpenScene pipeline on the work computer

Per-point CLIP embeddings on the map (OpenScene 2D fusion) → text queries → traversability cost map → A* path.
CLIP first to get it running; PE-Core swaps in later (only `car_semantic_map/clip_features.py` changes).

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
| `tools/rellis_frame_test.py` | Single real frame, 2D + 3D zero-shot test |

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
- `car_semantic_map/clip_features.py`: dense CLIP (ClearCLIP-style last layer), GPU if available. **The PE-Core swap goes here**
- `car_semantic_map/openscene_fusion.py`: project map points into views (depth-checked), average features; label majority vote
- `car_semantic_map/scene_io.py`: scene loader, projection, `check_conventions`
- `car_semantic_map/evaluate.py`: metrics, cost map
- `config/ontology.json`: Jaime's class JSON layout + his RELLIS prompts. **Traversability / μ are placeholders**
