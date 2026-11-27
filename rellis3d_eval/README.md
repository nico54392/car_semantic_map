# RELLIS-3D test of the perception stack

This tests the perception half of 01 on real data with human labels, before anything goes into Unity. It covers:
- 2D segmentation against image labels;
- 2D → 3D fusion onto the Ouster scans, scored against the LiDAR labels;
- a fused voxel map built from many scans, for the planner.

## Data (lab machine)

Get these from the RELLIS-3D README (Google Drive):

| What | Size | Needed for |
|---|---|---|
| Full Images + Full Image Annotations **ID format** + Image Split File | 11 GB + 94 MB | eval2d |
| Ouster LiDAR SemanticKITTI + Ouster Annotation SemanticKITTI + Ouster Scan Poses | 14 GB + 174 MB + 174 MB | check, eval3d, buildmap |
| Camera Intrinsic (`camera_info.txt`) + Basler Camera to Ouster (`transforms.yaml`) | a few KB | everything 3D |

Unzip everything into one root so that each sequence folder `00000`…`00004` holds:
- `pylon_camera_node/`
- `pylon_camera_node_label_id/`
- `os1_cloud_node_kitti_bin/`
- `os1_cloud_node_semantickitti_label_id/`
- `poses.txt`

Put the two calibration files in the root.

```
pip install open_clip_torch opencv-python-headless scipy pyyaml pillow torch
```

## Run order

```bash
R=/data/RELLIS-3D
# 1. calibration, frame pairing and pose convention. Look at out/check_*.png too
python rellis.py check  --root $R --seq 00000 --frames 0:2000:100 --out out/check
# 2. 2D on the official image test split (comparable to Jaime's SAM3 0.428 mIoU)
python rellis.py eval2d --root $R --split test.lst --backbone clip --out out/2d_clip
# 3. 3D: single view vs fused (5 views each side, 0.5 s apart)
python rellis.py eval3d --root $R --seq 00000 --frames 0:2800:20 --window 5 --view-step 5 --backbone clip --out out/3d_clip
# 4. fused map for one stretch
python rellis.py buildmap --root $R --seq 00000 --frames 1000:1300:3 --backbone clip --out out/map
# 5. hand-off to control: 0.5 m grid (trav/mu mean+std, probs, height, step, slope, seen) + A* sanity check
python to_grid.py out/map/rellis_map_00000_1000-1297.npz --plan
# 6. pictures: frames.png, map_views.png, interactive map3d.html (pip install plotly)
python viz.py out/map/rellis_map_00000_1000-1297.npz --root $R
```

Backbones: `pe-b` (default), `pe-l`, `clip` come from `car_semantic_map/clip_features.py`; `oracle` uses the GT image labels. The ontology defaults to the repo's `config/ontology.json` (RELLIS classes only; `--all-classes` keeps the goal objects). `dirt` is its own class (the official benchmark maps it to void). Grids from `to_grid.py` follow `control_grid.py` (`axes="yx"`, `control_grid.load` transposes) plus an extra `step_rigid` layer.

Run each command with `--backbone oracle` as well. The oracle uses the GT image labels as "features", so it gives the ceiling that projection, fusion and the metrics allow. Report CLIP results next to it.

Other options:
- `--classes grass,tree,...` restricts the ontology.
- `--ontology` takes the harness's `ontology.json` or Jaime's class JSON.
- `--pose-frame kitti` is for when `check` says so.

## Reading the numbers
- **mIoU** averages over classes present in the GT. **mIoU_all** averages over all 18; that's what the official RELLIS notebook does, with void dropped.
- Check which convention Jaime's 0.428 uses before comparing.
- **3D ceiling:** on the example frame, LiDAR labels agree with the image labels only ~77% of the time. That makes the oracle's 3D accuracy ~0.77, not 1.0. Judge CLIP in 3D against the oracle row.
- **Patch ceiling:** with perfect labels pooled to 16-px CLIP patches, 2D mIoU tops out at ~0.82 on the example frame. Small classes (vehicle, fence, pole) get lost. LSeg/OpenSeg or SAM3 masks fix this; CLIP prompts don't.
- **Coverage:** the camera sees only ~10% of each 360° scan. Fusing neighbouring views is what raises coverage.
- **Control metrics:** trav/μ MAE, obstacle F1 and cost-map MAE use the placeholder values in `ontology_rellis.json` until Jaime's numbers are in.

## Files
- `rellis.py`: the CLI.
- `rellis_io.py`: loaders, label map, calibration, projection with distortion and occlusion check, poses.
- `backbones.py`: the `clip` backbone (ClearCLIP-style dense open_clip, sliding windows) and the `oracle`. Add LSeg or SAM3+CLIP here.
- `metrics.py`: the paper's metrics plus control metrics, with a majority-class baseline row.
- `viz.py`: data and map pictures + interactive 3D map (`--seen-only` to hide what no camera saw).
- `to_grid.py`: voxel map → control grid. `--source gt` builds it from the LiDAR labels (clean map for controls). Obstacles are semantic: only classes marked `rigid` in the ontology (tree, log, rubble...) block by height; grass/bush height only costs via traversability (`--geometric` for the old any-height rule). `--ontology` swaps the trav/μ values without rebuilding the map; `--risk` weights the std in the A* cost.
- `ontology_rellis.json`: the 18 RELLIS classes with prompts. Traversability and μ are placeholders.

`buildmap` writes `rellis_map_*.npz` with these keys:
- `xyz` f32;
- `feat` f16, 512-d;
- `gt`, the RELLIS eval id;
- `seen`.

It also writes GT and prediction `.ply` files. If `embedmap.py` expects different key names, rename them when loading.
