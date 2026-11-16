# OpenScene 3D model (distill) on our point clouds

`tools/openscene3d.py` runs the official OpenScene 3D network (MinkowskiNet, pretrained on nuScenes by default)
on any point cloud. No images and no fused 2D features: every point gets a 768-d CLIP-space feature
(OpenSeg = CLIP ViT-L/14@336 text side), so you can classify or query with text.

Why not their `run/eval.sh`: it loads the fused 2D features even in `distill` mode (165 GB download
for nuScenes), and their code uses `collections.Sequence`, which is gone in Python 3.10. This script
imports their model + voxelizer, patches the import, and skips the fused features.

## Setup (once, env with MinkowskiEngine)
```bash
source ~/miniconda3/bin/activate && conda activate nico_minkowski
cd ~/nico_ws/src/openscene && pip install -r requirements.txt     # yaml, open3d, clip, ...
cd ~/nico_ws/src/car_semantic_map && git pull
bash tools/get_sample_data.sh                                       # RELLIS sample, if not there yet
```

## Run
```bash
OS=~/nico_ws/src/openscene
# 1. RELLIS LiDAR scan, scored against its ground truth
python tools/openscene3d.py --openscene $OS --rellis data/rellis_sample --out out/os3d_rellis \
    --query "mud" "standing water" "somewhere soft"

# 2. same scan with the 16 nuScenes classes the model was trained/evaluated on
python tools/openscene3d.py --openscene $OS --rellis data/rellis_sample --labelset nuscenes --out out/os3d_rellis_nusc

# 3. synthetic / Unity map (map.ply + scene.json next to it)
python tools/openscene3d.py --openscene $OS --ply out/synth01/map.ply --out out/os3d_synth
```
First run downloads the 3D checkpoint and CLIP ViT-L/14@336 (~1 GB each).

Outputs in `--out`: `metrics.json` (acc, mIoU, per class), `bev.png` (GT vs prediction, top-down),
`pred.ply` (colored, open in CloudCompare), `query_*.png/.ply`, `feat.npy` (fp16 features per point).

## Things to know
- `--model nuscenes` (default, outdoor LiDAR) | `scannet` | `matterport` (indoor) | path to a `.pth.tar`
- Input color isn't used (OpenScene's default), only geometry. So it recognises shape/context, not texture:
  expect trees/vehicles/buildings to work better than grass vs mud
- Voxel size 0.05 m (what the model was trained with). Our synthetic map.ply is 0.2 m spaced, much sparser
  than the LiDAR it saw in training: expect it to struggle there
- Compare with the 2D-fusion numbers (CLIP/PE) in CLIP.md: same RELLIS scan, same classes
