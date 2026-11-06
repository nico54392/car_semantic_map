# car_semantic_map

Stage 01: SAM3 embeddings on a known 3D map (see Obsidian `01 Map to Trajectory`).

Builds on Jaime's `ros2_segmentation_vlm` **without changing it**:
- his repo is only imported (his `server/` folder goes on the Python path), never edited
- new behaviour lives in subclasses here
- our server runs on port 8766, his stays on 8765
- separate workspace (`~/nico_ws`) and separate conda env (`nico_sam3`, cloned from his)

## Setup on the SAM3 machine
```bash
# 1. own env, cloned so nothing we install touches his
conda env list                                   # find his SAM3 env name
conda create -n nico_sam3 --clone <his_env> -y
conda activate nico_sam3

# 2. own workspace
mkdir -p ~/nico_ws/src && cd ~/nico_ws/src
git clone <this repo url> car_semantic_map

# 3. tell our code where his server folder is (read-only)
export JAIME_SERVER=~/go2_ws/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server
```

## Tools
- `tools/pe_probe.py`: PE dot-product test. Are SAM3's backbone features text-queryable per patch?
