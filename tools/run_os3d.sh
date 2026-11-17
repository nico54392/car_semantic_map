#!/usr/bin/env bash
# Unattended OpenScene 3D runs. Start with:  nohup bash tools/run_os3d.sh > ~/os3d_log.txt 2>&1 &
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nico_minkowski
export OMP_NUM_THREADS=12
cd "$(dirname "$0")/.."
OS=${OPENSCENE:-$HOME/nico_ws/src/openscene}
echo "=== 1. RELLIS scan, our classes + queries  $(date)"
python tools/openscene3d.py --openscene $OS --rellis data/rellis_sample --out out/os3d_rellis \
    --query "mud" "standing water" "somewhere soft"
echo "=== 2. RELLIS scan, nuScenes 16 classes  $(date)"
python tools/openscene3d.py --openscene $OS --rellis data/rellis_sample --labelset nuscenes --out out/os3d_rellis_nusc
echo "=== 3. synthetic map  $(date)"
python tools/openscene3d.py --openscene $OS --ply out/synth01/map.ply --out out/os3d_synth
echo "=== DONE  $(date)"
