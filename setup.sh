#!/usr/bin/env bash
# One-time setup on the SAM3 machine. Touches nothing of Jaime's.
# usage: source setup.sh <jaime_conda_env> [jaime_workspace]
# no set -e: this file is sourced, an error would close the terminal
JAIME_ENV="$1"
if [ -z "$JAIME_ENV" ]; then echo "usage: source setup.sh <his_sam3_env> [his_workspace]   (see: conda env list)"; return 1; fi
JAIME_WS="${2:-$HOME/go2_ws}"

if ! conda env list | grep -q "^nico_sam3 "; then
  echo "Cloning $JAIME_ENV -> nico_sam3 (his env stays untouched)"
  conda create -n nico_sam3 --clone "$JAIME_ENV" -y
fi
conda activate nico_sam3
pip install -q matplotlib

export JAIME_SERVER="$JAIME_WS/src/ros2_segmentation_vlm/ros2_segmentation_vlm/server"
if [ ! -f "$JAIME_SERVER/inference/sam3_multi_prompt.py" ]; then
  echo "Can't find Jaime's server at $JAIME_SERVER. Pass his workspace as 2nd arg."; return 1
fi
grep -q "JAIME_SERVER=" ~/.bashrc || echo "export JAIME_SERVER=$JAIME_SERVER" >> ~/.bashrc
python -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Ready. JAIME_SERVER=$JAIME_SERVER"
