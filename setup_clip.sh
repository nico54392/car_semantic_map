#!/usr/bin/env bash
# One-time setup for the CLIP / OpenScene pipeline. Separate env so nothing here
# can break the SAM3 env (open_clip pulls timm, which sam3 may pin).
# usage: source setup_clip.sh [env_to_clone]     (default: nico_sam3, it already has torch + CUDA)
# no set -e: this file is sourced, an error would close the terminal
BASE_ENV="${1:-nico_sam3}"

if ! conda env list | grep -q "^nico_clip "; then
  if conda env list | grep -q "^$BASE_ENV "; then
    echo "Cloning $BASE_ENV -> nico_clip"
    conda create -n nico_clip --clone "$BASE_ENV" -y
  else
    echo "$BASE_ENV not found, making a fresh env (install the CUDA torch build yourself if pip picks the wrong one)"
    conda create -n nico_clip python=3.10 -y
  fi
fi
conda activate nico_clip
python -c "import torch" 2>/dev/null || pip install torch torchvision
pip install -q open_clip_torch opencv-python-headless pillow scipy matplotlib pyyaml

# downloads the CLIP weights once (~600 MB, cached in ~/.cache/huggingface)
python - <<'PY'
import torch, open_clip
open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion400m_e32")
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
echo "Ready. Next: bash tools/smoke_test.sh"
