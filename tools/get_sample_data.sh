#!/usr/bin/env bash
# Grabs the one example frame that ships with the RELLIS-3D repo (image, 2D labels,
# Ouster scan + labels, calibration) into data/rellis_sample. ~4 MB.
# Used by rellis_frame_test.py and as texture source for make_synthetic.py.
set -e
D="$(cd "$(dirname "$0")/.." && pwd)/data/rellis_sample"
mkdir -p "$D"
BASE=https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/utils/example
for f in frame000104-1581624663_149.jpg frame000104-1581624663_149.png 000104.bin 000104.label \
         camera_info.txt transforms.yaml; do
  [ -f "$D/$f" ] || curl -sSfL -o "$D/$f" "$BASE/$f"
done
echo "RELLIS sample in $D"
