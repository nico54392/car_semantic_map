#!/usr/bin/env bash
# End-to-end check of the CLIP / OpenScene pipeline, no Unity needed.
# RELLIS sample frame -> fake Unity export -> check -> embedded map -> query -> plan -> metrics.
# Run from the repo root with nico_clip active. Outputs in out/.
set -e
cd "$(dirname "$0")/.."
S=out/synth01
R=data/rellis_sample/frame000104-1581624663_149

echo "== 1. sample data";            bash tools/get_sample_data.sh
echo "== 2. real frame test";        python tools/rellis_frame_test.py data/rellis_sample out/rellis_frame
echo "== 3. fake Unity export";      [ -d $S ] || python tools/make_synthetic.py $S --photo $R.jpg --photo_labels $R.png --views 12
echo "== 4. convention check";       python tools/scene_eval.py check $S
echo "== 5. embedded map";           python tools/openscene_map.py build $S
echo "== 6. text queries";           python tools/openscene_map.py query $S "standing water" "mud" "tall bushes"
echo "== 7. cost map + path";        python tools/openscene_map.py plan $S --start 5 -5 --goal 35 -35 --cell 0.5 \
                                       --classes grass mud puddle bush dirt gravel rock
echo "== 8. metrics vs GT";          python tools/scene_eval.py openscene $S
                                     python tools/scene_eval.py compare $S
echo
echo "Done. Look at: out/rellis_frame/*.png, $S/query.png, $S/plan.png, $S/compare_*.png, $S/report.md"
