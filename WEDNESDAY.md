# Wednesday: PE probe on Jaime's machine

## 0. Before leaving home
- [ ] 10–20 Unity screenshots of terrain (grass, dirt path, pavement, trees) in a folder

## 1. Get the code
```bash
mkdir -p ~/nico_ws/src && cd ~/nico_ws/src
git clone https://github.com/nico54392/car_semantic_map.git
cd car_semantic_map
```

## 2. Env (once)
```bash
conda env list                        # find his SAM3 env
source setup.sh <his_env_name>        # clones it to nico_sam3, sets JAIME_SERVER
```
If his workspace isn't ~/go2_ws: `source setup.sh <his_env> /path/to/his_ws`

## 3. Frames
- Real: realsense-viewer → RGB on → camera icon (snapshot), 5–10 shots of grass, dirt, pavement, trees. Close the viewer after
- Sim: copy the Unity screenshots over
- Put all in `~/frames`

## 4. Run
```bash
conda activate nico_sam3
python tools/pe_probe.py --images ~/frames \
  --prompts "grass . lawn" "dirt . path" "tree" "pavement" \
  --out out/pe_probe
```

## 5. Bring back
- `out/pe_probe/shapes.txt` (always, even if it errors)
- `out/pe_probe/results.csv`
- 2–3 of the `*_grid.png`
- the terminal output

## Reading it
- "no image map shares a channel size" → normal, send shapes.txt, needs a projection layer
- AUROC ~0.5 → noise, SAM3 features not text-queryable this way
- AUROC 0.8+ → usable for rough per-point queries (Path A)

## Also
- [ ] Ask Jaime how he made the activation maps in the paper
- [ ] Mention the RELLIS mIoU mismatch (text 0.381 vs table 0.428)
