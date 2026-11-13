# 01 Monday Plan: Unity export → prompt → robot drives there
Same plan as Obsidian `01 Monday Plan`. Details: [CLIP.md](CLIP.md).

**Goal for Monday:** first real Unity export through the whole chain, ending with "go to the bench" → path → robot drives it in Unity. That's the first "Done when" in the hub.

**Not Monday:** tuning on the synthetic scene, the rock false positives, the labels baseline, MinkowskiEngine, real traversability values. All parked below.

---
## Before Monday (Claude)
- [ ] `tools/goal_from_prompt.py`: text → per-point score → threshold + cluster → object position → nearest free cell next to it → A* path. Also prints the 3D mask for prompts that weren't in the ontology (2nd "Done when")
- [ ] Path export: waypoints in Unity coords + ROS (FLU) coords, JSON/CSV
- [ ] Simple ROS2 follower (pure pursuit on the waypoints → `/cmd_vel`), using whatever pose topic Unity publishes
- [ ] Object classes in `ontology.json` (bench, tree, building, car…) with prompts + traversability 0
- [ ] Test all of it on the synthetic scene

## Before Monday (Nico, if there's time)
- [ ] Check what the Unity robot publishes: `ros2 topic list` with the sim running. Need a pose (`/odom`, `/tf` or a pose topic) for the follower
- [ ] Where does the `/cmd_vel` driver script live, and does it take `geometry_msgs/Twist`?

---
## Monday

### 1. Test scene in Unity (~1–2 h)
- [ ] Terrain ~40×40 m with **4–5 layers**: grass, dirt (path), mud, gravel. Puddles if easy (flat dark patches)
- [ ] A few **goal objects**: bench, tree or two, a building or car. These are what we'll ask for by name
- [ ] Some rocks (real obstacles)
- [ ] **Colliders on everything** (terrain collider is automatic; MeshCollider on props). No collider = missing from depth and labels
- [ ] Robot spawn point known (needed as path start)

### 2. Hook up SceneCapture (~30 min)
- [ ] `SceneCapture.cs` + `TerrainClass.cs` into `Assets/Scripts`
- [ ] On a camera at robot camera height. **Physical Camera off**
- [ ] `classes`: same names as `ontology.json` (0 void, grass, dirt, mud, gravel, rock, bench, tree, building…)
- [ ] `terrainLayerToClass`: one id per terrain layer, in Paint Texture order
- [ ] `TerrainClass` on each prop with its id
- [ ] `autoCaptureOnStart`, `autoViews` = **80** (12 gave streaky maps), `cameraHeight` = robot camera height
- [ ] Play → `Captures/scene_<time>/`
- [ ] Sanity: open a few `*_rgb.png`. Black → tell Claude (URP render fix). Compile errors → send them

### 3. Move it to the work computer
- [ ] Copy the capture folder to `~/nico_ws/scenes/unity01` (scp / USB / OneDrive)

### 4. Check the export (~15 min)
```bash
conda activate nico_clip && cd ~/nico_ws/src/car_semantic_map && git pull
python tools/scene_eval.py check ~/nico_ws/scenes/unity01
```
- [ ] Says **OK** (~95%+). If not, send the output, it lists the usual causes

### 5. Embedded map + sanity queries (~15 min)
```bash
S=~/nico_ws/scenes/unity01
python tools/openscene_map.py build $S --backbone pe-b
python tools/openscene_map.py query $S "grass" "mud" "dirt path" "bench" "tree"
python tools/scene_eval.py openscene $S --backbone pe-b && python tools/scene_eval.py compare $S --backbone pe-b
```
- [ ] `query.png`: does "bench" light up the bench?
- [ ] `report.md`: first real numbers against Unity ground truth

### 6. Prompt → path (~30 min)
```bash
python tools/goal_from_prompt.py $S "go to the bench" --start <spawn x y>
python tools/goal_from_prompt.py $S "picnic table"      # NOT in the ontology → 3D mask still?
```
- [ ] Path goes to the bench and around bad terrain
- [ ] An unseen prompt gives a sensible 3D mask (2nd "Done when")

### 7. Drive it (~1 h)
- [ ] Sim running, follower node gets the waypoints, publishes `/cmd_vel`
- [ ] Robot reaches the bench → **record it** (screen capture for Jaime)

## Bring back
- `check` output, `report.md`, `query.png`, `plan.png`, the path JSON
- Video of the robot driving
- Anything that broke

---
## Parked (after Monday)
- [ ] Rock / obstacle false positives (prompt wording, confidence margin)
- [ ] Labels baseline: SAM3 per view → `fuse-labels` → compare (3rd "Done when")
- [ ] Real traversability / μ from Jaime's class JSON
- [ ] SAM3-mask + PE hybrid for sharper boundaries
- [ ] Class → wheel configuration table
- [ ] MinkowskiEngine / OpenScene 3D net
- [ ] Nav2 / MPC instead of A* + pure pursuit
