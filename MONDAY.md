# 01 Monday Plan: Unity export → prompt → robot drives there
Same plan as Obsidian `01 Monday Plan`. Details: [FUSION.md](FUSION.md).

**Goal for Monday:** first real Unity export through the whole chain, ending with "go to the bench" → path → robot drives it in Unity. That's the first "Done when" in the hub.

**Not Monday:** tuning on the synthetic scene, the rock false positives, the labels baseline, MinkowskiEngine, real traversability values. All parked below.

---
## Before Monday (Claude)
- [x] `tools/goal_from_prompt.py`: prompt → 3D mask (prompt vs all ontology classes, synonyms dropped) → blobs → goal on the ring around the object, closest to the start → A* path. Works for prompts not in the ontology
- [x] Path export: `goals/<prompt>/path.json` (ROS + Unity coords), `path.csv`, `mask.ply`, `goal.png`
- [x] `tools/ros2_follow_path.py`: pure pursuit → `/cmd_vel`, pose from `/odom_unity` (default)
- [x] Object classes in `ontology.json` (bench, car, table, sign, door, wall, trash can) + RELLIS tree/building/vehicle/person
- [x] Tested on the synthetic scene: "go to the puddle" and "standing water" (not in ontology) → goal + path; simulated robot reaches the goal, ≤0.23 m off the path

## Before Monday (Nico, if there's time)
- [x] Pose: `car_sim/Assets/Scripts/Publishers + Nav2/Nav2Publisher.cs` → `nav_msgs/Odometry` on `/odom_unity` (TF on `/tf_unity`). Same (z,-x,y) conversion as the map export → same frame, no offset
- [x] `/cmd_vel` driver: `car_sim/Assets/CAR/Scripts/CmdVelDriver.cs`, `geometry_msgs/Twist`, gives control back to the keyboard after 0.5 s of silence
- [ ] Confirm live: with the sim running, `ros2 topic list | grep -E "odom_unity|cmd_vel"` and `ros2 topic echo /odom_unity --once`

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
python tools/feature_map.py build $S --backbone pe-b
python tools/feature_map.py query $S "grass" "mud" "dirt path" "bench" "tree"
python tools/scene_eval.py fuse $S --backbone pe-b && python tools/scene_eval.py compare $S --backbone pe-b
```
- [ ] `query.png`: does "bench" light up the bench?
- [ ] `report.md`: first real numbers against Unity ground truth

### 6. Prompt → path (~30 min)
Start = robot spawn in the map frame: `ros2 topic echo /odom_unity --once` → `pose.pose.position.x/y`.
```bash
python tools/goal_from_prompt.py $S "go to the bench" --start <x> <y>
python tools/goal_from_prompt.py $S "picnic table" --start <x> <y>    # NOT in the ontology → 3D mask still?
```
Look at `$S/goals/<prompt>/goal.png`. Knobs: `--thresh` (mask), `--pick nearest`, `--standoff` (m from object), `--synonym`.
Open `mask.ply` in CloudCompare to see the 3D mask.
- [ ] Path goes to the bench and around bad terrain
- [ ] An unseen prompt gives a sensible 3D mask (2nd "Done when")

### 7. Drive it (~1 h)
```bash
source /opt/ros/humble/setup.bash          # same ROS_DOMAIN_ID / CycloneDDS as the Unity bridge
python3 tools/ros2_follow_path.py --ros-args -p path_file:=$S/goals/bench/path.json
```
- [ ] First log line prints the robot pose and the path start: they should be close. If yaw looks off, `-p heading_offset_deg:=90` (or -90/180)
- [ ] Too twitchy / cuts corners: `-p lookahead:=1.2 -p v_max:=0.3`
- [ ] Follower needs `rclpy` → run it with the system ROS python, not inside `nico_clip`
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
