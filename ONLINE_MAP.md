# Live semantic map: camera → PE-B → 3D map in RViz, queried by prompt

The robot builds its own map from what its camera sees. You type a prompt, and RViz lights up the
matching points (intensity = score, threshold = what counts as a match). The same nodes run on Unity
and on the real robot with the D435i; only the camera driver and the pose source change.

```
camera (RGB + aligned depth + camera_info)  ─┐
pose (TF map<-camera, or a pose topic)       ├─► semantic_mapper.py ─► /semantic_map/cloud    (x y z intensity rgb)
prompt / threshold (topic or ros2 param)    ─┘    (PE-B per keyframe,  /semantic_map/matches  (score >= threshold)
                                                   0.1 m voxels)       /semantic_map/keyframes (camera track)
                                                        │ save
                                                        ▼
                                     <out_dir>/embedded_map.npz ─► prompt_drive.py --map ai (drive to it)
```

The pieces, each replaceable:

| piece | Unity | real robot |
|---|---|---|
| camera | `unity/RGBDPublisher.cs` on the robot's camera | `realsense2_camera` with `align_depth.enable:=true` |
| pose | `/camera_pose_unity` (`pose_source:=topic`) | TF from RTAB-Map / VO (`pose_source:=tf`) |
| initial scan | `tools/scan_drive.py` (circle or spin) | same, with `--odom_topic` = the robot's odometry |
| map + query | `tools/semantic_mapper.py` | same |
| view | `rviz2 -d config/rviz/semantic_map.rviz` | same |
| drive to a prompt | `tools/prompt_drive.py <out_dir> --map ai` | same |

## Unity

1. Put `unity/RGBDPublisher.cs` on the **robot's camera**. It only reads that camera; SceneCapture is the one that moves its camera, so keep the two apart. Leave the defaults: 320×240 at 3 Hz. Physical Camera must be off.
2. Start the terminals. As before, the endpoint and RViz run without conda.
   ```bash
   # 1: ROS-TCP endpoint (workspace ~/nico_ws/ros)
   ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
   # 2: mapper (nico_clip + ROS sourced)
   python tools/semantic_mapper.py --ros-args -p pose_source:=topic -p max_depth:=8.0
   # 3: RViz
   rviz2 -d config/rviz/semantic_map.rviz
   # 4: initial scan
   python3 tools/scan_drive.py --laps 1 --radius 1.5
   ```
3. Watch the map build in RViz.
   - The cyan arrows are the keyframes the mapper kept.
   - The colour is the real RGB until you give it a prompt. Untick "map: prompt score" and tick "map: real colour" to see it that way.
4. Query it:
   ```bash
   ros2 topic pub --once /semantic_map/prompt std_msgs/String "data: 'puddle'"
   ros2 topic pub --once /semantic_map/prompt std_msgs/String "data: 'guy with a red hat'"
   ros2 param set /semantic_mapper threshold 0.6
   ```
   The yellow-to-dark colouring is the score; the red boxes are the points above the threshold.
5. Save the map, then drive to it:
   ```bash
   ros2 service call /semantic_map/save std_srvs/srv/Trigger
   python tools/prompt_drive.py ~/nico_ws/scenes/live_<time> --map ai
   ```
   The save message prints the exact folder.

## Real robot / handheld D435i

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=false
# pose: any SLAM/VO that publishes TF map -> camera_color_optical_frame, e.g. RTAB-Map's RGB-D odometry + mapping
python tools/semantic_mapper.py --ros-args \
  -p rgb_topic:=/camera/camera/color/image_raw \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p info_topic:=/camera/camera/color/camera_info \
  -p pose_source:=tf -p max_depth:=4.0
rviz2 -d config/rviz/semantic_map.rviz      # set the camera Image display to /camera/camera/color/image_raw
```

- **Topic names.** Recent `realsense2_camera` versions use the `/camera/camera/...` namespace. Check with `ros2 topic list`.
- **Depth.** It must be **aligned** to colour (`align_depth.enable:=true`). The mapper reads 16UC1 in millimetres.
- **No pose source, no map.** Without SLAM/VO publishing TF, the mapper waits and warns `no TF map <- ...`.

## Parameters worth knowing

| param | default | what it does |
|---|---|---|
| `voxel` | 0.1 | map resolution (m) |
| `max_depth` | 5.0 | ignore depth beyond this distance. Use ~4 for the D435i outdoors, more in Unity |
| `stride` | 4 | use every 4th pixel, which is enough at 0.1 m voxels |
| `kf_dist` / `kf_angle` / `kf_period` | 0.3 m / 15° / 0.5 s | when a frame becomes a keyframe |
| `prompt` / `threshold` | "" / 0.5 | what to look for, and what counts as a match |
| `mode` | prob | `prob` = P(prompt beats every ontology class); `cosine` = raw similarity |
| `backbone` | pe-b | any `clip_features.make_extractor` name |
| `out_dir` | `~/nico_ws/scenes/live_<time>` | where `save` writes |

## Limits right now

- **No loop closure or relocalisation in the mapper.** The map is only as good as the pose source: Unity is perfect, the real robot is only as good as RTAB-Map.
- **Moving things (people walking) leave trails.** Voxels average over time and are never removed. Use `clear` to start over.
- **Fine attributes depend on resolution.** "The red hat" works up close; far away it's a few 16 px patches.
- **There's no GT label image from Unity yet.** That means no per-class score for the live map. The offline route (`SceneCapture` → `scene_eval`) still does that.
