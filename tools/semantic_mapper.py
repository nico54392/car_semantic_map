#!/usr/bin/env python3
"""
Live semantic map over ROS 2: camera (RGB + aligned depth + camera_info) + pose -> PE-B embedding map,
published as a PointCloud2 whose `intensity` is how well each point matches the current prompt.
Same node for Unity (unity/RGBDPublisher.cs) and a real RealSense (realsense2_camera + any odometry/SLAM).

    python tools/semantic_mapper.py                                   # defaults: Unity topics, pose from TF
    python tools/semantic_mapper.py --ros-args -p rgb_topic:=/camera/camera/color/image_raw \\
        -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \\
        -p info_topic:=/camera/camera/color/camera_info                # RealSense D435i

Change what it looks for, any time (RViz updates within a second):
    ros2 topic pub --once /semantic_map/prompt std_msgs/String "data: 'guy with a red hat'"
    ros2 param set /semantic_mapper threshold 0.6                     # or: ros2 topic pub ... /semantic_map/threshold std_msgs/Float32 "data: 0.6"
    ros2 service call /semantic_map/save std_srvs/srv/Trigger         # -> <out_dir>/embedded_map.npz (prompt_drive.py --map ai reads it)
    ros2 service call /semantic_map/clear std_srvs/srv/Trigger

Publishes
  /semantic_map/cloud      PointCloud2 (x y z intensity rgb), whole map, intensity = prompt score (0 with no prompt)
  /semantic_map/matches    PointCloud2, only points with score >= threshold
  /semantic_map/keyframes  PoseArray, where each keyframe was taken (the "track")
Pose
  pose_source = tf     map_frame <- the depth image's frame_id (RealSense optical frame; Unity: see RGBDPublisher)
  pose_source = topic  geometry_msgs/PoseStamped of the camera (OpenCV optical axes) in the map frame
Keyframes: only when the camera moved >= kf_dist m or turned >= kf_angle deg (and at most every kf_period s).
Run it in the nico_clip env with ROS sourced (needs torch + open_clip for PE-B).
"""
import math, queue, sys, threading, time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import String, Float32
from std_srvs.srv import Trigger
import message_filters
from tf2_ros import Buffer, TransformListener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from car_semantic_map.online_map import OnlineEmbeddingMap          # noqa: E402


# ---------------------------------------------------------------- message helpers (no cv_bridge needed)

def image_to_numpy(msg):
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if "a" in enc[3:] else 3
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
        a = a[..., :3]
        return a[..., ::-1].copy() if enc.startswith("bgr") else a.copy()
    if enc in ("16uc1", "mono16"):
        a = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return a.astype(np.float32) * 0.001                  # RealSense: millimetres
    if enc == "32fc1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width].copy()
    if enc in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width].copy()
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def make_cloud(points, intensity, colors, frame, stamp=None):
    n = len(points)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("rgb", "<f4")])
    if n:
        rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
        rec["intensity"] = intensity
        c = colors.astype(np.uint32)
        rec["rgb"] = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
    msg = PointCloud2()
    msg.header.frame_id = frame
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height, msg.width = 1, n
    msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, k in enumerate(["x", "y", "z", "intensity", "rgb"])]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 20, 20 * n, True
    msg.data = rec.tobytes()
    return msg


# ---------------------------------------------------------------- node

class SemanticMapper(Node):
    def __init__(self):
        super().__init__("semantic_mapper")
        P = lambda n, d: self.declare_parameter(n, d).value
        self.rgb_topic = P("rgb_topic", "/camera/color/image_raw")
        self.depth_topic = P("depth_topic", "/camera/depth/image_raw")
        self.info_topic = P("info_topic", "/camera/color/camera_info")
        self.pose_source = P("pose_source", "tf")             # tf | topic
        self.pose_topic = P("pose_topic", "/camera_pose_unity")
        self.map_frame = P("map_frame", "map")
        self.backbone = P("backbone", "pe-b")
        self.voxel = P("voxel", 0.1)
        self.max_depth = P("max_depth", 5.0)                   # D435i: ~4-6 m usable outdoors
        self.stride = P("stride", 4)                           # use every stride-th pixel
        self.kf_dist = P("kf_dist", 0.3)
        self.kf_angle = P("kf_angle", 15.0)
        self.kf_period = P("kf_period", 0.5)
        self.prompt = P("prompt", "")
        self.threshold = P("threshold", 0.5)
        self.mode = P("mode", "prob")                          # prob | cosine
        self.publish_rate = P("publish_rate", 1.0)
        self.out_dir = P("out_dir", str(Path.home() / "nico_ws" / "scenes" / time.strftime("live_%Y%m%d_%H%M%S")))
        self.add_on_set_parameters_callback(self.on_params)

        self.get_logger().info(f"loading {self.backbone} ...")
        from car_semantic_map.clip_features import make_extractor
        self.map = OnlineEmbeddingMap(make_extractor(self.backbone), backbone=self.backbone, voxel=self.voxel,
                                      max_depth=self.max_depth, stride=self.stride)
        self.lock = threading.Lock()
        self.jobs = queue.Queue(maxsize=1)                     # newest keyframe only; PE-B runs in a worker
        self.last_kf, self.t_last_kf, self.cam_pose = None, 0.0, None
        self.dirty, self.score_key, self.scores, self.msgs = True, None, None, None
        self.warned = set()

        self.tf = Buffer(); TransformListener(self.tf, self)
        if self.pose_source == "topic":
            self.create_subscription(PoseStamped, self.pose_topic, self.on_pose, qos_profile_sensor_data)
        subs = [message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data),
                message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data),
                message_filters.Subscriber(self, CameraInfo, self.info_topic, qos_profile=qos_profile_sensor_data)]
        message_filters.ApproximateTimeSynchronizer(subs, 10, 0.1).registerCallback(self.on_frame)
        self.create_subscription(String, "/semantic_map/prompt", lambda m: self.set_prompt(m.data), 10)
        self.create_subscription(Float32, "/semantic_map/threshold", lambda m: self.set_threshold(m.data), 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "/semantic_map/cloud", 1)
        self.pub_match = self.create_publisher(PointCloud2, "/semantic_map/matches", 1)
        self.pub_kf = self.create_publisher(PoseArray, "/semantic_map/keyframes", 1)
        self.create_service(Trigger, "/semantic_map/save", self.srv_save)
        self.create_service(Trigger, "/semantic_map/clear", self.srv_clear)
        self.create_timer(1.0 / max(self.publish_rate, 0.1), self.publish)
        threading.Thread(target=self.worker, daemon=True).start()
        self.get_logger().info(f"waiting for {self.rgb_topic} + {self.depth_topic} + {self.info_topic}, "
                               f"pose from {'TF (' + self.map_frame + ' <- camera frame)' if self.pose_source == 'tf' else self.pose_topic}")

    # ---- parameters / prompt
    def on_params(self, params):
        for p in params:
            if p.name == "prompt":
                self.set_prompt(p.value, from_param=True)
            elif p.name == "threshold":
                self.set_threshold(p.value, from_param=True)
            elif p.name == "mode":
                self.mode, self.dirty = p.value, True
        return SetParametersResult(successful=True)

    def set_prompt(self, text, from_param=False):
        self.prompt, self.dirty = text.strip(), True
        self.get_logger().info(f'prompt: "{self.prompt}"' if self.prompt else "prompt cleared")

    def set_threshold(self, t, from_param=False):
        self.threshold, self.dirty = float(t), True
        self.get_logger().info(f"threshold: {self.threshold:.2f}")

    # ---- pose
    def on_pose(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        T = np.eye(4); T[:3, :3] = quat_to_R(o.x, o.y, o.z, o.w); T[:3, 3] = [p.x, p.y, p.z]
        self.cam_pose = T

    def pose_at(self, frame, stamp):
        if self.pose_source == "topic":
            return self.cam_pose
        from rclpy.time import Time
        from rclpy.duration import Duration
        for t in (Time.from_msg(stamp), Time()):               # exact stamp, else latest
            try:
                tr = self.tf.lookup_transform(self.map_frame, frame, t, timeout=Duration(seconds=0.05))
                q, v = tr.transform.rotation, tr.transform.translation
                T = np.eye(4); T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w); T[:3, 3] = [v.x, v.y, v.z]
                return T
            except Exception as e:
                err = e
        if "tf" not in self.warned:
            self.warned.add("tf"); self.get_logger().warn(f"no TF {self.map_frame} <- {frame} yet: {err}")
        return None

    # ---- frames -> keyframes
    def on_frame(self, rgb_msg, depth_msg, info_msg):
        now = time.time()
        if now - self.t_last_kf < self.kf_period:
            return
        T = self.pose_at(depth_msg.header.frame_id or info_msg.header.frame_id, depth_msg.header.stamp)
        if T is None:
            return
        if self.last_kf is not None:
            d = np.linalg.norm(T[:3, 3] - self.last_kf[:3, 3])
            ang = math.degrees(math.acos(np.clip((np.trace(self.last_kf[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
            if d < self.kf_dist and ang < self.kf_angle:
                return
        try:
            rgb, depth = image_to_numpy(rgb_msg), image_to_numpy(depth_msg)
        except ValueError as e:
            self.get_logger().error(str(e)); return
        K = np.array(info_msg.k, float).reshape(3, 3)
        if depth.shape != rgb.shape[:2]:
            if "size" not in self.warned:
                self.warned.add("size")
                self.get_logger().warn(f"depth {depth.shape} != rgb {rgb.shape[:2]}: use ALIGNED depth "
                                       "(RealSense: align_depth.enable:=true). Resizing depth for now.")
            ys = (np.arange(rgb.shape[0]) * depth.shape[0] / rgb.shape[0]).astype(int)
            xs = (np.arange(rgb.shape[1]) * depth.shape[1] / rgb.shape[1]).astype(int)
            depth = depth[ys][:, xs]
        if self.jobs.full():
            try:
                self.jobs.get_nowait()                         # drop the older pending keyframe
            except queue.Empty:
                pass
        self.jobs.put((rgb, depth, K, T))
        self.last_kf, self.t_last_kf = T, now

    def worker(self):
        while True:
            rgb, depth, K, T = self.jobs.get()
            t0 = time.time()
            feats = self.map.ex.image_features(rgb)            # GPU, outside the lock
            feats = feats.numpy() if hasattr(feats, "numpy") else feats
            with self.lock:
                n = self.map.insert(rgb, depth, K, T, features=feats)
                self.dirty = True
            self.get_logger().info(f"keyframe {len(self.map.keyframes)}: +{n} pts, {self.map.n} voxels, "
                                   f"{time.time() - t0:.2f} s")

    # ---- output
    def publish(self):
        """rebuild the clouds only when something changed; republish every tick so late RViz windows see them"""
        if self.map.n == 0:
            return
        if self.dirty:
            with self.lock:
                pts, cols = self.map.points, self.map.colors
                key = (self.prompt, self.mode, self.map.n, len(self.map.keyframes))
                if self.prompt and key != self.score_key:
                    self.scores, dropped = self.map.score(self.prompt, self.mode)
                    self.score_key = key
                    hit = (self.scores >= self.threshold).sum()
                    self.get_logger().info(f'"{self.prompt}": {hit} / {len(pts)} points >= {self.threshold:.2f} '
                                           f"(max {self.scores.max():.2f})" + (f"; not competing: {dropped}" if dropped else ""))
                kfs = list(self.map.keyframes)
                self.dirty = False
            s = self.scores if (self.prompt and self.scores is not None and len(self.scores) == len(pts)) \
                else np.zeros(len(pts), np.float32)
            m = s >= self.threshold if self.prompt else np.zeros(len(pts), bool)
            self.msgs = (make_cloud(pts, s, cols, self.map_frame, None), make_cloud(pts[m], s[m], cols[m], self.map_frame, None),
                         self.keyframe_msg(kfs))
        stamp = self.get_clock().now().to_msg()
        for msg, pub in zip(self.msgs, (self.pub_cloud, self.pub_match, self.pub_kf)):
            msg.header.stamp = stamp
            pub.publish(msg)

    def keyframe_msg(self, kfs):
        pa = PoseArray(); pa.header.frame_id = self.map_frame
        for T in kfs:
            ps = Pose(); ps.position.x, ps.position.y, ps.position.z = map(float, T[:3, 3])
            # PoseArray arrows point along +x: show the camera's viewing direction (optical +z)
            R = T[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T
            ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = R_to_quat(R)
            pa.poses.append(ps)
        return pa

    def srv_save(self, req, res):
        with self.lock:
            out = self.map.save(self.out_dir)
        res.success, res.message = True, f"saved {out} ({self.map.n} voxels). Use: python tools/prompt_drive.py {Path(out).parent} --map ai"
        self.get_logger().info(res.message)
        return res

    def srv_clear(self, req, res):
        with self.lock:
            self.map = OnlineEmbeddingMap(self.map.ex, backbone=self.backbone, voxel=self.voxel,
                                          max_depth=self.max_depth, stride=self.stride)
            self.last_kf, self.dirty = None, True
        res.success, res.message = True, "map cleared"
        return res


def main():
    rclpy.init()
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
