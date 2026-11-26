#!/usr/bin/env python3
"""
Follow a path from goal_from_prompt.py with pure pursuit: reads the robot pose, publishes /cmd_vel.

  source /opt/ros/humble/setup.bash     # + ROS_DOMAIN_ID=42 / CycloneDDS like the Unity bridge
  python3 tools/ros2_follow_path.py --ros-args \
      -p path_file:=<scene>/goals/bench/path.json \
      -p pose_topic:=/odom_unity

Parameters
  path_file      path.json from goal_from_prompt.py (uses waypoints_ros, map frame = ROS FLU)
  pose_topic     where the robot pose comes from. car_sim's Nav2Publisher.cs publishes nav_msgs/Odometry
                 on /odom_unity = Unity world pose converted with (z,-x,y), same frame as the map
                 export, so no offset is needed (default)
  pose_type      odom (nav_msgs/Odometry) | pose (geometry_msgs/PoseStamped)
                 | pose_cov (geometry_msgs/PoseWithCovarianceStamped)
  cmd_vel_topic  default /cmd_vel (geometry_msgs/Twist)
  lookahead, v_max, w_max, goal_tol   pure pursuit (m, m/s, rad/s, m)
  offset_x, offset_y, offset_yaw      only if the pose topic is NOT in the map frame, e.g. odom that
                 starts at (0,0) at spawn: map = R(offset_yaw) * odom + offset. Leave 0 if Unity
                 publishes world pose converted to ROS (FLU), which is what the map uses
  heading_offset_deg  if the robot model's forward isn't Unity +Z, yaw from odom is off by a constant.
                 Check: drive forward with the keyboard, odom x/y should move along the reported yaw
  rate           control loop Hz (default 20)

Publishes the path on /follow_path/path (nav_msgs/Path) so it shows in RViz.
Stops (zero Twist) on arrival, on Ctrl-C, and if the pose goes stale (> pose_timeout, default 1.0 s).
"""
import json, math, sys, time
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path as PathMsg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_semantic_map.pure_pursuit import PurePursuit   # noqa: E402


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Follower(Node):
    def __init__(self):
        super().__init__("follow_path")
        p = lambda n, d: self.declare_parameter(n, d).value
        path_file = p("path_file", "")
        self.pose_type = p("pose_type", "odom")
        pose_topic = p("pose_topic", "/odom_unity")
        self.heading_offset = math.radians(p("heading_offset_deg", 0.0))
        cmd_topic = p("cmd_vel_topic", "/cmd_vel")
        self.ox, self.oy, self.oyaw = p("offset_x", 0.0), p("offset_y", 0.0), p("offset_yaw", 0.0)
        if not path_file:
            raise SystemExit("set -p path_file:=.../path.json")
        d = json.load(open(path_file))
        wp = [w[:2] for w in d["waypoints_ros"]]
        self.pp = PurePursuit(wp, lookahead=p("lookahead", 0.8), v_max=p("v_max", 0.5),
                              w_max=p("w_max", 1.0), goal_tol=p("goal_tol", 0.3))
        self.get_logger().info(f'"{d["prompt"]}": {len(wp)} waypoints, {d["path_length_m"]:.1f} m, '
                               f'goal {d["goal_ros"][:2]}')

        msg_type = {"odom": Odometry, "pose": PoseStamped, "pose_cov": PoseWithCovarianceStamped}[self.pose_type]
        from rclpy.qos import qos_profile_sensor_data     # best effort: works with reliable and best-effort publishers
        self.create_subscription(msg_type, pose_topic, self.on_pose, qos_profile_sensor_data)
        self.pose_timeout = p("pose_timeout", 1.0)
        self.cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.path_pub = self.create_publisher(PathMsg, "/follow_path/path", 1)
        self.path_msg = PathMsg()
        self.path_msg.header.frame_id = "map"
        for x, y, z in d["waypoints_ros"]:
            ps = PoseStamped(); ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), float(z)
            self.path_msg.poses.append(ps)
        self.pose, self.t_pose, self.done = None, 0.0, False
        self.create_timer(1.0 / p("rate", 20.0), self.tick)
        self.create_timer(1.0, lambda: self.path_pub.publish(self.path_msg))
        self.get_logger().info(f"waiting for {pose_topic} ({self.pose_type}) ...")

    def on_pose(self, msg):
        pose = msg.pose.pose if self.pose_type in ("odom", "pose_cov") else msg.pose
        x, y, yaw = pose.position.x, pose.position.y, yaw_of(pose.orientation)
        c, s = math.cos(self.oyaw), math.sin(self.oyaw)          # pose frame -> map frame
        self.pose = (c * x - s * y + self.ox, s * x + c * y + self.oy, yaw + self.oyaw + self.heading_offset)
        if self.t_pose == 0.0:
            self.get_logger().info(f"first pose (map frame): x={self.pose[0]:.2f} y={self.pose[1]:.2f} "
                                   f"yaw={math.degrees(self.pose[2]):.0f} deg. Path starts at "
                                   f"{self.pp.wp[0].round(2).tolist()}: these should be close")
        self.t_pose = time.time()

    def tick(self):
        if self.done:
            return
        tw = Twist()
        if self.pose is None or time.time() - self.t_pose > self.pose_timeout:
            self.cmd.publish(tw)                                  # no / stale pose: stand still
            return
        v, w, self.done = self.pp.step(*self.pose)
        tw.linear.x, tw.angular.z = float(v), float(w)
        self.cmd.publish(tw)
        if self.done:
            self.get_logger().info("goal reached")

    def stop(self):
        self.cmd.publish(Twist())


def main():
    rclpy.init()
    node = Follower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
