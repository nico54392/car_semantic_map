#!/usr/bin/env python3
"""
Initial scan: drive a circle (or spin in place) so the camera sees all around, while
tools/semantic_mapper.py builds the map from the keyframes. Stops by itself after --laps turns.

    python3 tools/scan_drive.py                          # 1 lap, 1.5 m radius, 0.3 m/s
    python3 tools/scan_drive.py --radius 0 --w 0.4       # spin in place (no space / narrow street)
    python3 tools/scan_drive.py --laps 2 --radius 3

Heading comes from the odometry topic (counts the turned angle, so it stops after real laps even if
the wheels slip). Real robot: --odom_topic /odom (or whatever your odometry publishes).
Ctrl-C stops the robot.
"""
import argparse, math, time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Scan(Node):
    def __init__(self, a):
        super().__init__("scan_drive")
        self.a = a
        self.cmd = self.create_publisher(Twist, a.cmd_vel_topic, 10)
        self.create_subscription(Odometry, a.odom_topic, self.on_odom, qos_profile_sensor_data)
        self.yaw, self.turned, self.t_odom = None, 0.0, 0.0
        if a.radius > 0:
            self.v, self.w = a.v, a.v / a.radius * (1 if a.direction == "left" else -1)
        else:
            self.v, self.w = 0.0, a.w * (1 if a.direction == "left" else -1)
        self.goal = 2 * math.pi * a.laps
        self.t0 = time.time()
        self.done = False
        self.create_timer(0.05, self.tick)
        kind = f"circle r={a.radius} m at {self.v:.2f} m/s" if a.radius > 0 else f"spin in place at {abs(self.w):.2f} rad/s"
        self.get_logger().info(f"{kind}, {a.laps} lap(s); heading from {a.odom_topic}")

    def on_odom(self, m):
        y = yaw_of(m.pose.pose.orientation)
        if self.yaw is not None:
            d = (y - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.turned += abs(d)
        self.yaw, self.t_odom = y, time.time()

    def tick(self):
        if self.done:
            return
        if self.yaw is None:
            if time.time() - self.t0 > 3:
                self.get_logger().warn(f"no {self.a.odom_topic} yet: is the robot/odometry running?", throttle_duration_sec=5)
            return
        stale = time.time() - self.t_odom > 1.0
        if self.turned >= self.goal or time.time() - self.t0 > self.a.timeout:
            self.stop(f"done: turned {math.degrees(self.turned):.0f} deg in {time.time() - self.t0:.0f} s")
            return
        tw = Twist()
        if not stale:
            tw.linear.x, tw.angular.z = float(self.v), float(self.w)
        self.cmd.publish(tw)

    def stop(self, why):
        self.cmd.publish(Twist())
        self.done = True
        self.get_logger().info(why + ". Now: ros2 service call /semantic_map/save std_srvs/srv/Trigger")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=1.5, help="m; 0 = spin in place")
    ap.add_argument("--v", type=float, default=0.3, help="m/s along the circle")
    ap.add_argument("--w", type=float, default=0.4, help="rad/s when spinning in place")
    ap.add_argument("--direction", choices=["left", "right"], default="left")
    ap.add_argument("--odom_topic", default="/odom_unity")
    ap.add_argument("--cmd_vel_topic", default="/cmd_vel")
    ap.add_argument("--timeout", type=float, default=180.0, help="s, safety stop")
    a = ap.parse_args()
    rclpy.init()
    n = Scan(a)
    try:
        while rclpy.ok() and not n.done:
            rclpy.spin_once(n, timeout_sec=0.05)
    except KeyboardInterrupt:
        n.stop("stopped (Ctrl-C)")
    finally:
        n.cmd.publish(Twist()); n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
