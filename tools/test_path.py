#!/usr/bin/env python3
"""
Hand-made test paths for the follower, no map needed. Run on the ROS machine (no conda):

  source /opt/ros/humble/setup.bash
  python3 tools/test_path.py heading                 # drives forward 3 s, checks the reported yaw
  python3 tools/test_path.py make --shape s          # path from the robot's current pose
  python3 tools/ros2_follow_path.py --ros-args -p path_file:=$HOME/nico_ws/scenes/test/s/path.json

heading: publishes /cmd_vel forward (0.3 m/s) for 3 s, compares the direction the robot actually
  moved (from /odom_unity) with the yaw it reports. Prints the heading_offset_deg to pass to the
  follower if they don't match (|offset| > ~10 deg), and whether forward/backward are flipped.
make: reads the current pose once and writes <out>/<shape>/path.json in the follower's format.
  Shapes, all starting at the robot and heading where it faces:
    line    straight, --size m
    turn    straight half, then a 90 deg left arc, then straight
    s       S-curve (one sine period) --size m long, --amp m wide
    square  --size m sides, left turns, back to the start
  --heading_offset (deg) if `heading` said so.
Pick a flat open spot: it tests steering, not climbing.
"""
import argparse, json, math, sys, time
from pathlib import Path


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ros_node(name):
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    return rclpy, Node(name)


def get_poses(rclpy, node, topic, seconds=0.0, n=1):
    """collect (t, x, y, z, yaw) from an Odometry topic"""
    from nav_msgs.msg import Odometry
    got = []
    def cb(m):
        p = m.pose.pose
        got.append((time.time(), p.position.x, p.position.y, p.position.z, yaw_of(p.orientation)))
    sub = node.create_subscription(Odometry, topic, cb, 10)
    t0 = time.time()
    while (len(got) < n or time.time() - t0 < seconds) and time.time() - t0 < seconds + 5:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not got:
        sys.exit(f"no messages on {topic} in 5 s: is Unity playing and the bridge running?")
    return got


def heading(a):
    rclpy, node = ros_node("test_path_heading")
    from geometry_msgs.msg import Twist
    pub = node.create_publisher(Twist, a.cmd_vel, 10)
    p0 = get_poses(rclpy, node, a.pose_topic)[-1]
    tw = Twist(); tw.linear.x = 0.3
    t0 = time.time()
    while time.time() - t0 < 3.0:
        pub.publish(tw); rclpy.spin_once(node, timeout_sec=0.1)
    pub.publish(Twist())
    time.sleep(0.6)
    p1 = get_poses(rclpy, node, a.pose_topic)[-1]
    dx, dy = p1[1] - p0[1], p1[2] - p0[2]; dist = math.hypot(dx, dy)
    print(f"start x={p0[1]:.2f} y={p0[2]:.2f} yaw={math.degrees(p0[4]):.0f} deg")
    print(f"end   x={p1[1]:.2f} y={p1[2]:.2f} yaw={math.degrees(p1[4]):.0f} deg  moved {dist:.2f} m in 3 s (asked 0.9 m)")
    if dist < 0.2:
        print("barely moved: robot stuck (slope?) or /cmd_vel not reaching Unity. Check `ros2 topic info /cmd_vel`")
        return
    moved = math.atan2(dy, dx)
    off = math.degrees(wrap(moved - p0[4]))
    print(f"moved toward {math.degrees(moved):.0f} deg, reported yaw {math.degrees(p0[4]):.0f} deg -> offset {off:.0f} deg")
    if abs(off) < 10:
        print("OK: heading matches, no offset needed")
    elif abs(abs(off) - 180) < 15:
        print("robot drove BACKWARD relative to its yaw: forward is flipped. Use -p heading_offset_deg:=180 (or fix the model)")
    else:
        print(f"use -p heading_offset_deg:={off:.0f} with the follower, and --heading_offset {off:.0f} with `make`")
    rclpy.shutdown()


def shape_points(kind, size, amp, step=0.25):
    pts = []
    if kind == "line":
        pts = [(s, 0.0) for s in frange(0, size, step)]
    elif kind == "turn":
        h, r = size / 2, max(size / 4, 1.0)
        pts = [(s, 0.0) for s in frange(0, h, step)]
        for k in range(1, 19):                                  # 90 deg left arc
            t = k * math.pi / 36
            pts.append((h + r * math.sin(t), r * (1 - math.cos(t))))
        pts += [(h + r, r + s) for s in frange(step, h, step)]
    elif kind == "s":
        pts = [(s, amp * math.sin(2 * math.pi * s / size)) for s in frange(0, size, step)]
    elif kind == "square":
        corners = [(0, 0), (size, 0), (size, size), (0, size), (0, 0)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            L = math.hypot(x1 - x0, y1 - y0); n = max(int(L / step), 1)
            pts += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
        pts.append((0, 0))
    return pts


def frange(a, b, s):
    n = int(round((b - a) / s))
    return [a + i * s for i in range(n + 1)]


def make(a):
    rclpy, node = ros_node("test_path_make")
    t, x, y, z, yaw = get_poses(rclpy, node, a.pose_topic)[-1]
    rclpy.shutdown()
    yaw += math.radians(a.heading_offset)
    local = shape_points(a.shape, a.size, a.amp)
    c, s = math.cos(yaw), math.sin(yaw)
    ros = [[round(x + c * px - s * py, 3), round(y + s * px + c * py, 3), round(z, 3)] for px, py in local]
    length = sum(math.dist(p[:2], q[:2]) for p, q in zip(ros, ros[1:]))
    unity = [[-p[1], p[2], p[0]] for p in ros]
    out = Path(a.out).expanduser() / a.shape; out.mkdir(parents=True, exist_ok=True)
    d = dict(prompt=f"test path: {a.shape}", source="test_path.py (no map)", start_ros=[x, y], start_yaw_deg=math.degrees(yaw),
             goal_ros=ros[-1], goal_unity=unity[-1], path_length_m=length, waypoints_ros=ros, waypoints_unity=unity,
             frames="ros = /odom_unity frame (Unity world converted with (z,-x,y))")
    json.dump(d, open(out / "path.json", "w"), indent=1)
    print(f"robot at x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f} deg -> '{a.shape}' path, {len(ros)} waypoints, {length:.1f} m")
    print(f"saved {out / 'path.json'}")
    print(f"next: python3 tools/ros2_follow_path.py --ros-args -p path_file:={out / 'path.json'}"
          + (f" -p heading_offset_deg:={a.heading_offset:g}" if a.heading_offset else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("heading", "make"):
        p = sub.add_parser(name)
        p.add_argument("--pose_topic", default="/odom_unity")
        p.add_argument("--cmd_vel", default="/cmd_vel")
        if name == "make":
            p.add_argument("--shape", choices=["line", "turn", "s", "square"], default="s")
            p.add_argument("--size", type=float, default=6.0)
            p.add_argument("--amp", type=float, default=1.0)
            p.add_argument("--heading_offset", type=float, default=0.0)
            p.add_argument("--out", default="~/nico_ws/scenes/test")
    a = ap.parse_args()
    heading(a) if a.cmd == "heading" else make(a)


if __name__ == "__main__":
    main()
