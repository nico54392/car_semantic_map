#!/usr/bin/env python3
"""
Word -> goal -> Nav2 drives there, in one command. Reads the robot's pose, finds the goal for the word on the
map, sends it to Nav2 and logs the run (same files as nav2_goal.py, so plot_run.py works).

python3 tools/nav2_go.py ~/nico_ws/scenes/flat_mix dirt                 # nearest reachable dirt
python3 tools/nav2_go.py ~/nico_ws/scenes/flat_mix gravel --pick largest
python3 tools/nav2_go.py ~/nico_ws/scenes/flat_mix bench --standoff 1.5  # obstacle class: stop 1.5 m in front

Goal source for now = the Unity labels (oracle_plan.py --to). Later: SAM3 / the memory idea, same command.
Needs: Unity in Play, ROS-TCP endpoint, unity_nav2_bridge.py and Nav2 running.
Output: <scene>/goals/oracle_<word>/ (oracle path.json, goal.png) and .../nav2/ (Nav2 plan + run_*.csv).
"""
import argparse, json, math, re, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nav2_goal import GoalRun          # noqa: E402


def current_pose(timeout=5.0):
    node = Node("nav2_go_pose")
    got = {}
    node.create_subscription(Odometry, "/odom_unity", lambda m: got.setdefault("p", m.pose.pose.position),
                             qos_profile_sensor_data)
    t0 = node.get_clock().now()
    while rclpy.ok() and "p" not in got and (node.get_clock().now() - t0).nanoseconds < timeout * 1e9:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    if "p" not in got:
        raise SystemExit("no /odom_unity: is Unity in Play and the ROS-TCP endpoint running?")
    return got["p"].x, got["p"].y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("word", help="a class in the scene: dirt, gravel, asphalt, bench, ...")
    ap.add_argument("--pick", choices=["nearest", "largest"], default="nearest")
    ap.add_argument("--standoff", type=float, default=1.0)
    ap.add_argument("--min_trav", type=float, default=0.3)
    ap.add_argument("--dry", action="store_true", help="find the goal, don't drive")
    a = ap.parse_args()

    rclpy.init()
    x, y = current_pose()
    print(f"robot at ({x:.2f}, {y:.2f})")
    cmd = [sys.executable, str(HERE / "oracle_plan.py"), a.scene, "--start", f"{x:.3f}", f"{y:.3f}",
           "--to", a.word, "--pick", a.pick, "--standoff", str(a.standoff), "--min_trav", str(a.min_trav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit("goal finding failed (see above)")
    m = re.search(r"saved (.+path\.json)", r.stdout)
    pj = Path(m.group(1)) if m else None
    if not pj or not pj.exists():
        print(r.stdout); raise SystemExit("couldn't find oracle_plan's path.json")
    d = json.load(open(pj))
    gx, gy = d["goal_ros"][:2]
    print(f"'{a.word}': goal ({gx:.2f}, {gy:.2f}), {math.dist((x, y), (gx, gy)):.1f} m away "
          f"(oracle path {d['path_length_m']:.1f} m, preview {pj.parent / 'goal.png'})")
    if a.dry:
        return

    n = GoalRun(SimpleNamespace(x=gx, y=gy, out=str(pj.parent / "nav2")))
    try:
        n.send()
    except KeyboardInterrupt:
        n.get_logger().info("cancelled (Ctrl-C): the logger stops, cancel the goal in Nav2 to stop the robot")
    finally:
        n.save()


if __name__ == "__main__":
    main()
