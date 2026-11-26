#!/usr/bin/env python3
"""
Send Nav2 a goal and record the run in the same format as ros2_follow_path.py, so plot_run.py
compares Nav2's plan with what the robot actually drove.

python3 tools/nav2_goal.py 12 12                                     # -> ~/nico_ws/scenes/nav2_runs/<time>/
python3 tools/nav2_goal.py 12 12 --out ~/nico_ws/scenes/flat_mix/goals/nav2_12_12
python tools/plot_run.py <out>/run_....csv

Saves <out>/path.json (Nav2's FIRST plan, waypoints_ros) and <out>/run_<time>.csv (t,x,y,yaw,v_cmd,w_cmd).
Nav2 replans about once a second; the later plans go to plans.json, in case they drift.
"""
import argparse, json, math, time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from nav2_msgs.action import NavigateToPose


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GoalRun(Node):
    def __init__(self, a):
        super().__init__("nav2_goal")
        self.out = Path(a.out).expanduser() if a.out else \
            Path.home() / "nico_ws/scenes/nav2_runs" / time.strftime("%Y%m%d_%H%M%S")
        self.out.mkdir(parents=True, exist_ok=True)
        self.goal = (a.x, a.y)
        self.pose, self.cmd, self.log, self.t0 = None, (0.0, 0.0), [], None
        self.plans = []
        self.create_subscription(Odometry, "/odom_unity", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(PathMsg, "/plan", self.on_plan, 10)
        self.create_timer(0.05, self.tick)
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.done = False

    def on_odom(self, m):
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)

    def on_plan(self, m):
        pts = [[ps.pose.position.x, ps.pose.position.y, 0.0] for ps in m.poses]
        if pts:
            self.plans.append(pts)

    def tick(self):
        if self.pose is None or self.done:
            return
        if self.t0 is None:
            self.t0 = time.time()
        self.log.append((time.time() - self.t0, *self.pose, *self.cmd))

    def send(self):
        self.get_logger().info("waiting for Nav2 (navigate_to_pose) ...")
        self.ac.wait_for_server()
        g = NavigateToPose.Goal()
        g.pose = PoseStamped(); g.pose.header.frame_id = "map"
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = float(self.goal[0]), float(self.goal[1])
        g.pose.pose.orientation.w = 1.0
        self.get_logger().info(f"goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) sent")
        fut = self.ac.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            raise SystemExit("goal rejected")
        res = gh.get_result_async()
        while rclpy.ok() and not res.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        status = res.result().status
        self.get_logger().info("goal reached" if status == 4 else f"finished with status {status} (4 = succeeded)")

    def save(self):
        self.done = True
        if self.plans:
            P = self.plans[0]
            L = sum(math.dist(P[i][:2], P[i + 1][:2]) for i in range(len(P) - 1))
            json.dump(dict(prompt=f"nav2 goal {self.goal}", source="Nav2 SmacPlanner2D (first plan)",
                           path_length_m=L, goal_ros=P[-1], waypoints_ros=P),
                      open(self.out / "path.json", "w"), indent=1)
            json.dump(self.plans, open(self.out / "plans.json", "w"))
        if self.log:
            f = self.out / time.strftime("run_%Y%m%d_%H%M%S.csv")
            with open(f, "w") as fh:
                fh.write("t,x,y,yaw,v_cmd,w_cmd\n")
                for r in self.log:
                    fh.write(",".join(f"{v:.4f}" for v in r) + "\n")
            print(f"driven route saved: {f}\n  plot: python tools/plot_run.py {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=float); ap.add_argument("y", type=float)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rclpy.init()
    n = GoalRun(a)
    try:
        n.send()
    except KeyboardInterrupt:
        n.get_logger().info("cancelled (Ctrl-C): stopping")
    finally:
        n.save()


if __name__ == "__main__":
    main()
