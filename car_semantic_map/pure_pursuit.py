"""Pure pursuit on a list of xy waypoints. No ROS in here so it can be tested offline;
tools/ros2_follow_path.py wraps it in a node.

Robot = unicycle: (x, y, yaw) in the same frame as the waypoints. step() returns (v, w, done).
"""
import math
import numpy as np


class PurePursuit:
    def __init__(self, waypoints, lookahead=0.8, v_max=0.5, w_max=1.0, goal_tol=0.3, turn_in_place_deg=60):
        self.wp = np.asarray(waypoints, float)[:, :2]
        self.L = lookahead
        self.v_max, self.w_max, self.tol = v_max, w_max, goal_tol
        self.turn_in_place = math.radians(turn_in_place_deg)
        self.i = 0                                    # progress index, only moves forward

    def _target(self, p):
        # advance progress to the closest waypoint ahead of us (search a window so we never go back)
        win = self.wp[self.i:self.i + 50]
        self.i += int(np.argmin(np.linalg.norm(win - p, axis=1)))
        # first waypoint at least L away, else the goal
        d = np.linalg.norm(self.wp[self.i:] - p, axis=1)
        far = np.nonzero(d >= self.L)[0]
        return self.wp[self.i + far[0]] if len(far) else self.wp[-1]

    def step(self, x, y, yaw):
        p = np.array([x, y])
        dist_goal = np.linalg.norm(self.wp[-1] - p)
        if dist_goal < self.tol:
            return 0.0, 0.0, True
        tx, ty = self._target(p) - p
        # target in robot frame
        lx = math.cos(yaw) * tx + math.sin(yaw) * ty
        ly = -math.sin(yaw) * tx + math.cos(yaw) * ty
        alpha = math.atan2(ly, lx)
        if abs(alpha) > self.turn_in_place:          # target well behind/beside us: rotate first
            return 0.0, math.copysign(self.w_max, alpha), False
        Ld2 = lx * lx + ly * ly
        k = 2.0 * ly / max(Ld2, 1e-6)                 # curvature to hit the lookahead point
        v = self.v_max / (1.0 + 1.5 * abs(k))         # slow down in tight turns
        v = min(v, self.v_max * max(dist_goal / (2 * self.L), 0.3))   # and near the goal
        w = max(-self.w_max, min(self.w_max, v * k))
        return v, w, False


def simulate(waypoints, start_pose, dt=0.05, max_t=300, **kw):
    """Integrate a unicycle following the path. Returns trajectory (T,3) and whether it arrived."""
    pp = PurePursuit(waypoints, **kw)
    x, y, yaw = start_pose
    traj = [(x, y, yaw)]
    for _ in range(int(max_t / dt)):
        v, w, done = pp.step(x, y, yaw)
        if done:
            return np.array(traj), True
        x += v * math.cos(yaw) * dt; y += v * math.sin(yaw) * dt; yaw += w * dt
        traj.append((x, y, yaw))
    return np.array(traj), False
