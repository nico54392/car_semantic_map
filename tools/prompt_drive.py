#!/usr/bin/env python3
"""
Type a prompt, the robot drives there, type the next one.  "go to the puddle" -> ... -> "now the bench"

    python tools/prompt_drive.py <scene> --map gt          # Unity labels (no AI): tests planning + driving
    python tools/prompt_drive.py <scene> --map ai          # embedded map (feature_map.py build): open vocabulary
    python tools/prompt_drive.py <scene> --map gt --sim --start 0 0   # no ROS: simulated unicycle, for testing

Loop: pose from /odom_unity -> prompt -> 3D mask -> pick the blob (nearest by default) -> goal
(drivable class: on the blob; obstacle/object: on a ring --standoff m around it) -> A* on the
traversability grid from the robot's CURRENT position -> pure pursuit -> /cmd_vel -> next prompt.

At the prompt:  any text = go there  |  "x y" = go to map coordinates  |  stop  |  list  |  where  |  quit
While driving: type "stop" (or a new prompt, which replaces the current goal).

Same frames as goal_from_prompt.py / ros2_follow_path.py: map = ROS FLU from the Unity export, and
/odom_unity is published in that frame (Nav2Publisher.cs), so no offset by default.
Each goal writes <scene>/goals/drive_<prompt>/path.json (same format as goal_from_prompt.py) + plan.png.
"""
import argparse, json, math, queue, re, sys, threading, time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from car_semantic_map.pure_pursuit import PurePursuit          # noqa: E402
from car_semantic_map.evaluate import load_ontology            # noqa: E402
from feature_map import grid, astar, ONTOLOGY                  # noqa: E402
from goal_from_prompt import clean_prompt, clusters           # noqa: E402


# ---------------------------------------------------------------- planner (no ROS)

class PromptPlanner:
    def __init__(self, scene, mode="gt", cell=0.25, ontology=ONTOLOGY, unknown=0.5, thresh=0.5,
                 synonym=0.85, standoff=1.0, min_trav=0.3, pick="nearest", weights=None):
        self.scene, self.mode, self.cell = Path(scene), mode, cell
        self.unknown, self.thresh, self.synonym = unknown, thresh, synonym
        self.standoff, self.min_trav, self.pick = standoff, min_trav, pick
        self.onto = load_ontology(Path(ontology))
        if mode == "gt":
            from car_semantic_map.scene_io import load_scene
            sc = load_scene(scene, max_views=1)
            self.P, self.L, self.classes = sc.points, sc.point_labels, sc.classes
            tv = {cid: (unknown if cid == 0 or self.onto.get(n, {}).get("traversability") is None
                        else float(self.onto[n]["traversability"])) for cid, n in self.classes.items()}
            trav = np.array([tv.get(int(c), unknown) for c in self.L], float)
            present = sorted({self.classes.get(int(c), "?") for c in np.unique(self.L)} - {"void", "?"})
            self.names = [n for n in present]
        else:
            from feature_map import EmbeddedMap
            self.m = EmbeddedMap(scene, weights, ontology)
            self.P = self.m.points
            names, probs = self.m.classify()
            trav, _ = self.m.properties(names, probs, "traversability")
            self.names = names
        self.trav_pt = trav
        self.tg, self.lo = grid(self.P, trav, cell)
        self.zg, _ = grid(self.P, self.P[:, 2].astype(float), cell, (self.lo, self.P[:, :2].max(0)))
        self.shape = np.array(self.tg.shape)
        print(f"map: {len(self.P)} points, {self.tg.shape[0]}x{self.tg.shape[1]} cells of {cell} m, "
              f"x {self.lo[0]:.1f}..{self.lo[0] + self.shape[0] * cell:.1f}, y {self.lo[1]:.1f}..{self.lo[1] + self.shape[1] * cell:.1f}")

    def _reach(self, tg, start, cands, tries=25):
        for goal in sorted(cands, key=lambda c: np.hypot(c[0] - start[0], c[1] - start[1]))[:tries]:
            path, cost = astar(tg, start, goal, unknown_cost=self.unknown)
            if path:
                return path, cost
        return None, np.inf

    def cell_of(self, xy):
        return tuple(np.clip(np.floor((np.asarray(xy, float) - self.lo) / self.cell).astype(int), 0, self.shape - 1))

    def mask_for(self, text):
        """-> (point mask, is_drivable_target, message)"""
        if self.mode == "gt":
            ids = {n: c for c, n in self.classes.items()}
            name = text if text in ids else (text[:-1] if text.endswith("s") and text[:-1] in ids else None)
            if name is None:
                return None, False, f'"{text}" is not a class in this scene. Classes: {", ".join(self.names)}'
            mask = self.L == ids[name]
            t = self.onto.get(name, {}).get("traversability")
            return mask, (t is not None and t >= self.min_trav), f"{mask.sum()} points labelled {name}"
        from goal_from_prompt import target_probability
        p, best_other, dropped = target_probability(self.m, text, synonym=self.synonym)
        mask = np.nan_to_num(p) > self.thresh
        msg = f"{mask.sum()} points with P > {self.thresh} (max {np.nanmax(p):.2f})"
        if dropped:
            msg += f"; synonyms not competing: {dropped}"
        if not mask.any():
            top = best_other[np.nan_to_num(p) > np.nanpercentile(p, 99)]
            u, c = np.unique(top, return_counts=True)
            msg += f"; nothing found. Its best points belong to {dict(zip(u, c))}: try other words"
            return None, False, msg
        t = self.onto.get(text, {}).get("traversability")
        drivable = t is not None and t >= self.min_trav
        if t is None:                         # open vocabulary: drivable if what's under the mask is
            drivable = np.nanmean(self.trav_pt[mask]) >= self.min_trav
        return mask, drivable, msg

    def plan(self, text, start_xy):
        """-> dict(waypoints (N,3) ros, goal, ...) or (None, reason)"""
        start = self.cell_of(start_xy)
        tg = self.tg
        coords = re.fullmatch(r"\s*(-?\d+(\.\d+)?)[ ,]+(-?\d+(\.\d+)?)\s*", text)
        if coords:
            gxy = np.array([float(coords.group(1)), float(coords.group(3))])
            cands, label, drivable = [self.cell_of(gxy)], "xy", True
            hi = self.lo + self.shape * self.cell
            msg = "goal by coordinates" + ("" if np.all((gxy >= self.lo) & (gxy <= hi)) else
                  f" (OUTSIDE the map x {self.lo[0]:.1f}..{hi[0]:.1f}, y {self.lo[1]:.1f}..{hi[1]:.1f}: clamped to the edge)")
        else:
            text = clean_prompt(re.sub(r"^((now|then|next|and|ok|okay|alright)[,!]?\s+)+", "", text.strip().lower()))
            mask, drivable, msg = self.mask_for(text)
            if mask is None:
                return None, msg
            if not drivable:                  # object/obstacle: its own cells are walls
                tg = tg.copy()
                ij = np.floor((self.P[mask, :2] - self.lo) / self.cell).astype(int)
                tg[ij[:, 0], ij[:, 1]] = 0.0
            cl, lab = clusters(self.P, mask, self.cell, self.lo, self.tg.shape)
            if not cl:
                return None, msg + "; no blob bigger than one cell"
            for c in cl:
                c["d"] = float(np.linalg.norm(c["centroid"][:2] - np.asarray(start_xy)))
            order = sorted(cl, key=lambda c: c["d"]) if self.pick == "nearest" else \
                sorted(cl, key=lambda c: -c["n_points"])
            label = text
            path = None
            for k, obj in enumerate(order[:8]):           # nearest blob first; skip ones we can't reach
                blob = lab == obj["id"]
                if drivable:                              # drive onto it: its cells
                    cands = [tuple(c) for c in np.argwhere(blob)]
                else:                                     # stop in front of it
                    dist = ndimage.distance_transform_edt(~blob) * self.cell
                    ring = (dist > 0) & (dist <= self.standoff + self.cell / 2) & \
                           (np.nan_to_num(tg, nan=-1) >= self.min_trav)
                    cands = [tuple(c) for c in np.argwhere(ring)]
                path, cost = self._reach(tg, start, cands)
                if path:
                    msg += f"; {len(cl)} blob(s), going to " + \
                           (f"the {'nearest' if self.pick == 'nearest' else 'largest'}" if k == 0 else f"#{k + 1} (closer ones unreachable)") + \
                           f" ({obj['d']:.1f} m away, {'drive onto it' if drivable else f'stop {self.standoff} m in front'})"
                    break
            if not path:
                return None, msg + f"; none of the {min(len(order), 8)} nearest blobs is reachable (try --standoff / --min_trav)"
        if coords:
            path, cost = self._reach(tg, start, cands)
            if not path:
                return None, msg + "; A* found no route"
        cells = np.array(path)
        xy = self.lo + (cells + 0.5) * self.cell
        xy[0] = start_xy                      # start exactly where the robot is
        z = np.array([self.zg[tuple(c)] for c in cells]); z = np.where(np.isnan(z), np.nanmean(self.zg), z)
        ros = np.c_[xy, z]
        length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        res = dict(prompt=text, query=label, start_ros=list(map(float, start_xy)), goal_ros=list(map(float, ros[-1])),
                   goal_unity=[float(-ros[-1, 1]), float(ros[-1, 2]), float(ros[-1, 0])],
                   path_length_m=length, path_cost=float(cost),
                   mean_trav_on_path=float(np.nanmean([self.tg[tuple(c)] for c in cells])),
                   waypoints_ros=ros.tolist(), waypoints_unity=np.c_[-ros[:, 1], ros[:, 2], ros[:, 0]].tolist(),
                   map=self.mode, message=msg)
        out = self.scene / "goals" / ("drive_" + re.sub(r"[^a-z0-9]+", "_", label.lower())[:40])
        out.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out / "path.json", "w"), indent=1)
        self._png(out / "plan.png", xy, label)
        res["dir"] = str(out)
        return res, msg

    def _png(self, path, xy, label, traj=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ext = [self.lo[0], self.lo[0] + self.shape[0] * self.cell, self.lo[1], self.lo[1] + self.shape[1] * self.cell]
        im = ax.imshow(self.tg.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        ax.plot(xy[:, 0], xy[:, 1], "r-", lw=2, label="plan")
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], "w--", lw=1.5, label="driven")
        ax.plot(*xy[0], "wo"); ax.plot(*xy[-1], "r*", ms=14)
        ax.set_title(f'"{label}"  ({self.mode} map)'); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(loc="upper right"); plt.colorbar(im, ax=ax, fraction=0.04, label="traversability")
        plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


# ---------------------------------------------------------------- prompt reader

def reader(q):
    while True:
        try:
            q.put(input().strip())
        except EOFError:
            q.put("quit"); return


HELP = 'type a place ("go to the puddle"), "x y", stop, list, where, quit'


def handle_meta(cmd, planner, pose):
    if cmd in ("list", "help", "?"):
        print("classes:", ", ".join(planner.names), "\n" + HELP)
        return True
    if cmd == "where":
        print("robot at", "unknown" if pose is None else f"x={pose[0]:.2f} y={pose[1]:.2f} yaw={math.degrees(pose[2]):.0f} deg")
        return True
    return False


# ---------------------------------------------------------------- simulated robot (no ROS)

def run_sim(planner, a):
    pose = [a.start[0], a.start[1], math.radians(a.start_yaw)]
    q = queue.Queue(); threading.Thread(target=reader, args=(q,), daemon=True).start()
    print(f"SIM robot at ({pose[0]:.1f}, {pose[1]:.1f}). {HELP}")
    while True:
        print("> ", end="", flush=True)
        cmd = q.get()
        if cmd in ("quit", "exit", "q"):
            return
        if not cmd or cmd == "stop" or handle_meta(cmd, planner, pose):
            continue
        res, msg = planner.plan(cmd, pose[:2])
        print("  " + msg)
        if res is None:
            continue
        if res["path_length_m"] < 0.5:
            print("  already there"); continue
        print(f"  path {res['path_length_m']:.1f} m, mean traversability {res['mean_trav_on_path']:.2f} -> {res['dir']}")
        pp = PurePursuit([w[:2] for w in res["waypoints_ros"]], lookahead=a.lookahead, v_max=a.v_max,
                         w_max=a.w_max, goal_tol=a.goal_tol)
        traj, dt, t = [tuple(pose)], 0.05, 0.0
        while t < 600:
            v, w, done = pp.step(*pose)
            if done:
                break
            pose[0] += v * math.cos(pose[2]) * dt; pose[1] += v * math.sin(pose[2]) * dt; pose[2] += w * dt
            traj.append(tuple(pose)); t += dt
        traj = np.array(traj)
        print(f"  {'arrived' if done else 'did NOT arrive'} after {t:.0f} s (sim), now at ({pose[0]:.1f}, {pose[1]:.1f})")
        planner._png(Path(res["dir"]) / "plan.png", np.array(res["waypoints_ros"])[:, :2], res["query"], traj[:, :2])


# ---------------------------------------------------------------- ROS node

def run_ros(planner, a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry, Path as PathMsg

    class Driver(Node):
        def __init__(self):
            super().__init__("prompt_drive")
            self.create_subscription(Odometry, a.pose_topic, self.on_pose, qos_profile_sensor_data)
            self.cmd = self.create_publisher(Twist, a.cmd_vel_topic, 10)
            self.path_pub = self.create_publisher(PathMsg, "/prompt_drive/path", 1)
            self.pose, self.t_pose, self.pp, self.path_msg = None, 0.0, None, None
            self.create_timer(1.0 / a.rate, self.tick)
            self.create_timer(1.0, lambda: self.path_msg and self.path_pub.publish(self.path_msg))

        def on_pose(self, msg):
            p, o = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
            c, s = math.cos(a.offset[2]), math.sin(a.offset[2])
            self.pose = (c * p.x - s * p.y + a.offset[0], s * p.x + c * p.y + a.offset[1],
                         yaw + a.offset[2] + math.radians(a.heading_offset_deg))
            self.t_pose = time.time()

        def go(self, res):
            self.pp = PurePursuit([w[:2] for w in res["waypoints_ros"]], lookahead=a.lookahead, v_max=a.v_max,
                                  w_max=a.w_max, goal_tol=a.goal_tol)
            self.path_msg = PathMsg(); self.path_msg.header.frame_id = "map"
            for x, y, z in res["waypoints_ros"]:
                ps = PoseStamped(); ps.header.frame_id = "map"
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), float(z)
                self.path_msg.poses.append(ps)

        def halt(self):
            self.pp = None; self.cmd.publish(Twist())

        def tick(self):
            if self.pp is None:
                return
            tw = Twist()
            if self.pose is None or time.time() - self.t_pose > a.pose_timeout:
                self.cmd.publish(tw); return               # stale pose: stand still
            v, w, done = self.pp.step(*self.pose)
            tw.linear.x, tw.angular.z = float(v), float(w)
            self.cmd.publish(tw)
            if done:
                self.halt()
                print(f"\n  arrived at ({self.pose[0]:.1f}, {self.pose[1]:.1f}). next?\n> ", end="", flush=True)

    rclpy.init()
    node = Driver()
    q = queue.Queue(); threading.Thread(target=reader, args=(q,), daemon=True).start()
    print(f"waiting for {a.pose_topic} ...")
    try:
        while rclpy.ok() and node.pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"robot at ({node.pose[0]:.1f}, {node.pose[1]:.1f}). {HELP}\n> ", end="", flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                cmd = q.get_nowait()
            except queue.Empty:
                continue
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "stop":
                node.halt(); print("  stopped\n> ", end="", flush=True); continue
            if not cmd or handle_meta(cmd, planner, node.pose):
                print("> ", end="", flush=True); continue
            res, msg = planner.plan(cmd, node.pose[:2])
            print("  " + msg)
            if res is None or res["path_length_m"] < 0.5:
                print("  already there\n> " if res else "> ", end="", flush=True); continue
            print(f"  path {res['path_length_m']:.1f} m, mean traversability {res['mean_trav_on_path']:.2f}: driving "
                  f"(type stop or a new place)\n> ", end="", flush=True)
            node.go(res)
    except KeyboardInterrupt:
        pass
    finally:
        node.halt(); node.destroy_node(); rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--map", choices=["gt", "ai"], default="gt", help="gt = Unity labels, ai = embedded_map.npz")
    ap.add_argument("--sim", action="store_true", help="no ROS: simulated robot")
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0], help="--sim start x y (map frame)")
    ap.add_argument("--start_yaw", type=float, default=0.0, help="--sim start heading, deg")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--pick", choices=["nearest", "largest"], default="nearest")
    ap.add_argument("--standoff", type=float, default=1.0)
    ap.add_argument("--min_trav", type=float, default=0.3)
    ap.add_argument("--unknown", type=float, default=0.5)
    ap.add_argument("--thresh", type=float, default=0.5, help="ai map: P(prompt wins) for a point to count")
    ap.add_argument("--synonym", type=float, default=0.85)
    ap.add_argument("--ontology", default=str(ONTOLOGY))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pose_topic", default="/odom_unity")
    ap.add_argument("--cmd_vel_topic", default="/cmd_vel")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="pose frame -> map: x y yaw(rad)")
    ap.add_argument("--heading_offset_deg", type=float, default=0.0)
    ap.add_argument("--lookahead", type=float, default=0.8)
    ap.add_argument("--v_max", type=float, default=0.5)
    ap.add_argument("--w_max", type=float, default=1.0)
    ap.add_argument("--goal_tol", type=float, default=0.3)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--pose_timeout", type=float, default=1.0)
    a = ap.parse_args()
    planner = PromptPlanner(a.scene, a.map, a.cell, a.ontology, a.unknown, a.thresh, a.synonym,
                            a.standoff, a.min_trav, a.pick, a.weights)
    (run_sim if a.sim else run_ros)(planner, a)


if __name__ == "__main__":
    main()
