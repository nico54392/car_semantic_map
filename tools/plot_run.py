#!/usr/bin/env python3
"""
Planned path vs the route the robot actually drove (from ros2_follow_path.py's run_<time>.csv).

python tools/plot_run.py ~/nico_ws/scenes/test/s/run_20261126_150400.csv      # path.json next to it
python tools/plot_run.py <run.csv> --path <path.json> --out run.png

Prints and plots:
  cross-track error  distance from each driven pose to the planned path (mean / 95th pct / max)
  goal error         distance from the final pose to the path's end
  turn-in-place      share of time the follower had to stop and spin (v = 0, w != 0): the
                     "forward, stop, turn" pattern. High = the robot under-rotates while driving
  commanded vs actual turn rate  actual yaw rate / commanded w while moving: < 1 means it turns
                     less than asked -> raise CmdVelDriver's Turn Slip Factor by about 1/ratio
Needs numpy + matplotlib (e.g. conda activate nico_clip).
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--path", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run).expanduser()
    pj = Path(a.path).expanduser() if a.path else run.with_name("path.json")
    d = json.load(open(pj))
    P = np.array(d["waypoints_ros"])[:, :2]
    R = np.genfromtxt(run, delimiter=",", names=True)
    t, x, y, yaw, v, w = (R[k] for k in ("t", "x", "y", "yaw", "v_cmd", "w_cmd"))
    xy = np.c_[x, y]

    # cross-track: distance to the planned polyline
    def seg_dist(p):
        a_, b_ = P[:-1], P[1:]; ab = b_ - a_
        u = np.clip(((p - a_) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return np.min(np.linalg.norm(a_ + u[:, None] * ab - p, axis=1))
    cte = np.array([seg_dist(p) for p in xy])
    goal_err = float(np.linalg.norm(xy[-1] - P[-1]))
    spin = (np.abs(v) < 1e-6) & (np.abs(w) > 1e-6)
    dyaw = np.diff(np.unwrap(yaw)); dt = np.maximum(np.diff(t), 1e-3)
    moving = (np.abs(v[:-1]) > 0.05) & (np.abs(w[:-1]) > 0.1)
    ratio = float(np.median((dyaw / dt)[moving] / w[:-1][moving])) if moving.sum() > 5 else float("nan")
    dist = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    print(f"planned {d.get('path_length_m', 0):.1f} m | driven {dist:.1f} m in {t[-1]:.1f} s ({dist / max(t[-1], 1e-3):.2f} m/s avg)")
    print(f"cross-track error: mean {cte.mean():.2f} m, 95% {np.percentile(cte, 95):.2f} m, max {cte.max():.2f} m")
    print(f"goal error: {goal_err:.2f} m")
    print(f"turn-in-place: {spin.mean():.0%} of the time ({int(np.sum(np.diff(spin.astype(int)) == 1))} times)")
    if ratio == ratio:
        print(f"actual / commanded turn rate while driving: {ratio:.2f}"
              + (f"  -> under-rotates: try Turn Slip Factor x{1 / max(ratio, 0.1):.1f}" if ratio < 0.8 else "  (OK)"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(-P[:, 1], P[:, 0], "b--", lw=2, label="planned")
    sc = ax[0].scatter(-xy[:, 1], xy[:, 0], c=cte, s=8, cmap="plasma", label="driven (colour = error, m)")
    ax[0].plot(-xy[spin, 1], xy[spin, 0], "rx", ms=5, label="turn in place")
    ax[0].plot(-P[0, 1], P[0, 0], "go", ms=9); ax[0].plot(-P[-1, 1], P[-1, 0], "r*", ms=14)
    plt.colorbar(sc, ax=ax[0], fraction=0.046); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("-y (m)"); ax[0].set_ylabel("x (m)")
    ax[0].set_title(f"{d.get('prompt', '')}: mean error {cte.mean():.2f} m, goal {goal_err:.2f} m")
    ax[1].plot(t, v, label="v cmd (m/s)"); ax[1].plot(t, w, label="w cmd (rad/s)")
    ax[1].plot(t[1:], dyaw / dt, alpha=.6, label="actual yaw rate (rad/s)"); ax[1].plot(t, cte, "k:", label="cross-track (m)")
    ax[1].legend(fontsize=8); ax[1].set_xlabel("time (s)"); ax[1].set_title("commands vs response")
    fig.tight_layout()
    out = Path(a.out) if a.out else run.with_suffix(".png")
    fig.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main()
