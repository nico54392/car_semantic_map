#!/usr/bin/env python3
"""
"go to the bench" -> 3D mask on the embedded map -> object -> goal next to it -> A* path -> waypoints.

python tools/goal_from_prompt.py <scene> "go to the bench" --start 5 -5
python tools/goal_from_prompt.py <scene> "picnic table" --start 5 -5        # not in the ontology: still works

How the mask is made: the prompt competes against every ontology class (softmax over
cosine similarities, like classification). A point belongs to the mask if the prompt wins
with probability > --thresh. Comparing against the other classes instead of thresholding raw
cosine makes it work for prompts that aren't in the ontology too.
If the prompt *is* an ontology class name, its own prompts are used and it's dropped from
the competitors.

Then: mask points -> xy grid -> connected blobs -> pick one (largest, or nearest to start) ->
goal = free cell on the ring around the blob that's closest to the start -> A* on the
traversability cost map (same as feature_map.py plan).

Outputs in <scene>/goals/<slug>/:
  path.json   waypoints in ROS (FLU) and Unity coords, goal, object centroid, clusters
  path.csv    x,y,z ROS waypoints
  mask.ply    the 3D mask (ascii, x y z score), loads in CloudCompare / MeshLab
  goal.png    top-down picture
Coordinates: map frame = ROS FLU from the Unity export. Unity (x,y,z) = (-y_ros, z_ros, x_ros).
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from feature_map import EmbeddedMap, grid, astar, ONTOLOGY   # noqa: E402

FILLER = r"^(please\s+)?((go|drive|move|navigate|head)\s+(to|towards|toward)\s+)?(the|a|an)?\s*"


def clean_prompt(text):
    return re.sub(FILLER, "", text.strip().lower()).strip(" .") or text


def target_probability(m, text, temperature=100.0, synonym=0.85):
    """P(prompt beats every ontology class) per point.
    Ontology classes whose text embedding is >= `synonym` cosine to the prompt are dropped from the
    competitors (e.g. "standing water" vs class "water"), otherwise a synonym steals the mask."""
    onto = m.onto
    if text in onto:
        tq = m.enc.query_features(onto[text]["prompts"])
        neg = [n for n in onto if n != text]
        dropped = []
    else:
        tq = m.enc.query_features([text])
        names = list(onto)
        Qc, own = m.enc.class_features([onto[n]["prompts"] for n in names])
        tsim = (Qc @ tq[0]).numpy()
        cs = np.array([tsim[own == k].max() for k in range(len(names))])
        dropped = [(names[k], round(float(cs[k]), 2)) for k in np.nonzero(cs >= synonym)[0]]
        neg = [n for k, n in enumerate(names) if cs[k] < synonym]
        if not neg:                     # everything looks like a synonym: keep all but the closest few to compete
            keep = np.argsort(cs)[: max(1, len(names) - 3)]
            neg = [names[k] for k in sorted(keep)]
            dropped = [d for d in dropped if d[0] not in neg]
    t_score = (m.feat @ tq.T).max(1).values                      # best of the target's prompts
    Q, owner = m.enc.class_features([onto[n]["prompts"] for n in neg])
    sims = m.feat @ Q.T
    n_score = torch.stack([sims[:, torch.from_numpy(np.nonzero(owner == k)[0])].max(1).values
                           for k in range(len(neg))], 1)
    logits = torch.cat([t_score[:, None], n_score], 1) * temperature
    p = logits.softmax(1)[:, 0].numpy()
    best_other = np.array(neg)[n_score.argmax(1).numpy()]
    p[~m.seen] = np.nan
    return p, best_other, dropped


def clusters(points, mask, cell, lo, shape, min_cells=2):
    occ = np.zeros(shape, bool)
    ij = np.floor((points[mask, :2] - lo) / cell).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    out = []
    pt_lab = np.zeros(len(points), int)
    pt_lab[mask] = lab[ij[:, 0], ij[:, 1]]
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        if len(cells) < min_cells:
            continue
        pts = points[pt_lab == k]
        out.append(dict(id=k, cells=cells, n_cells=len(cells), n_points=len(pts),
                        centroid=pts.mean(0), top_z=float(pts[:, 2].max())))
    return out, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene"); ap.add_argument("prompt")
    ap.add_argument("--start", type=float, nargs=2, required=True, help="robot start x y (map frame, ROS FLU)")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--thresh", type=float, default=0.5, help="P(prompt wins) for a point to be in the mask")
    ap.add_argument("--synonym", type=float, default=0.85,
                    help="ontology classes this similar (text cosine) to an open prompt don't compete with it. "
                         "Tuned on CLIP; check the printed values with PE")
    ap.add_argument("--pick", choices=["largest", "nearest"], default="largest")
    ap.add_argument("--standoff", type=float, default=1.0, help="stop this far (m) from the object's edge")
    ap.add_argument("--min_trav", type=float, default=0.3, help="goal cell must be at least this traversable")
    ap.add_argument("--classes", nargs="*", help="ontology classes for the cost map (default: all)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ontology", default=str(ONTOLOGY))
    a = ap.parse_args()

    m = EmbeddedMap(a.scene, a.weights, a.ontology)
    P = m.points
    text = clean_prompt(a.prompt)
    in_onto = text in m.onto
    print(f'prompt "{a.prompt}" -> "{text}" ({"ontology class" if in_onto else "NOT in ontology, open-vocabulary"})')

    # 1. 3D mask
    p, best_other, dropped = target_probability(m, text, synonym=a.synonym)
    if dropped:
        print("synonyms dropped from the competition:", dropped)
    mask = np.nan_to_num(p) > a.thresh
    print(f"mask: {mask.sum()} / {m.seen.sum()} seen points (P > {a.thresh}), max P {np.nanmax(p):.2f}")
    if not mask.any():
        losers = dict(zip(*np.unique(best_other[np.nan_to_num(p) > np.nanpercentile(p, 99)], return_counts=True)))
        print("nothing found. Top-1% points are mostly claimed by:", losers, "-> try --thresh lower or another wording")
        sys.exit(1)

    # 2. cost map (same rule as feature_map.py plan)
    names, probs = m.classify(a.classes)
    trav, _ = m.properties(names, probs, "traversability")
    if not in_onto:
        trav = np.where(mask, 0.0, trav)          # unknown object: treat the mask as an obstacle too
    tg, lo = grid(P, trav, a.cell)
    zg, _ = grid(P, P[:, 2].astype(float), a.cell, (lo, P[:, :2].max(0)))

    # 3. pick the object
    cl, lab = clusters(P, mask, a.cell, lo, tg.shape)
    if not cl:
        print("mask has no blob bigger than 1 cell"); sys.exit(1)
    s_xy = np.array(a.start)
    for c in cl:
        c["dist_start"] = float(np.linalg.norm(c["centroid"][:2] - s_xy))
    obj = max(cl, key=lambda c: c["n_points"]) if a.pick == "largest" else min(cl, key=lambda c: c["dist_start"])
    print(f"{len(cl)} blob(s); picked #{obj['id']} ({a.pick}): {obj['n_points']} pts, "
          f"centroid ({obj['centroid'][0]:.2f}, {obj['centroid'][1]:.2f}, {obj['centroid'][2]:.2f}), "
          f"{obj['dist_start']:.1f} m from start")

    # 4. goal = free cell on the ring around the blob, closest to the start (= approach side)
    blob = lab == obj["id"]
    dist = ndimage.distance_transform_edt(~blob) * a.cell
    ring = (dist > 0) & (dist <= a.standoff + a.cell / 2) & (np.nan_to_num(tg, nan=-1) >= a.min_trav)
    cand = np.argwhere(ring)
    if not len(cand):
        print("no free cell around the object; try a larger --standoff or lower --min_trav"); sys.exit(1)
    centers = lo + (cand + 0.5) * a.cell
    order = np.argsort(np.linalg.norm(centers - s_xy, axis=1))
    start = tuple(np.floor((s_xy - lo) / a.cell).astype(int))
    path = None
    for k in order[:20]:                                   # nearest few, first one A* can reach
        goal = tuple(cand[k])
        path, cost = astar(tg, start, goal)
        if path:
            break
    if not path:
        print("A* found no route to any goal cell"); sys.exit(1)

    # 5. export
    cells = np.array(path)
    xy = lo + (cells + 0.5) * a.cell
    z = np.array([zg[tuple(c)] for c in cells])
    z = np.where(np.isnan(z), np.nanmean(zg), z)
    ros = np.c_[xy, z]
    unity = np.c_[-ros[:, 1], ros[:, 2], ros[:, 0]]
    slug = re.sub(r"[^a-z0-9]+", "_", text)[:40]
    out = Path(a.scene) / "goals" / slug; out.mkdir(parents=True, exist_ok=True)
    res = dict(prompt=a.prompt, query=text, in_ontology=in_onto, thresh=a.thresh, cell=a.cell,
               start_ros=list(map(float, s_xy)), goal_ros=list(map(float, ros[-1])),
               goal_unity=list(map(float, unity[-1])),
               object_centroid_ros=list(map(float, obj["centroid"])),
               path_cost=float(cost), path_length_m=float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
               mean_trav_on_path=float(np.nanmean([tg[tuple(c)] for c in cells])),
               clusters=[dict(id=c["id"], n_points=c["n_points"], centroid_ros=list(map(float, c["centroid"])),
                              dist_from_start=c["dist_start"]) for c in cl],
               frames="ros = map frame (FLU, x fwd, y left, z up) from the Unity export; unity = (-y, z, x)",
               waypoints_ros=ros.round(3).tolist(), waypoints_unity=unity.round(3).tolist())
    json.dump(res, open(out / "path.json", "w"), indent=1)
    np.savetxt(out / "path.csv", ros, fmt="%.3f", delimiter=",", header="x,y,z (ROS map frame)", comments="")
    mp = P[mask]
    with open(out / "mask.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(mp)}\nproperty float x\nproperty float y\n"
                f"property float z\nproperty float score\nend_header\n")
        np.savetxt(f, np.c_[mp, p[mask]], fmt="%.3f")
    print(f"path: {len(path)} cells, {res['path_length_m']:.1f} m, mean trav {res['mean_trav_on_path']:.2f}")
    print(f"goal ROS ({ros[-1][0]:.2f}, {ros[-1][1]:.2f}) | Unity ({unity[-1][0]:.2f}, {unity[-1][1]:.2f}, {unity[-1][2]:.2f})")

    # 6. picture
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X, Y = -P[:, 1], P[:, 0]
    ext = [-(lo[1] + tg.shape[1] * a.cell), -lo[1], lo[0], lo[0] + tg.shape[0] * a.cell]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    sc = ax[0].scatter(X, Y, c=p, s=1, cmap="magma", vmin=0, vmax=1, marker="s")
    plt.colorbar(sc, ax=ax[0], fraction=0.046); ax[0].set_title(f'P("{text}") per point · mask = P > {a.thresh}')
    for c in cl:
        cx, cy = -c["centroid"][1], c["centroid"][0]
        ax[0].plot(cx, cy, "c+" if c is not obj else "g*", ms=14 if c is obj else 9, mew=2)
    im = ax[1].imshow(tg[:, ::-1], origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].contour(np.linspace(ext[0], ext[1], tg.shape[1]), np.linspace(ext[2], ext[3], tg.shape[0]),
                  blob[:, ::-1].astype(float), levels=[0.5], colors="b", linewidths=2)
    ax[1].plot(-xy[:, 1], xy[:, 0], "k-", lw=2.5)
    ax[1].plot(-s_xy[1], s_xy[0], "ko", ms=9); ax[1].plot(-xy[-1, 1], xy[-1, 0], "g*", ms=16)
    ax[1].set_title(f'"{a.prompt}": path {res["path_length_m"]:.1f} m (blue = object, ★ = goal)')
    for x_ in ax:
        x_.set_aspect("equal"); x_.set_xlabel("-y [m]"); x_.set_ylabel("x [m]")
    plt.tight_layout(); plt.savefig(out / "goal.png", dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
