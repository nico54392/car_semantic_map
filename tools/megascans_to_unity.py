#!/usr/bin/env python3
"""
Megascans surfaces (as downloaded from Fab) -> textures ready for Unity URP terrain layers.

    python tools/megascans_to_unity.py --out <UnityProject>/Assets/Course/Surfaces \
        grass=~/Downloads/Megascans/Wild_Grass_xyz  dirt=~/Downloads/Megascans/Forest_Path_abc \
        mud=...  gravel=...  puddle=...

The name before "=" is the CLASS (must match config/ontology.json: grass, dirt, mud, gravel, puddle, rock...).
That name becomes the terrain layer name, so SceneCapture labels and TerrainFriction's mu line up by themselves.

Per class it writes <out>/<class>/:
  <class>_albedo.png   base colour (sRGB)
  <class>_normal.png   normal map (OpenGL / +Y, what Unity expects; Megascans ships DirectX -> green flipped,
                       turn off with --no-flip-green if bumps look inverted in Unity)
  <class>_mask.png     URP TerrainLit mask map: R metallic, G ambient occlusion, B height, A smoothness (1 - roughness)
and <out>/surfaces.json with the class order and each surface's real-world tile size (from the Megascans
json if present, else --tile metres). Then in Unity: CAR > Course > Create Terrain Layers (unity/Editor/CourseTools.cs).
Needs: pillow, numpy.
"""
import argparse, json, re, shutil
from pathlib import Path

import numpy as np
from PIL import Image

SUFFIX = {   # texture role -> filename patterns (Megascans / Fab naming varies a bit between exports)
    "albedo": r"(basecolor|base_color|albedo|diffuse|_col(or)?)\b",
    "normal": r"(normal|_nrm|_norm)(?!.*(lod|bump))",
    "rough": r"(roughness|_rough)\b",
    "ao": r"(ambientocclusion|_ao|occlusion)\b",
    "height": r"(displacement|height|_disp)\b",
    "metal": r"(metalness|metallic|_metal)\b",
    "gloss": r"(gloss|smoothness)\b",
}
IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".tga")


def find(folder: Path):
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG]
    got = {}
    for role, pat in SUFFIX.items():
        cand = [p for p in files if re.search(pat, p.stem.lower().replace("-", "_"))]
        # prefer the largest resolution, skip previews/thumbs
        cand = [p for p in cand if not re.search(r"(preview|thumb|icon)", p.stem.lower())]
        if cand:
            got[role] = max(cand, key=lambda p: p.stat().st_size)
    return got


def load(path, size=None, gray=True):
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.array(im, np.float32) / (65535.0 if im.mode.startswith("I;16") else max(np.array(im).max(), 1))
        im = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
    im = im.convert("L" if gray else "RGB")
    if size and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.array(im, np.float32) / 255.0


def tile_size(folder: Path, default):
    for j in folder.rglob("*.json"):
        try:
            d = json.load(open(j))
        except Exception:
            continue
        txt = json.dumps(d).lower()
        m = re.search(r'"(physicalsize|physical_size|size)"\s*:\s*"?([\d.]+)\s*x\s*([\d.]+)', txt)
        if m:
            return float(m.group(2))
        m = re.search(r'"meta".*?"scansarea".*?"([\d.]+)x([\d.]+)', txt)
        if m:
            return float(m.group(1))
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="CLASS=FOLDER")
    ap.add_argument("--out", required=True, help="e.g. <UnityProject>/Assets/Course/Surfaces")
    ap.add_argument("--tile", type=float, default=2.0, help="tile size (m) when the download has no metadata")
    ap.add_argument("--max-size", type=int, default=2048, help="downscale bigger textures to this (px)")
    ap.add_argument("--no-flip-green", dest="flip", action="store_false")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    surfaces = []
    for pair in a.pairs:
        cls, folder = pair.split("=", 1)
        cls, folder = cls.strip().lower(), Path(folder).expanduser()
        tex = find(folder)
        if "albedo" not in tex:
            print(f"{cls}: no base colour texture in {folder}, skipped"); continue
        alb = Image.open(tex["albedo"]).convert("RGB")
        if max(alb.size) > a.max_size:
            s = a.max_size / max(alb.size)
            alb = alb.resize((int(alb.size[0] * s), int(alb.size[1] * s)), Image.LANCZOS)
        size = alb.size
        d = out / cls; d.mkdir(exist_ok=True)
        alb.save(d / f"{cls}_albedo.png")
        if "normal" in tex:
            n = load(tex["normal"], size, gray=False)
            if a.flip:
                n[..., 1] = 1.0 - n[..., 1]
            Image.fromarray((n * 255).astype(np.uint8)).save(d / f"{cls}_normal.png")
        H, W = size[1], size[0]
        metal = load(tex["metal"], size) if "metal" in tex else np.zeros((H, W), np.float32)
        ao = load(tex["ao"], size) if "ao" in tex else np.ones((H, W), np.float32)
        height = load(tex["height"], size) if "height" in tex else np.full((H, W), 0.5, np.float32)
        if "gloss" in tex:
            smooth = load(tex["gloss"], size)
        elif "rough" in tex:
            smooth = 1.0 - load(tex["rough"], size)
        else:
            smooth = np.full((H, W), 0.2, np.float32)
        mask = np.stack([metal, ao, height, smooth], -1)
        Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8), "RGBA").save(d / f"{cls}_mask.png")
        ts = tile_size(folder, a.tile)
        surfaces.append(dict(cls=cls, tile=ts, source=str(folder),
                             found={k: v.name for k, v in tex.items()}))
        missing = [k for k in ("normal", "rough", "ao", "height") if k not in tex]
        print(f"{cls:8s} {size[0]}x{size[1]}, tile {ts:g} m, from {folder.name}" +
              (f"  (missing: {', '.join(missing)} -> defaults)" if missing else ""))
    json.dump(dict(order=[s["cls"] for s in surfaces], surfaces=surfaces), open(out / "surfaces.json", "w"), indent=1)
    print(f"-> {out} ({len(surfaces)} surfaces). In Unity: CAR > Course > Create Terrain Layers")


if __name__ == "__main__":
    main()
