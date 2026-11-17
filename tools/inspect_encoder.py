#!/usr/bin/env python3
"""
Look inside an image encoder: what input it expects, how it cuts the image into patches,
and the shape of the tensor at every layer.

python tools/inspect_encoder.py pe-b                 # PE-Core-B-16 (open_clip)
python tools/inspect_encoder.py pe-l --size 448      # try a size it wasn't trained at
python tools/inspect_encoder.py clip --weights /path/clip_b16.pt
python tools/inspect_encoder.py sam3                 # run in Jaime's sam3 env (read-only use)
python tools/inspect_encoder.py pe-b --depth 3       # show deeper modules

Prints: config (image size, patch size, width, output dim, position encoding),
then one line per module: name, type, input shape -> output shape.
"""
import argparse
import torch, torch.nn as nn

PE = {"pe-t": "PE-Core-T-16-384", "pe-s": "PE-Core-S-16-384", "pe-b": "PE-Core-B-16",
      "pe-l": "PE-Core-L-14-336", "pe-g": "PE-Core-bigG-14-448"}


def shp(x):
    if torch.is_tensor(x):
        return "x".join(map(str, x.shape))
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(shp(v) for v in x[:4]) + (", ..." if len(x) > 4 else "") + ")"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {shp(v)}" for k, v in list(x.items())[:4]) + "}"
    return type(x).__name__


def trace(model, x, depth):
    rows, hooks = [], []
    for name, m in model.named_modules():
        if name and name.count(".") < depth:
            hooks.append(m.register_forward_hook(
                lambda m, i, o, n=name: rows.append((n, type(m).__name__, shp(i), shp(o)))))
    with torch.no_grad():
        out = model(x)
    for h in hooks:
        h.remove()
    seen = set()
    for n, t, i, o in rows:
        if n in seen:          # blocks called once; skip repeats from shared modules
            continue
        seen.add(n)
        print(f"  {n:45s} {t:22s} {i:>22s} -> {o}")
    return out


def load(a):
    if a.model == "sam3":
        import sam3                                   # Jaime's package; we only read it
        from sam3 import model_builder as mb
        build = getattr(mb, "build_sam3_image_model", None)
        if build is None:
            raise SystemExit(f"no build_sam3_image_model in sam3.model_builder; has: {[n for n in dir(mb) if 'build' in n]}")
        m = build().eval()
        enc = m.backbone if hasattr(m, "backbone") else m
        print("SAM3 top-level children:", [n for n, _ in m.named_children()])
        return enc, a.size or 1008, None
    import open_clip
    if a.model == "clip":
        m, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained=a.weights or "laion400m_e32")
    else:
        m, _, pre = open_clip.create_model_and_transforms(PE[a.model], pretrained=a.weights or "meta")
    v = m.visual.eval()
    native = getattr(v, "image_size", None) or getattr(getattr(v, "trunk", None), "patch_embed", None)
    native = native if isinstance(native, (int, tuple)) else getattr(native, "img_size", None)
    native = native[0] if isinstance(native, tuple) else native
    return v, a.size or native or 224, m


def describe(enc, size, full):
    print(f"\n=== config ===")
    print(f"input tensor        : 1x3x{size}x{size}  (float, normalized RGB)")
    trunk = getattr(enc, "trunk", enc)
    pe = getattr(trunk, "patch_embed", None) or getattr(enc, "conv1", None)
    if pe is not None:
        ps = getattr(pe, "patch_size", None) or getattr(pe, "kernel_size", None)
        ps = ps[0] if isinstance(ps, tuple) else ps
        print(f"patch size          : {ps}  -> {size // ps} x {size // ps} = {(size // ps) ** 2} tokens")
        native = getattr(pe, "img_size", None) or getattr(enc, "image_size", None)
        print(f"trained image size  : {native}")
    width = getattr(trunk, "embed_dim", None) or getattr(enc, "width", None) or getattr(getattr(enc, "conv1", None), "out_channels", None)
    print(f"token width         : {width}")
    if getattr(trunk, "rope", None) is not None or any("rope" in n.lower() for n, _ in trunk.named_modules()):
        print("position encoding   : RoPE (relative, works at other sizes; quality drops away from the trained size)")
    elif hasattr(enc, "positional_embedding"):
        print(f"position encoding   : learned absolute {tuple(enc.positional_embedding.shape)} "
              f"(fixed grid; other sizes need interpolation)")
    elif getattr(trunk, "pos_embed", None) is not None:
        print(f"position encoding   : learned absolute {tuple(trunk.pos_embed.shape)}")
    if full is not None:
        out_dim = getattr(full, "text_projection", None)
        out_dim = out_dim.shape[-1] if torch.is_tensor(out_dim) else getattr(getattr(full, "text", None), "output_dim", None)
        print(f"shared image-text dim: {out_dim}   <- what a 3D point stores and what text is compared to")
    n = sum(p.numel() for p in enc.parameters())
    print(f"parameters          : {n / 1e6:.0f} M")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["clip", "sam3", *PE])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--size", type=int, default=None, help="input size to try (default: the trained size)")
    ap.add_argument("--depth", type=int, default=2, help="how deep into the module tree to print")
    a = ap.parse_args()
    enc, size, full = load(a)
    describe(enc, size, full)
    print(f"\n=== forward pass at {size}x{size} ===")
    try:
        out = trace(enc, torch.randn(1, 3, size, size), a.depth)
        print(f"\nfinal output: {shp(out)}")
    except Exception as e:
        print(f"\nfailed at {size}x{size}: {type(e).__name__}: {e}\n-> this size isn't accepted as-is")


if __name__ == "__main__":
    main()
