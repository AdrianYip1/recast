"""Objects the bank always has, made from code instead of a photo: a classic wine glass (clear glass, refracted
by the path tracer). ensure_examples() writes any that are missing; the worker calls it when it starts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .export import export_glass_glb

# (radius, height) in metres up the outside from the foot's centre to the rim, then back down the inside of the bowl.
# About a standard 350 ml wine glass: 21 cm tall, 7 cm foot, 8.6 cm bowl, 1.5 mm walls
WINE_GLASS = [(0.0, 0.0), (0.036, 0.0), (0.036, 0.003), (0.012, 0.006), (0.0045, 0.013), (0.0040, 0.085),
              (0.0085, 0.095), (0.025, 0.106), (0.038, 0.125), (0.043, 0.150), (0.0415, 0.180), (0.037, 0.205),
              (0.0355, 0.205), (0.040, 0.180), (0.0415, 0.150), (0.0365, 0.126), (0.0235, 0.108), (0.007, 0.0995),
              (0.0, 0.098)]


def wine_glass(sections: int = 72) -> trimesh.Trimesh:
    """The profile turned around the vertical axis: a closed, watertight glass, +Y up, its foot at y = 0."""
    profile = np.array(WINE_GLASS + [WINE_GLASS[0]])
    mesh = trimesh.creation.revolve(profile, sections=sections)          # around +Z
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # +Z up -> +Y up
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    return mesh


def wine_glass_thumbnail(path: Path, size: int = 128) -> None:
    """A clear-glass icon of the same profile, on transparent, for the bank's tile."""
    s = 4                                                     # drawn large, then scaled down (smooth edges)
    img = Image.new("RGBA", (size * s, size * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    height = max(y for _, y in WINE_GLASS)
    k = 0.86 * size * s / height
    cx, base = size * s / 2, size * s * 0.93
    outer = WINE_GLASS[:12]
    inner = WINE_GLASS[12:]
    to_px = lambda r, y, side: (cx + side * r * k, base - y * k)
    silhouette = [to_px(r, y, 1) for r, y in outer] + [to_px(r, y, -1) for r, y in reversed(outer)]
    bowl = [to_px(r, y, 1) for r, y in inner] + [to_px(r, y, -1) for r, y in reversed(inner)]
    draw.polygon(silhouette, fill=(200, 225, 240, 120), outline=(235, 245, 255, 255), width=3 * s // 2)
    draw.polygon(bowl, fill=(170, 200, 225, 70), outline=(210, 230, 245, 200), width=s)
    top = base - height * k
    draw.line([(cx - 0.020 * k, top + 0.035 * k), (cx - 0.032 * k, top + 0.075 * k)], fill=(255, 255, 255, 230), width=2 * s)
    img.resize((size, size), Image.LANCZOS).save(path)


def ensure_examples(bank: Path) -> list[Path]:
    """Writes the example objects missing from the bank; returns the folders it wrote."""
    written = []
    folder = Path(bank) / "example_wine_glass"
    if not (folder / "object_meta.json").exists():
        folder.mkdir(parents=True, exist_ok=True)
        mesh = wine_glass()
        export_glass_glb(mesh, folder / "object.glb")
        wine_glass_thumbnail(folder / "thumbnail.png")
        meta = {"mode": "example", "glass": True, "photo": "wine glass",
                "measurement": {"height_m": float(np.ptp(mesh.vertices[:, 1]))},
                "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        (folder / "object_meta.json").write_text(json.dumps(meta, indent=2))   # last: marks the entry complete
        written.append(folder)
    return written
