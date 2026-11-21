"""Normalize (pivot at the base, +Y up, real-world scale) and write object.glb + object_meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .texture import Texture


def normalize(mesh: trimesh.Trimesh, base: np.ndarray, height: float | None = None,
              width: float | None = None) -> tuple[trimesh.Trimesh, float]:
    """Move the pivot to the base (min y = 0, base's x/z at the origin) and scale to metres.

    Give the real `height` (or `width`, measured across the photo) in metres.
    Returns (new mesh, scale factor applied). +Y is already up: the world frame
    comes from the vanishing points.
    """
    m = mesh.copy()
    v = np.asarray(m.vertices, dtype=np.float64)
    pivot = np.array([base[0], v[:, 1].min(), base[2]])
    v = v - pivot
    if height:
        scale = height / max(np.ptp(v[:, 1]), 1e-12)
    elif width:
        horizontal = np.hypot(v[:, 0], v[:, 2])
        scale = width / max(2 * horizontal.max(), 1e-12)
    else:
        raise ValueError("give the real height or width in metres")
    m.vertices = v * scale
    return m, float(scale)


def export_glb(mesh: trimesh.Trimesh, tex: Texture, path: str | Path, alpha_cutout: bool = False,
               double_sided: bool = False, alpha_cutoff: float = 0.5) -> None:
    """Textured glb. Cutout modes (card, extrude) use alphaMode MASK; cards are double-sided."""
    rgba = tex.rgba if alpha_cutout else tex.rgba[..., :3]
    image = Image.fromarray(np.ascontiguousarray(rgba))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0,
        doubleSided=double_sided,
        alphaMode="MASK" if alpha_cutout else "OPAQUE",
        alphaCutoff=alpha_cutoff if alpha_cutout else None)
    # trimesh stores OpenGL-style UVs (v up) and flips them when writing glTF
    uv_gl = np.column_stack([tex.uv[:, 0], 1.0 - tex.uv[:, 1]])
    out = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv_gl, material=material))
    out.export(str(path), file_type="glb")


def write_meta(path: str | Path, meta: dict) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(meta, indent=2, default=default))
