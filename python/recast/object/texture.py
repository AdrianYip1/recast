"""Texture from the photo, shared by every mode.

- Projective UVs: each vertex's UV is where it lands in the photo, so the photo
  maps straight back onto the geometry it came from.
- Back-side fill: vertices facing away from the camera are mirrored through the
  vertical plane that contains the symmetry axis and faces the camera, and the
  mirrored point is projected instead (the back shows a mirrored front).
- Crop: the texture is the mask's bounding box (plus padding), mask -> alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh

from ..camera import Camera
from ..io import linear_to_srgb, luminance
from .placement import facing_frame


@dataclass
class Texture:
    rgba: np.ndarray   # (h, w, 4) uint8, sRGB + alpha
    uv: np.ndarray     # (V, 2) glTF convention: origin top-left, v down


def projective_uv(cam: Camera, vertices: np.ndarray) -> np.ndarray:
    """Pixel coordinates (V, 2) of each vertex in the photo."""
    return cam.project(np.asarray(vertices, dtype=np.float64))


def back_side_fill(cam: Camera, mesh: trimesh.Trimesh, axis_point: np.ndarray) -> np.ndarray:
    """Pixel coords per vertex, mirroring back-facing vertices through the facing plane."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    to_cam = cam.position - v
    back = np.einsum("ij,ij->i", n, to_cam) < 0
    _, normal = facing_frame(cam, axis_point)
    dist = (v - axis_point) @ normal
    mirrored = v - 2 * dist[:, None] * normal
    return projective_uv(cam, np.where(back[:, None], mirrored, v))


def neutral_light(img_linear: np.ndarray, exclude: np.ndarray | None = None, limit: float = 1.33) -> np.ndarray:
    """Per-channel gains (3,) that take out the colour of the light the photo was taken in (warm bulbs, blue shade),
    so the object's texture is its own colour and the new room's light tints it instead.

    The light's colour is estimated from the rest of the photo with shades of grey (p = 6, Finlayson & Trezzi 2004),
    skipping the object, clipped and near-black pixels. Brightness is kept (the gains average to 1 in luminance) and
    each gain stays within 1/limit..limit, so a room that really is one colour isn't bleached grey."""
    px = img_linear.reshape(-1, 3)
    keep = (px.max(axis=1) < 0.98) & (px.max(axis=1) > 0.005)
    if exclude is not None:
        keep &= ~exclude.reshape(-1)
    if keep.sum() < 500:
        return np.ones(3)
    illuminant = np.mean(px[keep] ** 6, axis=0) ** (1.0 / 6.0)
    gains = float(luminance(illuminant)) / np.maximum(illuminant, 1e-6)
    gains = np.clip(gains, 1.0 / limit, limit)
    return gains / float(luminance(gains))


def delight(img: np.ndarray, mask: np.ndarray, sigma_frac: float = 0.08) -> np.ndarray:
    """Rough de-lighting: divide by heavily blurred luminance (inside the mask only)."""
    m = mask.astype(np.float32)
    sigma = max(2.0, sigma_frac * max(mask.shape))
    lum = luminance(img).astype(np.float32)
    blurred = cv2.GaussianBlur(lum * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-6)
    ref = np.median(lum[mask]) if mask.any() else 1.0
    shading = np.clip(blurred / max(ref, 1e-6), 0.2, 5.0)
    return np.where(mask[..., None], img / shading[..., None], img).astype(np.float32)


def crop_texture(img_linear: np.ndarray, alpha: np.ndarray, uv_px: np.ndarray, pad: int = 8,
                 bleed: bool = True) -> Texture:
    """Cut the photo to the mask's bounding box and map pixel UVs into it.

    bleed: fill outside the mask with nearby object colours, so opaque meshes
    whose edges graze the outline don't pick up the background.
    """
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.5)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    srgb8 = (np.clip(linear_to_srgb(img_linear[y0:y1, x0:x1]), 0, 1) * 255 + 0.5).astype(np.uint8)
    a = alpha[y0:y1, x0:x1]
    if bleed and (a < 0.5).any():
        srgb8 = cv2.inpaint(srgb8, (a < 0.5).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    rgba = np.dstack([srgb8, (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)])
    # pixel centres: pixel x covers [x, x+1), so uv = (x + 0.5 - x0) / width
    uv = np.column_stack([(uv_px[:, 0] + 0.5 - x0) / (x1 - x0), (uv_px[:, 1] + 0.5 - y0) / (y1 - y0)])
    return Texture(rgba, np.clip(uv, 0.0, 1.0))
