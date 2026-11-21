"""Surface of revolution: lathe the silhouette's profile around a vertical axis.

For bottles, vases, mugs, lamps: at each height on the axis, the silhouette's
half-width in pixels, times depth / focal length, is the radius in metres.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.signal import savgol_filter

from ...camera import Camera
from ..placement import UP


def silhouette_half_width(mask: np.ndarray, uv: np.ndarray, perp: np.ndarray,
                          max_px: float | None = None, step: float = 0.5) -> float:
    """Average distance from `uv` to the mask edge along +perp and -perp, in pixels."""
    h, w = mask.shape
    max_px = max_px or float(np.hypot(h, w))
    dists = []
    for sign in (1.0, -1.0):
        t = np.arange(0.0, max_px, step)
        pts = uv[None, :] + sign * t[:, None] * perp[None, :]
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inside[inside] = mask[yi[inside], xi[inside]]
        out = np.flatnonzero(~inside)
        dists.append(t[out[0]] if len(out) else max_px)
    return float(np.mean(dists))


def lathe_profile(mask: np.ndarray, cam: Camera, base: np.ndarray, top: np.ndarray,
                  n_heights: int = 64, smooth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Heights along the axis and the radius at each, in world units."""
    axis = top - base
    length = float(np.linalg.norm(axis))
    s = np.linspace(0.0, 1.0, n_heights)
    # sample slightly inside the ends: right at the base/top the silhouette is a single row
    s_sample = np.clip(s, 0.5 / n_heights, 1 - 0.5 / n_heights)
    radii = []
    for si in s_sample:
        p = base + si * axis
        uv = cam.project(p[None])[0]
        # projected axis direction at this point -> perpendicular in the image
        uv2 = cam.project((p + 0.01 * length * axis / length)[None])[0]
        a = uv2 - uv
        perp = np.array([-a[1], a[0]]) / max(np.linalg.norm(a), 1e-9)
        depth = float(((p - cam.position) @ cam.R.T)[2])
        radii.append(silhouette_half_width(mask, uv, perp) * depth / cam.fx)
    radii = np.asarray(radii)
    if smooth and n_heights >= 7:
        window = min(n_heights // 4 * 2 + 1, n_heights - (1 - n_heights % 2))
        radii = savgol_filter(radii, max(window, 5), 3)
    return s * length, np.clip(radii, 1e-4 * length, None)


def revolve(heights: np.ndarray, radii: np.ndarray, base: np.ndarray, n_segments: int = 48) -> trimesh.Trimesh:
    """Closed mesh: a ring of vertices per height, capped at both ends."""
    theta = np.linspace(0.0, 2 * np.pi, n_segments, endpoint=False)
    n_h = len(heights)
    ring = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1)  # (S, 3)
    verts = (base[None, None, :] + radii[:, None, None] * ring[None, :, :]
             + heights[:, None, None] * UP[None, None, :]).reshape(-1, 3)
    bottom_c = len(verts)
    top_c = bottom_c + 1
    verts = np.vstack([verts, base + heights[0] * UP, base + heights[-1] * UP])

    def idx(i, j):
        return i * n_segments + (j % n_segments)

    faces = []
    for i in range(n_h - 1):
        for j in range(n_segments):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            faces += [[a, c, b], [a, d, c]]  # outward normals
    for j in range(n_segments):
        faces.append([bottom_c, idx(0, j), idx(0, j + 1)])
        faces.append([top_c, idx(n_h - 1, j + 1), idx(n_h - 1, j)])
    return trimesh.Trimesh(verts, np.asarray(faces), process=False)


def build_revolution(mask: np.ndarray, cam: Camera, base: np.ndarray, top: np.ndarray,
                     n_heights: int = 64, n_segments: int = 48) -> trimesh.Trimesh:
    heights, radii = lathe_profile(mask, cam, base, top, n_heights)
    return revolve(heights, radii, base, n_segments)
