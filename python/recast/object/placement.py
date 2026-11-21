"""Turning clicks into 3D points on and above the floor.

The object photo's camera is solved like the scene's (vanishing points of the
floor/table the object stands on). The world scale there is arbitrary: the
geometry is built in those units and rescaled to the user's measurement at export.
"""

from __future__ import annotations

import numpy as np

from ..camera import Camera

UP = np.array([0.0, 1.0, 0.0])


def base_point(cam: Camera, uv) -> np.ndarray:
    """Where the object touches the floor: ray through the clicked pixel hits y = 0."""
    p = cam.pixel_on_floor(np.asarray(uv, dtype=np.float64).reshape(1, 2))[0]
    if np.isnan(p).any():
        raise ValueError("ground contact click is above the horizon")
    return p


def height_from_click(cam: Camera, base: np.ndarray, uv) -> float:
    """Height h such that base + h*up best matches the clicked pixel.

    Closest points between the vertical line through `base` and the click's ray.
    """
    d = cam.rays(np.asarray(uv, dtype=np.float64).reshape(1, 2))[0]
    w0 = base - cam.position
    b = UP @ d
    denom = 1.0 - b * b
    if denom < 1e-9:
        raise ValueError("click ray is parallel to the vertical axis")
    return float((b * (d @ w0) - (UP @ w0)) / denom)


def facing_frame(cam: Camera, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (tangent, normal) of the vertical plane through `base` facing the camera.

    normal points horizontally from the base towards the camera; tangent = up x normal.
    """
    to_cam = cam.position - base
    to_cam[1] = 0.0
    n = np.linalg.norm(to_cam)
    normal = to_cam / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    tangent = np.cross(UP, normal)
    return tangent, normal


def on_facing_plane(cam: Camera, base: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Intersect pixel rays (N, 2) with the vertical plane through `base` facing the camera."""
    _, normal = facing_frame(cam, base)
    d = cam.rays(np.atleast_2d(uv))
    denom = d @ normal
    t = ((base - cam.position) @ normal) / denom
    return cam.position + d * t[:, None]


def mask_bottom_center(mask: np.ndarray) -> np.ndarray:
    """Default ground contact: the middle of the lowest row of the mask."""
    ys, xs = np.nonzero(mask)
    y = ys.max()
    row = xs[ys == y]
    return np.array([(row.min() + row.max()) / 2.0, float(y)])


def mask_top_above(mask: np.ndarray, base_uv: np.ndarray) -> np.ndarray:
    """Default axis top: the highest mask pixel in the base's column (or the mask's top)."""
    ys, xs = np.nonzero(mask)
    col = int(round(base_uv[0]))
    in_col = ys[np.abs(xs - col) <= 1]
    top = in_col.min() if len(in_col) else ys.min()
    return np.array([base_uv[0], float(top)])
