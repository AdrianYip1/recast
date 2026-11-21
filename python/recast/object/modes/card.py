"""Card: a textured upright quad, the quickest way to get any object into the scene.

It stands at the ground contact point, facing the camera, sized so it exactly
covers the mask's bounding box. The texture's alpha cuts out the object's shape
(engine: alpha cutout + double-sided).
"""

from __future__ import annotations

import numpy as np
import trimesh

from ...camera import Camera
from ..placement import UP, facing_frame, on_facing_plane


def build_card(mask: np.ndarray, cam: Camera, base: np.ndarray, pad_px: float = 2.0) -> trimesh.Trimesh:
    ys, xs = np.nonzero(mask)
    x0, x1 = xs.min() - pad_px, xs.max() + 1 + pad_px
    y0, y1 = ys.min() - pad_px, ys.max() + 1 + pad_px
    tangent, _ = facing_frame(cam, base)

    # The bbox's corners land on the card's plane as a convex quad (perspective);
    # the upright rectangle around that quad projects over the whole bbox.
    corners = on_facing_plane(cam, base, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
    u = (corners - base) @ tangent
    u_left, u_right = float(u.min()), float(u.max())
    height = max(float(corners[:, 1].max()), 1e-6)

    bottom = base.copy()
    bottom[1] = 0.0
    v = np.array([
        bottom + u_left * tangent,
        bottom + u_right * tangent,
        bottom + u_right * tangent + height * UP,
        bottom + u_left * tangent + height * UP,
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # counter-clockwise seen from the camera side
    return trimesh.Trimesh(v, faces, process=False)
