"""Box: from the visible floor corners and one top corner.

The world axes come from the vanishing points, so snapping the edges to the
VP directions means the box is axis-aligned in world space.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ...camera import Camera
from ..placement import base_point, height_from_click


def build_box(cam: Camera, floor_corners_uv, top_uv) -> trimesh.Trimesh:
    """floor_corners_uv: 2-4 clicked corners where the box meets the floor.
    top_uv: a clicked top corner (directly above one of the floor corners).
    """
    corners = np.array([base_point(cam, uv) for uv in np.asarray(floor_corners_uv, float).reshape(-1, 2)])
    if len(corners) < 2:
        raise ValueError("click at least 2 floor corners")

    # height: try the top click against each floor corner, keep the one it fits best
    best = None
    for c in corners:
        h = height_from_click(cam, c, top_uv)
        err = np.linalg.norm(cam.project((c + [0, h, 0])[None])[0] - np.asarray(top_uv, float))
        if h > 0 and (best is None or err < best[1]):
            best = (h, err)
    if best is None:
        raise ValueError("top click must be above a floor corner")
    height = best[0]

    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    ext = hi - lo
    # only one edge visible (2 corners): assume a square footprint, extending away from the camera
    for a, b in ((0, 2), (2, 0)):
        if ext[a] < 0.05 * ext[b]:
            centre = (lo + hi) / 2
            away = np.sign(centre[a] - cam.position[a]) or 1.0
            if away > 0:
                hi[a] = lo[a] + ext[b]
            else:
                lo[a] = hi[a] - ext[b]
    lo[1], hi[1] = 0.0, height

    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    # separate vertices per face so each face gets its own projected texture coords
    box.unmerge_vertices()
    return box
