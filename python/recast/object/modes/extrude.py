"""Extruded silhouette: the outline as a polygon, given thickness, standing upright.

For flat-ish things (signs, books seen face-on, cut-outs). Thickness defaults
to 10% of the width.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon

from ...camera import Camera
from ..placement import UP, facing_frame, on_facing_plane


def outline(mask: np.ndarray, simplify: float = 0.004) -> np.ndarray:
    """Largest outer contour, simplified (epsilon = simplify * perimeter). (N, 2) pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    c = max(contours, key=cv2.contourArea)
    eps = simplify * cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)


def build_extrude(mask: np.ndarray, cam: Camera, base: np.ndarray, thickness: float | None = None,
                  simplify: float = 0.004) -> trimesh.Trimesh:
    tangent, normal = facing_frame(cam, base)
    pts = on_facing_plane(cam, base, outline(mask, simplify))
    local = np.column_stack([(pts - base) @ tangent, pts[:, 1]])

    poly = Polygon(local).buffer(0)  # fixes any self-touching from the simplification
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    width = poly.bounds[2] - poly.bounds[0]
    t = thickness if thickness else 0.1 * width

    mesh = trimesh.creation.extrude_polygon(poly, height=t)  # x, y in plane, z in [0, t]
    v = mesh.vertices
    world = (base[None, :] * [1, 0, 1] + v[:, :1] * tangent + v[:, 1:2] * UP
             + (v[:, 2:3] - t / 2) * normal)
    # local (x, y, z) -> (tangent, up, normal) keeps handedness since tangent = up x normal
    return trimesh.Trimesh(world, mesh.faces, process=False)
