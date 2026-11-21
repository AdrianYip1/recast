"""Checks: reproject the mesh onto the photo, and compare against a known mesh.

- silhouette + IoU: the mesh drawn back through the solved camera should cover
  the mask. A low IoU means the camera solve or the clicks are off.
- overlay: photo with the mask outline (green) and mesh outline (red).
- compare_meshes: height error, Chamfer distance and silhouette IoU against a
  ground-truth mesh (e.g. a known GLB the engine took a screenshot of).
- render_mesh: a small rasteriser standing in for that engine screenshot.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..camera import Camera
from ..io import linear_to_srgb


def _projected_triangles(cam: Camera, mesh: trimesh.Trimesh):
    """(pixel coords per vertex, indices of faces fully in front of the camera, camera depth per vertex)."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    cam_z = ((v - cam.position) @ cam.R.T)[:, 2]
    in_front = np.flatnonzero((cam_z[np.asarray(mesh.faces)] > 1e-6).all(axis=1))
    return cam.project(v), in_front, cam_z


def silhouette(cam: Camera, mesh: trimesh.Trimesh) -> np.ndarray:
    uv, keep, _ = _projected_triangles(cam, mesh)
    faces = np.asarray(mesh.faces)[keep]
    out = np.zeros((cam.height, cam.width), np.uint8)
    # 4 bits of sub-pixel precision so thin triangles still land in the right place.
    # One call per triangle: fillPoly with many polygons uses even-odd filling, so
    # overlapping front/back triangles would cancel out and leave holes.
    tris = np.round(uv[faces] * 16).astype(np.int32)
    for t in tris:
        cv2.fillConvexPoly(out, t, 1, lineType=cv2.LINE_8, shift=4)
    return out.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def overlay(img_linear: np.ndarray, mask: np.ndarray, sil: np.ndarray) -> np.ndarray:
    """sRGB uint8 image: mask outline green, reprojected mesh outline red."""
    srgb = (np.clip(linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8).copy()
    for m, colour in ((mask, (0, 220, 0)), (sil, (255, 40, 40))):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(srgb, contours, -1, colour, max(1, round(max(m.shape) / 500)))
    return srgb


def render_mesh(cam: Camera, mesh: trimesh.Trimesh, colour=(0.6, 0.15, 0.08),
                background=(0.25, 0.25, 0.25), light_dir=(-0.4, 0.8, 0.5)) -> tuple[np.ndarray, np.ndarray]:
    """Flat-shaded render (painter's algorithm) -> (linear RGB image, mask)."""
    uv, keep, cam_z = _projected_triangles(cam, mesh)
    faces = np.asarray(mesh.faces)[keep]
    normals = mesh.face_normals[keep]
    L = np.asarray(light_dir, float) / np.linalg.norm(light_dir)
    shade = 0.25 + 0.75 * np.clip(normals @ L, 0, None)
    order = np.argsort(-cam_z[faces].mean(axis=1))  # far to near
    img = np.zeros((cam.height, cam.width, 3), np.float32)
    img[:] = background
    tris = np.round(uv[faces] * 16).astype(np.int32)
    for i in order:
        c = tuple(float(x) for x in np.asarray(colour) * shade[i])
        cv2.fillConvexPoly(img, tris[i], c, lineType=cv2.LINE_8, shift=4)
    return img, silhouette(cam, mesh)


def chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 20000, seed: int = 0) -> float:
    """Symmetric mean nearest-surface distance between two meshes (sampled)."""
    pa = trimesh.sample.sample_surface(a, n, seed=seed)[0]
    pb = trimesh.sample.sample_surface(b, n, seed=seed + 1)[0]
    da = cKDTree(pb).query(pa)[0]
    db = cKDTree(pa).query(pb)[0]
    return float((da.mean() + db.mean()) / 2)


def compare_meshes(pred: trimesh.Trimesh, truth: trimesh.Trimesh, cam: Camera | None = None) -> dict:
    """Both meshes in the same frame (e.g. both normalized: pivot at base, metres)."""
    h_pred = float(np.ptp(pred.vertices[:, 1]))
    h_true = float(np.ptp(truth.vertices[:, 1]))
    out = {
        "height_pred_m": h_pred,
        "height_true_m": h_true,
        "height_error_pct": 100.0 * abs(h_pred - h_true) / max(h_true, 1e-12),
        "chamfer_m": chamfer(pred, truth),
    }
    out["chamfer_pct_of_height"] = 100.0 * out["chamfer_m"] / max(h_true, 1e-12)
    if cam is not None:
        out["silhouette_iou"] = iou(silhouette(cam, pred), silhouette(cam, truth))
    return out
