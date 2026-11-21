"""Undo radial lens distortion so straight lines in the room stay straight.

Vanishing points assume a pinhole camera; barrel distortion from wide phone
lenses bends long edges and throws them off. We use the Brown model with two
radial terms (k1, k2), the same one OpenCV calibrates.
"""

from __future__ import annotations

import cv2
import numpy as np


def intrinsics(width: int, height: int, focal_px: float) -> np.ndarray:
    """Pinhole K with the principal point at the image centre."""
    return np.array(
        [[focal_px, 0.0, (width - 1) / 2.0],
         [0.0, focal_px, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def undistort(img: np.ndarray, focal_px: float, k1: float, k2: float = 0.0) -> np.ndarray:
    """Remove radial distortion. Output has the same size and focal length."""
    if k1 == 0.0 and k2 == 0.0:
        return img
    h, w = img.shape[:2]
    K = intrinsics(w, h, focal_px)
    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(img, K, dist, None, K)


def undistort_lensfun(img: np.ndarray, make: str | None, model: str | None, lens_make: str | None,
                      lens_model: str | None, focal_mm: float | None, f_number: float | None = None,
                      distance_m: float = 3.0) -> np.ndarray | None:
    """Undistort using the lensfun database (camera + lens from EXIF).

    Returns None if lensfunpy isn't installed or the camera/lens isn't in the
    database, so the caller can fall back to `undistort` with known k1/k2.
    The scale is fixed at 1, so the focal length in pixels doesn't change.
    """
    if not (make and model and lens_model and focal_mm):
        return None
    try:
        import lensfunpy
    except ImportError:
        return None

    db = lensfunpy.Database()
    cams = db.find_cameras(make, model)
    if not cams:
        return None
    lenses = db.find_lenses(cams[0], lens_make, lens_model)
    if not lenses:
        return None

    h, w = img.shape[:2]
    mod = lensfunpy.Modifier(lenses[0], cams[0].crop_factor, w, h)
    mod.initialize(focal_mm, f_number or 8.0, distance_m, scale=1.0,
                   pixel_format=np.float32, flags=lensfunpy.ModifyFlags.DISTORTION)
    coords = mod.apply_geometry_distortion()
    if coords is None:
        return None
    return cv2.remap(img, coords.astype(np.float32), None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def undistort_points(pts: np.ndarray, width: int, height: int, focal_px: float,
                     k1: float, k2: float = 0.0) -> np.ndarray:
    """Undistort (N, 2) pixel coordinates, e.g. hand-clicked line endpoints."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    K = intrinsics(width, height, focal_px)
    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def estimate_k1(lines_px: list[np.ndarray], width: int, height: int, focal_px: float,
                search: tuple[float, float] = (-0.4, 0.4), steps: int = 81) -> float:
    """Pick k1 that makes the given polylines straightest.

    `lines_px` is a list of (N, 2) arrays, each a set of points the user traced
    along one edge that should be straight (door frame, wall corner, ...).
    Brute force over k1 is plenty fast for a handful of polylines.
    """
    def straightness_error(k1: float) -> float:
        err = 0.0
        for pts in lines_px:
            if len(pts) < 3:
                continue
            p = undistort_points(pts, width, height, focal_px, k1)
            p = p - p.mean(axis=0)
            # smallest singular value = RMS distance from best-fit line
            err += np.linalg.svd(p, compute_uv=False)[-1] ** 2 / len(p)
        return err

    candidates = np.linspace(search[0], search[1], steps)
    errors = [straightness_error(k) for k in candidates]
    return float(candidates[int(np.argmin(errors))])
