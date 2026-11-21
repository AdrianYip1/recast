"""Line segments -> vanishing points.

Parallel 3D edges (floor tiles, wall corners, table sides) meet at one point in
the image. In a room there are usually three such directions, one per axis.

Lines and points are handled in homogeneous coordinates: the line through two
points is their cross product, and the intersection of two lines is theirs.
"""

from __future__ import annotations

import cv2
import numpy as np

# A segment is (x1, y1, x2, y2) in pixels.


def detect_segments(img_linear: np.ndarray, min_length: float = 30.0) -> np.ndarray:
    """Detect straight segments with OpenCV's LSD. Returns (N, 4)."""
    gray = cv2.cvtColor(np.clip(img_linear, 0, 1), cv2.COLOR_RGB2GRAY)
    # LSD works better on perceptual brightness than on linear values
    gray8 = (np.sqrt(gray) * 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(gray8)[0]
    if segs is None:
        return np.zeros((0, 4), dtype=np.float64)
    segs = segs.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length]


def segment_lines(segs: np.ndarray) -> np.ndarray:
    """Homogeneous line for each segment, normalised so (a, b) is a unit normal."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    ones = np.ones(len(segs))
    p1 = np.column_stack([segs[:, 0], segs[:, 1], ones])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], ones])
    lines = np.cross(p1, p2)
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def vp_from_segments(segs: np.ndarray) -> np.ndarray:
    """Least-squares vanishing point of a group of segments (homogeneous, unit norm).

    Finds v minimising sum (l_i . v)^2, i.e. the smallest singular vector of the
    stacked lines. Works for points at infinity too (v[2] ~ 0).
    """
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    if len(segs) < 2:
        raise ValueError("need at least 2 segments for a vanishing point")
    lines = segment_lines(segs)
    # weight by length so long, reliable edges count more
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    v = np.linalg.svd(lines * np.sqrt(lengths)[:, None])[2][-1]
    return v / np.linalg.norm(v)


def angular_error(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle (radians) between each segment and the line from its midpoint to vp."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    mid = (segs[:, :2] + segs[:, 2:]) / 2.0
    d_seg = segs[:, 2:] - segs[:, :2]
    if abs(vp[2]) < 1e-9:
        d_vp = np.broadcast_to(vp[:2], d_seg.shape)  # direction for a point at infinity
    else:
        d_vp = vp[:2] / vp[2] - mid
    cos = np.abs(np.sum(d_seg * d_vp, axis=1)) / (
        np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-12)
    return np.arccos(np.clip(cos, 0.0, 1.0))


def ransac_vp(segs: np.ndarray, threshold_deg: float = 2.0, iterations: int = 2000,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One vanishing point by RANSAC. Returns (vp, inlier_mask)."""
    rng = rng or np.random.default_rng(0)
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    n = len(segs)
    if n < 2:
        raise ValueError("not enough segments")
    lines = segment_lines(segs)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    thresh = np.radians(threshold_deg)

    best_score, best_mask = -1.0, None
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        v = np.cross(lines[i], lines[j])
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        mask = angular_error(segs, v / norm) < thresh
        score = lengths[mask].sum()
        if score > best_score:
            best_score, best_mask = score, mask

    if best_mask is None or best_mask.sum() < 2:
        raise RuntimeError("RANSAC found no vanishing point")
    vp = vp_from_segments(segs[best_mask])
    # refit inliers once with the refined point
    mask = angular_error(segs, vp) < thresh
    if mask.sum() >= 2:
        vp = vp_from_segments(segs[mask])
    return vp, mask


def find_vanishing_points(segs: np.ndarray, count: int = 3, threshold_deg: float = 2.0,
                          rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy: find the strongest vanishing point, remove its inliers, repeat.

    Returns a list of (vp, indices_into_segs). May return fewer than `count`
    if the image runs out of lines.
    """
    rng = rng or np.random.default_rng(0)
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    remaining = np.arange(len(segs))
    out = []
    for _ in range(count):
        if len(remaining) < 4:
            break
        try:
            vp, mask = ransac_vp(segs[remaining], threshold_deg, rng=rng)
        except RuntimeError:
            break
        out.append((vp, remaining[mask]))
        remaining = remaining[~mask]
    return out


def assign_axes(segs: np.ndarray, found: list[tuple[np.ndarray, np.ndarray]]):
    """Decide which found vanishing point is vertical (world Y) for an upright photo.

    The vertical group is the one whose segments are closest to straight up/down
    in the image. Of the rest, the two strongest (most total segment length) are
    the horizontal X and Z axes. Returns (vp_x, vp_y, vp_z); vp_y may be None.
    """
    if len(found) < 2:
        raise ValueError("need at least 2 vanishing points")
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    d = segs[:, 2:] - segs[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    verticalness = np.abs(d[:, 1]) / np.maximum(length, 1e-9)

    scores = [np.average(verticalness[idx], weights=length[idx]) for _, idx in found]
    strength = [length[idx].sum() for _, idx in found]
    order = list(range(len(found)))

    vp_y = None
    if len(found) >= 3 or max(scores) > 0.9:
        y = int(np.argmax(scores))
        if scores[y] > 0.8:
            vp_y = found[y][0]
            order.remove(y)
    horiz = sorted(order, key=lambda i: -strength[i])[:2]
    if len(horiz) < 2:
        raise ValueError("found only one horizontal direction")
    return found[horiz[0]][0], vp_y, found[horiz[1]][0]


def to_pixels(vp: np.ndarray) -> np.ndarray | None:
    """Homogeneous vp -> (x, y), or None if it's at infinity."""
    if abs(vp[2]) < 1e-9:
        return None
    return vp[:2] / vp[2]
