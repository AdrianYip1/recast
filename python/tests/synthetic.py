"""Helpers that build fake rooms with a known camera, for checking the solvers."""

import numpy as np

from recast.camera import Camera


def look_at_camera(width=1280, height=960, focal_px=1000.0, yaw_deg=35.0, pitch_deg=-12.0,
                   camera_height=1.5) -> Camera:
    """Camera at (0, h, 0), turned by yaw around +Y and pitched down."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    # camera-to-world in OpenCV axes: start looking down world +Z... built from glTF-style basis
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    # rows of R are the camera axes in world coords (x right, y down, z forward)
    R = np.vstack([right, down, fwd])
    if np.linalg.det(R) < 0:
        R[0] = -R[0]
    return Camera(width, height, focal_px, R, np.array([0.0, camera_height, 0.0]))


def room_segments(cam: Camera, rng=None, noise_px=0.3, n_per_axis=25):
    """Random 3D segments parallel to each world axis, projected into the image.

    Returns (segments (N, 4), axis label (N,)).
    """
    rng = rng or np.random.default_rng(0)
    segs, labels = [], []
    for axis in range(3):
        count = 0
        while count < n_per_axis:
            start = cam.position + cam.R[2] * rng.uniform(2, 6) + rng.uniform(-2, 2, 3)
            start[1] = rng.uniform(0.0, 2.5)
            end = start.copy()
            end[axis] += rng.uniform(0.5, 1.5)
            uv = cam.project(np.vstack([start, end]))
            cam_pts = (np.vstack([start, end]) - cam.position) @ cam.R.T
            if (cam_pts[:, 2] <= 0.1).any():
                continue
            if not ((uv >= 0).all() and (uv[:, 0] < cam.width).all() and (uv[:, 1] < cam.height).all()):
                continue
            if np.linalg.norm(uv[1] - uv[0]) < 30:
                continue
            uv = uv + rng.normal(0, noise_px, uv.shape)
            segs.append(uv.ravel())
            labels.append(axis)
            count += 1
    return np.array(segs), np.array(labels)


def true_vp(cam: Camera, axis: int) -> np.ndarray:
    """Homogeneous vanishing point of world axis `axis`."""
    v = cam.K @ cam.R[:, axis]
    return v / np.linalg.norm(v)
