"""Camera calibration from vanishing points (Hartley & Zisserman ch. 8, as in fSpy).

Two coordinate frames are used internally:

* OpenCV camera frame: x right, y down, z forward (into the scene).
* World frame: right-handed, +Y up, floor is y = 0, units in meters.

`R` maps world directions to camera directions (x_cam = R @ (X_world - C)).
For export we convert to the glTF camera convention (looks down -Z, +Y up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OpenCV camera axes -> glTF camera axes (flip y and z)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0])


@dataclass
class Camera:
    width: int
    height: int
    focal_px: float          # fx; fy defaults to the same (square pixels)
    R: np.ndarray            # (3, 3) world -> OpenCV camera
    position: np.ndarray     # (3,) camera centre in world
    principal: np.ndarray | None = None  # (2,) pixels (cx, cy), default image centre
    fy: float | None = None

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=np.float64)
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.principal is None:
            self.principal = np.array([(self.width - 1) / 2.0, (self.height - 1) / 2.0])
        self.principal = np.asarray(self.principal, dtype=np.float64)
        if self.fy is None:
            self.fy = self.focal_px

    @property
    def fx(self) -> float:
        return self.focal_px

    @property
    def K(self) -> np.ndarray:
        cx, cy = self.principal
        return np.array([[self.fx, 0, cx], [0, self.fy, cy], [0, 0, 1.0]])

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * np.arctan(self.height / (2.0 * self.fy))

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2)."""
        X = np.atleast_2d(X)
        cam = (X - self.position) @ self.R.T
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> unit ray directions in world space (N, 3)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        h = np.column_stack([uv, np.ones(len(uv))])
        d_cam = h @ np.linalg.inv(self.K).T
        d = d_cam @ self.R  # R^T applied to each row
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_on_floor(self, uv: np.ndarray, floor_y: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the floor plane y = floor_y. NaN if the ray misses."""
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (floor_y - self.position[1]) / d[:, 1]
        t = np.where(t > 0, t, np.nan)
        return self.position + d * t[:, None]

    def camera_to_world_gl(self) -> np.ndarray:
        """3x3 rotation, glTF camera convention (local -Z forward, +Y up) -> world."""
        return self.R.T @ _CV_TO_GL

    def to_dict(self) -> dict:
        """The scene.json camera block (image size lives at the top level)."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.principal[0]),
            "cy": float(self.principal[1]),
            "rotation": [[float(v) for v in row] for row in self.camera_to_world_gl()],
            "position": [float(v) for v in self.position],
        }

    @classmethod
    def from_dict(cls, d: dict, image_size) -> "Camera":
        R_gl = np.asarray(d["rotation"], dtype=np.float64)
        R = (R_gl @ _CV_TO_GL).T  # back to world -> OpenCV camera
        return cls(int(image_size[0]), int(image_size[1]), float(d["fx"]), R,
                   np.asarray(d["position"], dtype=np.float64),
                   np.array([d["cx"], d["cy"]], dtype=np.float64), float(d["fy"]))


def focal_from_vps(v1: np.ndarray, v2: np.ndarray, principal: np.ndarray) -> float:
    """Focal length from two vanishing points of orthogonal directions.

    For orthogonal directions d1, d2: (K^-1 v1) . (K^-1 v2) = 0, which with
    square pixels and zero skew reduces to f^2 = -(v1 - p) . (v2 - p).
    """
    f2 = -float(np.dot(np.asarray(v1) - principal, np.asarray(v2) - principal))
    if f2 <= 0:
        raise ValueError("vanishing points are not consistent with orthogonal directions")
    return float(np.sqrt(f2))


def principal_from_3_vps(v1, v2, v3) -> np.ndarray:
    """Principal point = orthocentre of the triangle formed by 3 finite vanishing points."""
    v1, v2, v3 = (np.asarray(v, dtype=np.float64) for v in (v1, v2, v3))
    # altitude from v1 is perpendicular to (v3 - v2): (p - v1) . (v3 - v2) = 0
    A = np.array([v3 - v2, v3 - v1])
    b = np.array([np.dot(v1, v3 - v2), np.dot(v2, v3 - v1)])
    return np.linalg.solve(A, b)


def _orthonormalize(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _direction(vp_h: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    d = K_inv @ np.asarray(vp_h, dtype=np.float64)
    return d / np.linalg.norm(d)


def solve_camera(width: int, height: int, vp_x: np.ndarray, vp_z: np.ndarray,
                 vp_y: np.ndarray | None = None, focal_px: float | None = None,
                 camera_height: float = 1.5) -> Camera:
    """Solve focal length, rotation and position.

    vp_x, vp_z: homogeneous vanishing points of two horizontal, perpendicular
                directions (e.g. along the two walls). These become world X and Z.
    vp_y:       optional vertical vanishing point. If all three are finite, the
                principal point is solved too; otherwise it's the image centre.
                The focal length needs any two of the three to be finite.
    focal_px:   skip solving the focal length (e.g. it came from EXIF).
    camera_height: distance from the floor in meters. This sets the scale of the
                whole scene, since a single photo can't tell big rooms from small.
    """
    vp_x = np.asarray(vp_x, dtype=np.float64)
    vp_z = np.asarray(vp_z, dtype=np.float64)
    principal = np.array([(width - 1) / 2.0, (height - 1) / 2.0])

    finite = [abs(v[2]) > 1e-9 for v in (vp_x, vp_z)]
    if vp_y is not None:
        vp_y = np.asarray(vp_y, dtype=np.float64)
        if all(finite) and abs(vp_y[2]) > 1e-9 and focal_px is None:
            principal = principal_from_3_vps(vp_x[:2] / vp_x[2], vp_y[:2] / vp_y[2], vp_z[:2] / vp_z[2])

    if focal_px is None:
        # any two perpendicular directions with finite vanishing points will do; a photo
        # looking straight down a room has vp_x at infinity but finite vp_z and vp_y
        candidates = [vp_x, vp_z] + ([vp_y] if vp_y is not None else [])
        usable = [v[:2] / v[2] for v in candidates if abs(v[2]) > 1e-9]
        if len(usable) < 2:
            raise ValueError("need two finite vanishing points (or a known focal length)")
        focal_px = focal_from_vps(usable[0], usable[1], principal)

    K = np.array([[focal_px, 0, principal[0]], [0, focal_px, principal[1]], [0, 0, 1.0]])
    K_inv = np.linalg.inv(K)

    dx = _direction(vp_x, K_inv)
    dz = _direction(vp_z, K_inv)
    if vp_y is not None:
        dy = _direction(vp_y, K_inv)
    else:
        dy = np.cross(dz, dx)

    # world +Y is up, which is -y in the OpenCV camera frame for an upright photo
    if dy[1] > 0:
        dy = -dy
    # pick X/Z signs so X points right-ish and the frame is right-handed
    if dx[0] < 0:
        dx = -dx
    if np.dot(np.cross(dx, dy), dz) < 0:
        dz = -dz

    # columns are world axes expressed in camera coords
    R = _orthonormalize(np.column_stack([dx, dy, dz]))
    position = np.array([0.0, camera_height, 0.0])
    return Camera(width, height, float(focal_px), R, position, principal)
