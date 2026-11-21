import numpy as np

from recast.lens import estimate_k1, intrinsics, undistort_points


def _distort(pts, w, h, f, k1):
    """Apply radial distortion to ideal pixel coords (inverse of undistort)."""
    K = intrinsics(w, h, f)
    norm = (pts - K[:2, 2]) / f
    r2 = (norm ** 2).sum(axis=1, keepdims=True)
    return norm * (1 + k1 * r2) * f + K[:2, 2]


def test_undistort_points_inverts_distortion():
    w, h, f, k1 = 1200, 800, 900.0, -0.15
    ideal = np.array([[100.0, 100.0], [600, 400], [1100, 700], [50, 750]])
    distorted = _distort(ideal, w, h, f, k1)
    assert np.allclose(undistort_points(distorted, w, h, f, k1), ideal, atol=0.5)


def test_estimate_k1_recovers_barrel_distortion():
    w, h, f, k1 = 1200, 800, 900.0, -0.12
    # three straight lines near the edges, where distortion bends them most
    t = np.linspace(0, 1, 15)[:, None]
    lines = [
        np.hstack([60 + 0 * t, 50 + 700 * t]),
        np.hstack([50 + 1100 * t, 70 + 0 * t]),
        np.hstack([1140 + 0 * t, 50 + 700 * t]),
    ]
    bent = [_distort(l, w, h, f, k1) for l in lines]
    assert abs(estimate_k1(bent, w, h, f) - k1) < 0.02


def test_undistort_identity_when_no_distortion():
    from recast.lens import undistort
    img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    assert undistort(img, 25.0, 0.0) is img
