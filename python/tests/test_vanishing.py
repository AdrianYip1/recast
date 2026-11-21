import numpy as np

from recast.vanishing import (angular_error, assign_axes, find_vanishing_points, segment_lines, to_pixels,
                              vp_from_segments)
from synthetic import look_at_camera, room_segments, true_vp


def _pixel_dist(a, b):
    pa, pb = to_pixels(a), to_pixels(b)
    return np.linalg.norm(pa - pb) / np.linalg.norm(pb)


def test_lines_through_segment_endpoints():
    segs = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 5.0, 8.0]])
    lines = segment_lines(segs)
    for s, l in zip(segs, lines):
        assert abs(l @ [s[0], s[1], 1]) < 1e-9
        assert abs(l @ [s[2], s[3], 1]) < 1e-9


def test_vp_from_exact_segments():
    vp = np.array([400.0, 300.0])
    rng = np.random.default_rng(3)
    segs = []
    for _ in range(10):
        p = rng.uniform(0, 1000, 2)
        q = p + (vp - p) * rng.uniform(0.2, 0.6)
        segs.append([*p, *q])
    est = to_pixels(vp_from_segments(np.array(segs)))
    assert np.allclose(est, vp, atol=1e-6)


def test_find_three_vanishing_points_with_outliers():
    cam = look_at_camera()
    rng = np.random.default_rng(7)
    segs, labels = room_segments(cam, rng)
    clutter = rng.uniform(0, [cam.width, cam.height, cam.width, cam.height], (15, 4))
    all_segs = np.vstack([segs, clutter])

    found = find_vanishing_points(all_segs, count=3, rng=rng)
    assert len(found) == 3

    truth = [true_vp(cam, a) for a in range(3)]
    for vp, idx in found:
        # each found point should match one true axis and mostly hold that axis' segments
        errs = [np.max(angular_error(segs[labels == a], vp)) for a in range(3)]
        axis = int(np.argmin(errs))
        assert np.degrees(errs[axis]) < 3.0
        t = truth[axis]
        if abs(t[2]) > 1e-6 and abs(vp[2]) > 1e-6:
            assert _pixel_dist(vp, t) < 0.05
        inlier_labels = labels[idx[idx < len(segs)]]
        assert np.mean(inlier_labels == axis) > 0.9


def test_assign_axes_finds_vertical():
    cam = look_at_camera(pitch_deg=-8)
    rng = np.random.default_rng(9)
    segs, labels = room_segments(cam, rng)
    found = find_vanishing_points(segs, count=3, rng=rng)
    vx, vy, vz = assign_axes(segs, found)
    assert vy is not None
    # the vertical one is world Y's vanishing point
    assert np.degrees(np.max(angular_error(segs[labels == 1], vy))) < 3.0
    for v in (vx, vz):
        assert np.degrees(np.max(angular_error(segs[labels == 1], v))) > 10.0
