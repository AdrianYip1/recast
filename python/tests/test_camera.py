import numpy as np
import pytest

from recast.camera import focal_from_vps, principal_from_3_vps, solve_camera
from recast.vanishing import to_pixels, vp_from_segments
from synthetic import look_at_camera, room_segments, true_vp


def test_focal_from_exact_vps():
    cam = look_at_camera(focal_px=1100.0)
    f = focal_from_vps(to_pixels(true_vp(cam, 0)), to_pixels(true_vp(cam, 2)), cam.principal)
    assert np.isclose(f, 1100.0, rtol=1e-6)


def test_principal_point_is_orthocentre():
    cam = look_at_camera(yaw_deg=40, pitch_deg=-25)
    vps = [to_pixels(true_vp(cam, a)) for a in range(3)]
    assert np.allclose(principal_from_3_vps(*vps), cam.principal, atol=1e-4)


@pytest.mark.parametrize("yaw,pitch", [(35, -12), (-50, -20), (20, -30)])
def test_solve_camera_from_noisy_segments(yaw, pitch):
    cam = look_at_camera(yaw_deg=yaw, pitch_deg=pitch, focal_px=950.0)
    segs, labels = room_segments(cam, np.random.default_rng(11))
    vx, vy, vz = (vp_from_segments(segs[labels == a]) for a in range(3))

    est = solve_camera(cam.width, cam.height, vx, vz, vy, camera_height=1.5)

    assert abs(est.focal_px - 950.0) / 950.0 < 0.03
    # world up must come out exactly; horizontal axes may differ by a sign flip
    assert np.dot(est.R[:, 1], cam.R[:, 1]) > 0.999
    for a in (0, 2):
        assert abs(np.dot(est.R[:, a], cam.R[:, a])) > 0.999
    assert np.allclose(est.R @ est.R.T, np.eye(3), atol=1e-9)


def test_floor_points_roundtrip():
    cam = look_at_camera()
    floor = np.array([[0.5, 0.0, 3.0], [-1.0, 0.0, 4.0], [1.5, 0.0, 2.5]])
    floor = cam.position + (floor - [0, 1.5, 0]) @ np.eye(3)  # keep in front of the camera
    floor[:, 1] = 0.0
    floor = floor @ np.array([[np.cos(0.6), 0, np.sin(0.6)], [0, 1, 0], [-np.sin(0.6), 0, np.cos(0.6)]]).T
    uv = cam.project(floor)
    assert np.allclose(cam.pixel_on_floor(uv), floor, atol=1e-9)


def test_gl_rotation_looks_forward():
    cam = look_at_camera(yaw_deg=0, pitch_deg=0)
    R_gl = cam.camera_to_world_gl()
    forward_world = R_gl @ [0, 0, -1]
    assert np.allclose(forward_world, cam.R[2], atol=1e-12)
    assert np.allclose(R_gl @ [0, 1, 0], [0, 1, 0], atol=1e-12)


def test_one_point_perspective():
    # looking straight down the room: world X is parallel to the image, its vanishing point is at infinity
    cam = look_at_camera(yaw_deg=0, pitch_deg=-10, focal_px=415.7)
    vx, vy, vz = (true_vp(cam, a) for a in range(3))
    assert abs(vx[2]) < 1e-9
    est = solve_camera(cam.width, cam.height, vx, vz, vy)
    assert np.isclose(est.focal_px, 415.7, rtol=1e-6)
    assert np.allclose(np.abs(est.R), np.abs(cam.R), atol=1e-9)
