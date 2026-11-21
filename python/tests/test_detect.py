"""Automatic walls/objects: geometry from (fake) model output, and the detect command.

The models are replaced by what they would ideally return for a synthetic room (an
outline per proxy and a relative disparity map with an unknown scale and offset), so
these tests check our geometry, not the networks.
"""

import json

import numpy as np
import pytest

from recast import detect
from recast.cli import main
from recast.io import save_jpg
from recast.scene import BoxProxy, Material, PlaneProxy, Scene, floor_proxy, render_gbuffer
from synthetic import look_at_camera

TABLE = ((-0.6, 0.0, 2.4), (0.6, 0.75, 3.2))
VASE = ((-0.1, 0.75, 2.7), (0.1, 1.0, 2.9))


def _room():
    cam = look_at_camera(width=320, height=240, focal_px=260.0, yaw_deg=-20, pitch_deg=-18, camera_height=1.3)
    truth = Scene(cam, proxies=[
        floor_proxy(0),
        PlaneProxy(1, (0.0, 0.0, 6.0), (0.0, 0.0, -1.0)),       # back wall
        PlaneProxy(2, (-3.0, 0.0, 0.0), (1.0, 0.0, 0.0)),       # left wall
        BoxProxy(3, *TABLE, Material((0.4, 0.3, 0.2))),
        BoxProxy(4, *VASE, Material((0.7, 0.7, 0.6))),
    ])
    return cam, truth


def _fake_models(cam, truth):
    """What ideal models would return: labelled outlines + disparity = a / z + b."""
    _, ids, t = render_gbuffer(truth)
    rays = cam.rays(np.column_stack([np.mgrid[0:cam.height, 0:cam.width][1].ravel(),
                                     np.mgrid[0:cam.height, 0:cam.width][0].ravel()]).astype(float))
    z = t.ravel() * (rays @ cam.R[2])
    disparity = np.where(np.isfinite(z), 2.0 / np.where(np.isfinite(z), z, 1.0) + 0.3, 0.3).reshape(ids.shape)
    names = {0: "floor", 1: "wall", 2: "wall", 3: "table", 4: "vase"}
    segments = [detect.Segment(names[i], 0.95, ids == i) for i in names if (ids == i).any()]
    return segments, disparity


def test_depth_is_put_in_metres_from_the_floor():
    cam, truth = _room()
    segments, disparity = _fake_models(cam, truth)
    floor = next(s.mask for s in segments if s.label == "floor")
    P, fit = detect.depth_to_metres(cam, disparity, floor)
    assert fit["a"] == pytest.approx(2.0, rel=1e-3) and fit["b"] == pytest.approx(0.3, abs=1e-3)
    _, ids, t = render_gbuffer(truth)
    hit = ids >= 0
    true_pts = cam.position + cam.rays(np.argwhere(hit)[:, ::-1].astype(float)) * t[hit][:, None]
    assert np.nanmax(np.abs(P[hit] - true_pts)) < 1e-6


def test_boxes_and_walls_are_rebuilt():
    cam, truth = _room()
    segments, disparity = _fake_models(cam, truth)
    det = detect.build_proxies(cam, segments, disparity, first_id=1)
    boxes = {o["label"]: [p for p in det.proxies if p.id in o["proxy_ids"]] for o in det.objects}
    assert set(boxes) == {"table", "vase"}

    (table,), (vase,) = boxes["table"], boxes["vase"]
    assert table.min[1] == 0.0                                 # its feet touch the floor
    np.testing.assert_allclose(table.min, TABLE[0], atol=0.08)
    np.testing.assert_allclose(table.max, TABLE[1], atol=0.08)
    assert vase.min[1] == pytest.approx(table.max[1])           # resting on the table, not floating
    np.testing.assert_allclose(vase.max, VASE[1], atol=0.08)

    walls = {(w["axis"], w["side"]): w["value"] for w in det.walls}
    assert walls[(2, -1.0)] == pytest.approx(6.0, abs=0.1)      # back wall, facing the camera
    assert walls[(0, 1.0)] == pytest.approx(-3.0, abs=0.1)      # left wall
    ids = [p.id for p in det.proxies]
    assert ids == list(range(1, 1 + len(ids)))


def test_detect_command_keeps_hand_made_proxies_and_reseats_the_object(tmp_path, monkeypatch):
    cam, truth = _room()
    fake = _fake_models(cam, truth)
    calls = []
    monkeypatch.setattr(detect, "run_models", lambda rgb8, min_score=0.5: calls.append(rgb8.shape[:2]) or fake)

    sdir = tmp_path / "data" / "t"
    sdir.mkdir(parents=True)
    save_jpg(sdir / "photo.jpg", np.full((cam.height, cam.width, 3), 0.3))
    hand = BoxProxy(1, (2.0, 0.0, 4.0), (2.5, 0.5, 4.5))
    Scene(cam, proxies=[floor_proxy(0), hand]).save(sdir / "scene.json")
    data = json.loads((sdir / "scene.json").read_text())
    # placed on the floor behind the table, at a pixel that shows the table top
    on_top = np.array([0.2, 0.75, 2.9])
    ray = (on_top - cam.position) / np.linalg.norm(on_top - cam.position)
    centre = list(cam.position + ray * (-cam.position[1] / ray[1]))
    data["object"] = {"path": "object.glb", "transform": {"position": centre, "rotation_y_deg": 0, "scale": 1}}
    (sdir / "scene.json").write_text(json.dumps(data))

    args = ["detect", "--name", "t", "--data-root", str(tmp_path / "data")]
    main(args)
    main(args)                                                  # a rerun replaces its own proxies
    assert calls == [(cam.height, cam.width)] * 2

    scene = Scene.load(sdir / "scene.json")
    ids = [p.id for p in scene.proxies]
    assert ids[:2] == [0, 1] and len(ids) == len(set(ids))      # floor and the hand-made box kept
    info = json.loads((sdir / "detect.json").read_text())
    assert sorted(o["label"] for o in info["objects"]) == ["table", "vase"]
    assert set(info["proxy_ids"]) == set(ids[2:])
    for o in info["objects"]:
        assert (sdir / o["mask"]).exists()
    assert (sdir / "detect_check.jpg").exists()

    # the object stays at the same spot in the photo, now standing on the table top
    pos = np.array(scene.object.to_dict()["transform"]["position"])
    table = next(p for p in scene.proxies if p.id in next(o for o in info["objects"] if o["label"] == "table")["proxy_ids"])
    assert pos[1] == pytest.approx(table.max[1] + 0.002, abs=1e-3)
    np.testing.assert_allclose(cam.project(pos), cam.project(np.array(centre)), atol=1.5)


def test_depth_edges_mark_jumps_only():
    from recast.detect import depth_edges
    d = np.full((200, 300), 1.0)
    d[:, 150:] = 3.0                                  # a depth jump down the middle
    d += np.linspace(0.0, 0.2, 200)[:, None]          # and a gentle slope like a floor
    edges = depth_edges(d, model_px=100)
    assert edges[:, 145:155].any(axis=1).all()
    assert not edges[:, :120].any() and not edges[:, 180:].any()


def test_scale_from_a_table():
    from recast.detect import Detection, scale_from_known_heights, scaled
    from recast.scene import BoxProxy, PlaneProxy
    table = BoxProxy(1, (0.0, 0.0, -3.0), (0.8, 0.9, -2.2))      # 0.90 m tall: the room came out too big
    wall = PlaneProxy(2, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0))
    det = Detection([table, wall], [{"label": "table", "score": 1.0, "proxy_ids": [1]}], [], {})
    k, used = scale_from_known_heights(det)
    assert np.isclose(k, 0.74 / 0.9) and used
    small = scaled(det, k)
    assert np.isclose(small.proxies[0].max[1], 0.74)
    assert np.isclose(small.proxies[1].point[2], -5.0 * k)
