import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from recast.scene import (AreaLight, BoxProxy, Lighting, Material, ObjectPlacement, Scene, box_on_floor,
                          cast, floor_proxy, render_gbuffer, update_scene_file, wall_through)
from synthetic import look_at_camera

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schema" / "scene.schema.json").read_text())


def _scene():
    cam = look_at_camera(width=160, height=120, focal_px=130.0)
    return Scene(
        camera=cam,
        object=ObjectPlacement("object.glb", (0.2, 0, 3.5), 30.0, 1.0),
        proxies=[floor_proxy(0, Material((0.3, 0.25, 0.2), 0.3)),
                 BoxProxy(1, (0.5, 0, 3), (1.5, 0.75, 4))],
        lights=[AreaLight(np.array([[0, 2.4, 3], [1, 2.4, 3], [1, 2.4, 4], [0, 2.4, 4]]), (8, 8, 7))],
        lighting=Lighting(1, np.arange(12, dtype=float).reshape(4, 3) / 30),
    )


def test_matches_schema():
    jsonschema.validate(_scene().to_dict(), SCHEMA)
    jsonschema.validate(Scene(look_at_camera()).to_dict(), SCHEMA)  # fresh scene, no SH yet


def test_schema_rejects_bad_camera():
    d = _scene().to_dict()
    del d["camera"]["fx"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, SCHEMA)


def test_json_roundtrip(tmp_path):
    s = _scene()
    p = tmp_path / "scene.json"
    s.save(p)
    back = Scene.load(p)
    assert np.allclose(back.camera.R, s.camera.R)
    assert np.allclose(back.camera.K, s.camera.K)
    assert np.allclose(back.lighting.sh, s.lighting.sh)
    assert back.proxies[0].material.glossy and not back.proxies[1].material.glossy
    assert back.to_dict() == s.to_dict()


def test_keeps_unknown_keys_and_engine_edits(tmp_path):
    p = tmp_path / "scene.json"
    _scene().save(p)
    data = json.loads(p.read_text())
    data["engine_state"] = {"selected": 3}
    data["proxies"].append({"type": "plane", "id": 7, "point": [0, 0, 5], "normal": [0, 0, -1],
                            "material": {"albedo": [1, 1, 1], "roughness": 1}})
    p.write_text(json.dumps(data))

    update_scene_file(p, lighting={"sh_order": 1, "sh": [[1, 1, 1]] * 4})
    after = json.loads(p.read_text())
    assert after["engine_state"] == {"selected": 3}
    assert len(after["proxies"]) == 3
    assert after["lighting"]["sh"][0] == [1, 1, 1]
    assert Scene.load(p).to_dict()["engine_state"] == {"selected": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_camera_rotation_is_proper():
    R = np.asarray(_scene().to_dict()["camera"]["rotation"])
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_box_hit_and_normal():
    box = BoxProxy(0, (-1, 0, 4), (1, 1, 5))
    t, n, ids = cast(np.array([0.0, 0.5, 0.0]), np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), [box])
    assert np.isclose(t[0], 4.0) and np.allclose(n[0], [0, 0, -1]) and ids[0] == 0
    assert np.isinf(t[1]) and ids[1] == -1


def test_gbuffer_floor_wall_box():
    s = _scene()
    cam = s.camera
    wall = wall_through(2, cam, (20, 90), (140, 90))
    s.proxies.append(wall)
    normals, ids, depth = render_gbuffer(s)
    assert (ids[-1] >= 0).all()
    assert (ids[0] == 2).all()  # top row: the wall
    assert np.allclose(normals[ids == 0], [0, 1, 0])
    assert np.isfinite(depth[ids >= 0]).all()
    # every hit point really lies on the floor for floor pixels
    ys, xs = np.nonzero(ids == 0)
    pts = cam.position + cam.rays(np.column_stack([xs, ys]).astype(float)) * depth[ys, xs][:, None]
    assert np.allclose(pts[:, 1], 0.0, atol=1e-9)


def test_box_on_floor_from_clicks():
    cam = look_at_camera()
    corners = np.array([[0.5, 0, 3.0], [1.2, 0, 3.0], [1.2, 0, 3.6]])
    box = box_on_floor(5, cam, cam.project(corners), 0.8)
    assert np.allclose(box.min, [0.5, 0, 3.0], atol=1e-6)
    assert np.allclose(box.max, [1.2, 0.8, 3.6], atol=1e-6)
