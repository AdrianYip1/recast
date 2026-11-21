"""End to end: stage A -> B -> C -> D through the command line, on synthetic images."""

import json

import numpy as np
from PIL import Image

from recast.cli import main
from recast.io import linear_to_srgb
from recast.scene import Scene
from synthetic import look_at_camera, room_segments


def _save_srgb(path, linear):
    Image.fromarray((np.clip(linear_to_srgb(linear), 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def test_pipeline_end_to_end(tmp_path):
    root = tmp_path / "data"
    cam = look_at_camera(width=320, height=240, focal_px=260.0, yaw_deg=30, pitch_deg=-15)

    # stage A with hand-traced lines (auto detection needs a real photo)
    segs, labels = room_segments(cam, np.random.default_rng(5))
    lines = {"x": segs[labels == 0].tolist(), "y": segs[labels == 1].tolist(), "z": segs[labels == 2].tolist()}
    (tmp_path / "lines.json").write_text(json.dumps(lines))
    _save_srgb(tmp_path / "room.png", np.full((240, 320, 3), 0.3))
    main(["scene", str(tmp_path / "room.png"), "--name", "t", "--data-root", str(root),
          "--lines", str(tmp_path / "lines.json"), "--camera-height", "1.5", "--no-lensfun"])

    sdir = root / "t"
    assert (sdir / "photo.hdr").exists()
    scene = Scene.load(sdir / "scene.json")
    assert abs(scene.camera.focal_px - 260.0) / 260.0 < 0.03
    assert scene.proxies[0].id == 0 and scene.scale_reference["metres"] == 1.5

    # stage B: a ball on a grey background, cut out with GrabCut
    yy, xx = np.mgrid[0:200, 0:200]
    ball = np.full((200, 200, 3), 0.15)
    inside = (yy - 100) ** 2 + (xx - 100) ** 2 < 60 ** 2
    ball[inside] = [0.6, 0.1, 0.05]
    _save_srgb(tmp_path / "ball.png", ball)
    main(["object", str(tmp_path / "ball.png"), "--name", "t", "--data-root", str(root),
          "--rect", "20", "20", "160", "160", "--height", "0.2"])
    assert (sdir / "object.glb").exists() and (sdir / "object_mask.png").exists()
    placed = json.loads((sdir / "scene.json").read_text())["object"]
    assert placed["path"] == "object.glb"
    assert placed["transform"]["position"][1] == 0.0

    # stage C, with the Python stand-in gbuffer
    main(["light", "--name", "t", "--data-root", str(root), "--python-gbuffer"])
    lighting = json.loads((sdir / "scene.json").read_text())["lighting"]
    assert lighting["sh_order"] == 1 and len(lighting["sh"]) == 4
    assert (sdir / "gbuffer" / "ids.png").exists()

    # stage D: the "render" is the photo itself, compared with itself
    _save_srgb(sdir / "render.png", np.full((240, 320, 3), 0.3))
    main(["eval", "--name", "t", "--data-root", str(root), "--truth", str(tmp_path / "room.png")])
    scores = json.loads((sdir / "eval.json").read_text())
    assert scores["ssim"] > 0.99
