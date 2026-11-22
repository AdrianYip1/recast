"""Integration with the engine side: Adrian's tools/make_test_scene.py writes a scene folder
(photo.hdr, object.glb, scene.json, ground_truth.json); the Python stages must read it."""

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

from recast.cli import main
from recast.scene import Scene

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "make_test_scene.py"
SCHEMA = json.loads((REPO / "schema" / "scene.schema.json").read_text())


def test_python_stages_run_on_engine_test_scene(tmp_path):
    import pytest
    if not TOOL.exists():
        pytest.skip("tools/make_test_scene.py not merged from main")
    sdir = tmp_path / "engine_sample"
    subprocess.run([sys.executable, str(TOOL), str(sdir)], check=True, capture_output=True)

    scene = Scene.load(sdir / "scene.json")
    assert scene.photo.endswith(".hdr") and scene.objects[0].path == "object.glb"

    # stage C on the engine's scene, then the file must still match the shared schema
    main(["light", "--name", sdir.name, "--data-root", str(tmp_path)])
    data = json.loads((sdir / "scene.json").read_text())
    jsonschema.validate(data, SCHEMA)
    assert len(data["lighting"]["sh"]) == 4

    # stage D with the truth taken from ground_truth.json; the "render" is the truth itself
    truth = json.loads((sdir / "ground_truth.json").read_text())["photo_with_object"]
    main(["eval", "--name", sdir.name, "--data-root", str(tmp_path), "--render", str(sdir / truth)])
    assert json.loads((sdir / "eval.json").read_text())["ssim"] > 0.99
