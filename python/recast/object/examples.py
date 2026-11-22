"""Objects the bank always has, shipped with the code instead of made from a photo: a glass dragon (amber glass that
the path tracer refracts and colours by thickness, see tools/make_glass_dragon.py). ensure_examples() writes any that
are missing and removes retired ones; the worker calls it when it starts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import trimesh

ASSETS = Path(__file__).resolve().parent / "assets"

# bank folder -> (asset file name, name on the bank tile)
EXAMPLES = {"example_glass_dragon": ("glass_dragon", "glass dragon")}


def remove_retired(bank: Path) -> None:
    for folder in bank.iterdir() if bank.exists() else []:
        meta = folder / "object_meta.json"
        if folder.name in EXAMPLES or not meta.exists():
            continue
        try:
            retired = json.loads(meta.read_text()).get("mode") == "example"
        except (OSError, json.JSONDecodeError):
            retired = False
        if retired:
            shutil.rmtree(folder, ignore_errors=True)


def ensure_examples(bank: Path) -> list[Path]:
    """Writes the example objects missing from the bank; returns the folders it wrote."""
    bank = Path(bank)
    remove_retired(bank)
    written = []
    for name, (asset, label) in EXAMPLES.items():
        folder = bank / name
        if (folder / "object_meta.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ASSETS / f"{asset}.glb", folder / "object.glb")
        shutil.copyfile(ASSETS / f"{asset}.png", folder / "thumbnail.png")
        mesh = trimesh.load(folder / "object.glb", force="mesh")
        meta = {"mode": "example", "glass": True, "photo": label,
                "measurement": {"height_m": float(mesh.extents[1])},
                "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        (folder / "object_meta.json").write_text(json.dumps(meta, indent=2))   # last: marks the entry complete
        written.append(folder)
    return written
