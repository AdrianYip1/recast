# recast / python

Takes the photos and builds everything the C++ engine needs: camera, room
geometry, lighting, and the inserted object as a glTF mesh.

## Setup

```powershell
cd python
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Optional: `pip install -e ".[raw]"` to load camera RAW files (better than JPEG,
since the colours are already linear).

## Pipeline

| Step | Module | What it does |
|---|---|---|
| 1 | `recast.io` | load photo, read focal length from EXIF, sRGB <-> linear |
| 2 | `recast.lens` | undo radial lens distortion |
| 3 | `recast.vanishing` | detect line segments, find vanishing points |
| 4 | `recast.camera` | solve focal length + rotation from vanishing points |
| 5 | `recast.scene` | planes, boxes, materials, lights -> `scene.json` |
| 6 | `recast.lighting` | fit soft lighting (9 spherical harmonics) from shading |
| 7 | `recast.object3d` | cut out object, inflate to mesh, remove its lighting, export `.glb` |

No ML models or AI APIs anywhere, just geometry and least squares.

## Output for the engine

```
output/
  scene.json       camera, planes, boxes, lights, object transform
  background.hdr   scene photo, undistorted, LINEAR colour (stbi_loadf can read it)
  object.glb       object mesh + albedo texture
```

Conventions (same as glTF):

- right-handed, **+Y up**, units in **meters**
- floor is the plane `y = 0`
- camera looks down its local **-Z**, local +Y is up
- `camera.rotation` is the camera-to-world 3x3 matrix, row-major
- colours in `scene.json` are linear RGB

`examples/` has a small fake room (`scene.json`, `background.hdr`, `object.glb`)
to test the engine's loader with. Regenerate it with `python examples/make_example.py`.
