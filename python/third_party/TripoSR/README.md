Vendored from https://github.com/VAST-AI-Research/TripoSR (MIT, see LICENSE) at commit 107cefd.
Only `tsr/` is kept. recast changes:
- `tsr/models/isosurface.py`: scikit-image marching cubes instead of torchmcubes (no compiling).
- `tsr/utils.py`: rembg and imageio imported only where used.
- `tsr/bake_texture.py` removed (moderngl); recast bakes the texture itself (`recast/object/modes/generated.py`).
