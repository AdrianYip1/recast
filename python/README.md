# recast / python

Takes the photos and builds everything the engine needs. The connection to C++
is a folder of files, not bindings: Python writes them, the engine loads them
(and reloads when they change).

## Setup

```powershell
cd python
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,lens]"
pytest
```

Extras: `lens` = lensfun database for undistortion, `raw` = camera RAW files
(better than JPEG, the colours are already linear), `models` = the pretrained
models for `recast detect` (CPU torch first, ~0.5 GB with the models):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[models]"
```

## Scene folder (the contract)

```
data/<scene_name>/
  scene.json        camera, paths, scale, proxies, lights, SH   (Python writes, engine adds proxies/lights)
  photo.jpg         undistorted scene photo, sRGB               (Python)
  object.glb        extracted object, metres, +Y up, pivot at base (Python)
  object_mask.jpg   cutout, for debugging                       (Python)
  object_check.jpg  mesh outline reprojected on the object photo (Python)
  detect.json       what `detect` found: labels, proxy ids, masks, depth fit (Python)
  masks/            one cutout per detected object (white = object) (Python)
  detect_check.jpg  detected objects (left), their boxes + walls (right) (Python)
  gbuffer/          normals.hdr, ids.png, depth.hdr             (engine, for the lighting fit)
  render.jpg        final composite (render.png also accepted)  (engine)
```

Photos go in and come out as **.jpg** (quality 95, no chroma subsampling).
`photo.jpg` is sRGB: the engine should load it with `stbi_load` and convert to
linear (or use an `_SRGB` texture format) before lighting maths. The lighting fit
also reads a linear `photo.hdr` if `scene.json` points at one. The gbuffer stays
`.hdr`/`.png`: JPEG compression would corrupt proxy ids, normals and depths.

Engine: `foto.exe data/<scene_name>/scene.json`

`scene.json` is described by [`schema/scene.schema.json`](../schema/scene.schema.json)
(both sides can validate against it). Rules: metres; world +Y up, right-handed;
camera `rotation` is camera-to-world in engine convention (looks down -Z);
pixel origin top-left; colours linear RGB. Python only rewrites the sections it
owns, keeps any keys it doesn't know, and writes atomically (temp file + rename),
so hot reload never sees half a file. Until stage B has run there is no `object`
key at all (the engine's loader rejects `"object": null`).

Lights: Python's area lights and the engine's point/directional lights share
`lights` (the engine detects them when a scene has none, and they can be
added, dragged, removed and dimmed by hand in its debug panel). Python keeps
every engine light and `lighting.ambient` as written, and only replaces the SH;
the object's `scale` (the engine's size slider) is kept too. If the engine's
light detection takes over 10 s or finds nothing, it says "no light source
detected" and opens its light editor.

Checked against the engine: every scene.json the Python writes loads in
`sandbox/scene/sceneFile.cpp`, and `light`/`eval` run on the sample from
`tools/make_test_scene.py` (`eval` picks up its `ground_truth.json`;
see `tests/test_engine_scene.py`).

`data/example/` is a complete fake room (made by `python examples/make_example.py`)
for testing the loader before real photos exist.

### gbuffer encoding (proposal, Radiance .hdr can't hold negative numbers)

| file | contents |
|---|---|
| `normals.hdr` | world-space normal stored as `n * 0.5 + 0.5`, black = no proxy |
| `ids.png` | 16-bit greyscale, `proxy id + 1`, 0 = no proxy |
| `depth.hdr` | distance along the camera ray in metres (same in R, G, B), 0 = no proxy |

Until the engine exports it, `light --python-gbuffer` renders the same thing from the proxies.

## Pipeline

```powershell
python -m recast scene  room.jpg --name living_room --camera-height 1.45   # A
python -m recast detect --name living_room                                  # A2 (optional)
python -m recast object mug.jpg  --name living_room --mode revolution --height 0.12        # B
#   ... engine: click proxies + lights, export gbuffer/ ...
python -m recast light  --name living_room                                  # C
#   ... engine: render.jpg ...
python -m recast eval   --name living_room --truth with_real_mug.jpg        # D
```

**A. Scene photo (.jpg) -> photo.jpg + camera** (`io`, `lens`, `vanishing`, `camera`)
1. Decode to linear (RAW via rawpy, or inverse sRGB for JPEG).
2. Undistort with lensfun (camera + lens from EXIF), or `--k1/--k2`.
3. Focal length: EXIF when the photo has it (`--solve-focal` to solve anyway), otherwise solved from two
   vanishing points that aren't near infinity, otherwise a typical phone lens (0.9 x the long side).
   The principal point is always the image centre. Photos sent through chat apps usually lose their EXIF.
4. Rotation from the vertical and the strongest horizontal vanishing point (LSD segments + RANSAC; `--lines` to trace them by hand).
5. Scale from `--camera-height`; floor is y = 0.
6. OpenCV camera -> engine convention (`R_engine = R_cv * diag(1, -1, -1)`), written to scene.json.

**A2. Walls and objects, automatically** (`detect`). The only step with
pretrained models, both run offline on the CPU (~30 s, downloaded once):
OneFormer ADE20K swin-tiny (MIT) outlines and names everything (floor, wall,
ceiling, sofa, table, vase, ...), Depth Anything V2 Base (CC-BY-NC-4.0, non-commercial; Small is Apache-2.0) gives
relative depth. The rest is ours:
1. Depth in metres: fit `disparity = a / z + b` on the floor pixels, whose true
   depth the solved camera gives exactly.
2. Each object -> its 3D points -> an axis-aligned box (two stacked boxes when it
   has a clear seat and back). Objects whose outline meets the floor (checked
   against depth, so a table top's overhang doesn't count) are pinned to y = 0
   and their feet widen the footprint; the outline's top edge, put on the plane
   of the top by the camera, fixes the far side that depth sees edge-on.
3. Each box rests on the floor or on the box under it (vase on table); boxes
   inside bigger ones (cushions in a sofa) are dropped.
4. Pixels where the depth jumps (outlines) are skipped: the model blurs depth across them, which
   stretches boxes. An object touching the floor has its depth scaled so its feet land where the
   camera puts them.
5. Scale: the camera height is a guess, so tables, desks, coffee tables and counters (whose height
   hardly varies) rescale the whole scene, camera height included, to their usual height.
6. Labels miss things, so geometry adds them: "wall" pixels standing over 0.3 m in front of every wall
   plane (pillars, shelving), "floor" pixels rising over 6 cm (steps, ledges, platforms) and unlabelled
   pixels standing in the room become objects too, boxed the same way.
7. Wall and ceiling pixels -> axis-aligned planes, pushed behind every object. Things seen through
   windows (trees, street lamps, cars, ...) and boxes floating over 2.2 m up get no box.
   These also keep the engine's shadow map sized to the room: without them the
   floor reaches the horizon and the object's shadow is too small to show.

Rerunning replaces what the last run added and keeps the floor and any proxies
made by hand (`--clear` drops those too). An object already placed stays at
the same spot in the photo, standing on whatever surface is there now.

**B. Object photo -> object.glb** (`recast.object`), no ML. One photo, a few
clicks, one real measurement:

```powershell
python -m recast.object mug.jpg --mode revolution --height 0.12 --out ..\data\room1
```

1. Load, shrink to `--max-size` (1600 px) and calibrate like the scene photo (vanishing points of the floor/table;
   `--lines` to trace them, or a level camera with `--pitch-deg` as fallback).
   World scale is arbitrary until the measurement fixes it.
2. Segment: GrabCut from one dragged box (`--refine` for brush fixes, `--chroma green|blue` for backdrop shots).
3. Ground contact = bottom-centre of the cutout, revolution axis top = top of the cutout above it
   (`--click` to click them instead, or pass `--base`, `--top`). Box mode always asks for its floor corners + a top corner.
4. Geometry, one module per mode in `recast/object/modes/`:

   | mode | for | how |
   |---|---|---|
   | `card` | anything, fastest | upright quad at the base facing the camera, alpha cutout, double-sided |
   | `revolution` | bottles, vases, mugs | silhouette half-width x depth / focal per height, smoothed, lathed |
   | `box` | boxes, books, furniture | floor corners hit y = 0, edges snapped to the VP axes |
   | `extrude` | signs, flat cut-outs | outline polygon given 10%-of-width thickness |
   | `inflate` | rounded blobs | older rounded silhouette (`recast.object3d`) |

5. Texture: projective UVs (each vertex samples the photo where it projects);
   back-facing vertices mirror through the axis plane (back = mirrored front);
   cropped to the mask; `--delight` divides out blurred luminance.
6. Export: pivot at the base centre, +Y up, scaled to `--height` (or `--width`),
   `object.glb` + `object_meta.json` (mode, clicks, camera, measurement, scores). `--glass` exports clear
   glass (`KHR_materials_transmission` + `KHR_materials_ior`) instead of the photo's colours, for bottles and glasses.
   A new object stands on the first floor or box top in view (searched from the bottom-middle of the photo).
7. Check: the mesh is reprojected onto the photo; `object_check.jpg` shows mask
   (green) vs mesh (red) and the IoU is printed. `--compare truth.glb` adds
   height error and Chamfer distance against a known mesh (e.g. a GLB the engine
   screenshotted from a known camera; `tests/test_object_modes.py` does the same with a Python render).

Engine side for cutout modes: `card`/`extrude` glbs use `alphaMode: MASK`
(cutoff 0.5) and cards are `doubleSided`, so the engine needs an alpha-cutoff
discard in the fragment shader and a no-culling path for double-sided materials.

**C. Lighting fit** (`lighting`): loads photo.jpg + gbuffer, masks pixels that
are off the proxies, clipped, highlights on glossy proxies (roughness < 0.5), or
in the shadow of a clicked light; then one least-squares solve for the SH
coefficients (order 1 = 4 by default, `--order 2` = 9) **and** each clicked area
light's radiance (exact polygon-light irradiance). Writes `lighting` and the
lights' `radiance` back into scene.json.

**D. Evaluation** (`evaluate`): PSNR/SSIM of render.jpg vs. a ground-truth
photo (real object placed, same tripod position), over the whole image and the
object region (object + shadow, found where the render differs from the empty room).
