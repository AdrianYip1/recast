"""Command line for the pipeline. Everything for one scene lives in data/<scene_name>/.

    python -m recast scene  room.jpg   --name living_room --camera-height 1.45
    python -m recast object mug.jpg    --name living_room --mode revolution --height 0.12
    python -m recast light             --name living_room              (after the engine wrote gbuffer/)
    python -m recast eval              --name living_room --truth with_real_mug.jpg

The engine is started with:  foto.exe data/<scene_name>/scene.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import evaluate, lens
from .camera import solve_camera
from .gbuffer import load_gbuffer, save_gbuffer
from .io import Photo, linear_to_srgb, load_image_linear, load_photo, save_jpg
from .lighting import fit_scene_lighting
from .scene import (PHOTO_NAME, Lighting, Scene, floor_proxy, render_gbuffer, update_scene_file,
                    write_json_atomic)

PREPROCESS_FILE = "preprocess.json"
from .vanishing import assign_axes, detect_segments, find_vanishing_points, vp_from_segments

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def scene_dir(args) -> Path:
    return Path(args.data_root) / args.name


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"missing {path}\n{hint}")
    return path


# --- shared: undistortion has to be identical for the scene photo and the ground truth ---

def preprocess(photo: Photo, settings: dict) -> np.ndarray:
    img = photo.pixels
    if settings.get("lensfun"):
        e = photo.exif or {}
        out = lens.undistort_lensfun(img, e.get("make"), e.get("model"), e.get("lens_make"),
                                     e.get("lens_model"), photo.focal_mm, e.get("f_number"))
        if out is not None:
            return out
    k1, k2 = settings.get("k1", 0.0), settings.get("k2", 0.0)
    if k1 or k2:
        return lens.undistort(img, settings["focal_px"], k1, k2)
    return img


# --- stage A ----------------------------------------------------------------------

def cmd_scene(args) -> None:
    out = scene_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    photo = load_photo(args.photo)
    w, h = photo.width, photo.height

    focal_px = args.focal_px or photo.focal_px(args.sensor_width_mm)
    if args.focal_mm and args.sensor_width_mm:
        focal_px = args.focal_mm / args.sensor_width_mm * w

    settings = {"lensfun": not args.no_lensfun, "k1": args.k1, "k2": args.k2,
                "focal_px": focal_px or 0.9 * max(w, h)}
    img = preprocess(photo, settings)
    print(f"photo {w}x{h}, EXIF focal {'%.1f px' % focal_px if focal_px else 'unknown'}"
          f"{', undistorted' if img is not photo.pixels else ''}")

    if args.lines:
        lines = json.loads(Path(args.lines).read_text())
        vp_x = vp_from_segments(np.asarray(lines["x"]))
        vp_z = vp_from_segments(np.asarray(lines["z"]))
        vp_y = vp_from_segments(np.asarray(lines["y"])) if lines.get("y") else None
    else:
        segs = detect_segments(img)
        found = find_vanishing_points(segs, count=3)
        print(f"{len(segs)} line segments, {len(found)} vanishing points")
        try:
            vp_x, vp_y, vp_z = assign_axes(segs, found)
        except ValueError as e:
            sys.exit(f"not enough straight edges to find the room's directions ({e}).\n"
                     "Trace a few edges per direction by hand and pass them with --lines lines.json:\n"
                     '  {"x": [[x1,y1,x2,y2], ...], "z": [...], "y": [...]}   (x, z horizontal; y vertical, optional)')

    try:
        cam = solve_camera(w, h, vp_x, vp_z, vp_y, focal_px=focal_px if args.trust_exif else None,
                           camera_height=args.camera_height)
    except ValueError as e:
        if not focal_px:
            sys.exit(f"camera solve failed ({e}); pass --focal-px, or trace more edges with --lines")
        print(f"could not solve focal length ({e}), using EXIF")
        cam = solve_camera(w, h, vp_x, vp_z, vp_y, focal_px=focal_px, camera_height=args.camera_height)
    print(f"camera: f = {cam.focal_px:.1f} px ({np.degrees(cam.fov_y):.1f} deg vertical fov), "
          f"height {args.camera_height} m")

    save_jpg(out / PHOTO_NAME, img)
    scale_ref = {"kind": "camera_height", "metres": args.camera_height}
    preprocess_info = {"lensfun": settings["lensfun"], "k1": args.k1, "k2": args.k2,
                       "focal_px": settings["focal_px"]}
    # the engine rewrites scene.json with only its own keys when the object is moved,
    # so the undistortion settings eval needs also live in a file only Python writes
    write_json_atomic(out / PREPROCESS_FILE, preprocess_info)

    path = out / "scene.json"
    if path.exists() and not args.overwrite:
        # keep the engine's proxies/lights/object; only the camera side is ours
        update_scene_file(path, photo=PHOTO_NAME, image_size=[w, h], camera=cam.to_dict(),
                          scale_reference=scale_ref, preprocess=preprocess_info)
        print(f"updated camera in {path}")
    else:
        Scene(cam, photo=PHOTO_NAME, scale_reference=scale_ref, proxies=[floor_proxy(0)],
              extra={"preprocess": preprocess_info}).save(path)
        print(f"wrote {path}")


# --- stage C ----------------------------------------------------------------------

def cmd_light(args) -> None:
    out = scene_dir(args)
    path = require(out / "scene.json", "run `python -m recast scene` (stage A) first")
    scene = Scene.load(path)
    photo = load_image_linear(out / scene.photo)

    gdir = out / "gbuffer"
    if args.python_gbuffer or not (gdir / "ids.png").exists():
        print("no engine gbuffer, rendering proxies in Python")
        n, ids, depth = render_gbuffer(scene)
        save_gbuffer(gdir, n, ids, depth)
    normals, ids, depth = load_gbuffer(gdir)

    sh, rad, used = fit_scene_lighting(photo, normals, ids, depth, scene, order=args.order)
    print(f"fit from {int(used.sum())} pixels: SH order {args.order}, {len(rad)} area light(s)")

    data = json.loads(path.read_text())
    lights = data.get("lights", [])
    j = 0
    for light in lights:  # update radiance in place, keeping whatever else the engine stores
        if light.get("type", "area") == "area" and j < len(rad):
            light["radiance"] = [float(v) for v in rad[j]]
            j += 1
    # only the SH is ours: keep the engine's ambient (and anything else it stores there)
    lighting = dict(data.get("lighting") or {})
    lighting.update(Lighting(args.order, sh).to_dict())
    update_scene_file(path, lighting=lighting, lights=lights)
    print(f"updated lighting in {path}")


# --- stage D ----------------------------------------------------------------------

def _load_srgb(path: Path) -> np.ndarray:
    if Path(path).suffix.lower() == ".hdr":
        return np.clip(linear_to_srgb(load_image_linear(path)), 0, 1).astype(np.float64)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def cmd_eval(args) -> None:
    out = scene_dir(args)
    data = json.loads(require(out / "scene.json", "run `python -m recast scene` (stage A) first").read_text())
    if args.render:
        render_path = Path(args.render)
    else:  # render.jpg, or render.png if that's what the engine wrote
        render_path = next((out / n for n in ("render.jpg", "render.png") if (out / n).exists()), out / "render.jpg")
    render = _load_srgb(require(render_path, "the engine writes render.jpg; or pass --render"))

    # default truth: the one tools/make_test_scene.py (engine side) lists in ground_truth.json
    truth_path = args.truth
    if truth_path is None:
        gt = out / "ground_truth.json"
        if not gt.exists():
            sys.exit("pass --truth (a photo of the scene with the real object in it)")
        truth_path = out / json.loads(gt.read_text())["photo_with_object"]
    # the ground truth went through the same lens, so undistort it the same way
    # (.hdr truths are synthetic renders: already linear, no EXIF)
    truth_photo = Photo(load_image_linear(truth_path)) if Path(truth_path).suffix.lower() == ".hdr" \
        else load_photo(truth_path)
    pre_path = out / PREPROCESS_FILE
    settings = json.loads(pre_path.read_text()) if pre_path.exists() else data.get("preprocess", {})
    truth = np.clip(linear_to_srgb(preprocess(truth_photo, settings)), 0, 1)
    if truth.shape != render.shape:
        truth = cv2.resize(truth, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)

    if args.mask:
        region = np.asarray(Image.open(args.mask).convert("L")) > 127
    else:
        empty = np.clip(linear_to_srgb(load_image_linear(out / data.get("photo", PHOTO_NAME))), 0, 1)
        if empty.shape != render.shape:
            empty = cv2.resize(empty, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
        region = evaluate.change_mask(render, empty)

    s = evaluate.score(render, truth, region)
    print(f"whole image : PSNR {s.psnr:.2f} dB, SSIM {s.ssim:.4f}")
    if s.region_psnr is not None:
        rs = f"{s.region_ssim:.4f}" if s.region_ssim is not None else "n/a"
        print(f"object area : PSNR {s.region_psnr:.2f} dB, SSIM {rs} ({s.region_pixels} px)")
    (out / "eval.json").write_text(json.dumps(s.to_dict(), indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="recast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--name", required=True, help="scene folder name under data/")
        sp.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))

    a = sub.add_parser("scene", help="stage A: scene photo (.jpg) -> photo.jpg + camera")
    common(a)
    a.add_argument("photo")
    a.add_argument("--camera-height", type=float, default=1.5, help="metres above the floor (sets scale)")
    a.add_argument("--focal-px", type=float)
    a.add_argument("--focal-mm", type=float)
    a.add_argument("--sensor-width-mm", type=float)
    a.add_argument("--trust-exif", action="store_true", help="use the EXIF focal length instead of solving it")
    a.add_argument("--k1", type=float, default=0.0)
    a.add_argument("--k2", type=float, default=0.0)
    a.add_argument("--no-lensfun", action="store_true")
    a.add_argument("--lines", help='JSON {"x": [[x1,y1,x2,y2],...], "z": [...], "y": [...]} instead of auto lines')
    a.add_argument("--overwrite", action="store_true", help="start scene.json fresh")
    a.set_defaults(func=cmd_scene)

    # stage B has its own command line (modes, clicks, checks): python -m recast.object --help
    sub.add_parser("object", help="stage B: object photo -> object.glb (python -m recast.object --help)",
                   add_help=False)

    c = sub.add_parser("light", help="stage C: fit lighting from photo + gbuffer")
    common(c)
    c.add_argument("--order", type=int, choices=(1, 2), default=1)
    c.add_argument("--python-gbuffer", action="store_true", help="render the gbuffer here instead of the engine")
    c.set_defaults(func=cmd_light)

    d = sub.add_parser("eval", help="stage D: PSNR/SSIM against a ground-truth photo")
    common(d)
    d.add_argument("--truth", help="ground-truth photo (.jpg, or .hdr); default: ground_truth.json's photo_with_object")
    d.add_argument("--render")
    d.add_argument("--mask", help="object region (white) instead of the automatic one")
    d.set_defaults(func=cmd_eval)

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["object"]:
        from .object.cli import build_parser, run
        args = build_parser("python -m recast object").parse_args(argv[1:])
        if not (args.out or args.name):
            sys.exit("give --out DIR or --name SCENE")
        run(args)
        return
    args = p.parse_args(argv)
    args.func(args)
