"""Command line for the pipeline. Everything for one scene lives in data/<scene_name>/.

    python -m recast scene  room.jpg   --name living_room --camera-height 1.45
    python -m recast object mug.jpg    --name living_room --mode revolution --height 0.12
    python -m recast detect            --name living_room              (find walls + objects, optional)
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
from .camera import Camera, solve_camera
from .gbuffer import load_gbuffer, save_gbuffer
from .io import Photo, linear_to_srgb, load_image_linear, load_photo, save_hdr, save_jpg, to_srgb8
from .lighting import fit_scene_lighting
from .scene import (PHOTO_NAME, Lighting, Scene, floor_proxy, render_gbuffer, scene_objects, update_scene_file,
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
        # straight edges from a 1600 px copy: plenty for vanishing points, and much faster than the full photo
        shrink = min(1.0, 1600 / max(img.shape[:2]))
        small = img if shrink == 1.0 else cv2.resize(img.astype(np.float32), (round(img.shape[1] * shrink), round(img.shape[0] * shrink)),
                                                    interpolation=cv2.INTER_AREA)
        segs = detect_segments(small) / shrink
        found = find_vanishing_points(segs, count=3)
        print(f"{len(segs)} line segments, {len(found)} vanishing points")
        try:
            vp_x, vp_y, vp_z = assign_axes(segs, found)
        except ValueError as e:
            sys.exit(f"not enough straight edges to find the room's directions ({e}).\n"
                     "Trace a few edges per direction by hand and pass them with --lines lines.json:\n"
                     '  {"x": [[x1,y1,x2,y2], ...], "z": [...], "y": [...]}   (x, z horizontal; y vertical, optional)')

    try:
        cam = solve_camera(w, h, vp_x, vp_z, vp_y, focal_px=None if args.solve_focal else focal_px,
                           camera_height=args.camera_height)
    except ValueError as e:
        # no EXIF and no usable vanishing points (e.g. a photo taken square to a wall): a typical phone lens
        fallback = focal_px or 0.9 * max(w, h)
        print(f"could not solve the focal length ({e}), using {'EXIF' if focal_px else 'a typical phone lens'}")
        cam = solve_camera(w, h, vp_x, vp_z, vp_y, focal_px=fallback, camera_height=args.camera_height)
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


# --- stage A2: walls and objects, found automatically ---------------------------------

DETECT_FILE = "detect.json"


def cmd_detect(args) -> None:
    from . import detect

    out = scene_dir(args)
    path = require(out / "scene.json", "run `python -m recast scene` (stage A) first")
    scene = Scene.load(path)
    cam = scene.camera
    with Image.open(out / scene.photo) as header:
        if header.size != (cam.width, cam.height):
            sys.exit(f"{scene.photo} is {header.size[0]}x{header.size[1]} but the camera is {cam.width}x{cam.height}")

    # the models see ~512 px anyway: everything runs on a smaller copy with the same camera, scaled.
    # Boxes come out in metres, so they don't depend on the working size
    small = load_photo(out / scene.photo, max_side=args.max_size or None)
    photo_linear = small.pixels
    rgb8 = to_srgb8(photo_linear)
    full_cam = cam
    if small.scale < 1.0:
        size = (photo_linear.shape[1], photo_linear.shape[0])
        cam = Camera(size[0], size[1], full_cam.fx * size[0] / full_cam.width, full_cam.R, full_cam.position.copy(),
                     (full_cam.principal + 0.5) * np.array(size) / [full_cam.width, full_cam.height] - 0.5,
                     full_cam.fy * size[1] / full_cam.height)

    print("running the segmentation and depth models on the CPU")
    segments, disparity = detect.run_models(rgb8, min_score=args.min_score)
    print(f"{len(segments)} regions: " + ", ".join(sorted({detect.short_label(s.label) for s in segments})))

    # replace what the last detect run added; keep the floor and anything made by hand
    old = json.loads((out / DETECT_FILE).read_text()) if (out / DETECT_FILE).exists() else {}
    drop = set(old.get("proxy_ids", []))
    data = json.loads(path.read_text())
    kept = [p for p in data.get("proxies", [])
            if p["id"] not in drop and not (args.clear and p.get("id") != 0)]
    first_id = max((p["id"] for p in kept), default=-1) + 1

    try:
        det = detect.build_proxies(cam, segments, disparity, first_id, photo_linear, min_area=args.min_area)
    except ValueError as e:
        sys.exit(f"detect failed: {e}")
    fit = det.depth_fit
    print(f"depth in metres from {fit['floor_pixels']} floor pixels "
          f"(typical error {100 * fit['floor_depth_error']:.0f}%)")

    # the camera height was a guess: tables and counters say how big the room really is. The correction is kept within
    # 0.8-1.25x of the original guess in total, so running detect again doesn't compound it
    k, used = detect.scale_from_known_heights(det)
    guess = float(scene.scale_reference.get("guess") or scene.scale_reference.get("metres") or cam.position[1])
    k = float(np.clip(cam.position[1] * k, 0.8 * guess, 1.25 * guess) / cam.position[1])
    rescaled = abs(k - 1.0) > 0.03
    if rescaled:
        det = detect.scaled(det, k)
        cam.position = cam.position * k
        full_cam.position = full_cam.position * k
        print(f"scale from {', '.join(used)}: camera height {cam.position[1] / k:.2f} -> {cam.position[1]:.2f} m")
        for p in kept:
            for key in ("point", "min", "max"):
                if key in p:
                    p[key] = [float(v) * k for v in p[key]]
        for light in data.get("lights", []):
            if "position" in light:
                light["position"] = [float(v) * k for v in light["position"]]
        for obj in scene_objects(data):
            if obj.get("transform"):
                obj["transform"]["position"] = [float(v) * k for v in obj["transform"]["position"]]

    det = detect.open_below_slabs(det)

    mdir = out / "masks"
    mdir.mkdir(exist_ok=True)
    for f in mdir.glob("*.png"):
        f.unlink()
    objects = []
    for o in det.objects:
        name = f"{o['proxy_ids'][0]:02d}_{o['label'].replace(' ', '_')}.png"
        cv2.imwrite(str(mdir / name), o["segment"].mask.astype(np.uint8) * 255)
        boxes = [scene_box for scene_box in det.proxies if scene_box.id in o["proxy_ids"]]
        lo = np.min([b.min for b in boxes], axis=0)
        hi = np.max([b.max for b in boxes], axis=0)
        size = hi - lo
        rests = "on the floor" if lo[1] < 1e-6 else f"on the {lo[1]:.2f} m up"
        if o["label"] in detect.OPEN_BELOW and o.get("shape") != "seat_back":
            rests = f"top at {hi[1]:.2f} m (open below)"
        print(f"  {o['label']:<14} {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, {rests}"
              f"{'  (2 boxes: seat + back)' if o.get('shape') == 'seat_back' else '  (fitted shape)' if o.get('shape') == 'hull' else ''}")
        objects.append({"label": o["label"], "score": o["score"], "proxy_ids": o["proxy_ids"], "shape": o.get("shape", "box"),
                        "mask": f"masks/{name}"})
    for w in det.walls:
        kind = "ceiling" if w["axis"] == 1 else "wall"
        print(f"  {kind:<14} at {'xyz'[w['axis']]} = {w['value']:.2f} m")

    # windows and lamps in the photo: the engine fits their strength (a new room means new lighting)
    emitters = detect.find_emitters(cam, det.points, segments, det.walls)
    for light in emitters:
        print(f"  {'window light' if light['type'] == 'area' else 'lamp light':<14} " +
              (f"{np.ptp(np.array(light['corners'])[:, [0, 2]], axis=0).max():.2f} x {np.ptp(np.array(light['corners'])[:, 1]):.2f} m"
               if light["type"] == "area" else "at " + ", ".join(f"{v:.2f}" for v in light["position"])))

    # the photo's own depth (distance from the camera, metres): the engine hides the object only where the photo really
    # has something in front of it, instead of wherever a box stands in for the furniture
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(det.points - cam.position, axis=2)
    distance = np.where(np.isfinite(distance), distance, 1e4).astype(np.float32)
    save_hdr(out / "photo_depth.hdr", np.repeat(distance[..., None], 3, axis=2))

    proxies = kept + [p.to_dict() for p in det.proxies]
    sections = {"proxies": proxies, "lights": emitters}
    if rescaled:
        sections.update(camera=full_cam.to_dict(),
                        scale_reference={"kind": "camera_height", "metres": float(cam.position[1]), "guess": guess, "from": used})
    placed_objects = scene_objects(data)
    if placed_objects:
        # the surfaces changed under the objects: keep each at the same spot in the photo and stand it
        # on whatever surface is there now, like dragging it there in the engine does
        from .scene import cast, proxy_from_dict
        casters = [proxy_from_dict(p) for p in proxies]
        for obj in placed_objects:
            if not obj.get("transform"):
                continue
            pos = np.asarray(obj["transform"]["position"], float)
            ray = cam.rays(cam.project(pos))
            t, n, hit_id = cast(cam.position, ray, casters)
            if hit_id[0] >= 0 and n[0][1] > 0.7:
                pos = cam.position + ray[0] * t[0]
                pos[1] += 0.002   # just above the surface: the engine pushes overlapping objects aside
                obj["transform"]["position"] = [round(float(v), 4) for v in pos]
        sections["objects"] = placed_objects
    update_scene_file(path, **sections)
    write_json_atomic(out / DETECT_FILE, {
        "models": {"segmentation": detect.SEG_MODEL, "depth": detect.DEPTH_MODEL},
        "depth_fit": fit, "proxy_ids": [p.id for p in det.proxies], "objects": objects,
        "walls": [{k: v for k, v in w.items() if k != "points"} for w in det.walls]})
    cv2.imwrite(str(out / "detect_check.jpg"), cv2.cvtColor(detect.draw_check(rgb8, cam, det), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"added {len(det.proxies)} proxies to {path}; see detect_check.jpg")


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
    a.add_argument("--trust-exif", action="store_true", help="(default now) use the EXIF focal length when the photo has one")
    a.add_argument("--solve-focal", action="store_true", help="solve the focal length from vanishing points even with EXIF")
    a.add_argument("--k1", type=float, default=0.0)
    a.add_argument("--k2", type=float, default=0.0)
    a.add_argument("--no-lensfun", action="store_true")
    a.add_argument("--lines", help='JSON {"x": [[x1,y1,x2,y2],...], "z": [...], "y": [...]} instead of auto lines')
    a.add_argument("--overwrite", action="store_true", help="start scene.json fresh")
    a.set_defaults(func=cmd_scene)

    # stage B has its own command line (modes, clicks, checks): python -m recast.object --help
    sub.add_parser("object", help="stage B: object photo -> object.glb (python -m recast.object --help)",
                   add_help=False)

    a2 = sub.add_parser("detect", help="find walls, ceiling and objects in the scene photo (pretrained models)")
    common(a2)
    a2.add_argument("--min-score", type=float, default=0.5, help="drop detections less certain than this")
    a2.add_argument("--min-area", type=float, default=0.001, help="ignore objects smaller than this share of the photo")
    a2.add_argument("--clear", action="store_true", help="also remove boxes/walls made by hand (keeps the floor)")
    a2.add_argument("--max-size", type=int, default=1280, help="work on the photo shrunk to this many pixels (0 = full size)")
    a2.set_defaults(func=cmd_detect)

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
