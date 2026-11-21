"""python -m recast.object photo.jpg --mode revolution --height 0.32 --out data/room1/

One box drag in an OpenCV window cuts the object out (or --rect). Ground contact
is the bottom-centre of the cutout and the axis top is the top of the cutout above
it; --click asks for them instead, --refine opens the brush to fix the cutout.
Box mode always needs its corner clicks. --glass makes a clear glass object.
Photos are shrunk to --max-size pixels first, which keeps the cutout fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import object3d
from ..camera import Camera, solve_camera
from ..io import load_photo, save_jpg, to_srgb8
from ..scene import ObjectPlacement, update_scene_file
from ..vanishing import assign_axes, detect_segments, find_vanishing_points, vp_from_segments
from . import MODES, annotate, check
from .export import export_glass_glb, export_glb, normalize, write_meta
from .modes import build_box, build_card, build_extrude, build_revolution
from .placement import base_point, height_from_click, mask_bottom_center, mask_top_above
from .segment import chroma_key, grabcut, salient_alpha, soft_alpha
from .texture import back_side_fill, crop_texture, delight, projective_uv

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def level_camera(w: int, h: int, focal_px: float, pitch_deg: float) -> Camera:
    """Fallback when no vanishing points: camera 1 unit up, looking along +Z, pitched."""
    p = np.radians(pitch_deg)
    fwd = np.array([0.0, np.sin(p), np.cos(p)])
    right = np.array([-1.0, 0.0, 0.0])
    return Camera(w, h, focal_px, np.vstack([right, np.cross(fwd, right), fwd]), np.array([0.0, 1.0, 0.0]))


def solve_object_camera(img: np.ndarray, args, exif_focal: float | None) -> tuple[Camera, str]:
    """Camera for the object photo. World scale is arbitrary (camera 1 unit up);
    the real measurement fixes it at export."""
    h, w = img.shape[:2]
    focal = args.focal_px or exif_focal
    vps = None
    if args.lines:
        lines = json.loads(Path(args.lines).read_text())
        vps = (vp_from_segments(np.asarray(lines["x"])),
               vp_from_segments(np.asarray(lines["y"])) if lines.get("y") else None,
               vp_from_segments(np.asarray(lines["z"])))
    elif not args.no_auto_lines:
        segs = detect_segments(img)
        try:
            vps = assign_axes(segs, find_vanishing_points(segs, count=3))
        except ValueError:
            vps = None
    if vps is not None:
        vx, vy, vz = vps
        try:
            return solve_camera(w, h, vx, vz, vy, focal_px=None if args.solve_focal else focal,
                                camera_height=1.0), "vanishing points" + (" + EXIF focal" if focal and not args.solve_focal else "")
        except ValueError:
            fallback = focal or 0.9 * max(w, h)
            return solve_camera(w, h, vx, vz, vy, focal_px=fallback, camera_height=1.0), \
                "vanishing points + " + ("EXIF focal" if focal else "typical phone focal")
    focal = focal or 0.9 * max(w, h)
    return level_camera(w, h, focal, args.pitch_deg), f"level camera fallback (pitch {args.pitch_deg} deg)"


def _pt(v, scale: float = 1.0) -> np.ndarray | None:
    return None if v is None else np.asarray(v, dtype=np.float64) * scale


def run(args) -> dict:
    out = Path(args.out) if args.out else Path(args.data_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    ui = not args.no_ui

    photo = load_photo(args.photo)
    from ..cli import preprocess  # same undistortion as the scene photo
    img = preprocess(photo, {"lensfun": not args.no_lensfun, "k1": args.k1, "k2": args.k2,
                             "focal_px": args.focal_px or photo.focal_px() or 0.9 * max(photo.width, photo.height)})

    # everything after this works in the shrunk photo's pixels, command line clicks included
    shrink = min(1.0, args.max_size / max(img.shape[:2])) if args.max_size else 1.0
    if shrink < 1.0:
        img = cv2.resize(img.astype(np.float32), (round(img.shape[1] * shrink), round(img.shape[0] * shrink)),
                         interpolation=cv2.INTER_AREA).astype(img.dtype)
    exif_focal = photo.focal_px() * shrink if photo.focal_px() else None
    if args.focal_px:
        args.focal_px *= shrink

    # --- segment
    rect = tuple(int(round(v * shrink)) for v in args.rect) if args.rect else None
    alpha = None
    if args.mode == "generated" and not (args.mask or args.chroma):
        if rect is None and ui:
            rect = annotate.select_rect(img)
        alpha = salient_alpha(img, rect)
        mask = alpha > 0.5
    elif args.mask:
        mask = np.asarray(Image.open(args.mask).convert("L")) > 127
    elif args.chroma:
        mask = chroma_key(img, args.chroma)
    else:
        if rect is None:
            if not ui:
                sys.exit("give --rect x y w h (or --mask / --chroma)")
            rect = annotate.select_rect(img)
        mask = grabcut(img, rect)
        if ui and args.refine:
            mask = annotate.refine_mask(img, mask, rect)
    if not mask.any():
        sys.exit("the mask is empty")

    # --- generated: TripoSR builds the whole shape and texture, no camera or clicks needed
    if args.mode == "generated":
        return _run_generated(args, out, img, mask, alpha, shrink)

    # --- inflate: the older rounded-silhouette mesh, no camera needed
    if args.mode == "inflate":
        height = args.height or (annotate.ask_float("Real height in metres: ") if ui and sys.stdin.isatty() else None)
        if not height:
            sys.exit("give --height in metres")
        obj = object3d.build_object(img, mask, height_m=height, depth_ratio=args.depth_ratio,
                                    delight=args.delight)
        object3d.export_glb(obj, out / "object.glb")
        save_jpg(out / "object_mask.jpg", mask.astype(np.uint8) * 255, linear=False)
        meta = {"mode": "inflate", "measurement": {"height_m": height}, "photo": str(args.photo)}
        write_meta(out / "object_meta.json", meta)
        _place_in_scene(out)
        print(f"wrote {out / 'object.glb'} (inflate, {height} m tall)")
        return meta

    cam, how = solve_object_camera(img, args, exif_focal)
    print(f"camera: {how}, f = {cam.focal_px:.1f} px")

    # --- clicks
    base_uv = _pt(args.base, shrink)
    if base_uv is None:
        base_uv = annotate.click_points(img, ["ground contact"], mask)[0] if ui and args.click else mask_bottom_center(mask)
    base = base_point(cam, base_uv)
    clicks = {"ground_contact": base_uv}

    if args.mode == "card":
        mesh = build_card(mask, cam, base)
    elif args.mode == "revolution":
        top_uv = _pt(args.top, shrink)
        if top_uv is None:
            top_uv = annotate.click_points(img, ["top of the axis"], mask)[0] if ui and args.click else mask_top_above(mask, base_uv)
        clicks["axis_top"] = top_uv
        h = height_from_click(cam, base, top_uv)
        mesh = build_revolution(mask, cam, base, base + np.array([0.0, h, 0.0]), args.n_heights, args.n_segments)
    elif args.mode == "box":
        corners = np.asarray(args.corners, float).reshape(-1, 2) * shrink if args.corners else None
        top_uv = _pt(args.top, shrink)
        if corners is None or top_uv is None:
            if not ui:
                sys.exit("box mode needs --corners (2-4 floor corners) and --top (a top corner)")
            pts = annotate.click_points(img, [f"floor corner {i}" for i in range(1, 5)], mask, optional_after=2)
            corners = np.array(pts)
            top_uv = annotate.click_points(img, ["top corner above a floor corner"], mask)[0]
        clicks["floor_corners"], clicks["top_corner"] = corners, top_uv
        mesh = build_box(cam, corners, top_uv)
        base = np.asarray(mesh.vertices).mean(axis=0) * [1, 0, 1]
    else:  # extrude
        mesh = build_extrude(mask, cam, base, thickness=None)

    # --- texture
    tex_img = delight(img, mask) if args.delight else img
    cutout = args.mode in ("card", "extrude")
    if args.mode == "card":
        uv_px = projective_uv(cam, mesh.vertices)
    else:
        uv_px = back_side_fill(cam, mesh, base)
    tex = crop_texture(tex_img, soft_alpha(mask) if cutout else mask.astype(np.float32), uv_px, bleed=not cutout)

    # --- check: reproject and compare with the mask
    sil = check.silhouette(cam, mesh)
    score = check.iou(sil, mask)
    save_jpg(out / "object_check.jpg", check.overlay(img, mask, sil), linear=False)
    print(f"reprojection IoU {score:.3f} (green = mask, red = mesh; see object_check.jpg)")
    if score < 0.8:
        print("  low IoU: check the ground contact / axis clicks or the camera solve")

    # --- normalize + export
    height, width = args.height, args.width
    if not (height or width):
        if not (ui and sys.stdin.isatty()):
            sys.exit("give --height or --width in metres")
        height = annotate.ask_float("Real height in metres: ")
    world_mesh = mesh
    mesh, scale = normalize(mesh, base, height=height, width=width)
    if args.glass:
        export_glass_glb(mesh, out / "object.glb")
    else:
        export_glb(mesh, tex, out / "object.glb", alpha_cutout=cutout, double_sided=args.mode == "card")
    save_jpg(out / "object_mask.jpg", mask.astype(np.uint8) * 255, linear=False)

    meta = {
        "mode": args.mode,
        "glass": bool(args.glass),
        "photo": str(args.photo),
        "working_scale": shrink,
        "measurement": {"height_m": height} if height else {"width_m": width},
        "clicks_px": clicks,
        "rect": rect,
        "camera": {"how": how, **cam.to_dict(), "image_size": [cam.width, cam.height]},
        "scale_units_to_m": scale,
        "reprojection_iou": score,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }
    if args.compare:
        import trimesh
        truth = trimesh.load(args.compare, force="mesh")
        meta["compare"] = check.compare_meshes(mesh, truth)
        c = meta["compare"]
        print(f"vs {args.compare}: height error {c['height_error_pct']:.2f}%, "
              f"Chamfer {c['chamfer_m'] * 100:.2f} cm ({c['chamfer_pct_of_height']:.2f}% of height)")
    write_meta(out / "object_meta.json", meta)
    _place_in_scene(out)
    size = np.ptp(mesh.vertices, axis=0)
    print(f"wrote {out / 'object.glb'} ({args.mode}, {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m)")
    meta["_world_mesh"] = world_mesh  # for tests; not written
    meta["_camera"] = cam
    return meta


def _run_generated(args, out: Path, img: np.ndarray, mask: np.ndarray, alpha, shrink: float) -> dict:
    from .modes.generated import build_generated

    height, width = args.height, args.width
    if not (height or width):
        sys.exit("give --height or --width in metres")
    a = alpha if alpha is not None else mask.astype(np.float32)
    mesh, tex, info = build_generated(to_srgb8(img), a, remove_lighting=not args.keep_shading)
    lo, hi = mesh.bounds
    base = np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])
    mesh, scale = normalize(mesh, base, height=height, width=width)
    if args.glass:
        export_glass_glb(mesh, out / "object.glb")
    else:
        export_glb(mesh, tex, out / "object.glb")
    save_jpg(out / "object_mask.jpg", mask.astype(np.uint8) * 255, linear=False)
    meta = {"mode": "generated", "glass": bool(args.glass), "photo": str(args.photo), "working_scale": shrink,
            "measurement": {"height_m": height} if height else {"width_m": width}, "model": info,
            "scale_units_to_m": scale, "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
    write_meta(out / "object_meta.json", meta)
    _place_in_scene(out)
    size = np.ptp(mesh.vertices, axis=0)
    print(f"wrote {out / 'object.glb'} (generated on {info['device']}, {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m)")
    return meta


def _in_view(cam: Camera, point: np.ndarray) -> bool:
    local = (np.asarray(point, dtype=np.float64) - cam.position) @ cam.R.T
    if local[2] <= 0.1:
        return False
    u, v = cam.project(np.asarray(point, dtype=np.float64)[None])[0]
    return 0 <= u < cam.width and 0 <= v < cam.height


def _first_hit(origin: np.ndarray, d: np.ndarray, boxes, walls) -> tuple[float, bool]:
    """Distance to the first floor / box / wall along the ray, and whether that surface faces up."""
    best, up = np.inf, False
    if d[1] < -1e-6:
        best, up = -origin[1] / d[1], True
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - origin) / d, (hi - origin) / d
        entry = np.minimum(t1, t2)
        near, far = np.nanmax(entry), np.nanmin(np.maximum(t1, t2))
        if far >= max(near, 0.0) and 0.0 < near < best:
            best, up = near, int(np.nanargmax(entry)) == 1 and d[1] < 0
    for p0, n in walls:
        facing = float(d @ n)
        if facing < -1e-6:
            t = float((p0 - origin) @ n) / facing
            if 0.0 < t < best:
                best, up = t, False
    return best, up


def _support_spot(cam: Camera, boxes: list[tuple[np.ndarray, np.ndarray]],
                  walls: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The first floor or box top the camera sees, searched from the bottom-middle of the photo outwards,
    or the floor 2.5 m in front of the camera when nothing to stand on is in view."""
    for v in np.arange(0.95, 0.45, -0.02):
        for u in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9):
            d = cam.rays(np.array([[cam.width * u, cam.height * v]]))[0]
            t, up = _first_hit(cam.position, d, boxes, walls)
            spot = cam.position + d * t
            if up and 0.8 < np.hypot(*(spot - cam.position)[[0, 2]]) < 15.0:
                return spot
    forward = cam.R[2] * [1.0, 0.0, 1.0]
    length = np.linalg.norm(forward)
    forward = forward / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    return cam.position * [1.0, 0.0, 1.0] + forward * 2.5


def _place_in_scene(out: Path) -> None:
    """If this folder has a scene.json, point it at object.glb. A new object (or one the camera can't see)
    stands on the floor or a box top in view; one the engine placed in view stays where it is."""
    path = out / "scene.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    cam = Camera.from_dict(data["camera"], data["image_size"])
    placed = data.get("object")
    if placed and _in_view(cam, placed["transform"]["position"]):
        placed["path"] = "object.glb"
    else:
        boxes = [(np.asarray(p["min"], float), np.asarray(p["max"], float))
                 for p in data.get("proxies", []) if p.get("type") == "box"]
        walls = [(np.asarray(p["point"], float), np.asarray(p["normal"], float))
                 for p in data.get("proxies", []) if p.get("type") == "plane" and abs(p["normal"][1]) < 0.5]
        spot = _support_spot(cam, boxes, walls)
        placed = ObjectPlacement("object.glb", (float(spot[0]), float(spot[1]), float(spot[2]))).to_dict()
        # the photographed side (+Z in object.glb) faces the camera
        to_cam = cam.position - spot
        placed["transform"]["rotation_y_deg"] = round(float(np.degrees(np.arctan2(to_cam[0], to_cam[2]))), 2)
    update_scene_file(path, object=placed)


def build_parser(prog: str = "python -m recast.object") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo")
    p.add_argument("--mode", choices=MODES, default="card")
    p.add_argument("--out", help="scene folder to write into (e.g. data/room1/)")
    p.add_argument("--name", help="scene folder name under data/ (instead of --out)")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    g = p.add_argument_group("measurement (sets the real scale)")
    g.add_argument("--height", type=float, help="real height in metres")
    g.add_argument("--width", type=float, help="real width in metres (instead of height)")
    g = p.add_argument_group("segmentation")
    g.add_argument("--rect", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    g.add_argument("--mask", help="ready-made cutout (white = object)")
    g.add_argument("--chroma", choices=("green", "blue"), help="backdrop colour to key out")
    g.add_argument("--refine", action="store_true", help="open the brush to fix the cutout")
    g.add_argument("--max-size", type=int, default=1600, help="shrink the photo to this many pixels on its long side (0 = full size)")
    g = p.add_argument_group("clicks (pixels in the photo)")
    g.add_argument("--click", action="store_true", help="click the ground contact / axis top instead of taking them from the cutout")
    g.add_argument("--base", type=float, nargs=2, metavar=("U", "V"), help="ground contact")
    g.add_argument("--top", type=float, nargs=2, metavar=("U", "V"), help="revolution: axis top; box: a top corner")
    g.add_argument("--corners", type=float, nargs="+", help="box: 2-4 floor corners as u v u v ...")
    g = p.add_argument_group("camera")
    g.add_argument("--lines", help='JSON {"x": [[x1,y1,x2,y2],...], "z": [...], "y": [...]} of the floor/table')
    g.add_argument("--no-auto-lines", action="store_true")
    g.add_argument("--focal-px", type=float)
    g.add_argument("--trust-exif", action="store_true", help="(default now) use the EXIF focal length when there is one")
    g.add_argument("--solve-focal", action="store_true", help="solve the focal length from vanishing points even with EXIF")
    g.add_argument("--pitch-deg", type=float, default=0.0, help="fallback camera tilt if no vanishing points")
    g.add_argument("--k1", type=float, default=0.0)
    g.add_argument("--k2", type=float, default=0.0)
    g.add_argument("--no-lensfun", action="store_true")
    g = p.add_argument_group("shape")
    g.add_argument("--n-heights", type=int, default=64)
    g.add_argument("--n-segments", type=int, default=48)
    g.add_argument("--depth-ratio", type=float, default=0.5, help="inflate only")
    g.add_argument("--delight", action="store_true", help="flatten baked-in shading (rough)")
    g.add_argument("--glass", action="store_true", help="clear glass instead of the photo's colours (bottles, glasses)")
    g.add_argument("--keep-shading", action="store_true", help="generated: keep the photo's lighting in the texture")
    p.add_argument("--compare", help="ground-truth .glb (normalized: metres, pivot at base) to score against")
    p.add_argument("--no-ui", action="store_true", help="never open windows; use defaults or fail")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.out or args.name):
        sys.exit("give --out DIR or --name SCENE")
    run(args)
