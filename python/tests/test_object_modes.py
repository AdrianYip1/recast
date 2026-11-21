"""Ground-truth tests: render a known mesh from a known camera, rebuild it from the
"photo" with the same clicks a user would make, and compare.

This is the Python stand-in for "load a known GLB in the engine, screenshot it".
"""

import json

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import Polygon

from recast.io import save_jpg
from recast.object import check
from recast.object.cli import main as object_main
from recast.object.export import export_glb, normalize
from recast.object.modes import build_box, build_card, build_extrude, build_revolution
from recast.object.modes.revolution import revolve
from recast.object.placement import UP, base_point, facing_frame, height_from_click
from recast.object.texture import back_side_fill, crop_texture, projective_uv
from synthetic import look_at_camera, room_segments


def _cam():
    return look_at_camera(width=640, height=480, focal_px=600.0, yaw_deg=25, pitch_deg=-22, camera_height=1.0)


def _spot(cam, dist=2.2):
    fwd = cam.R[2].copy()
    fwd[1] = 0
    fwd /= np.linalg.norm(fwd)
    p = cam.position + dist * fwd
    p[1] = 0.0
    return p


def _vase(base):
    h = np.linspace(0, 0.5, 40)
    r = 0.11 + 0.05 * np.sin(np.pi * h / 0.5) - 0.03 * (h / 0.5) ** 2
    return revolve(h, r, base, 64)


def _normalized(mesh, base, height):
    return normalize(mesh, base, height=height)[0]


def _score(pred_world, truth_world, base_pred, base_true, height):
    pred = _normalized(pred_world, base_pred, height)
    truth = _normalized(truth_world, base_true, height)
    return check.compare_meshes(pred, truth)


def test_height_from_click_is_exact():
    cam = _cam()
    base = _spot(cam)
    top = base + [0, 0.37, 0]
    assert np.isclose(height_from_click(cam, base, cam.project(top[None])[0]), 0.37, atol=1e-9)
    assert np.allclose(base_point(cam, cam.project(base[None])[0]), base, atol=1e-9)


@pytest.mark.parametrize("shape", ["cylinder", "vase"])
def test_revolution(shape):
    cam = _cam()
    base = _spot(cam)
    if shape == "cylinder":
        truth = trimesh.creation.cylinder(radius=0.15, height=0.5, sections=64)
        truth.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        truth.apply_translation(base + [0, 0.25, 0])
    else:
        truth = _vase(base)
    _, mask = check.render_mesh(cam, truth)

    base_uv = cam.project(base[None])[0]
    top_uv = cam.project((base + [0, np.ptp(truth.vertices[:, 1]), 0])[None])[0]
    b = base_point(cam, base_uv)
    top = b + [0, height_from_click(cam, b, top_uv), 0]
    pred = build_revolution(mask, cam, b, top)

    assert pred.is_watertight and pred.volume > 0
    iou = check.iou(check.silhouette(cam, pred), mask)
    s = _score(pred, truth, b, base, 0.5)
    assert iou > 0.95, iou
    assert s["chamfer_pct_of_height"] < 2.0, s


def test_box():
    cam = _cam()
    base = _spot(cam)
    lo = base + [-0.2, 0, -0.15]
    hi = base + [0.2, 0.3, 0.15]
    truth = trimesh.creation.box(extents=hi - lo)
    truth.apply_translation((lo + hi) / 2)
    _, mask = check.render_mesh(cam, truth)

    floor = np.array([[lo[0], 0, lo[2]], [hi[0], 0, lo[2]], [hi[0], 0, hi[2]], [lo[0], 0, hi[2]]])
    # the 3 floor corners nearest the camera are the visible ones
    vis = floor[np.argsort(np.linalg.norm(floor - cam.position, axis=1))[:3]]
    corners_uv = cam.project(vis)
    top_uv = cam.project((vis[0] + [0, 0.3, 0])[None])[0]
    pred = build_box(cam, corners_uv, top_uv)

    assert np.allclose(pred.bounds, truth.bounds, atol=1e-6)
    assert check.iou(check.silhouette(cam, pred), mask) > 0.99


def test_box_from_one_edge_assumes_square():
    cam = _cam()
    base = _spot(cam)
    a, b = base + [-0.2, 0, -0.2], base + [0.2, 0, -0.2]
    pred = build_box(cam, cam.project(np.array([a, b])), cam.project((a + [0, 0.3, 0])[None])[0])
    ext = pred.extents
    assert np.isclose(ext[0], 0.4, atol=1e-6) and np.isclose(ext[2], 0.4, atol=1e-6)
    assert np.isclose(ext[1], 0.3, atol=1e-6)


def test_card_stands_on_base_and_covers_mask():
    cam = _cam()
    base = _spot(cam)
    tangent, normal = facing_frame(cam, base)
    # truth: a flat disc standing up, facing the camera
    disc = trimesh.creation.cylinder(radius=0.2, height=0.01, sections=64)
    R = np.column_stack([tangent, UP, normal])  # cylinder axis (z) -> facing normal
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = base + [0, 0.2, 0]
    disc.apply_transform(T)
    _, mask = check.render_mesh(cam, disc)

    card = build_card(mask, cam, base_point(cam, cam.project(base[None])[0]))
    v = card.vertices
    assert np.isclose(v[:, 1].min(), 0.0)
    assert abs(np.ptp(v[:, 1]) - 0.4) < 0.02           # as tall as the disc
    # at least as wide; a bit wider is expected (it must cover the whole mask under
    # perspective, and the alpha cuts the rest away)
    assert 0.4 <= np.ptp(v @ tangent) < 0.46
    # covers the mask; the disc is 1 cm thick, so its front edge dips a pixel below
    # the floor contact row, which a card standing on y = 0 can't reach
    missed = mask & ~check.silhouette(cam, card)
    assert missed.sum() <= 0.002 * mask.sum()
    assert np.nonzero(missed)[0].min(initial=10**9) >= np.nonzero(mask)[0].max() - 1
    assert card.face_normals[0] @ normal > 0.999         # faces the camera


def test_extrude():
    cam = _cam()
    base = _spot(cam)
    tangent, normal = facing_frame(cam, base)
    # truth: an L-shaped sign, 10% of its width thick, standing on the floor facing the camera
    poly = Polygon([(-0.2, 0), (0.2, 0), (0.2, 0.12), (-0.05, 0.12), (-0.05, 0.4), (-0.2, 0.4)])
    t = 0.04
    m = trimesh.creation.extrude_polygon(poly, t)
    w = base * [1, 0, 1] + m.vertices[:, :1] * tangent + m.vertices[:, 1:2] * UP + (m.vertices[:, 2:3] - t / 2) * normal
    truth = trimesh.Trimesh(w, m.faces)
    _, mask = check.render_mesh(cam, truth)

    pred = build_extrude(mask, cam, base_point(cam, cam.project(base[None])[0]))
    assert pred.is_watertight
    assert check.iou(check.silhouette(cam, pred), mask) > 0.93
    s = _score(pred, truth, base, base, 0.4)
    assert s["chamfer_pct_of_height"] < 3.0, s


def test_projective_uv_and_back_fill():
    cam = _cam()
    base = _spot(cam)
    truth = _vase(base)
    uv = projective_uv(cam, truth.vertices)
    assert np.allclose(uv, cam.project(truth.vertices))
    filled = back_side_fill(cam, truth, base)
    _, normal = facing_frame(cam, base)
    front = np.einsum("ij,ij->i", truth.vertex_normals, cam.position - truth.vertices) > 0
    assert np.allclose(filled[front], uv[front])
    # back vertices land inside the object's silhouette (they show the mirrored front)
    _, mask = check.render_mesh(cam, truth)
    px = np.round(filled[~front]).astype(int)
    assert mask[px[:, 1], px[:, 0]].mean() > 0.95


def test_card_glb_has_cutout_material(tmp_path):
    cam = _cam()
    img, mask = check.render_mesh(cam, _vase(_spot(cam)))
    card = build_card(mask, cam, _spot(cam))
    tex = crop_texture(img, mask.astype(np.float32), projective_uv(cam, card.vertices), bleed=False)
    export_glb(normalize(card, _spot(cam), height=0.5)[0], tex, tmp_path / "c.glb", alpha_cutout=True,
               double_sided=True)
    raw = (tmp_path / "c.glb").read_bytes()
    gltf = json.loads(raw[20:20 + int.from_bytes(raw[12:16], "little")])
    mat = gltf["materials"][0]
    assert mat["alphaMode"] == "MASK" and mat["doubleSided"] is True
    loaded = trimesh.load(tmp_path / "c.glb", force="mesh")
    assert np.isclose(loaded.bounds[1, 1] - loaded.bounds[0, 1], 0.5, rtol=1e-6)


def test_cli_revolution_end_to_end(tmp_path):
    """photo -> object.glb + meta + check overlay, scored against the true mesh."""
    cam = _cam()
    base = _spot(cam)
    truth = _vase(base)
    img, mask = check.render_mesh(cam, truth)
    save_jpg(tmp_path / "vase.jpg", img)
    normalize(truth, base, height=0.5)[0].export(tmp_path / "truth.glb")

    # floor/table lines the user would trace, from the same camera
    segs, labels = room_segments(cam, np.random.default_rng(3), noise_px=0.2)
    (tmp_path / "lines.json").write_text(json.dumps(
        {"x": segs[labels == 0].tolist(), "y": segs[labels == 1].tolist(), "z": segs[labels == 2].tolist()}))

    ys, xs = np.nonzero(mask)
    rect = [xs.min() - 15, ys.min() - 15, np.ptp(xs) + 30, np.ptp(ys) + 30]
    base_uv = cam.project(base[None])[0]
    top_uv = cam.project((base + [0, 0.5, 0])[None])[0]
    out = tmp_path / "scene"
    object_main([str(tmp_path / "vase.jpg"), "--mode", "revolution", "--height", "0.5", "--out", str(out),
                 "--rect", *map(str, rect), "--base", *map(str, base_uv), "--top", *map(str, top_uv),
                 "--lines", str(tmp_path / "lines.json"), "--no-ui", "--no-lensfun",
                 "--compare", str(tmp_path / "truth.glb")])

    meta = json.loads((out / "object_meta.json").read_text())
    assert meta["mode"] == "revolution"
    assert meta["reprojection_iou"] > 0.9
    assert meta["compare"]["height_error_pct"] < 1.0
    assert meta["compare"]["chamfer_pct_of_height"] < 4.0
    assert (out / "object.glb").exists()
    for f in ("object_mask.jpg", "object_check.jpg"):
        assert Image.open(out / f).format == "JPEG"
    loaded = trimesh.load(out / "object.glb", force="mesh")
    assert np.isclose(loaded.bounds[0, 1], 0.0, atol=1e-6)
