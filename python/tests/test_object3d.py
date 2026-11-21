import numpy as np
import trimesh

from recast.io import luminance
from recast.object3d import build_object, clean_mask, cutout, export_glb, inflate


def _disk_mask(h=200, w=200, r=70):
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < r ** 2


def _lit_ball(h=200, w=200, r=70, colour=(0.6, 0.2, 0.1)):
    """Sphere lit from the upper left, on a grey background."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = (xx - w / 2) / r, -(yy - h / 2) / r
    inside = nx ** 2 + ny ** 2 < 1
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, None))
    light = np.array([-0.5, 0.5, 0.7]); light /= np.linalg.norm(light)
    shading = 0.15 + np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, None)
    img = np.full((h, w, 3), 0.18, np.float32)
    img[inside] = (np.array(colour) * shading[inside, None]).astype(np.float32)
    return img, inside


def test_clean_mask_fills_holes_and_drops_specks():
    m = _disk_mask()
    m[100, 100] = False           # hole
    m[95:105, 95:105] = False     # bigger hole
    m[5, 5] = True                # speck
    out = clean_mask(m)
    assert out[100, 100] and out[98, 98]
    assert not out[5, 5]


def test_inflate_is_closed_and_rounded():
    v, f, _ = inflate(_disk_mask(), height_m=0.3, depth_ratio=1.0, max_res=60)
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0  # outward-facing normals
    # sits on the floor, about 0.3m tall, and a disk inflates to roughly a sphere
    assert np.isclose(v[:, 1].min(), 0.0, atol=0.01)
    assert np.isclose(v[:, 1].max() - v[:, 1].min(), 0.3, rtol=0.05)
    depth = v[:, 2].max() - v[:, 2].min()
    assert 0.25 < depth < 0.33


def test_grabcut_finds_ball():
    img, inside = _lit_ball()
    mask = cutout(img, (20, 20, 160, 160))
    iou = (mask & inside).sum() / (mask | inside).sum()
    assert iou > 0.9


def test_delighting_flattens_shading():
    img, inside = _lit_ball()
    obj = build_object(img, inside, max_res=60)
    lit = luminance(img[inside])
    flat = luminance(obj.texture[obj.mask])
    # relative brightness spread should drop a lot once shading is divided out
    assert np.std(flat) / np.mean(flat) < 0.5 * np.std(lit) / np.mean(lit)
    # and the hue (red > green > blue) survives
    mean = obj.texture[obj.mask].mean(axis=0)
    assert mean[0] > mean[1] > mean[2]


def test_export_glb_loads_back(tmp_path):
    img, inside = _lit_ball()
    obj = build_object(img, inside, max_res=40)
    p = tmp_path / "ball.glb"
    export_glb(obj, p)
    loaded = trimesh.load(p, force="mesh")
    assert len(loaded.faces) == len(obj.faces)
    assert loaded.visual.uv is not None
    assert np.isclose(loaded.bounds[1, 1] - loaded.bounds[0, 1], 0.3, rtol=0.05)
