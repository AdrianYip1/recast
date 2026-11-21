import numpy as np
from PIL import Image

from recast.io import linear_to_srgb, load_hdr, load_photo, save_hdr, srgb_to_linear


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_srgb_known_values():
    assert np.isclose(srgb_to_linear(np.float32(0.5)), 0.21404, atol=1e-4)
    assert srgb_to_linear(np.float32(0.0)) == 0.0
    assert np.isclose(srgb_to_linear(np.float32(1.0)), 1.0)


def test_load_photo_is_linear(tmp_path):
    img = np.full((8, 12, 3), 128, np.uint8)
    p = tmp_path / "grey.png"
    Image.fromarray(img).save(p)
    photo = load_photo(p)
    assert photo.pixels.shape == (8, 12, 3)
    assert np.allclose(photo.pixels, srgb_to_linear(np.float32(128 / 255)), atol=1e-5)
    assert photo.focal_px() is None


def test_focal_px_from_35mm():
    from recast.io import Photo
    # 36x24 frame at 3600x2400 px: a 36mm-equivalent lens is 3600 px
    photo = Photo(np.zeros((2400, 3600, 3), np.float32), focal_35mm=36.0)
    assert np.isclose(photo.focal_px(), 3600, rtol=1e-3)


def test_hdr_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    img = (rng.random((6, 5, 3)) * 4).astype(np.float32)
    p = tmp_path / "x.hdr"
    save_hdr(p, img)
    back = load_hdr(p)
    # RGBE: 8-bit mantissas sharing one exponent, so error scales with the brightest channel
    tol = img.max(axis=-1, keepdims=True) / 64
    assert (np.abs(back - img) <= tol).all()


def test_jpg_roundtrip_is_close(tmp_path):
    from recast.io import load_image_linear, save_jpg
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.dstack([xx / 63, yy / 63, np.full_like(xx, 0.5, dtype=float)]).astype(np.float32) ** 2.2
    p = tmp_path / "x.jpg"
    save_jpg(p, img)
    assert Image.open(p).format == "JPEG"
    back = load_image_linear(p)
    assert back.shape == img.shape
    assert np.abs(back - img).mean() < 0.01


def test_load_image_linear_reads_hdr_and_jpg(tmp_path):
    from recast.io import load_image_linear, save_jpg
    img = np.full((8, 8, 3), 0.25, np.float32)
    save_hdr(tmp_path / "a.hdr", img)
    save_jpg(tmp_path / "a.jpg", img)
    assert np.allclose(load_image_linear(tmp_path / "a.hdr"), 0.25, atol=0.01)
    assert np.allclose(load_image_linear(tmp_path / "a.jpg"), 0.25, atol=0.01)
