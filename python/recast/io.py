"""Loading photos, reading EXIF, and converting between sRGB and linear colour.

Everything downstream (lighting fit, compositing) works in linear RGB float32,
values roughly in [0, 1] for a normally exposed photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}

# EXIF tag ids
_MAKE = 0x010F
_MODEL = 0x0110
_EXIF_IFD = 0x8769
_FNUMBER = 0x829D
_FOCAL_LENGTH = 0x920A
_FOCAL_35MM = 0xA405
_LENS_MAKE = 0xA433
_LENS_MODEL = 0xA434


@dataclass
class Photo:
    """A photo in linear RGB with whatever we could learn about the camera."""

    pixels: np.ndarray  # (H, W, 3) float32, linear RGB
    focal_mm: float | None = None
    focal_35mm: float | None = None
    exif: dict | None = None  # make, model, lens_make, lens_model, f_number (strings/floats, may be missing)

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    def focal_px(self, sensor_width_mm: float | None = None) -> float | None:
        """Focal length in pixels.

        With the sensor width: f_px = f_mm / sensor_width_mm * image_width.
        Otherwise from the 35mm-equivalent focal length, scaled by the diagonal
        (a 35mm frame's diagonal is 43.27mm) so it works for any aspect ratio.
        """
        if sensor_width_mm and self.focal_mm:
            return self.focal_mm / sensor_width_mm * self.width
        if not self.focal_35mm:
            return None
        diag_px = float(np.hypot(self.width, self.height))
        return self.focal_35mm * diag_px / 43.2666


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, None)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of a linear RGB image."""
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _read_exif(img: Image.Image) -> tuple[float | None, float | None, dict]:
    exif = img.getexif()
    if not exif:
        return None, None, {}
    tags = dict(exif)
    try:
        tags.update(exif.get_ifd(_EXIF_IFD))
    except Exception:
        pass

    def text(tag):
        v = tags.get(tag)
        return str(v).strip("\x00 ").strip() or None if v is not None else None

    focal = tags.get(_FOCAL_LENGTH)
    focal35 = tags.get(_FOCAL_35MM)
    fnum = tags.get(_FNUMBER)
    info = {
        "make": text(_MAKE),
        "model": text(_MODEL),
        "lens_make": text(_LENS_MAKE),
        "lens_model": text(_LENS_MODEL),
        "f_number": float(fnum) if fnum else None,
    }
    return (float(focal) if focal else None, float(focal35) if focal35 else None, info)


def load_photo(path: str | Path) -> Photo:
    """Load a JPG/PNG/TIFF (or camera RAW if rawpy is installed) as linear RGB."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw(path)

    with Image.open(path) as img:
        focal, focal35, info = _read_exif(img)
        # respect the orientation flag so lines/vanishing points are in the right place
        orient = img.getexif().get(ExifTags.Base.Orientation, 1)
        img = img.convert("RGB")
        if orient == 3:
            img = img.rotate(180, expand=True)
        elif orient == 6:
            img = img.rotate(270, expand=True)
        elif orient == 8:
            img = img.rotate(90, expand=True)
        srgb = np.asarray(img, dtype=np.float32) / 255.0

    return Photo(srgb_to_linear(srgb), focal, focal35, info)


def _load_raw(path: Path) -> Photo:
    try:
        import rawpy
    except ImportError as e:
        raise ImportError("install rawpy to load RAW files: pip install -e .[raw]") from e

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
    pixels = rgb.astype(np.float32) / 65535.0

    # rawpy doesn't expose EXIF, but Pillow can usually read it from the same file
    focal = focal35 = None
    info: dict = {}
    try:
        with Image.open(path) as img:
            focal, focal35, info = _read_exif(img)
    except Exception:
        pass
    return Photo(pixels, focal, focal35, info)


def save_hdr(path: str | Path, linear_rgb: np.ndarray) -> None:
    """Write linear RGB as Radiance .hdr (readable from C++ with stbi_loadf)."""
    bgr = cv2.cvtColor(np.asarray(linear_rgb, dtype=np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"could not write {path}")


def load_hdr(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_png(path: str | Path, linear_rgb: np.ndarray) -> None:
    """Write linear RGB as an 8-bit sRGB PNG (for previews)."""
    srgb = np.clip(linear_to_srgb(linear_rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(srgb).save(path)
