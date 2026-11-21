"""Stage D: compare the engine's composite against a ground-truth photo.

The ground truth is a photo with the real object physically placed in the
scene, taken from the same tripod position. Scores are PSNR and SSIM, over the
whole image and inside the object's region (object plus its shadow, which is
where the render differs from the empty-room photo).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class Scores:
    psnr: float
    ssim: float
    region_psnr: float | None
    region_ssim: float | None
    region_pixels: int

    def to_dict(self) -> dict:
        return asdict(self)


def change_mask(render: np.ndarray, empty_photo: np.ndarray, threshold: float = 0.02,
                grow_px: int = 8) -> np.ndarray:
    """Pixels where the render differs from the empty room: the object and its shadow."""
    diff = np.abs(np.asarray(render, np.float64) - np.asarray(empty_photo, np.float64)).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    if grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def score(render: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> Scores:
    """render/truth: (H, W, 3) in [0, 1], same colour space (sRGB as displayed)."""
    render = np.clip(np.asarray(render, np.float64), 0, 1)
    truth = np.clip(np.asarray(truth, np.float64), 0, 1)
    if render.shape != truth.shape:
        raise ValueError(f"size mismatch: render {render.shape} vs truth {truth.shape}")

    psnr = peak_signal_noise_ratio(truth, render, data_range=1.0)
    ssim = structural_similarity(truth, render, data_range=1.0, channel_axis=-1)

    r_psnr = r_ssim = None
    n = 0
    if region is not None and region.any():
        n = int(region.sum())
        mse = np.mean((truth[region] - render[region]) ** 2)
        r_psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
        # SSIM needs a window, so score the region's bounding box, averaging only masked pixels
        ys, xs = np.nonzero(region)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if min(y1 - y0, x1 - x0) >= 7:
            _, full = structural_similarity(truth[y0:y1, x0:x1], render[y0:y1, x0:x1], data_range=1.0,
                                            channel_axis=-1, full=True)
            r_ssim = float(full.mean(axis=-1)[region[y0:y1, x0:x1]].mean())
    return Scores(float(psnr), float(ssim), r_psnr, r_ssim, n)
