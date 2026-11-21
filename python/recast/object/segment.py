"""Object mask: GrabCut from a dragged rectangle (with brush fixes), or chroma key."""

from __future__ import annotations

import cv2
import numpy as np

from ..io import linear_to_srgb
from ..object3d import clean_mask

KEY_HUES = {"green": (35, 85), "blue": (90, 130)}  # OpenCV hue is 0-179


def _bgr8(img_linear: np.ndarray) -> np.ndarray:
    srgb8 = (np.clip(linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(srgb8, cv2.COLOR_RGB2BGR)


def grabcut(img_linear: np.ndarray, rect=None, fg_strokes: np.ndarray | None = None,
            bg_strokes: np.ndarray | None = None, init: np.ndarray | None = None,
            iterations: int = 5) -> np.ndarray:
    """GrabCut. Start from `rect` (x, y, w, h), and/or refine `init` with brush strokes.

    fg_strokes / bg_strokes: bool masks the user painted as definitely object /
    definitely background.
    """
    bgr = _bgr8(img_linear)
    h, w = bgr.shape[:2]
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    if init is None and fg_strokes is None and bg_strokes is None:
        if rect is None:
            raise ValueError("need a rectangle or a starting mask")
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(bgr, gc, tuple(int(v) for v in rect), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    else:
        if init is not None:
            gc = np.where(init, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        else:
            gc = np.full((h, w), cv2.GC_BGD, np.uint8)
            x, y, rw, rh = (int(v) for v in rect)
            gc[y:y + rh, x:x + rw] = cv2.GC_PR_FGD
        if fg_strokes is not None:
            gc[fg_strokes] = cv2.GC_FGD
        if bg_strokes is not None:
            gc[bg_strokes] = cv2.GC_BGD
        cv2.grabCut(bgr, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    return clean_mask((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))


def chroma_key(img_linear: np.ndarray, key: str = "green", min_saturation: int = 60,
               min_value: int = 40) -> np.ndarray:
    """Object = everything that isn't the backdrop colour (HSV threshold + cleanup)."""
    if key not in KEY_HUES:
        raise ValueError(f"key must be one of {list(KEY_HUES)}")
    hsv = cv2.cvtColor(_bgr8(img_linear), cv2.COLOR_BGR2HSV)
    lo, hi = KEY_HUES[key]
    backdrop = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) &
                (hsv[..., 1] >= min_saturation) & (hsv[..., 2] >= min_value))
    m = (~backdrop).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return clean_mask(m.astype(bool))


def soft_alpha(mask: np.ndarray, feather_px: float = 1.5) -> np.ndarray:
    """Feathered edge in [0, 1] for the texture's alpha channel."""
    if feather_px <= 0:
        return mask.astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_px), 0.0, 1.0)
