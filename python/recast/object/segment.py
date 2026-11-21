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
    # JPEG ringing around edges reads as "not background" to GrabCut's colour model and
    # the mask bleeds outward; a 5x5 median removes the speckle but keeps edges.
    # (Only GrabCut sees the filtered image; the texture uses the original.)
    bgr = cv2.medianBlur(_bgr8(img_linear), 5)
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


SAM_MODEL = "facebook/sam-vit-base"   # Apache-2.0
_SAM = None


def _sam():
    """SAM (Segment Anything) on the GPU when there is one, loaded once and kept (the worker preloads it)."""
    global _SAM
    if _SAM is None:
        import torch
        from transformers import SamModel, SamProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def load(cls):
            try:
                return cls.from_pretrained(SAM_MODEL, local_files_only=True)
            except OSError:
                print(f"downloading {SAM_MODEL} (once)")
                return cls.from_pretrained(SAM_MODEL)
        _SAM = (load(SamProcessor), load(SamModel).to(device).eval(), device)
    return _SAM


def salient_alpha(img_linear: np.ndarray, rect=None) -> np.ndarray:
    """Soft cutout (H, W) in [0, 1] of the object inside `rect` (x, y, w, h; the whole photo when None), from SAM
    prompted with that box. About 0.3 s on a GPU, and the same cutout as BiRefNet (IoU 0.98 on a bottle photo)."""
    import torch

    h, w = img_linear.shape[:2]
    x, y, bw, bh = rect if rect is not None else (0, 0, w, h)
    processor, model, device = _sam()
    srgb8 = (np.clip(linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8)
    inputs = processor(images=srgb8, input_boxes=[[[float(x), float(y), float(x + bw), float(y + bh)]]], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    logits = processor.image_processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu(),
                                                          inputs["reshaped_input_sizes"].cpu(), binarize=False)[0][0, 0]
    return torch.sigmoid(logits).numpy().astype(np.float32)
