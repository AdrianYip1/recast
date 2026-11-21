"""Reading/writing gbuffer/ (the engine renders it through the photo's camera).

Proposed encoding (Radiance .hdr can't hold negative numbers):

    normals.hdr   world-space normal n stored as n * 0.5 + 0.5, (0, 0, 0) = no proxy
    ids.png       16-bit greyscale, proxy id + 1, 0 = no proxy
    depth.hdr     distance along the camera ray in metres (same value in R, G, B), 0 = no proxy
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import load_hdr, save_hdr


def save_gbuffer(folder: str | Path, normals: np.ndarray, ids: np.ndarray, depth: np.ndarray) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hit = ids >= 0
    enc = np.where(hit[..., None], normals * 0.5 + 0.5, 0.0).astype(np.float32)
    save_hdr(folder / "normals.hdr", enc)
    cv2.imwrite(str(folder / "ids.png"), np.where(hit, ids + 1, 0).astype(np.uint16))
    d = np.where(hit, depth, 0.0).astype(np.float32)
    save_hdr(folder / "depth.hdr", np.repeat(d[..., None], 3, axis=-1))


def load_gbuffer(folder: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns normals (H, W, 3) unit or zero, ids (H, W) with -1 = none, depth (H, W)."""
    folder = Path(folder)
    raw_ids = cv2.imread(str(folder / "ids.png"), cv2.IMREAD_UNCHANGED)
    if raw_ids is None:
        raise IOError(f"missing {folder / 'ids.png'}")
    if raw_ids.ndim == 3:
        raw_ids = raw_ids[..., 0]
    ids = raw_ids.astype(np.int64) - 1

    n = load_hdr(folder / "normals.hdr") * 2.0 - 1.0
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(length > 0.5, n / np.maximum(length, 1e-9), 0.0)
    n[ids < 0] = 0.0

    depth = load_hdr(folder / "depth.hdr")[..., 0].astype(np.float64)
    depth[ids < 0] = np.inf
    return n, ids, depth
