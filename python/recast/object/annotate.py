"""OpenCV click UI: drag a box, fix the mask with a brush, click a few points.

Controls (shown in each window's title):
  rectangle : drag, then Enter/Space            (Esc cancels)
  fix mask  : left-drag = object, right-drag = background, R = re-run GrabCut,
              [ / ] = brush size, Enter = done
  points    : left-click to place, U = undo, Enter = done
Large photos are shown scaled down; all coordinates returned are full-size pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io import linear_to_srgb
from .segment import grabcut

MAX_SIDE = 1100
ENTER = (13, 10)
ESC = 27


def _display(img_linear: np.ndarray) -> tuple[np.ndarray, float]:
    srgb = (np.clip(linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR)
    s = min(1.0, MAX_SIDE / max(bgr.shape[:2]))
    if s < 1.0:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return bgr, s


def _tint(bgr: np.ndarray, mask_small: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[~mask_small] = (out[~mask_small] * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask_small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (0, 220, 0), 1)
    return out


def select_rect(img_linear: np.ndarray, title: str = "Drag a box around the object, then Enter") -> tuple:
    bgr, s = _display(img_linear)
    x, y, w, h = cv2.selectROI(title, bgr, showCrosshair=True)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        raise SystemExit("cancelled")
    return tuple(int(round(v / s)) for v in (x, y, w, h))


def refine_mask(img_linear: np.ndarray, mask: np.ndarray, rect=None) -> np.ndarray:
    """Brush fixes on top of GrabCut. Returns the final full-size mask."""
    title = "Fix mask: L-drag object, R-drag background, R key re-run, [ ] brush, Enter done"
    bgr, s = _display(img_linear)
    H, W = mask.shape
    fg = np.zeros((H, W), bool)
    bg = np.zeros((H, W), bool)
    state = {"mask": mask.copy(), "brush": max(3, int(0.01 * max(H, W))), "paint": None}

    def small(m):
        return cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    def paint(x, y, target):
        cx, cy = int(round(x / s)), int(round(y / s))
        stroke = np.zeros((H, W), np.uint8)
        cv2.circle(stroke, (cx, cy), state["brush"], 1, -1)
        target |= stroke.astype(bool)
        if target is fg:
            state["mask"] |= stroke.astype(bool)
        else:
            state["mask"] &= ~stroke.astype(bool)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["paint"] = fg
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["paint"] = bg
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["paint"] = None
        if state["paint"] is not None and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            paint(x, y, state["paint"])

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        cv2.imshow(title, _tint(bgr, small(state["mask"])))
        key = cv2.waitKey(20) & 0xFF
        if key in ENTER or key == ESC:
            break
        if key in (ord("r"), ord("R")):
            state["mask"] = grabcut(img_linear, rect, fg_strokes=fg, bg_strokes=bg, init=state["mask"])
        elif key == ord("["):
            state["brush"] = max(1, state["brush"] - 2)
        elif key == ord("]"):
            state["brush"] += 2
    cv2.destroyWindow(title)
    return state["mask"]


def click_points(img_linear: np.ndarray, labels: list[str], mask: np.ndarray | None = None,
                 optional_after: int | None = None) -> list[np.ndarray]:
    """Ask for one click per label, in order. Returns full-size pixel coords.

    optional_after: allow finishing early once this many points are placed
    (e.g. box corners: 2 required, up to 4).
    """
    bgr, s = _display(img_linear)
    base = _tint(bgr, cv2.resize(mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0) if mask is not None else bgr
    pts: list[tuple[float, float]] = []
    need = optional_after if optional_after is not None else len(labels)

    def title():
        nxt = labels[len(pts)] if len(pts) < len(labels) else "done"
        return f"Click: {nxt}   (U undo, Enter done)"

    win = "recast: click"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(labels):
            pts.append((x / s, y / s))

    cv2.setMouseCallback(win, on_mouse)
    while True:
        frame = base.copy()
        for i, (x, y) in enumerate(pts):
            p = (int(round(x * s)), int(round(y * s)))
            cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(frame, labels[i], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.setWindowTitle(win, title())
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("u"), ord("U")) and pts:
            pts.pop()
        elif key in ENTER and len(pts) >= need:
            break
        elif key == ESC:
            cv2.destroyWindow(win)
            raise SystemExit("cancelled")
    cv2.destroyWindow(win)
    return [np.array(p) for p in pts]


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  please type a number, e.g. 0.32")
