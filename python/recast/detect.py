"""Find the room and everything in it from the scene photo, with no clicks.

Two pretrained models do the "seeing" (downloaded once, then run offline on the CPU):
  * OneFormer (ADE20K, swin-tiny, MIT licence): an outline and a name for every
    object, plus which pixels are floor, wall and ceiling.
  * Depth Anything V2 Small (Apache-2.0): relative depth for every pixel.

Everything after that is our own geometry:
  1. depth_to_metres: the model's depth has no scale, but the floor's true depth is
     known exactly from the solved camera, so fitting the two on floor pixels turns
     the whole depth map into metres in the same frame as the camera.
  2. every object outline -> 3D points -> an axis-aligned box (two stacked boxes for
     things like sofas, whose seat is lower than their back), resting on the floor
     or on whatever box is under it.
  3. wall and ceiling pixels -> planes, which also keep the engine's shadow map
     sized to the room.

build_proxies only needs numpy arrays, so it is tested without the models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera
from .scene import BoxProxy, Material, PlaneProxy

SEG_MODEL = "shi-labs/oneformer_ade20k_swin_tiny"
# Base is CC-BY-NC-4.0 (non-commercial); Small (Apache-2.0) is the drop-in fallback for a commercial build
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

FLOOR = {"floor", "rug", "earth"}
# pillars stand on the floor and things go behind them, so they get boxes like furniture
WALL = {"wall", "window", "door", "curtain", "painting", "mirror", "blind", "screen door", "poster",
        "bulletin board", "radiator", "sconce", "clock", "flag"}
CEILING = {"ceiling"}
IGNORE = {"sky", "light", "chandelier", "fan", "person", "animal"}  # hanging or moving things
# seen through windows: not part of the room
OUTDOOR = {"tree", "grass", "plant life", "palm", "street lamp", "streetlight", "car", "bus", "truck", "van",
           "bicycle", "minibike", "building", "skyscraper", "house", "tower", "hovel", "road", "sidewalk", "path",
           "dirt track", "field", "land", "mountain", "hill", "rock", "sand", "sea", "river", "lake",
           "water", "bridge", "fence", "railing", "pole", "traffic light", "awning", "boat", "ship", "airplane"}

FLOOR_SNAP = 0.12        # m: a bottom this close to the floor rests on it
SUPPORT_SNAP = 0.15      # m: a bottom this close to a box top rests on that box
FLOATING = 2.2           # m: an unsupported bottom this high is a depth error or something outside
MAX_DEPTH = 40.0         # m: anything farther counts as unknown
LEVEL_GAP = 0.2          # m: seat vs back height difference that splits an object in two


def short_label(label: str) -> str:
    return label.split(",")[0].strip()


@dataclass
class Segment:
    label: str
    score: float
    mask: np.ndarray            # (H, W) bool


@dataclass
class Detection:
    proxies: list               # new PlaneProxy / BoxProxy, ids already assigned
    objects: list[dict]         # {"label", "score", "proxy_ids", "segment"}
    walls: list[dict]
    depth_fit: dict
    points: np.ndarray | None = field(default=None, repr=False)   # (H, W, 3) world points, NaN = unknown


# --- models ------------------------------------------------------------------------

_MODELS: dict = {}

def run_models(rgb8: np.ndarray, min_score: float = 0.5) -> tuple[list[Segment], np.ndarray]:
    """sRGB uint8 (H, W, 3) -> (segments, relative disparity (H, W), bigger = nearer)."""
    try:
        import torch
        from PIL import Image
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                                  OneFormerForUniversalSegmentation, OneFormerProcessor)
        from transformers.utils import logging as hf_logging
    except ImportError as e:
        raise SystemExit("the detect step needs the model libraries (CPU only):\n"
                         "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                         "  pip install -e \".[models]\"") from e
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(cls, name):
        key = (cls.__name__, name)
        if key not in _MODELS:                 # kept loaded, so a long-running worker pays for this once
            try:                               # downloaded once; after that, no network at all
                model = cls.from_pretrained(name, local_files_only=True)
            except OSError:
                print(f"downloading {name} (once)")
                model = cls.from_pretrained(name)
            _MODELS[key] = model.to(device).eval() if hasattr(model, "eval") else model
        return _MODELS[key]

    img = Image.fromarray(rgb8)
    h, w = rgb8.shape[:2]

    proc = load(OneFormerProcessor, SEG_MODEL)
    model = load(OneFormerForUniversalSegmentation, SEG_MODEL)
    with torch.no_grad():
        inputs = proc(images=img, task_inputs=["panoptic"], return_tensors="pt")
        out = model(**{k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()})
    r = proc.post_process_panoptic_segmentation(out, target_sizes=[(h, w)], threshold=min_score,
                                                label_ids_to_fuse=set())[0]
    seg = r["segmentation"].cpu().numpy()
    segments = [Segment(model.config.id2label[s["label_id"]], float(s["score"]), seg == s["id"])
                for s in r["segments_info"]]

    dproc = load(AutoImageProcessor, DEPTH_MODEL)
    dmodel = load(AutoModelForDepthEstimation, DEPTH_MODEL)
    with torch.no_grad():
        d = dmodel(**{k: v.to(device) for k, v in dproc(images=img, return_tensors="pt").items()}).predicted_depth.float().cpu()
    disparity = torch.nn.functional.interpolate(d[None], size=(h, w), mode="bicubic",
                                                align_corners=False)[0, 0].numpy()
    return segments, disparity.astype(np.float64)


# --- depth in metres ---------------------------------------------------------------

def _pixel_grid(cam: Camera) -> np.ndarray:
    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k) > 0


def depth_edges(disparity: np.ndarray, model_px: int = 518, jump: float = 0.06) -> np.ndarray:
    """Pixels where the depth jumps (object outlines). The model blurs depth across those, so their
    points float between the object and what's behind it and stretch the boxes. `jump` is the relative
    change across one model pixel that counts as an edge."""
    h, w = disparity.shape
    step = max(1, int(round(max(h, w) / model_px)))
    d = np.maximum(disparity, 1e-6)
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    gx[:, step:-step] = np.abs(d[:, 2 * step:] - d[:, :-2 * step])
    gy[step:-step, :] = np.abs(d[2 * step:, :] - d[:-2 * step, :])
    edges = (np.maximum(gx, gy) / d) > jump
    return cv2.dilate(edges.astype(np.uint8), np.ones((2 * step + 1, 2 * step + 1), np.uint8)) > 0


def depth_to_metres(cam: Camera, disparity: np.ndarray, floor_mask: np.ndarray):
    """Fit disparity = a / z + b on floor pixels, whose depth z (along the optical axis)
    the camera already gives. Returns (world points (H, W, 3) with NaN = unknown, fit info)."""
    rays = cam.rays(_pixel_grid(cam)).reshape(cam.height, cam.width, 3)
    fwd = cam.R[2]
    cos = rays @ fwd                                     # z = t * cos for a hit at distance t
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = -cam.position[1] / rays[..., 1]
    z_floor = t_floor * cos
    use = _erode(floor_mask, 2) & (rays[..., 1] < -1e-3) & (z_floor > 0.05) & (z_floor < MAX_DEPTH)
    n = int(use.sum())
    if n < 200:
        raise ValueError(f"only {n} floor pixels visible; the depth can't be put in metres")
    x = 1.0 / z_floor[use]
    y = disparity[use]
    if n > 40000:
        pick = np.random.default_rng(0).choice(n, 40000, replace=False)
        x, y = x[pick], y[pick]
    keep = np.ones(len(x), bool)
    for _ in range(4):                                   # least squares, dropping outliers
        A = np.column_stack([x[keep], np.ones(keep.sum())])
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        res = y - (a * x + b)
        mad = np.median(np.abs(res[keep])) + 1e-9
        keep = np.abs(res) < 3.0 * 1.4826 * mad
    if a <= 0:
        raise ValueError("the depth model's floor runs the wrong way; can't put it in metres")
    inv_z = (disparity - b) / a
    z = np.where(inv_z > 1.0 / MAX_DEPTH, 1.0 / np.maximum(inv_z, 1e-9), np.nan)
    pts = cam.position + rays * (z / cos)[..., None]
    rel = np.abs(1.0 / ((y[keep] - b) / a) - 1.0 / x[keep]) * x[keep]   # relative depth error on the floor
    info = {"a": float(a), "b": float(b), "floor_pixels": n,
            "floor_depth_error": float(np.median(rel))}
    return pts, info


# --- walls -------------------------------------------------------------------------

def fit_walls(P: np.ndarray, cam: Camera, max_walls: int = 3, tol: float = 0.25) -> list[dict]:
    """Axis-aligned vertical planes through wall points (N, 3): at most one per facing."""
    P = P[np.isfinite(P).all(axis=1)]
    n0 = len(P)
    walls, used = [], set()
    while len(walls) < max_walls and len(P) >= max(300, 0.08 * n0):
        best = None
        for axis in (0, 2):
            v = P[:, axis]
            lo, hi = np.percentile(v, [1, 99])
            if hi - lo < 1e-6:
                hist_peak = float(np.median(v))
            else:
                bins = max(int((hi - lo) / 0.05), 1)
                hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
                hist = np.convolve(hist, np.ones(5) / 5, mode="same")
                k = int(hist.argmax())
                hist_peak = 0.5 * (edges[k] + edges[k + 1])
            inl = np.abs(v - hist_peak) < tol
            if best is None or inl.sum() > best[2].sum():
                best = (axis, hist_peak, inl)
        axis, _, inl = best
        value = float(np.median(P[inl, axis]))
        side = 1.0 if cam.position[axis] > value else -1.0       # normal faces the camera
        if inl.sum() < max(300, 0.08 * n0):
            break
        if abs(cam.position[axis] - value) > 0.5 and (axis, side) not in used:
            used.add((axis, side))
            walls.append({"axis": axis, "value": value, "side": side, "points": int(inl.sum())})
        P = P[~inl]
    return walls


# --- objects -------------------------------------------------------------------------

def _boxes_for(pts: np.ndarray, cam: Camera) -> list[tuple[np.ndarray, np.ndarray]]:
    """One box around an object's visible points, or two (low + high part) when the
    object has a clear seat/back step like a sofa or armchair."""
    lo = np.array([np.percentile(pts[:, 0], 1.5), np.percentile(pts[:, 1], 3), np.percentile(pts[:, 2], 1.5)])
    hi = np.array([np.percentile(pts[:, 0], 98.5), np.percentile(pts[:, 1], 98), np.percentile(pts[:, 2], 98.5)])
    lo[1] = max(lo[1], 0.0)
    # only the front is visible: give a thin-looking object some depth, away from the camera
    for a, o in ((0, 2), (2, 0)):
        ext, other = hi[a] - lo[a], hi[o] - lo[o]
        if ext < 0.5 * other:
            if 0.5 * (lo[a] + hi[a]) >= cam.position[a]:
                hi[a] = lo[a] + 0.5 * other
            else:
                lo[a] = hi[a] - 0.5 * other
    boxes = [(lo, hi)]

    # seat/back split from a top-down height map on a 10 cm grid
    if max(hi[0] - lo[0], hi[2] - lo[2]) >= 0.4:
        cell = 0.1
        inside = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 2] >= lo[2]) & (pts[:, 2] <= hi[2])
        q = pts[inside]
        ij = np.floor((q[:, [0, 2]] - lo[[0, 2]]) / cell).astype(int)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.ravel()
        tops = np.array([np.percentile(q[inv == k, 1], 90) for k in range(len(keys))])
        if len(tops) >= 6:
            s = np.sort(tops)
            best_t, best_score = None, 0.0
            for i in range(1, len(s)):                   # Otsu split of the cell heights
                a_, b_ = s[:i], s[i:]
                score = len(a_) * len(b_) * (b_.mean() - a_.mean()) ** 2
                if score > best_score:
                    best_score, best_t = score, 0.5 * (s[i - 1] + s[i])
            low, high = tops < best_t, tops >= best_t
            if (best_t is not None and low.mean() >= 0.2 and high.mean() >= 0.15
                    and tops[high].mean() - tops[low].mean() > LEVEL_GAP and tops[low].mean() > 0.15):
                seat = float(np.percentile(tops[low], 70))
                hk = keys[high]
                h_lo = lo[[0, 2]] + hk.min(axis=0) * cell
                h_hi = np.minimum(lo[[0, 2]] + (hk.max(axis=0) + 1) * cell, hi[[0, 2]])
                boxes = [(np.array([lo[0], lo[1], lo[2]]), np.array([hi[0], seat, hi[2]])),
                         (np.array([h_lo[0], lo[1], h_lo[1]]), np.array([h_hi[0], hi[1], h_hi[1]]))]
    return boxes


def extend_far_side(cam: Camera, lo, hi, mask: np.ndarray, hidden: np.ndarray, max_grow: float = 1.0):
    """Depth sees a top almost edge-on, so its far part has few pixels and the box comes out
    short. The outline's top edge *is* the far edge of the top, though, and the camera puts
    those pixels on the plane y = top exactly. Only ever grows the box (by at most max_grow m),
    and skips columns where something in front hides the top."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    if cam.position[1] - hi[1] < 0.3:              # top seen too edge-on to place it
        return lo, hi
    cols = np.nonzero(mask.any(axis=0))[0]
    top = np.argmax(mask[:, cols], axis=0)
    ok = (top > 0) & ~hidden[np.maximum(top - 2, 0), cols]
    if ok.sum() < 5:
        return lo, hi
    rays = cam.rays(np.column_stack([cols[ok], top[ok]]).astype(np.float64))
    t = (hi[1] - cam.position[1]) / rays[:, 1]
    pts = (cam.position + rays * t[:, None])[t > 0]
    if len(pts) < 5:
        return lo, hi
    for a in (0, 2):
        if 0.5 * (lo[a] + hi[a]) >= cam.position[a]:
            hi[a] = min(max(hi[a], np.percentile(pts[:, a], 97)), hi[a] + max_grow)
        else:
            lo[a] = max(min(lo[a], np.percentile(pts[:, a], 3)), lo[a] - max_grow)
    return lo, hi


def floor_contacts(mask: np.ndarray, floor_mask: np.ndarray, gap: int = 4) -> np.ndarray:
    """Pixels (N, 2) where the bottom of the object's outline meets floor right below it:
    those touch the floor, so the camera alone says exactly where they are."""
    h = mask.shape[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.zeros((0, 2))
    bottom = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    below = np.zeros(len(cols), bool)
    for k in range(1, gap + 1):
        y = np.minimum(bottom + k, h - 1)
        below |= floor_mask[y, cols] & (bottom + k < h)
    return np.column_stack([cols[below], bottom[below]]).astype(np.float64)


def _overlap_xz(a_lo, a_hi, b_lo, b_hi) -> float:
    """Fraction of a's footprint that lies over b's."""
    w = max(0.0, min(a_hi[0], b_hi[0]) - max(a_lo[0], b_lo[0]))
    d = max(0.0, min(a_hi[2], b_hi[2]) - max(a_lo[2], b_lo[2]))
    area = max((a_hi[0] - a_lo[0]) * (a_hi[2] - a_lo[2]), 1e-9)
    return w * d / area


def _inside_fraction(a_lo, a_hi, b_lo, b_hi) -> float:
    """Fraction of box a's volume inside box b."""
    inter = np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0, None).prod()
    return float(inter / max(np.prod(a_hi - a_lo), 1e-12))


def _albedo(photo_linear, mask, floor_ref):
    """Surface colour relative to the floor (whose albedo is taken as 0.5)."""
    if photo_linear is None or mask.sum() == 0:
        return (0.5, 0.5, 0.5)
    c = np.median(photo_linear[mask], axis=0)
    return tuple(float(v) for v in np.clip(0.5 * c / np.maximum(floor_ref, 1e-4), 0.03, 0.9))


STANDS_UP = 0.06         # m above the floor that counts as standing on it, not part of it
OFF_WALL = 0.3           # m in front of every wall plane that counts as standing in the room


def geometry_segments(cam: Camera, P: np.ndarray, segments: list[Segment], kind: list[str], floor_mask: np.ndarray,
                      edges: np.ndarray, min_area: float = 0.003) -> list[Segment]:
    """Things the labels missed, found by where their pixels are in 3D rather than what they're called:
    "wall" that stands well in front of every wall plane (pillars, shelving, recesses), "floor" that rises
    off the floor (steps, ledges, platforms), and unlabelled pixels standing in the room. Each connected
    region becomes an object and gets a box like any recognised furniture."""
    h, w = floor_mask.shape
    finite = np.isfinite(P).all(axis=2) & ~edges
    y = np.where(finite, P[..., 1], np.nan)
    covered = np.zeros((h, w), bool)
    wall_mask = np.zeros((h, w), bool)
    for s, k in zip(segments, kind):
        covered |= s.mask
        if k == "wall":
            wall_mask |= s.mask
    walls = fit_walls(P[_erode(wall_mask, 2)], cam)
    on_a_wall = np.zeros((h, w), bool)
    in_room = finite.copy()
    for wall in walls:
        a, side = wall["axis"], wall["side"]
        dist = (P[..., a] - wall["value"]) * side          # > 0 on the camera's side of the wall
        on_a_wall |= np.abs(dist) < OFF_WALL
        in_room &= dist > -OFF_WALL
    with np.errstate(invalid="ignore"):
        standing = in_room & ~on_a_wall & (y > STANDS_UP) & (y < 3.0)
        candidates = {
            "pillar": wall_mask & standing,
            "step": floor_mask & in_room & (y > STANDS_UP) & (y < 1.2),
            "object": ~covered & standing,
        }
    out = []
    kernel = np.ones((5, 5), np.uint8)
    for label, mask in candidates.items():
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area * h * w:
                name = label
                if label == "pillar" and stats[i, cv2.CC_STAT_HEIGHT] < 1.5 * stats[i, cv2.CC_STAT_WIDTH]:
                    name = "ledge"                                  # wide and low: sills, shelves, counters
                out.append(Segment(name, 0.0, labels == i))
    return out


def build_proxies(cam: Camera, segments: list[Segment], disparity: np.ndarray, first_id: int,
                  photo_linear: np.ndarray | None = None, min_area: float = 0.001) -> Detection:
    h, w = cam.height, cam.width
    if disparity.shape != (h, w):
        raise ValueError(f"depth is {disparity.shape[::-1]}, photo is {(w, h)}")
    kind = []
    for s in segments:
        name = short_label(s.label)
        kind.append("floor" if name in FLOOR else "wall" if name in WALL else "ceiling" if name in CEILING
                    else "ignore" if name in IGNORE or name in OUTDOOR else "object")
    floor_mask = np.zeros((h, w), bool)
    for s, k in zip(segments, kind):
        if k == "floor":
            floor_mask |= s.mask
    P, fit = depth_to_metres(cam, disparity, floor_mask)
    edges = depth_edges(disparity)
    floor_ref = (np.median(photo_linear[floor_mask], axis=0)
                 if photo_linear is not None else np.full(3, 0.5))
    extra = geometry_segments(cam, P, segments, kind, floor_mask, edges)
    segments, kind = list(segments) + extra, list(kind) + ["object"] * len(extra)

    # objects -> boxes
    found = []
    for s, k in zip(segments, kind):
        if k != "object" or s.mask.mean() < min_area:
            continue
        m = _erode(s.mask, 2 if s.mask.sum() > 4000 else 1)
        if m.sum() < 0.4 * s.mask.sum():          # thin parts (legs) would vanish
            m = _erode(s.mask, 1)
        clean = m & ~edges
        if clean.sum() >= 0.4 * m.sum():
            m = clean
        pts = P[m]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 40:
            continue
        # standing on the floor? then its feet are exact
        contacts = floor_contacts(s.mask, floor_mask)
        if len(contacts):
            # an overhanging edge (a table top) also has floor below it in the photo; a real
            # contact is near floor height in the depth, a few pixels inside the outline
            u = contacts[:, 0].astype(int)
            v = np.maximum(contacts[:, 1].astype(int) - 3, 0)
            y_at = P[v, u, 1]
            dist = np.linalg.norm(P[v, u] - cam.position, axis=1)
            contacts = contacts[np.isfinite(y_at) & (y_at < np.maximum(0.08, 0.04 * dist))]
        if len(contacts) >= 3:
            # the floor fit can be off at this object's distance (narrow strip of floor, depth model bias):
            # scale the object's depth about the camera so its feet land where the camera says they are
            u = contacts[:, 0].astype(int)
            v = np.maximum(contacts[:, 1].astype(int) - 3, 0)
            true = np.linalg.norm(cam.pixel_on_floor(contacts) - cam.position, axis=1)
            model = np.linalg.norm(P[v, u] - cam.position, axis=1)
            ok = np.isfinite(true) & np.isfinite(model) & (model > 1e-3)
            if ok.sum() >= 3:
                k = float(np.clip(np.median(true[ok] / model[ok]), 0.5, 2.0))
                pts = cam.position + (pts - cam.position) * k
        # drop the depth "halo" where an outline meets the background
        d = np.linalg.norm(pts - cam.position, axis=1)
        med = np.median(d)
        pts = pts[np.abs(d - med) < max(4.0 * 1.4826 * np.median(np.abs(d - med)), 0.05)]
        boxes = [(lo, hi) for lo, hi in _boxes_for(pts, cam) if np.all(hi - lo > 0.02)]
        feet = np.zeros((0, 3))
        if len(contacts) and boxes:
            feet = cam.pixel_on_floor(contacts)
            e_lo = np.min([b[0] for b in boxes], axis=0)
            e_hi = np.max([b[1] for b in boxes], axis=0)
            margin = 0.5 * (e_hi - e_lo) + 0.3                # depth is close; a stray "foot" isn't
            near = np.isfinite(feet).all(axis=1)
            near[near] &= np.all((feet[near][:, [0, 2]] > (e_lo - margin)[[0, 2]])
                                 & (feet[near][:, [0, 2]] < (e_hi + margin)[[0, 2]]), axis=1)
            feet = feet[near]
        cols = np.count_nonzero(s.mask.any(axis=0))
        on_floor = len(feet) >= max(4, 0.02 * cols)          # thin legs only touch in a few columns
        if on_floor:
            if len(feet) >= 3:
                f_lo, f_hi = np.percentile(feet, 2, axis=0), np.percentile(feet, 98, axis=0)
                e_lo = np.min([b[0] for b in boxes], axis=0)
                e_hi = np.max([b[1] for b in boxes], axis=0)
                n_lo, n_hi = e_lo.copy(), e_hi.copy()
                for a in (0, 2):
                    span = e_hi[a] - e_lo[a]
                    if f_hi[a] - f_lo[a] >= 0.5 * span:      # the feet show this side; a top can overhang them
                        n_lo[a], n_hi[a] = min(f_lo[a], e_lo[a]), max(f_hi[a], e_hi[a])
                    elif cam.position[a] > 0.5 * (e_lo[a] + e_hi[a]):   # camera on the hi side
                        n_hi[a] = f_hi[a]
                        n_lo[a] = f_hi[a] - span
                    else:
                        n_lo[a] = f_lo[a]
                        n_hi[a] = f_lo[a] + span
                for lo, hi in boxes:                        # move every part with the envelope
                    for a in (0, 2):
                        k = (n_hi[a] - n_lo[a]) / max(e_hi[a] - e_lo[a], 1e-6)
                        lo[a], hi[a] = n_lo[a] + (lo[a] - e_lo[a]) * k, n_lo[a] + (hi[a] - e_lo[a]) * k
                    lo[1] = 0.0
        if boxes:
            # cut off by the bottom of the photo: it carries on below the frame, down to the floor
            cut_off = bool(s.mask[-3:].any())
            found.append({"segment": s, "boxes": boxes, "on_floor": on_floor or cut_off,
                          "albedo": _albedo(photo_linear, s.mask, floor_ref)})

    # what does each rest on? lowest first, so supports are settled before what's on them
    flat = [(i, j) for i, f in enumerate(found) for j in range(len(f["boxes"]))]
    flat.sort(key=lambda ij: found[ij[0]]["boxes"][ij[1]][0][1])
    all_objects = np.zeros((h, w), bool)
    for f in found:
        all_objects |= f["segment"].mask
    dist = np.linalg.norm(P - cam.position, axis=2)
    for f in found:
        m = f["segment"].mask
        near = np.nanmedian(dist[m]) - 0.15
        # only things in front can hide part of this object; things behind it count as spill
        occluders = all_objects & ~m & (dist < near)
        f["hidden"] = occluders
    settled: list[tuple[np.ndarray, np.ndarray, int]] = []
    for i, j in flat:
        lo, hi = found[i]["boxes"][j]
        if found[i]["on_floor"] or lo[1] < FLOOR_SNAP:
            lo[1] = 0.0
        else:
            tops = [b_hi[1] for b_lo, b_hi, owner in settled
                    if owner != i and abs(b_hi[1] - lo[1]) < SUPPORT_SNAP and _overlap_xz(lo, hi, b_lo, b_hi) > 0.3]
            if tops:
                lo[1] = max(tops)
        if lo[1] > FLOATING:
            continue
        if hi[1] - lo[1] < 0.02:
            hi[1] = lo[1] + 0.02
        f = found[i]
        if len(f["boxes"]) == 1:
            lo[:], hi[:] = extend_far_side(cam, lo, hi, f["segment"].mask, f["hidden"])
        settled.append((lo, hi, i))

    # boxes that sit inside a bigger one (cushions in a sofa, books in a table) add nothing
    all_boxes = [(lo, hi, i) for lo, hi, i in settled]
    keep = []
    for lo, hi, i in all_boxes:
        vol = np.prod(hi - lo)
        if any(o != i and np.prod(bh - bl) > vol and _inside_fraction(lo, hi, bl, bh) > 0.8
               for bl, bh, o in all_boxes):
            continue
        keep.append((lo, hi, i))

    proxies, objects = [], []
    next_id = first_id
    by_obj: dict[int, list[int]] = {}
    for lo, hi, i in keep:
        f = found[i]
        proxies.append(BoxProxy(next_id, tuple(float(v) for v in lo), tuple(float(v) for v in hi),
                                Material(f["albedo"], 0.8)))
        by_obj.setdefault(i, []).append(next_id)
        next_id += 1
    for i, ids in by_obj.items():
        s = found[i]["segment"]
        objects.append({"label": short_label(s.label), "score": round(s.score, 3), "proxy_ids": ids, "segment": s})

    # walls: never in front of an object (nothing stands behind a wall)
    wall_mask = np.zeros((h, w), bool)
    for s, k in zip(segments, kind):
        if k == "wall":
            wall_mask |= s.mask
    wall_albedo = _albedo(photo_linear, wall_mask, floor_ref)
    walls = fit_walls(P[_erode(wall_mask, 2)], cam)
    for wall in walls:
        a, side = wall["axis"], wall["side"]
        for p in proxies:
            if not isinstance(p, BoxProxy):
                continue
            far = p.min[a] if side > 0 else p.max[a]
            wall["value"] = min(wall["value"], far) if side > 0 else max(wall["value"], far)
        point = [0.0, 0.0, 0.0]
        point[a] = wall["value"]
        normal = [0.0, 0.0, 0.0]
        normal[a] = side
        wall["id"] = next_id
        proxies.append(PlaneProxy(next_id, tuple(point), tuple(normal), Material(wall_albedo, 0.9)))
        next_id += 1

    ceil_mask = np.zeros((h, w), bool)
    for s, k in zip(segments, kind):
        if k == "ceiling":
            ceil_mask |= s.mask
    cp = P[_erode(ceil_mask, 2)]
    cp = cp[np.isfinite(cp).all(axis=1)]
    if len(cp) > 200:
        y = float(np.median(cp[:, 1]))
        y = max(y, max((p.max[1] for p in proxies if isinstance(p, BoxProxy)), default=0.0))
        if y > cam.position[1] + 0.2:
            walls.append({"axis": 1, "value": y, "side": -1.0, "points": int(len(cp)), "id": next_id})
            proxies.append(PlaneProxy(next_id, (0.0, y, 0.0), (0.0, -1.0, 0.0),
                                      Material(_albedo(photo_linear, ceil_mask, floor_ref), 0.9)))
            next_id += 1
    return Detection(proxies, objects, walls, fit, P)


# --- lights seen in the photo ----------------------------------------------------------------

WINDOWS = {"window", "windowpane", "skylight"}
LAMPS = {"lamp", "light", "chandelier", "sconce", "spotlight"}
EMITTER_OFFSET = 0.05    # m in front of the wall, so the wall's own plane doesn't shadow its window


def find_emitters(cam: Camera, P: np.ndarray, segments: list[Segment], walls: list[dict],
                  max_windows: int = 2, max_lamps: int = 1) -> list[dict]:
    """Windows become area lights on the wall they're in (one rectangle per wall, from where the window pixels' rays
    meet that wall, not from depth, which sees through glass), lamps become small point lights at their 3D centre.
    Strengths are left at zero: the engine fits them to the photo. At most 2 windows + 1 lamp, leaving the engine's
    fourth light for a sun it may add."""
    h, w = P.shape[:2]
    window_mask = np.zeros((h, w), bool)
    lamps = []
    for s in segments:
        name = short_label(s.label)
        if name in WINDOWS:
            window_mask |= s.mask
        elif name in LAMPS:
            pts = P[s.mask]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) >= 20:
                lamps.append((int(s.mask.sum()), np.median(pts, axis=0)))

    windows = []
    vertical = [wall for wall in walls if wall["axis"] != 1]
    if vertical and window_mask.any():
        ys, xs = np.nonzero(window_mask)
        step = max(1, len(ys) // 20000)
        rays = cam.rays(np.column_stack([xs[::step], ys[::step]]).astype(np.float64))
        hits = []
        for wall in vertical:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wall["value"] - cam.position[wall["axis"]]) / rays[:, wall["axis"]]
            hits.append(np.where(t > 0, t, np.inf))
        hits = np.stack(hits)
        first = hits.argmin(axis=0)
        for k, wall in enumerate(vertical):
            mine = np.isfinite(hits[k]) & (first == k)
            if mine.sum() < 50:
                continue
            pts = cam.position + rays[mine] * hits[k, mine][:, None]
            a = wall["axis"]
            o = 2 - a                                            # the wall's horizontal direction
            h_lo, h_hi = np.percentile(pts[:, o], [2, 98])
            y_lo, y_hi = np.percentile(pts[:, 1], [2, 98])
            y_lo, y_hi = max(float(y_lo), 0.0), min(float(y_hi), 4.0)
            if h_hi - h_lo < 0.2 or y_hi - y_lo < 0.2:
                continue
            normal = np.zeros(3)
            normal[a] = wall["side"]                             # into the room
            along = np.cross([0.0, 1.0, 0.0], normal)           # corner 0 -> 1, so it shines along `normal`
            centre = np.zeros(3)
            centre[a] = wall["value"] + wall["side"] * EMITTER_OFFSET
            centre[o] = 0.5 * (h_lo + h_hi)
            centre[1] = 0.5 * (y_lo + y_hi)
            half_w = along * 0.5 * (h_hi - h_lo)
            half_h = np.array([0.0, 0.5 * (y_hi - y_lo), 0.0])
            corners = [centre - half_w - half_h, centre + half_w - half_h, centre + half_w + half_h, centre - half_w + half_h]
            windows.append(((h_hi - h_lo) * (y_hi - y_lo),
                            {"type": "area", "corners": [[round(float(v), 4) for v in c] for c in corners],
                             "radiance": [0.0, 0.0, 0.0]}))

    lights = [light for _, light in sorted(windows, key=lambda x: -x[0])[:max_windows]]
    lights += [{"type": "point", "position": [round(float(v), 4) for v in p], "intensity": [0.0, 0.0, 0.0], "radius": 0.1}
               for _, p in sorted(lamps, key=lambda x: -x[0])[:max_lamps]]
    return lights


# --- scale ----------------------------------------------------------------------------

# Furniture whose height hardly varies (m, top above the floor). A single photo can't tell a big room
# from a small one; the camera height is only a guess, and these fix it
KNOWN_TOP = {"table": 0.74, "desk": 0.75, "coffee table": 0.43, "counter": 0.91, "kitchen island": 0.91,
             "countertop": 0.91, "pool table": 0.80, "work surface": 0.91}


def scale_from_known_heights(det: Detection) -> tuple[float, list[str]]:
    """World scale factor that brings the known furniture to its usual height (1.0 = none found)."""
    boxes = {p.id: p for p in det.proxies if isinstance(p, BoxProxy)}
    ratios, used = [], []
    for o in det.objects:
        known = KNOWN_TOP.get(o["label"])
        parts = [boxes[i] for i in o["proxy_ids"] if i in boxes]
        if known is None or not parts or min(b.min[1] for b in parts) > 1e-6:
            continue
        top = max(b.max[1] for b in parts)
        if top > 0.05:
            ratios.append(known / top)
            used.append(f"{o['label']} {top:.2f} m (usually {known:.2f})")
    if not ratios:
        return 1.0, used
    return float(np.clip(np.median(ratios), 0.6, 1.6)), used


def scaled(det: Detection, k: float) -> Detection:
    """The whole detection scaled about the world origin (the floor below the camera stays put)."""
    from dataclasses import replace
    proxies = []
    for p in det.proxies:
        if isinstance(p, BoxProxy):
            proxies.append(replace(p, min=tuple(float(v) * k for v in p.min), max=tuple(float(v) * k for v in p.max)))
        else:
            proxies.append(replace(p, point=tuple(float(v) * k for v in p.point)))
    walls = [{**w, "value": w["value"] * k} for w in det.walls]
    return replace(det, proxies=proxies, walls=walls, points=None if det.points is None else det.points * k)


# --- check image -------------------------------------------------------------------

def _draw_segment(img, cam: Camera, a, b, colour, width=2, n=30):
    pts = np.linspace(a, b, n)
    pts = pts[(pts - cam.position) @ cam.R[2] > 0.05]
    if len(pts) < 2:
        return
    uv = cam.project(pts)
    for p, q in zip(uv[:-1], uv[1:]):
        if np.all(np.abs(p) < 1e5) and np.all(np.abs(q) < 1e5):
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), colour, width, cv2.LINE_AA)


def draw_check(rgb8: np.ndarray, cam: Camera, det: Detection) -> np.ndarray:
    """Left: every object found, coloured and named. Right: its boxes and the walls over the photo."""
    left = rgb8.copy()
    rng = np.random.default_rng(3)
    for o in det.objects:
        c = rng.integers(60, 255, 3)
        left[o["segment"].mask] = (0.45 * left[o["segment"].mask] + 0.55 * c).astype(np.uint8)
    for o in det.objects:
        ys, xs = np.nonzero(o["segment"].mask)
        org = (int(xs.mean()) - 4 * len(o["label"]), int(ys.mean()))
        cv2.putText(left, o["label"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(left, o["label"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    right = rgb8.copy()
    for wall in det.walls:
        if wall["axis"] == 1:
            continue
        a, o = wall["axis"], 2 - wall["axis"]
        for y in (0.0, 2.5):
            p, q = np.zeros(3), np.zeros(3)
            p[a] = q[a] = wall["value"]
            p[1] = q[1] = y
            p[o], q[o] = -12.0, 12.0
            _draw_segment(right, cam, p, q, (0, 200, 255), 2, n=200)
    for p in det.proxies:
        if not isinstance(p, BoxProxy):
            continue
        lo, hi = np.array(p.min), np.array(p.max)
        corners = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                    _draw_segment(right, cam, corners[i], corners[j], (255, 0, 255))
    return np.hstack([left, right])
