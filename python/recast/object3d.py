"""Object photo -> textured 3D mesh (.glb), no ML.

1. Cutout:   GrabCut (OpenCV) from a rough box around the object.
2. Shape:    "inflate" the silhouette into a closed, rounded mesh. Height at
             each pixel follows a sphere profile of its distance to the outline,
             so round things (mugs, fruit, bottles) come out about right.
3. Delight:  fit the object photo's lighting with the same SH fit used for the
             room, divide the shading out, and paint over clipped highlights.
             Only brightness is divided out, so the object keeps its colour.

Mesh space: +Y up, meters, origin at the bottom-centre of the object so it
sits on the floor when placed at y = 0. The front faces the camera (+Z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from .io import linear_to_srgb, luminance, srgb_to_linear
from .lighting import fit_sh, highlight_mask, irradiance


@dataclass
class ObjectMesh:
    vertices: np.ndarray   # (V, 3) meters
    faces: np.ndarray      # (F, 3) int
    uv: np.ndarray         # (V, 2) glTF convention: origin top-left, v down
    texture: np.ndarray    # (h, w, 3) linear RGB albedo, cropped to the object
    mask: np.ndarray       # (h, w) bool, object pixels in the texture crop


def cutout(img_linear: np.ndarray, rect: tuple[int, int, int, int], iterations: int = 5) -> np.ndarray:
    """GrabCut segmentation. rect = (x, y, w, h) roughly around the object. Returns bool mask."""
    srgb8 = (np.clip(linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(srgb8, cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, tuple(int(v) for v in rect), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return clean_mask((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest blob, fill its holes, and remove 1-pixel slivers."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == biggest).astype(np.uint8)
    # fill holes: flood the background from outside, whatever isn't reached is a hole
    padded = np.pad(m, 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return (m.astype(bool) | holes)


def _heights(mask: np.ndarray) -> np.ndarray:
    """Sphere-profile height field in pixel units (0 outside and on the outline)."""
    # exact distances: the fast 5x5 approximation leaves creases that show up in the normals
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    r = dt.max()
    if r <= 0:
        return dt
    return np.sqrt(np.clip(dt * (2 * r - dt), 0, None))


def _normals_from_heights(h: np.ndarray) -> np.ndarray:
    """Per-pixel front-surface normals in mesh space (x right, y up, z towards camera)."""
    # the outline is a pixel staircase, which leaves small ridges in the distance field;
    # blur a little so they don't turn into streaks in the normals
    sigma = max(1.0, 0.015 * max(h.shape))
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), sigma)
    dh_drow, dh_dcol = np.gradient(h)
    n = np.dstack([-dh_dcol, dh_drow, np.ones_like(h)])
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def remove_lighting(img: np.ndarray, mask: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Divide out the object photo's shading, keeping colour and average brightness."""
    # a rounded object shows normals in every direction, so order 2 is worth it here
    sh = fit_sh(img, normals, albedo=0.5, mask=mask, order=2)
    shading = luminance(irradiance(normals, sh))
    shading = np.clip(shading / np.median(shading[mask]), 0.2, None)
    albedo = img / shading[..., None]

    # specular highlights aren't albedo: paint over them from their surroundings
    hl = np.zeros_like(mask)
    hl[mask] = highlight_mask(img[mask], percentile=99.5)
    if hl.any():
        # cv2.inpaint needs 8-bit; only the filled-in pixels are taken from its output
        srgb8 = (np.clip(linear_to_srgb(albedo), 0, 1) * 255).astype(np.uint8)
        filled = cv2.inpaint(srgb8, hl.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        albedo = np.where(hl[..., None], srgb_to_linear(filled / 255.0), albedo)

    albedo = np.clip(albedo, 0.0, 1.0)
    albedo[~mask] = 0.0
    return albedo.astype(np.float32)


def inflate(mask: np.ndarray, height_m: float = 0.3, depth_ratio: float = 0.5,
            max_res: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed mesh from a silhouette.

    Returns (vertices, faces, grid_rc) where grid_rc are the (row, col)
    positions in the mask's pixel space for each vertex.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty mask")
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[r0:r1, c0:c1]

    # mesh on a coarser grid so the vertex count stays small
    step = max(1.0, max(crop.shape) / max_res)
    gh, gw = max(2, int(round(crop.shape[0] / step))), max(2, int(round(crop.shape[1] / step)))
    grid = cv2.resize(crop.astype(np.uint8) * 255, (gw, gh), interpolation=cv2.INTER_AREA) > 127
    grid = clean_mask(np.pad(grid, 1))  # pad so the outline is never on the edge
    gh, gw = grid.shape
    sy, sx = crop.shape[0] / (gh - 2), crop.shape[1] / (gw - 2)

    h = _heights(grid) * depth_ratio

    # vertex grid: one vertex per mask pixel
    idx = -np.ones(grid.shape, np.int64)
    rr, cc = np.nonzero(grid)
    idx[rr, cc] = np.arange(len(rr))

    # quads where all four corners are inside
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, 1:]; d = idx[1:, :-1]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    # image rows go down but mesh y goes up, so (a, d, c) is counter-clockwise seen from +Z
    front = np.concatenate([np.column_stack([a, d, c]), np.column_stack([a, c, b])])

    # outline vertices: on an edge used by only one front triangle
    edges = np.sort(np.concatenate([front[:, [0, 1]], front[:, [1, 2]], front[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    on_outline = np.zeros(len(rr), bool)
    on_outline[uniq[counts == 1].ravel()] = True
    # a triangle with all 3 corners on the outline would have a back copy lying
    # exactly on top of it; drop both and the neighbours close the surface
    front = front[~on_outline[front].all(axis=1)]

    # back copy shares the outline vertices, so the surface is closed with no stitching
    n = len(rr)
    back_index = np.where(on_outline, np.arange(n), n + np.cumsum(~on_outline) - 1)
    z = np.where(on_outline, 0.0, h[rr, cc])

    px_to_m = height_m / crop.shape[0]
    # grid -> original-mask pixel coords (centre of each grid cell)
    rows = r0 + (rr - 1 + 0.5) * sy
    cols = c0 + (cc - 1 + 0.5) * sx
    x = (cols - (c0 + c1) / 2.0) * px_to_m
    y = (r1 - rows) * px_to_m
    zm = z * max(sx, sy) * px_to_m

    front_v = np.column_stack([x, y, zm])
    back_v = np.column_stack([x, y, -zm])[~on_outline]
    vertices = np.concatenate([front_v, back_v])
    back = back_index[front][:, ::-1]  # reversed winding faces -Z
    faces = np.concatenate([front, back])

    grid_rc = np.concatenate([np.column_stack([rows, cols]), np.column_stack([rows, cols])[~on_outline]])
    return vertices, faces, grid_rc


def build_object(img_linear: np.ndarray, mask: np.ndarray, height_m: float = 0.3,
                 depth_ratio: float = 0.5, max_res: int = 160, delight: bool = True) -> ObjectMesh:
    """Mask + photo -> textured mesh, cropped to the object's bounding box."""
    mask = clean_mask(mask)
    vertices, faces, grid_rc = inflate(mask, height_m, depth_ratio, max_res)

    ys, xs = np.nonzero(mask)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tex = img_linear[r0:r1, c0:c1].astype(np.float32)
    tmask = mask[r0:r1, c0:c1]

    if delight:
        # surface normals at texture resolution, from the same sphere profile
        h = _heights(tmask) * depth_ratio
        tex = remove_lighting(tex, tmask, _normals_from_heights(h))

    th, tw = tmask.shape
    uv = np.column_stack([(grid_rc[:, 1] - c0) / tw, (grid_rc[:, 0] - r0) / th])
    return ObjectMesh(vertices, faces, np.clip(uv, 0, 1), tex, tmask)


def to_trimesh(obj: ObjectMesh) -> trimesh.Trimesh:
    # bleed colours past the outline so texture filtering doesn't pull in black
    srgb8 = (np.clip(linear_to_srgb(obj.texture), 0, 1) * 255).astype(np.uint8)
    if (~obj.mask).any():
        srgb8 = cv2.inpaint(srgb8, (~obj.mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    image = Image.fromarray(srgb8)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=False)
    # trimesh uses OpenGL-style UVs (v up) and flips them when writing glTF
    uv_gl = np.column_stack([obj.uv[:, 0], 1.0 - obj.uv[:, 1]])
    visual = trimesh.visual.TextureVisuals(uv=uv_gl, material=material)
    return trimesh.Trimesh(obj.vertices, obj.faces, visual=visual, process=False)


def export_glb(obj: ObjectMesh, path: str | Path) -> None:
    to_trimesh(obj).export(str(path), file_type="glb")
