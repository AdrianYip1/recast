"""Full 3D shape and texture from the cutout with TripoSR (MIT, runs locally, vendored in third_party/TripoSR).

The cutout is centred on grey like TripoSR's training images and the model predicts a triplane; marching
cubes gives the surface and xatlas unwraps it. The texture is the photo itself wherever the surface faced
the camera (projected through TripoSR's input camera), and the model's predicted colour on the back.
The photo's lighting is then divided out: an order-1 spherical harmonic shading fit over the surface
normals (Ramamoorthi & Hanrahan 2001) keeps the colour and drops the light, so the engine can relight it.
Finally the object is stood straight up, since photos are rarely taken level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from ...io import linear_to_srgb, srgb_to_linear
from ..texture import Texture

TSR_DIR = Path(__file__).resolve().parents[3] / "third_party" / "TripoSR"
MODEL = "stabilityai/TripoSR"
# TripoSR's input view (its run.py renders the input view with these): on +X, looking at the origin, +Z up
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40.0
CANVAS = 1024          # the input crop is kept at this size for the photo texture

# transformers 5 renamed the ViT encoder's modules; the checkpoint was saved with the old names
_VIT_RENAMES = [(".encoder.layer.", ".layers."), (".attention.attention.query.", ".attention.q_proj."),
                (".attention.attention.key.", ".attention.k_proj."), (".attention.attention.value.", ".attention.v_proj."),
                (".attention.output.dense.", ".attention.o_proj."), (".intermediate.dense.", ".mlp.fc1."),
                (".output.dense.", ".mlp.fc2.")]


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _match_checkpoint(ckpt: dict, expected: set) -> dict:
    out = {}
    for key, value in ckpt.items():
        new = key
        if key not in expected and key.startswith("image_tokenizer.model.encoder."):
            for old, renamed in _VIT_RENAMES:
                new = new.replace(old, renamed)
        out[new if new in expected else key] = value
    return out


_MODEL = None


def load_model():
    """TripoSR on the GPU when there is one. Downloaded once (~1.7 GB), then works offline. Kept loaded
    after the first call, so the worker (recast.worker) pays for it once."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    if str(TSR_DIR) not in sys.path:
        sys.path.insert(0, str(TSR_DIR))
    from tsr.system import TSR

    def fetch(name):
        try:
            return hf_hub_download(MODEL, name, local_files_only=True)
        except Exception:
            print(f"downloading {MODEL}/{name} (once)")
            return hf_hub_download(MODEL, name)

    cfg = OmegaConf.load(fetch("config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(fetch("model.ckpt"), map_location="cpu")
    model.load_state_dict(_match_checkpoint(ckpt, set(model.state_dict().keys())))
    model.renderer.set_chunk_size(8192)
    return model.to(_device()).eval()


def model_input(rgb8: np.ndarray, alpha: np.ndarray, ratio: float = 0.85) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Square crop around the cutout, object filling `ratio` of it, on 50% grey (TripoSR's training look).
    Returns the 512 px model input plus the same framing at CANVAS px (sRGB 0-1, alpha) for the texture."""
    ys, xs = np.nonzero(alpha > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = rgb8[y0:y1, x0:x1].astype(np.float32) / 255.0
    a = alpha[y0:y1, x0:x1].astype(np.float32)
    side = int(round(max(rgb.shape[:2]) / ratio))
    canvas = np.full((side, side, 3), 0.5, np.float32)
    canvas_a = np.zeros((side, side), np.float32)
    oy, ox = (side - rgb.shape[0]) // 2, (side - rgb.shape[1]) // 2
    canvas[oy:oy + rgb.shape[0], ox:ox + rgb.shape[1]] = rgb * a[..., None] + 0.5 * (1.0 - a[..., None])
    canvas_a[oy:oy + rgb.shape[0], ox:ox + rgb.shape[1]] = a
    image = Image.fromarray((canvas * 255.0 + 0.5).astype(np.uint8)).resize((512, 512), Image.LANCZOS)
    hi = cv2.resize(canvas, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA if side > CANVAS else cv2.INTER_CUBIC)
    hi_a = cv2.resize(canvas_a, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA if side > CANVAS else cv2.INTER_LINEAR)
    return image, np.clip(hi, 0, 1), np.clip(hi_a, 0, 1)


def project_input_view(points: np.ndarray, size: int = CANVAS) -> tuple[np.ndarray, np.ndarray]:
    """TripoSR frame points -> (pixel (N, 2) in the input crop, depth along the view)."""
    f = 0.5 * size / np.tan(np.radians(COND_FOVY_DEG) / 2)
    depth = COND_DISTANCE - points[:, 0]                 # camera on +X looking towards -X
    u = size / 2 + f * points[:, 1] / depth              # image right = +Y
    v = size / 2 - f * points[:, 2] / depth              # image up = +Z
    return np.column_stack([u, v]), depth


def _depth_map(verts: np.ndarray, faces: np.ndarray, size: int = CANVAS) -> np.ndarray:
    """Nearest surface depth per pixel of the input view (painter's order: far faces drawn first)."""
    px, depth = project_input_view(verts, size)
    face_depth = depth[faces].mean(axis=1)
    zbuf = np.full((size, size), np.inf, np.float32)
    for i in np.argsort(-face_depth):
        cv2.fillConvexPoly(zbuf, np.round(px[faces[i]]).astype(np.int32), float(face_depth[i]))
    return zbuf


def silhouette_iou(verts: np.ndarray, faces: np.ndarray, alpha: np.ndarray) -> float:
    """How well the mesh's outline in the input view matches the cutout (checks the camera assumptions)."""
    px, _ = project_input_view(verts, alpha.shape[0])
    sil = np.zeros(alpha.shape, np.uint8)
    for tri in faces:
        cv2.fillConvexPoly(sil, np.round(px[tri]).astype(np.int32), 1)
    a = alpha > 0.5
    return float((a & (sil > 0)).sum() / max((a | (sil > 0)).sum(), 1))


def _orient_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Marching cubes' winding is consistent but its direction isn't guaranteed, and the volume sign is
    unreliable on a mesh with holes: point the normals away from the centre on (area-weighted) average."""
    out = (mesh.face_normals * (mesh.triangles_center - mesh.centroid)).sum(1) @ mesh.area_faces
    if out >= 0:
        return mesh
    # a fresh mesh: trimesh's invert() keeps stale cached normals
    return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)[:, ::-1].copy(), process=False)


def _largest_piece(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    pieces = mesh.split(only_watertight=False)
    return max(pieces, key=lambda m: m.area) if len(pieces) > 1 else mesh


def _rasterize_uv(uv: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per texel: the face it lies in (-1 = none) and its barycentric coordinates."""
    ids = np.full((size, size), -1, np.int32)
    px = uv * size - 0.5
    for f, tri in enumerate(faces):
        cv2.fillConvexPoly(ids, np.round(px[tri]).astype(np.int32), int(f))
    ty, tx = np.nonzero(ids >= 0)
    f = ids[ty, tx]
    a, b, c = px[faces[f, 0]], px[faces[f, 1]], px[faces[f, 2]]
    p = np.column_stack([tx, ty]).astype(np.float64)
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-12, 1e-12, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.column_stack([1.0 - v - w, v, w]), 0.0, 1.0)
    bary /= bary.sum(1, keepdims=True)
    return ids, bary


def _pad(tex: np.ndarray, filled: np.ndarray, steps: int = 16) -> np.ndarray:
    """Grow the texels across the empty gaps so filtering at seams doesn't pull in black."""
    tex, m = tex.copy(), filled.astype(np.float32)
    for _ in range(steps):
        num = cv2.blur(tex * m[..., None], (3, 3))
        den = cv2.blur(m, (3, 3))
        grow = (m == 0) & (den > 0)
        tex[grow] = num[grow] / den[grow, None]
        m[grow] = 1.0
        if not grow.any():
            break
    tex[m == 0] = tex[m > 0].mean(axis=0)
    return tex


def _sample(img: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Bilinear lookup of img (H, W, C) at pixels (N, 2)."""
    n, cols = len(px), 1024                              # remap's maps must stay under 32767 rows
    rows = -(-n // cols)
    grid = np.zeros((rows * cols, 2), np.float32)
    grid[:n] = px
    m = cv2.remap(img.astype(np.float32), grid[:, 0].reshape(rows, cols), grid[:, 1].reshape(rows, cols),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.reshape(rows * cols, -1)[:n]


def delight(colors_linear: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide out smooth lighting: luminance ~ c0 + c . n (order-1 spherical harmonics), fitted robustly.

    Returns (albedo-like colours, the 4 coefficients). The overall brightness is kept, only the
    directional part (light from one side, dark underneath) is removed."""
    lum = colors_linear @ np.array([0.2126, 0.7152, 0.0722])
    A = np.column_stack([np.ones(len(normals)), normals])
    keep = lum > 1e-4
    coef = np.zeros(4)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], lum[keep], rcond=None)
        res = lum - A @ coef
        mad = np.median(np.abs(res[keep])) + 1e-6
        keep = (lum > 1e-4) & (np.abs(res) < 3.0 * 1.4826 * mad)
    shading = A @ coef
    ratio = np.clip(shading / max(float(np.mean(shading[keep])), 1e-6), 0.5, 2.0)
    return np.clip(colors_linear / ratio[:, None], 0.0, 1.0), coef


def upright(vertices: np.ndarray, max_tilt_deg: float = 45.0, iterations: int = 4) -> np.ndarray:
    """Rotation (3x3) that stands the object straight: the centres of its horizontal slices end up on one
    vertical line. TripoSR assumes a level camera, so a photo taken from above leans the object back by
    the camera's tilt; this undoes it for bottles, mugs, vases, boxes, lamps and the like."""
    R = np.eye(3)
    for _ in range(iterations):
        p = vertices @ R.T
        y = p[:, 1]
        edges = np.linspace(y.min(), y.max(), 25)[2:-2]          # the very top and bottom are ragged
        rows = [p[(y >= a) & (y < b)] for a, b in zip(edges[:-1], edges[1:])]
        centres = np.array([r.mean(axis=0) for r in rows if len(r) >= 10])
        if len(centres) < 4:
            break
        sx = np.polyfit(centres[:, 1], centres[:, 0], 1)[0]
        sz = np.polyfit(centres[:, 1], centres[:, 2], 1)[0]
        axis = np.array([sx, 1.0, sz]) / np.sqrt(sx * sx + 1.0 + sz * sz)
        if np.degrees(np.arccos(np.clip(axis[1], -1, 1))) < 0.2:
            break
        R = trimesh.geometry.align_vectors(axis, [0.0, 1.0, 0.0])[:3, :3] @ R
    if np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))) > max_tilt_deg:
        return np.eye(3)                                          # not a believable camera tilt: leave it
    return R


def build_generated(rgb8: np.ndarray, alpha: np.ndarray, resolution: int | None = None, texture_size: int = 1024,
                    remove_lighting: bool = True, model=None, max_faces: int = 20000) -> tuple[trimesh.Trimesh, Texture, dict]:
    """Cutout (sRGB uint8 + alpha in [0, 1]) -> textured mesh, +Y up, front towards +Z, arbitrary scale."""
    import torch
    import xatlas

    device = _device()
    model = model or load_model()
    resolution = resolution or (256 if device == "cuda" else 160)
    image, photo, photo_alpha = model_input(rgb8, alpha)
    with torch.no_grad():
        codes = model([image], device=device)
    mesh = _orient_outward(_largest_piece(model.extract_mesh(codes, False, resolution=resolution)[0]))
    to_gltf = _to_gltf()
    stand = upright(np.asarray(mesh.vertices) @ to_gltf[:3, :3].T)   # on the full mesh: its base is densest
    if len(mesh.faces) > max_faces:
        # marching cubes over-tessellates; fewer faces keep the unwrap fast and the path tracer light
        mesh = _orient_outward(mesh.simplify_quadric_decimation(face_count=max_faces))

    # unwrap, then bake every texel: the photo where the surface faced the camera, the model's colour elsewhere
    vmap, faces, uv = xatlas.parametrize(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32))
    faces = faces.astype(np.int64)
    verts = np.asarray(mesh.vertices)[vmap]
    normals = np.asarray(mesh.vertex_normals)[vmap]
    ids, bary = _rasterize_uv(uv, faces, texture_size)
    ty, tx = np.nonzero(ids >= 0)
    tri = faces[ids[ty, tx]]
    points = (verts[tri] * bary[..., None]).sum(1)
    texel_normals = (normals[tri] * bary[..., None]).sum(1)
    texel_normals /= np.maximum(np.linalg.norm(texel_normals, axis=1, keepdims=True), 1e-9)
    with torch.no_grad():
        predicted = model.renderer.query_triplane(model.decoder, torch.from_numpy(points.astype(np.float32)).to(device),
                                                  codes[0])["color"].float().cpu().numpy()
    predicted = np.clip(predicted, 0.0, 1.0)

    px, depth = project_input_view(points)
    to_cam = np.array([COND_DISTANCE, 0.0, 0.0]) - points
    facing = (texel_normals * to_cam).sum(1) / np.linalg.norm(to_cam, axis=1)
    zbuf = _depth_map(verts, faces)
    inside = (px[:, 0] >= 0) & (px[:, 0] < CANVAS - 1) & (px[:, 1] >= 0) & (px[:, 1] < CANVAS - 1)
    nearest = np.full(len(points), np.inf, np.float32)
    nearest[inside] = zbuf[px[inside, 1].astype(int), px[inside, 0].astype(int)]
    visible = inside & (depth <= nearest + 0.01)
    weight = np.clip((facing - 0.15) / 0.35, 0.0, 1.0) * visible * _sample(photo_alpha[..., None], px)[:, 0]
    color = weight[:, None] * _sample(photo, px) + (1.0 - weight[:, None]) * predicted

    info = {"resolution": resolution, "device": device, "faces": int(len(faces)), "texels": int(len(points)),
            "photo_texels": float((weight > 0.5).mean()), "facing_texels": float((facing > 0.5).mean()),
            "visible_texels": float(visible.mean()),
            "silhouette_iou": silhouette_iou(verts, faces, photo_alpha)}

    rotation = stand @ to_gltf[:3, :3]
    if remove_lighting:
        albedo, coef = delight(srgb_to_linear(color), texel_normals @ rotation.T)
        color = linear_to_srgb(albedo)
        info["removed_lighting_sh1"] = [float(c) for c in coef]

    tex = np.zeros((texture_size, texture_size, 3), np.float32)
    tex[ty, tx] = color
    tex = _pad(tex, ids >= 0)
    rgba = np.concatenate([(np.clip(tex, 0, 1) * 255.0 + 0.5).astype(np.uint8),
                           np.full((texture_size, texture_size, 1), 255, np.uint8)], axis=2)

    out = trimesh.Trimesh(verts @ rotation.T, faces, process=False)
    info["upright_tilt_deg"] = float(np.degrees(np.arccos(np.clip(stand[1, 1], -1.0, 1.0))))
    return out, Texture(rgba=rgba, uv=uv.astype(np.float64)), info


def _to_gltf() -> np.ndarray:
    """TripoSR's frame (+Z up, input camera on +X) -> +Y up with the photographed side facing +Z."""
    return trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]) @ \
        trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
