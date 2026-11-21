"""Synthetic stand-in for the Python pipeline's output.

Renders a simple room lit by one point light (with shadows) through an exactly known pinhole camera, plus a placeholder
object (a textured can) as a GLB. Also renders the ground truth: the same room with the real can in it.

Outputs (in the target folder):
  photo.hdr              scene photo without the object        (what the pipeline delivers)
  object.glb             placeholder object, metres, +Y up, pivot at base centre
  scene.json             camera, proxies, object path + transform
  photo_with_object.hdr  ground truth with the object physically present
  ground_truth.json      true light + object placement (for evaluation only, the engine doesn't read it)

usage: python tools/make_test_scene.py data/sample
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 1600, 1000
FX = FY = 1400.0
CX, CY = 812.0, 488.0
CAMERA_POS = np.array([0.6, 1.5, 3.2])
TARGET = np.array([-0.3, 0.6, -0.5])

FLOOR_Y = 0.0
BACK_WALL_Z = -2.0
LEFT_WALL_X = -2.0
BOX_MIN = np.array([0.2, 0.0, -1.2])
BOX_MAX = np.array([0.9, 0.75, -0.6])

LIGHT_POS = np.array([-1.2, 1.6, 0.6])
LIGHT_INTENSITY = np.array([3.4, 3.2, 2.9])
AMBIENT = np.array([0.10, 0.11, 0.13])

CAN_RADIUS = 0.12
CAN_HEIGHT = 0.30
CAN_POS = np.array([-0.55, 0.0, -0.35])
CAN_ROTATION_Y_DEG = 25.0
CAN_SEGMENTS = 48
TEXTURE_SIZE = 256


def look_at_rotation(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


def can_label(u, v):
    """Procedural can texture: red with white bands and a dark seam at u=0 so rotation is visible."""
    color = np.empty(np.broadcast(u, v).shape + (3,))
    color[...] = [0.70, 0.08, 0.06]
    band = (np.abs(v - 0.5) < 0.12) | (np.abs(v - 0.12) < 0.03) | (np.abs(v - 0.88) < 0.03)
    color[band] = [0.85, 0.85, 0.82]
    seam = (u < 0.02) | (u > 0.98)
    color[seam] = [0.05, 0.05, 0.08]
    return color


def write_hdr(path, rgb):
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    brightest = rgb.max(axis=2)
    mantissa, exponent = np.frexp(brightest)
    scale = np.where(brightest > 1e-32, mantissa * 256.0 / np.maximum(brightest, 1e-32), 0.0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(brightest > 1e-32, exponent + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {rgb.shape[0]} +X {rgb.shape[1]}\n".encode())
        f.write(rgbe.tobytes())


def png_bytes(rgb8):
    h, w = rgb8.shape[:2]
    rows = b"".join(b"\0" + rgb8[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def linear_to_srgb8(linear):
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(np.clip(linear, 0, None), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def grid_lines(a, b, spacing=0.5, width=0.012):
    da = np.abs((a / spacing) - np.round(a / spacing)) * spacing
    db = np.abs((b / spacing) - np.round(b / spacing)) * spacing
    return (da < width) | (db < width)


def intersect_box(origins, directions):
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        t0 = (BOX_MIN - origins) * inverse
        t1 = (BOX_MAX - origins) * inverse
    t_near = np.nanmax(np.minimum(t0, t1), axis=-1)
    t_far = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = (t_near <= t_far) & (t_far > 1e-4)
    t = np.where(t_near > 1e-4, t_near, t_far)
    axis = np.argmax(np.minimum(t0, t1), axis=-1)
    return np.where(hit, t, np.inf), axis


def intersect_can(origins, directions):
    """Vertical cylinder at CAN_POS with a top cap. Returns t, normal, (u, v) texture coordinates."""
    o = origins - CAN_POS
    d = directions
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (o[..., 0] * d[..., 0] + o[..., 2] * d[..., 2])
    c = o[..., 0] ** 2 + o[..., 2] ** 2 - CAN_RADIUS ** 2
    discriminant = b * b - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.maximum(discriminant, 0.0))
        t_side = (-b - root) / (2 * a)
        t_side_far = (-b + root) / (2 * a)
    t_side = np.where(t_side > 1e-4, t_side, t_side_far)
    y_side = o[..., 1] + d[..., 1] * t_side
    side_ok = (discriminant > 0) & (a > 1e-12) & (t_side > 1e-4) & (y_side >= 0) & (y_side <= CAN_HEIGHT)
    t_side = np.where(side_ok, t_side, np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_cap = (CAN_HEIGHT - o[..., 1]) / d[..., 1]
    px = o[..., 0] + d[..., 0] * t_cap
    pz = o[..., 2] + d[..., 2] * t_cap
    cap_ok = np.isfinite(t_cap) & (t_cap > 1e-4) & (px * px + pz * pz <= CAN_RADIUS ** 2)
    t_cap = np.where(cap_ok, t_cap, np.inf)

    t = np.minimum(t_side, t_cap)
    p = o + d * t[..., None]
    normal = np.zeros_like(o)
    is_side = t_side <= t_cap
    normal[..., 0] = np.where(is_side, p[..., 0] / CAN_RADIUS, 0.0)
    normal[..., 1] = np.where(is_side, 0.0, 1.0)
    normal[..., 2] = np.where(is_side, p[..., 2] / CAN_RADIUS, 0.0)

    rotation = np.radians(CAN_ROTATION_Y_DEG)
    angle = np.arctan2(p[..., 2], p[..., 0]) + rotation
    u = np.mod(-angle / (2 * np.pi), 1.0)
    v = np.where(is_side, 1.0 - np.clip(p[..., 1] / CAN_HEIGHT, 0, 1), 0.0)
    return t, normal, u, v, is_side


def render(rotation, with_can):
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    directions = np.stack([(u - CX) / FX, -(v - CY) / FY, -np.ones_like(u)], axis=-1) @ rotation.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(CAMERA_POS, directions.shape)

    best_t = np.full(u.shape, np.inf)
    albedo = np.zeros(u.shape + (3,))
    normal = np.zeros(u.shape + (3,))

    def consider(t, surface_albedo, surface_normal):
        nonlocal best_t
        closer = np.isfinite(t) & (t > 1e-4) & (t < best_t)
        best_t = np.where(closer, t, best_t)
        albedo[closer] = surface_albedo[closer] if np.ndim(surface_albedo) == 3 else surface_albedo
        normal[closer] = surface_normal[closer] if np.ndim(surface_normal) == 3 else surface_normal

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (FLOOR_Y - CAMERA_POS[1]) / directions[..., 1]
        p = CAMERA_POS + directions * t[..., None]
        checker = ((np.floor(p[..., 0] / 0.5) + np.floor(p[..., 2] / 0.5)) % 2)[..., None]
        consider(np.where(t > 0, t, np.inf), np.where(checker > 0, [0.55, 0.52, 0.48], [0.30, 0.28, 0.26]), [0, 1, 0])

        t = (BACK_WALL_Z - CAMERA_POS[2]) / directions[..., 2]
        p = CAMERA_POS + directions * t[..., None]
        wall = np.where(grid_lines(p[..., 0], p[..., 1])[..., None], [0.15, 0.15, 0.18], [0.62, 0.66, 0.72])
        consider(np.where((t > 0) & (p[..., 1] >= FLOOR_Y), t, np.inf), wall, [0, 0, 1])

        t = (LEFT_WALL_X - CAMERA_POS[0]) / directions[..., 0]
        p = CAMERA_POS + directions * t[..., None]
        wall = np.where(grid_lines(p[..., 2], p[..., 1])[..., None], [0.18, 0.12, 0.10], [0.72, 0.60, 0.52])
        consider(np.where((t > 0) & (p[..., 1] >= FLOOR_Y) & (p[..., 2] >= BACK_WALL_Z), t, np.inf), wall, [1, 0, 0])

    t_box, axis = intersect_box(origins, directions)
    box_normal = np.zeros(u.shape + (3,))
    for a in range(3):
        box_normal[..., a] = np.where(axis == a, -np.sign(directions[..., a]), 0.0)
    consider(t_box, [0.60, 0.18, 0.14], box_normal)

    if with_can:
        t_can, can_normal, cu, cv, is_side = intersect_can(origins, directions)
        can_albedo = np.where(is_side[..., None], can_label(cu, cv), [0.75, 0.75, 0.78])
        consider(t_can, can_albedo, can_normal)

    hit = np.isfinite(best_t)
    points = CAMERA_POS + directions * np.where(hit, best_t, 0.0)[..., None]
    to_light = LIGHT_POS - points
    distance = np.linalg.norm(to_light, axis=-1)
    light_dir = to_light / np.maximum(distance, 1e-6)[..., None]

    shadow_origins = points + normal * 1e-3
    blocked = intersect_box(shadow_origins, light_dir)[0] < distance
    if with_can:
        blocked |= intersect_can(shadow_origins, light_dir)[0] < distance

    cosine = np.clip(np.sum(normal * light_dir, axis=-1), 0.0, None)
    direct = LIGHT_INTENSITY * (cosine * ~blocked / np.maximum(distance, 1e-3) ** 2)[..., None]
    image = albedo * (AMBIENT + direct)
    image[~hit] = [0.02, 0.02, 0.025]
    return image


def write_can_glb(path):
    texture_u, texture_v = np.meshgrid((np.arange(TEXTURE_SIZE) + 0.5) / TEXTURE_SIZE, (np.arange(TEXTURE_SIZE) + 0.5) / TEXTURE_SIZE)
    png = png_bytes(linear_to_srgb8(can_label(texture_u, texture_v)))

    positions, normals, uvs, indices = [], [], [], []
    for ring_y, ring_v in ((0.0, 1.0), (CAN_HEIGHT, 0.0)):
        for s in range(CAN_SEGMENTS + 1):
            angle = -2 * np.pi * s / CAN_SEGMENTS
            x, z = np.cos(angle), np.sin(angle)
            positions.append([CAN_RADIUS * x, ring_y, CAN_RADIUS * z])
            normals.append([x, 0.0, z])
            uvs.append([s / CAN_SEGMENTS, ring_v])
    ring = CAN_SEGMENTS + 1
    for s in range(CAN_SEGMENTS):
        b0, b1, t0, t1 = s, s + 1, ring + s, ring + s + 1
        indices += [b0, b1, t0, b1, t1, t0]

    for cap_y, cap_normal in ((CAN_HEIGHT, 1.0), (0.0, -1.0)):
        centre = len(positions)
        positions.append([0.0, cap_y, 0.0])
        normals.append([0.0, cap_normal, 0.0])
        uvs.append([0.5, 0.5])
        for s in range(CAN_SEGMENTS + 1):
            angle = -2 * np.pi * s / CAN_SEGMENTS
            positions.append([CAN_RADIUS * np.cos(angle), cap_y, CAN_RADIUS * np.sin(angle)])
            normals.append([0.0, cap_normal, 0.0])
            uvs.append([0.5, 0.5])
        for s in range(CAN_SEGMENTS):
            a, b = centre + 1 + s, centre + 2 + s
            indices += [centre, a, b] if cap_normal > 0 else [centre, b, a]

    positions = np.array(positions, np.float32)
    normals = np.array(normals, np.float32)
    uvs = np.array(uvs, np.float32)
    indices = np.array(indices, np.uint32)

    def pad4(data, fill=b"\0"):
        return data + fill * ((4 - len(data) % 4) % 4)

    blobs = [positions.tobytes(), normals.tobytes(), uvs.tobytes(), indices.tobytes(), png]
    offsets, offset = [], 0
    for blob in blobs:
        offsets.append(offset)
        offset += len(pad4(blob))
    binary = b"".join(pad4(blob) for blob in blobs)

    gltf = {
        "asset": {"version": "2.0", "generator": "foto make_test_scene"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "can", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": "label", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.6}}],
        "textures": [{"source": 0}], "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": offsets[i], "byteLength": len(blobs[i])} for i in range(5)],
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    glb = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
           + struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
           + struct.pack("<II", len(binary), 0x004E4942) + binary)
    Path(path).write_bytes(glb)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
    out.mkdir(parents=True, exist_ok=True)

    rotation = look_at_rotation(CAMERA_POS, TARGET)
    write_hdr(out / "photo.hdr", render(rotation, with_can=False))
    write_hdr(out / "photo_with_object.hdr", render(rotation, with_can=True))
    write_can_glb(out / "object.glb")

    object_transform = {"position": CAN_POS.tolist(), "rotation_y_deg": CAN_ROTATION_Y_DEG, "scale": 1.0}

    scene = {
        "version": 1,
        "photo": "photo.hdr",
        "image_size": [WIDTH, HEIGHT],
        "camera": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "rotation": rotation.tolist(), "position": CAMERA_POS.tolist()},
        "scale_reference": {"kind": "camera_height", "metres": float(CAMERA_POS[1])},
        "object": {"path": "object.glb", "transform": object_transform},
        "proxies": [
            {"type": "plane", "id": 0, "point": [0, FLOOR_Y, 0], "normal": [0, 1, 0], "material": {"albedo": [0.42, 0.40, 0.37], "roughness": 0.9}},
            {"type": "plane", "id": 1, "point": [0, 0, BACK_WALL_Z], "normal": [0, 0, 1], "material": {"albedo": [0.62, 0.66, 0.72], "roughness": 0.9}},
            {"type": "plane", "id": 2, "point": [LEFT_WALL_X, 0, 0], "normal": [1, 0, 0], "material": {"albedo": [0.72, 0.60, 0.52], "roughness": 0.9}},
            {"type": "box", "id": 3, "min": BOX_MIN.tolist(), "max": BOX_MAX.tolist(), "material": {"albedo": [0.60, 0.18, 0.14], "roughness": 0.5}},
        ],
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")

    ground_truth = {
        "light": {"type": "point", "position": LIGHT_POS.tolist(), "intensity": LIGHT_INTENSITY.tolist()},
        "ambient": AMBIENT.tolist(),
        "object_transform": object_transform,
        "photo_with_object": "photo_with_object.hdr",
    }
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2) + "\n")
    print(f"wrote photo.hdr, photo_with_object.hdr, object.glb, scene.json, ground_truth.json to {out}")


if __name__ == "__main__":
    main()
