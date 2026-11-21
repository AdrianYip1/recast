"""Synthetic stand-in for the Python pipeline's output: a pinhole render of a simple room with an exactly known
camera, written as photo.hdr + scene.json. Used to check the engine's camera/proxy alignment.

usage: python tools/make_test_scene.py data/sample
"""
import json
import sys
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


def look_at_rotation(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


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


def grid_lines(a, b, spacing=0.5, width=0.012):
    da = np.abs((a / spacing) - np.round(a / spacing)) * spacing
    db = np.abs((b / spacing) - np.round(b / spacing)) * spacing
    return (da < width) | (db < width)


def render(rotation):
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    directions_camera = np.stack([(u - CX) / FX, -(v - CY) / FY, -np.ones_like(u)], axis=-1)
    directions = directions_camera @ rotation.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    best_t = np.full(u.shape, np.inf)
    color = np.zeros(u.shape + (3,))
    normal = np.zeros(u.shape + (3,))

    def consider(t, valid, surface_color, surface_normal):
        nonlocal best_t
        closer = valid & (t > 1e-4) & (t < best_t)
        best_t = np.where(closer, t, best_t)
        color[closer] = surface_color[closer] if surface_color.ndim == 3 else surface_color
        normal[closer] = surface_normal

    def hit_points(t):
        return CAMERA_POS + directions * t[..., None]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (FLOOR_Y - CAMERA_POS[1]) / directions[..., 1]
        p = hit_points(t)
        checker = ((np.floor(p[..., 0] / 0.5) + np.floor(p[..., 2] / 0.5)) % 2)[..., None]
        floor_color = np.where(checker > 0, [0.55, 0.52, 0.48], [0.30, 0.28, 0.26])
        consider(t, np.isfinite(t), floor_color, [0, 1, 0])

        t = (BACK_WALL_Z - CAMERA_POS[2]) / directions[..., 2]
        p = hit_points(t)
        wall_color = np.where(grid_lines(p[..., 0], p[..., 1])[..., None], [0.15, 0.15, 0.18], [0.62, 0.66, 0.72])
        consider(t, np.isfinite(t) & (p[..., 1] >= FLOOR_Y), wall_color, [0, 0, 1])

        t = (LEFT_WALL_X - CAMERA_POS[0]) / directions[..., 0]
        p = hit_points(t)
        wall_color = np.where(grid_lines(p[..., 2], p[..., 1])[..., None], [0.18, 0.12, 0.10], [0.72, 0.60, 0.52])
        consider(t, np.isfinite(t) & (p[..., 1] >= FLOOR_Y) & (p[..., 2] >= BACK_WALL_Z), wall_color, [1, 0, 0])

        inverse = 1.0 / directions
        t0 = (BOX_MIN - CAMERA_POS) * inverse
        t1 = (BOX_MAX - CAMERA_POS) * inverse
        t_near = np.minimum(t0, t1).max(axis=-1)
        t_far = np.maximum(t0, t1).min(axis=-1)
        box_hit = (t_near <= t_far) & (t_far > 0)
        axis = np.minimum(t0, t1).argmax(axis=-1)
        box_normal = np.zeros(u.shape + (3,))
        for a in range(3):
            sign = -np.sign(directions[..., a])
            box_normal[..., a] = np.where(axis == a, sign, 0.0)
        closer = box_hit & (t_near < best_t)
        best_t = np.where(closer, t_near, best_t)
        color[closer] = [0.60, 0.18, 0.14]
        normal[closer] = box_normal[closer]

    light_direction = np.array([0.4, 0.9, 0.5])
    light_direction /= np.linalg.norm(light_direction)
    shading = 0.35 + 0.65 * np.clip(normal @ light_direction, 0.0, 1.0)
    image = color * shading[..., None]
    image[~np.isfinite(best_t)] = [0.05, 0.05, 0.06]
    return image


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
    out.mkdir(parents=True, exist_ok=True)

    rotation = look_at_rotation(CAMERA_POS, TARGET)
    write_hdr(out / "photo.hdr", render(rotation))

    scene = {
        "version": 1,
        "photo": "photo.hdr",
        "image_size": [WIDTH, HEIGHT],
        "camera": {
            "fx": FX, "fy": FY, "cx": CX, "cy": CY,
            "rotation": rotation.tolist(),
            "position": CAMERA_POS.tolist(),
        },
        "scale_reference": {"kind": "camera_height", "metres": float(CAMERA_POS[1])},
        "proxies": [
            {"type": "plane", "id": 0, "point": [0, FLOOR_Y, 0], "normal": [0, 1, 0], "material": {"albedo": [0.42, 0.40, 0.37], "roughness": 0.9}},
            {"type": "plane", "id": 1, "point": [0, 0, BACK_WALL_Z], "normal": [0, 0, 1], "material": {"albedo": [0.62, 0.66, 0.72], "roughness": 0.9}},
            {"type": "plane", "id": 2, "point": [LEFT_WALL_X, 0, 0], "normal": [1, 0, 0], "material": {"albedo": [0.72, 0.60, 0.52], "roughness": 0.9}},
            {"type": "box", "id": 3, "min": BOX_MIN.tolist(), "max": BOX_MAX.tolist(), "material": {"albedo": [0.60, 0.18, 0.14], "roughness": 0.5}},
        ],
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    print(f"wrote {out / 'photo.hdr'} and {out / 'scene.json'}")


if __name__ == "__main__":
    main()
