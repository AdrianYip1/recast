"""Builds the bank's example glass dragon from Khronos' DragonAttenuation sample.

Keeps only the dragon (no cloth backdrop, no surface colour variant), bakes its node transform into the vertices so the
file is in metres, +Y up, base centre at the origin, and rescales the volume so the colour reads the same at that size.
Also draws the bank tile's thumbnail.

usage: python tools/make_glass_dragon.py [DragonAttenuation.glb]   (downloads it when no path is given)
writes: python/recast/object/assets/glass_dragon.glb, glass_dragon.png
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SOURCE_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DragonAttenuation/glTF-Binary/DragonAttenuation.glb"
HEIGHT_M = 0.20
OUT = Path(__file__).resolve().parent.parent / "python" / "recast" / "object" / "assets"

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def read_glb(data: bytes):
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_length])
    bin_start = 20 + json_length + 8
    return gltf, data[bin_start:]


def read_accessor(gltf, binary, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(DTYPES[accessor["componentType"]])
    width = COMPONENTS[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    count = accessor["count"]
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * width, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * width), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(count, width)


def quaternion_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def write_glb(path, positions, normals, indices, material):
    blobs = [positions.astype("<f4").tobytes(), normals.astype("<f4").tobytes(), indices.astype("<u4").tobytes()]
    views, offset = [], 0
    for blob, target in zip(blobs, (34962, 34962, 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target})
        offset += len(blob)
    gltf = {
        "asset": {"version": "2.0", "generator": "recast tools/make_glass_dragon.py",
                  "copyright": "1996 Stanford University Computer Graphics Laboratory; 2017 Morgan McGuire"},
        "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior", "KHR_materials_volume"],
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "glass dragon"}],
        "meshes": [{"name": "glass dragon",
                    "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [material],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    chunk = json.dumps(gltf, separators=(",", ":")).encode()
    chunk += b" " * (-len(chunk) % 4)
    body = b"".join(blobs)
    out = struct.pack("<II", len(chunk), 0x4E4F534A) + chunk + struct.pack("<II", len(body), 0x004E4942) + body
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(out)) + out)


def draw_thumbnail(path, positions, normals, indices, color, size=128):
    """Amber glass seen from the front three quarters: thick middles deep orange, thin edges pale, a few glints."""
    s = 4
    yaw, pitch = np.radians(-35.0), np.radians(-15.0)
    rotate = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    tilt = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    view = tilt @ rotate
    p = positions @ view.T
    n = normals @ view.T
    lo, hi = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
    k = 0.88 * size * s / (hi - lo).max()
    centre = (lo + hi) / 2
    xy = np.stack([size * s / 2 + (p[:, 0] - centre[0]) * k, size * s / 2 - (p[:, 1] - centre[1]) * k], axis=1)

    triangles = indices.reshape(-1, 3)
    facing = n[triangles].mean(axis=1)
    facing /= np.linalg.norm(facing, axis=1, keepdims=True) + 1e-9
    order = np.argsort(p[triangles, 2].mean(axis=1))                     # far first (the camera looks down -z)
    deep, pale = np.array(color) ** 2.2 * 0.9, np.array([1.0, 0.95, 0.75])
    light = np.array([-0.4, 0.7, 0.6]) / np.linalg.norm([-0.4, 0.7, 0.6])

    rgb = Image.new("RGB", (size * s, size * s))
    mask = Image.new("L", (size * s, size * s))
    draw_rgb, draw_mask = ImageDraw.Draw(rgb), ImageDraw.Draw(mask)
    for t in order:
        f = facing[t]
        thickness = abs(f[2]) ** 0.7
        c = pale + (deep - pale) * thickness
        glint = max(0.0, float(np.dot(f, (light + [0, 0, 1]) / np.linalg.norm(light + [0, 0, 1])))) ** 40
        c = np.clip(c + glint, 0, 1) ** (1 / 2.2)
        corners = [tuple(xy[i]) for i in triangles[t]]
        draw_rgb.polygon(corners, fill=tuple(int(v * 255) for v in c))
        draw_mask.polygon(corners, fill=235)
    image = rgb.convert("RGBA")
    image.putalpha(mask)
    image.resize((size, size), Image.LANCZOS).save(path)


def main(argv):
    data = Path(argv[0]).read_bytes() if argv else urllib.request.urlopen(SOURCE_URL).read()
    gltf, binary = read_glb(data)
    node = next(n for n in gltf["nodes"] if n.get("name") == "Dragon")
    primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
    source = gltf["materials"][primitive["material"]]

    positions = read_accessor(gltf, binary, primitive["attributes"]["POSITION"]).astype(np.float64)
    normals = read_accessor(gltf, binary, primitive["attributes"]["NORMAL"]).astype(np.float64)
    indices = read_accessor(gltf, binary, primitive["indices"]).reshape(-1).astype(np.uint32)

    rotation = quaternion_matrix(node.get("rotation", [0, 0, 0, 1]))
    node_scale = node.get("scale", [1, 1, 1])[0]
    positions = positions @ rotation.T
    normals = normals @ rotation.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    mesh_to_metres = HEIGHT_M / np.ptp(positions[:, 1])
    positions *= mesh_to_metres
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    positions -= [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]

    volume = source["extensions"]["KHR_materials_volume"]
    world_scale = mesh_to_metres / node_scale                           # how much smaller than in the Khronos scene
    material = {
        "name": "amber glass",
        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.0},
        "extensions": {
            "KHR_materials_transmission": {"transmissionFactor": 1.0},
            "KHR_materials_ior": {"ior": 1.5},
            "KHR_materials_volume": {"thicknessFactor": volume["thicknessFactor"] * mesh_to_metres,
                                     "attenuationColor": volume["attenuationColor"],
                                     "attenuationDistance": volume["attenuationDistance"] * world_scale},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_glb(OUT / "glass_dragon.glb", positions, normals, indices, material)
    draw_thumbnail(OUT / "glass_dragon.png", positions, normals, indices, volume["attenuationColor"])
    size = np.ptp(positions, axis=0)
    print(f"glass dragon: {len(positions)} vertices, {len(indices) // 3} triangles, "
          f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m, attenuation distance "
          f"{material['extensions']['KHR_materials_volume']['attenuationDistance']:.4f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
