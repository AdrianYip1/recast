"""Builds data/example/: a fake room in the exact layout the engine reads.

    scene.json, photo.hdr, gbuffer/,
    object.glb, object_mask.png, object_meta.json, object_check.png

The "photo" is rendered here from known lighting (soft SH light + a window
area light with shadows), then stage C is run on it, so scene.json shows the
fitted lighting next to the truth printed below. The object is a vase
rendered from a known camera and rebuilt with the revolution mode.

Run from the python/ folder:  python examples/make_example.py
"""

import tempfile
from pathlib import Path

import numpy as np

from recast.camera import Camera
from recast.gbuffer import save_gbuffer
from recast.io import save_hdr, save_png
from recast.lighting import fit_scene_lighting, irradiance, light_irradiance
from recast.object import check
from recast.object.cli import main as object_main
from recast.object.modes.revolution import revolve
from recast.scene import (AreaLight, BoxProxy, Lighting, Material, ObjectPlacement, PlaneProxy, Scene,
                          floor_proxy, render_gbuffer)

OUT = Path(__file__).resolve().parents[2] / "data" / "example"


def camera() -> Camera:
    # 1.5m up, looking along +Z, tilted 10 degrees down, ~60 degree vertical fov
    pitch = np.radians(-10)
    fwd = np.array([0.0, np.sin(pitch), np.cos(pitch)])
    right = np.array([-1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    return Camera(640, 480, 415.7, np.vstack([right, down, fwd]), np.array([0.0, 1.5, 0.0]))


def make_object():
    """A vase "photographed" from a known camera, rebuilt with the revolution mode."""
    pitch = np.radians(-20)
    fwd = np.array([0.0, np.sin(pitch), np.cos(pitch)])
    right = np.array([-1.0, 0.0, 0.0])
    obj_cam = Camera(640, 480, 600.0, np.vstack([right, np.cross(fwd, right), fwd]), np.array([0.0, 1.0, 0.0]))
    base = np.array([0.0, 0.0, 2.2])

    h = np.linspace(0, 0.5, 40)
    vase = revolve(h, 0.11 + 0.05 * np.sin(np.pi * h / 0.5) - 0.03 * (h / 0.5) ** 2, base, 64)
    img, _ = check.render_mesh(obj_cam, vase, colour=(0.15, 0.3, 0.6))
    shot = Path(tempfile.mkdtemp()) / "vase.png"
    save_png(shot, img)

    b = obj_cam.project(base[None])[0]
    t = obj_cam.project((base + [0, 0.5, 0])[None])[0]
    object_main([str(shot), "--mode", "revolution", "--height", "0.3", "--out", str(OUT),
                 "--rect", "220", "140", "200", "260", "--base", *map(str, b), "--top", *map(str, t),
                 "--focal-px", "600", "--pitch-deg", "-20", "--no-auto-lines", "--no-ui", "--no-lensfun"])


def render_photo(scene: Scene, sh_true, radiance_true):
    """Diffuse render of the proxies: SH light + area lights with shadows."""
    normals, ids, depth = render_gbuffer(scene)
    cam = scene.camera
    hit = ids >= 0
    ys, xs = np.nonzero(hit)
    n = normals[ys, xs]
    pos = cam.position + cam.rays(np.column_stack([xs, ys]).astype(float)) * depth[ys, xs][:, None]
    albedo = np.array([scene.proxy(int(i)).material.albedo for i in ids[ys, xs]])

    E = irradiance(n, sh_true)
    for light, rad in zip(scene.lights, radiance_true):
        e, _ = light_irradiance(pos, n, light, samples=6, proxies=scene.proxies)
        E = E + e[:, None] * np.asarray(rad)
    img = np.zeros((cam.height, cam.width, 3), np.float32)
    img[ys, xs] = albedo / np.pi * E
    return img, (normals, ids, depth)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wall = Material((0.7, 0.7, 0.68), 0.9)
    window = AreaLight(np.array([[2.45, 0.9, 3.0], [2.45, 0.9, 4.2], [2.45, 2.1, 4.2], [2.45, 2.1, 3.0]]))
    scene = Scene(
        camera=camera(),
        scale_reference={"kind": "camera_height", "metres": 1.5},
        object=ObjectPlacement("object.glb", (0.5, 0.0, 3.0), 0.0, 1.0),
        proxies=[
            floor_proxy(0, Material((0.35, 0.25, 0.18), 0.3)),               # glossy wood floor
            PlaneProxy(1, (0.0, 0.0, 5.0), (0.0, 0.0, -1.0), wall),            # back wall
            PlaneProxy(2, (2.5, 0.0, 0.0), (-1.0, 0.0, 0.0), wall),            # left wall (window)
            BoxProxy(3, (-1.4, 0.0, 3.2), (-0.4, 0.75, 4.0), Material((0.4, 0.3, 0.2), 0.6)),  # table
        ],
        lights=[window],
    )

    sh_true = np.array([[0.30, 0.30, 0.33], [0.10, 0.10, 0.11], [0.0, 0.0, 0.0], [0.04, 0.03, 0.02]])
    radiance_true = [(3.0, 2.8, 2.4)]
    photo, gbuf = render_photo(scene, sh_true, radiance_true)
    save_hdr(OUT / "photo.hdr", photo)
    save_gbuffer(OUT / "gbuffer", *gbuf)

    sh, rad, used = fit_scene_lighting(photo, *gbuf, scene, order=1)
    scene.lighting = Lighting(1, sh)
    scene.lights[0].radiance = tuple(float(v) for v in rad[0])
    scene.save(OUT / "scene.json")

    make_object()

    print("wrote", OUT)
    print("window radiance  true", radiance_true[0], " fitted", np.round(rad[0], 2).tolist())
    print("SH L00           true", sh_true[0].tolist(), " fitted", np.round(sh[0], 3).tolist())


if __name__ == "__main__":
    main()
