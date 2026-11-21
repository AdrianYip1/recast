import numpy as np
import pytest

from recast.camera import Camera
from recast.gbuffer import load_gbuffer, save_gbuffer
from recast.lighting import (dominant_direction, fit_scene_lighting, fit_sh, irradiance, light_irradiance,
                             polygon_irradiance, sh_basis, shade)
from recast.scene import AreaLight, BoxProxy, Material, PlaneProxy, Scene, floor_proxy, render_gbuffer


def _random_normals(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sh9():
    """Soft light: bright from above, a warm window off to +X."""
    sh = np.zeros((9, 3))
    sh[0] = [0.8, 0.8, 0.85]
    sh[1] = [0.35, 0.35, 0.4]   # +Y
    sh[3] = [0.25, 0.18, 0.1]   # +X, warm
    sh[6] = [0.05, 0.05, 0.05]
    return sh


def test_basis_is_orthonormal():
    n = _random_normals(400_000, np.random.default_rng(0))
    Y = sh_basis(n, order=2)
    gram = Y.T @ Y * (4 * np.pi / len(n))
    assert np.allclose(gram, np.eye(9), atol=0.02)


def test_order1_is_prefix_of_order2():
    n = _random_normals(10, np.random.default_rng(1))
    assert np.allclose(sh_basis(n, 1), sh_basis(n, 2)[:, :4])


def test_constant_light_gives_pi_irradiance():
    sh = np.zeros((4, 3))
    sh[0] = 1.0 / 0.282095
    n = _random_normals(50, np.random.default_rng(1))
    assert np.allclose(irradiance(n, sh), np.pi, atol=1e-4)


@pytest.mark.parametrize("order", [1, 2])
def test_fit_recovers_lighting(order):
    rng = np.random.default_rng(2)
    sh = _sh9()[:(order + 1) ** 2]
    n = _random_normals(5000, rng)
    pixels = shade(n, 0.5, sh) + rng.normal(0, 0.002, (5000, 3))
    est = fit_sh(pixels, n, albedo=0.5, order=order, exclude_highlights=False, robust_iterations=0, ridge=0)
    assert np.allclose(est, sh, atol=0.02)


def test_fit_ignores_highlights():
    rng = np.random.default_rng(3)
    sh = _sh9()
    n = _random_normals(5000, rng)
    pixels = shade(n, 0.5, sh)
    spot = rng.choice(5000, 40, replace=False)
    pixels[spot] = 1.0  # a glossy spot blown out to white
    clean = fit_sh(pixels, n, albedo=0.5, order=2)
    naive = fit_sh(pixels, n, albedo=0.5, order=2, exclude_highlights=False, robust_iterations=0, ridge=0)
    err_clean = np.abs(irradiance(n, clean) - irradiance(n, sh)).mean()
    err_naive = np.abs(irradiance(n, naive) - irradiance(n, sh)).mean()
    assert err_clean < err_naive / 3


def test_polygon_irradiance_matches_sampling():
    light = AreaLight(np.array([[2.45, 0.9, 3.0], [2.45, 0.9, 4.2], [2.45, 2.1, 4.2], [2.45, 2.1, 3.0]]))
    pts = np.array([[0, 0, 3.5], [2.3, 1.0, 3.5], [1.0, 1.5, 3.6], [2.4, 0.0, 2.0]])
    ns = np.array([[0, 1, 0], [1, 0, 0], [0.3, -1, 0], [0.6, 0.8, 0]], float)  # 3rd is clipped by its horizon
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)

    m = 300
    t = (np.arange(m) + 0.5) / m
    u, v = (a.ravel()[:, None] for a in np.meshgrid(t, t))
    c = light.corners
    s = (1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3]
    brute = []
    for p, n in zip(pts, ns):
        to = s - p
        d2 = (to ** 2).sum(1)
        w = to / np.sqrt(d2)[:, None]
        brute.append((light.area / len(s) * np.clip(w @ n, 0, None) * np.abs(w @ light.normal) / d2).sum())
    assert np.allclose(polygon_irradiance(pts, ns, light.corners), brute, rtol=1e-3)


def test_light_behind_surface_gives_nothing():
    light = AreaLight(np.array([[0, -1, 0], [1, -1, 0], [1, -1, 1], [0, -1, 1.0]]))
    assert polygon_irradiance(np.zeros((1, 3)), np.array([[0, 1.0, 0]]), light.corners)[0] == 0


def test_box_casts_shadow():
    light = AreaLight(np.array([[-0.1, 3, -0.1], [0.1, 3, -0.1], [0.1, 3, 0.1], [-0.1, 3, 0.1]]))
    table = BoxProxy(0, (-0.5, 0.8, -0.5), (0.5, 1.0, 0.5))
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    up = np.array([[0, 1.0, 0], [0, 1.0, 0]])
    E, vis = light_irradiance(pts, up, light, proxies=[table])
    assert vis[0] == 0 and E[0] == 0
    assert vis[1] == 1 and E[1] > 0


def _room():
    pitch = np.radians(-10)
    fwd = np.array([0.0, np.sin(pitch), np.cos(pitch)])
    right = np.array([-1.0, 0.0, 0.0])
    cam = Camera(200, 150, 130.0, np.vstack([right, np.cross(fwd, right), fwd]), np.array([0.0, 1.5, 0.0]))
    wall = Material((0.7, 0.7, 0.68), 0.9)
    return Scene(
        camera=cam,
        proxies=[floor_proxy(0, Material((0.35, 0.25, 0.18), 0.3)),
                 PlaneProxy(1, (0, 0, 5), (0, 0, -1), wall),
                 PlaneProxy(2, (2.5, 0, 0), (-1, 0, 0), wall),
                 BoxProxy(3, (-1.4, 0, 3.2), (-0.4, 0.75, 4.0), Material((0.4, 0.3, 0.2), 0.6))],
        lights=[AreaLight(np.array([[2.45, 0.9, 3.0], [2.45, 0.9, 4.2], [2.45, 2.1, 4.2], [2.45, 2.1, 3.0]]))],
    )


def _render(scene, sh, radiance):
    normals, ids, depth = render_gbuffer(scene)
    cam = scene.camera
    ys, xs = np.nonzero(ids >= 0)
    n = normals[ys, xs]
    pos = cam.position + cam.rays(np.column_stack([xs, ys]).astype(float)) * depth[ys, xs][:, None]
    albedo = np.array([scene.proxy(int(i)).material.albedo for i in ids[ys, xs]])
    e, _ = light_irradiance(pos, n, scene.lights[0], samples=6, proxies=scene.proxies)
    img = np.zeros((cam.height, cam.width, 3))
    img[ys, xs] = albedo / np.pi * (irradiance(n, sh) + e[:, None] * radiance)
    return img, (normals, ids, depth)


def test_scene_fit_recovers_sh_and_window(tmp_path):
    scene = _room()
    sh = np.array([[0.30, 0.30, 0.33], [0.10, 0.10, 0.11], [0.0, 0.0, 0.0], [0.04, 0.03, 0.02]])
    radiance = np.array([3.0, 2.8, 2.4])
    photo, gbuf = _render(scene, sh, radiance)

    # go through the on-disk gbuffer format, like the engine handoff
    save_gbuffer(tmp_path, *gbuf)
    normals, ids, depth = load_gbuffer(tmp_path)
    assert (ids == gbuf[1]).all()

    est_sh, est_rad, used = fit_scene_lighting(photo, normals, ids, depth, scene, order=1)
    assert np.allclose(est_rad[0], radiance, rtol=0.05)
    assert np.allclose(est_sh, sh, atol=0.02)
    # the table's shadow from the window is excluded
    assert used.sum() < (ids >= 0).sum()


def test_dominant_direction():
    d = dominant_direction(_sh9())
    assert d[1] > 0.7 and d[0] > 0.3
