"""Lighting fit from the photo's shading (stage C).

Soft light: Ramamoorthi & Hanrahan (SIGGRAPH 2001). For a diffuse surface the
light arriving from all directions (irradiance) is smooth, so a few spherical
harmonic (SH) coefficients describe it:

    pixel = albedo / pi * E(n),   E(n) = sum_k A_k * L_k * Y_k(n)

Order 1 (4 coefficients) is the default; order 2 (9) only helps if the visible
surfaces face many different directions.

Clicked area lights (Karsch et al. 2011) add a term that depends on position:

    pixel += albedo / pi * radiance_j * E_j(p, n)

where E_j is the irradiance from light j at unit radiance. Everything is
linear in the unknowns, so SH coefficients and light radiances come out of one
least-squares solve. Pixels that break the diffuse model are left out: off the
proxies, clipped, highlights on glossy proxies, and in the shadow of a clicked light.
"""

from __future__ import annotations

import numpy as np

from .io import luminance

# Cosine-lobe convolution constants per band (l = 0, 1, 2)
_A_BAND = [np.pi, 2 * np.pi / 3, np.pi / 4]


def num_coeffs(order: int) -> int:
    return (order + 1) ** 2


def a_hat(order: int) -> np.ndarray:
    return np.array([_A_BAND[l] for l in range(order + 1) for _ in range(2 * l + 1)])


def sh_basis(normals: np.ndarray, order: int = 2) -> np.ndarray:
    """Real SH basis up to `order` (1 or 2). normals (..., 3) -> (..., 4 or 9).

    Order: Y00, Y1-1 (y), Y10 (z), Y11 (x), Y2-2 (xy), Y2-1 (yz), Y20, Y21 (xz), Y22.
    """
    if order not in (1, 2):
        raise ValueError("SH order must be 1 or 2")
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    bands = [
        np.full_like(x, 0.282095),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
    ]
    if order == 2:
        bands += [
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3 * z * z - 1),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ]
    return np.stack(bands, axis=-1)


def _order_of(sh: np.ndarray) -> int:
    k = np.asarray(sh).shape[0]
    if k == 4:
        return 1
    if k == 9:
        return 2
    raise ValueError(f"expected 4 or 9 SH coefficients, got {k}")


def irradiance(normals: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """Irradiance E(n) for normals (..., 3) and radiance SH (4 or 9, 3) -> (..., 3)."""
    order = _order_of(sh)
    return (sh_basis(normals, order) * a_hat(order)) @ np.asarray(sh)


def shade(normals: np.ndarray, albedo, sh: np.ndarray) -> np.ndarray:
    """Outgoing radiance of a diffuse surface: albedo / pi * E(n)."""
    return np.asarray(albedo) / np.pi * irradiance(normals, sh)


def highlight_mask(img: np.ndarray, saturation: float = 0.98, percentile: float = 99.0) -> np.ndarray:
    """True for pixels to EXCLUDE: clipped, or brighter than the given luminance percentile."""
    clipped = (img >= saturation).any(axis=-1)
    lum = luminance(img)
    bright = lum > np.percentile(lum, percentile) if lum.size else np.zeros_like(lum, bool)
    return clipped | bright


def _solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design
    lhs = gram + ridge * np.trace(gram) / len(gram) * np.eye(len(gram))
    return np.linalg.solve(lhs, design.T @ target)


def fit_sh(pixels: np.ndarray, normals: np.ndarray, albedo=0.5, mask: np.ndarray | None = None,
           order: int = 1, exclude_highlights: bool = True, robust_iterations: int = 2,
           ridge: float = 1e-6) -> np.ndarray:
    """Fit radiance SH coefficients (no area lights). Returns (4 or 9, 3).

    pixels:  (..., 3) linear RGB
    normals: (..., 3) unit normals for the same pixels (zeros = unknown, skipped)
    albedo:  scalar, (3,), or per-pixel (..., 3). Light vs. surface brightness is
             ambiguous from one photo, so a guess (0.5 grey) just scales the result.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    albedo = _per_pixel(albedo, len(pixels))

    use = np.linalg.norm(normals, axis=1) > 0.5
    if mask is not None:
        use &= np.asarray(mask).reshape(-1)
    if exclude_highlights and use.any():
        hl = np.zeros_like(use)
        hl[use] = highlight_mask(pixels[use])
        use &= ~hl

    Y = sh_basis(normals, order) * a_hat(order) / np.pi
    return _fit(pixels, [Y * albedo[:, c:c + 1] for c in range(3)], use, robust_iterations, ridge,
                n_sh=Y.shape[1], n_lights=0)[0]


def _per_pixel(albedo, n: int) -> np.ndarray:
    a = np.asarray(albedo, dtype=np.float64)
    if a.ndim <= 1:
        return np.broadcast_to(a, (n, 3)).astype(np.float64)
    return a.reshape(-1, 3)


def _fit(pixels, designs, use, robust_iterations, ridge, n_sh, n_lights):
    """Per-channel least squares with outlier trimming and non-negative light radiance."""
    if use.sum() < designs[0].shape[1] + 1:
        raise ValueError("not enough usable pixels to fit lighting")
    sh = np.zeros((n_sh, 3))
    rad = np.zeros((n_lights, 3))
    for c in range(3):
        design = designs[c]
        active = np.ones(design.shape[1], bool)  # lights forced to 0 get switched off
        sel = use.copy()
        for it in range(robust_iterations + 1):
            while True:
                coef = np.zeros(design.shape[1])
                coef[active] = _solve(design[sel][:, active], pixels[sel, c], ridge)
                neg = active.copy()
                neg[:n_sh] = False
                neg &= coef < 0
                if not neg.any():
                    break
                active &= ~neg
            if it == robust_iterations:
                break
            # drop clear outliers (leftover speculars, clutter not in the proxies):
            # residuals beyond ~2.5 sigma, with sigma estimated robustly from the median
            resid = np.abs(design @ coef - pixels[:, c])
            sigma = 1.4826 * np.median(resid[sel])
            floor = 1e-3 * np.mean(np.abs(pixels[sel, c]))  # noise-free data: don't trim on rounding error
            sel = use & (resid <= max(2.5 * sigma, floor))
        sh[:, c] = coef[:n_sh]
        rad[:, c] = coef[n_sh:]
    return sh, rad


# --- area lights -----------------------------------------------------------------

def _edge_term(a: np.ndarray, b: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Lambert's per-edge term: angle(a, b) * n . normalize(a x b), for (N, 3) arrays."""
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(np.einsum("ij,ij->i", an, bn), -1.0, 1.0))
    g = np.cross(an, bn)
    g_len = np.linalg.norm(g, axis=1)
    return np.where(g_len > 1e-12, theta * np.einsum("ij,ij->i", normals, g) / np.maximum(g_len, 1e-12), 0.0)


def polygon_irradiance(positions: np.ndarray, normals: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Exact irradiance from a flat polygon light of unit radiance, ignoring shadows.

    Lambert's formula E = 1/2 * sum_edges angle_i * (n . gamma_i), with the polygon
    clipped to the part above each point's tangent plane. The light is two-sided.
    """
    corners = np.asarray(corners, dtype=np.float64)
    k = len(corners)
    v = corners[None, :, :] - positions[:, None, :]           # (N, k, 3)
    h = np.einsum("nkj,nj->nk", v, normals)                    # height above the tangent plane
    total = np.zeros(len(positions))
    exit_pt = np.zeros_like(positions)
    entry_pt = np.zeros_like(positions)
    for i in range(k):
        a, b = v[:, i], v[:, (i + 1) % k]
        ha, hb = h[:, i], h[:, (i + 1) % k]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_pt = a + (b - a) * (ha / (ha - hb))[:, None]  # where the edge meets the plane
        above_a, above_b = ha >= 0, hb >= 0
        # keep the part of the edge that is above the plane
        start = np.where((above_a | ~above_b)[:, None], a, cross_pt)
        end = np.where((above_b | ~above_a)[:, None], b, cross_pt)
        keep = above_a | above_b
        total += np.where(keep, _edge_term(start, end, normals), 0.0)
        leaving = above_a & ~above_b
        entering = ~above_a & above_b
        exit_pt[leaving] = cross_pt[leaving]
        entry_pt[entering] = cross_pt[entering]
    # clipped polygons are closed by an edge along the horizon, from exit to entry
    clipped = (h >= 0).any(axis=1) & (h < 0).any(axis=1)
    if clipped.any():
        total[clipped] += _edge_term(exit_pt[clipped], entry_pt[clipped], normals[clipped])
    return 0.5 * np.abs(total)


def light_irradiance(positions: np.ndarray, normals: np.ndarray, light, samples: int = 4,
                     proxies=None) -> tuple[np.ndarray, np.ndarray]:
    """Irradiance at each point from `light` at unit radiance, and the visible fraction.

    Unshadowed irradiance is exact (polygon formula). If `proxies` is given,
    samples*samples shadow rays estimate how much of the light is visible, and
    E is scaled by that. Returns (E (N,), visibility (N,) in [0, 1]).
    """
    from .scene import cast  # local import: scene imports camera, keep this module light

    E = polygon_irradiance(positions, normals, light.corners)
    if not proxies:
        return E, np.ones(len(positions))

    pts = light.samples(samples)
    vis = np.zeros(len(positions))
    origin = positions + normals * 1e-3
    for s in pts:
        to = s - origin
        d = np.linalg.norm(to, axis=1)
        w = to / d[:, None]
        t, _, _ = cast(origin, w, proxies)
        vis += t >= d - 1e-3
    vis /= len(pts)
    return E * vis, vis


def fit_scene_lighting(photo: np.ndarray, normals: np.ndarray, ids: np.ndarray, depth: np.ndarray,
                       scene, order: int = 1, max_pixels: int = 150_000, glossy_percentile: float = 97.0,
                       shadow_samples: int = 4, robust_iterations: int = 2, ridge: float = 1e-6,
                       rng: np.random.Generator | None = None):
    """Stage C: fit SH + clicked light radiances from the photo and the proxy gbuffer.

    photo (H, W, 3) linear; normals/ids/depth from the gbuffer (ids -1 = no proxy).
    Returns (sh (4 or 9, 3), radiances (num_lights, 3), used_mask (H, W)).
    """
    rng = rng or np.random.default_rng(0)
    cam = scene.camera
    h, w = ids.shape
    photo = np.asarray(photo, dtype=np.float64)

    use = (ids >= 0) & np.isfinite(depth) & (np.linalg.norm(normals, axis=-1) > 0.5)
    use &= ~(photo >= 0.98).any(axis=-1)

    # highlights: top few percent of brightness on each glossy proxy
    lum = luminance(photo)
    for p in scene.proxies:
        if p.material.glossy:
            on = use & (ids == p.id)
            if on.sum() > 10:
                use &= ~(on & (lum > np.percentile(lum[on], glossy_percentile)))

    # big photos: a random subset of pixels is plenty for ~10 unknowns
    flat = np.flatnonzero(use)
    if len(flat) > max_pixels:
        flat = rng.choice(flat, max_pixels, replace=False)
    ys, xs = np.unravel_index(flat, (h, w))

    px = photo[ys, xs]
    n = normals[ys, xs]
    rays = cam.rays(np.column_stack([xs, ys]).astype(np.float64))
    pos = cam.position + rays * depth[ys, xs][:, None]
    albedo_by_id = {p.id: np.asarray(p.material.albedo, dtype=np.float64) for p in scene.proxies}
    albedo = np.array([albedo_by_id.get(int(i), np.full(3, 0.5)) for i in ids[ys, xs]])

    # clicked lights: irradiance columns, and drop anything they don't fully light
    light_cols = []
    lit = np.ones(len(flat), bool)
    for light in scene.lights:
        E, vis = light_irradiance(pos, n, light, shadow_samples, scene.proxies)
        light_cols.append(E / np.pi)
        lit &= vis >= 1.0

    Y = sh_basis(n, order) * a_hat(order) / np.pi
    L = np.column_stack(light_cols) if light_cols else np.zeros((len(flat), 0))
    designs = [np.hstack([Y, L]) * albedo[:, c:c + 1] for c in range(3)]
    sh, rad = _fit(px, designs, lit, robust_iterations, ridge, n_sh=Y.shape[1], n_lights=len(light_cols))

    used = np.zeros((h, w), bool)
    used[ys[lit], xs[lit]] = True
    return sh, rad, used


def dominant_direction(sh: np.ndarray) -> np.ndarray:
    """Direction the light mostly comes from, read off the band-1 coefficients."""
    lum = np.asarray(sh) @ np.array([0.2126, 0.7152, 0.0722])
    d = np.array([lum[3], lum[1], lum[2]])  # (x, y, z)
    n = np.linalg.norm(d)
    return d / n if n > 0 else np.array([0.0, 1.0, 0.0])
