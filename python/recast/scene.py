"""scene.json: the contract between the Python pipeline and the engine.

Layout of a scene folder (data/<scene_name>/):

    scene.json        camera, paths, scale, proxies, lights, SH   (Python writes, engine adds proxies/lights)
    photo.jpg         undistorted scene photo, sRGB               (Python)
    object.glb        extracted object, metres, +Y up, pivot at base (Python)
    object_mask.jpg   cutout, for debugging                       (Python)
    object_check.jpg  mesh outline reprojected on the object photo (Python)
    gbuffer/          normals.hdr, ids.png, depth.hdr             (engine, for the lighting fit)
    render.jpg        final composite                             (engine)

Rules: metres; world +Y up, right-handed; camera rotation is camera-to-world in
engine convention (looks down -Z); pixel origin top-left; colours linear RGB.

Both sides edit scene.json, so Python only rewrites the sections it owns and
keeps everything else (including keys it doesn't know about), and writes
atomically so the engine's hot reload never sees half a file.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import Camera

SCHEMA_VERSION = 1
PHOTO_NAME = "photo.jpg"
GLOSSY_ROUGHNESS = 0.5  # proxies smoother than this get highlight masking in the lighting fit


def _vec(v) -> list[float]:
    return [float(x) for x in v]


@dataclass
class Material:
    albedo: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = 0.9

    @property
    def glossy(self) -> bool:
        return self.roughness < GLOSSY_ROUGHNESS

    def to_dict(self) -> dict:
        return {"albedo": _vec(self.albedo), "roughness": float(self.roughness)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Material":
        d = d or {}
        return cls(tuple(d.get("albedo", (0.5, 0.5, 0.5))), float(d.get("roughness", 0.9)))


@dataclass
class PlaneProxy:
    """Infinite plane through `point` with `normal` pointing into the room."""

    id: int
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    material: Material = field(default_factory=Material)

    def unit_normal(self) -> np.ndarray:
        n = np.asarray(self.normal, dtype=np.float64)
        return n / np.linalg.norm(n)

    def to_dict(self) -> dict:
        return {"type": "plane", "id": self.id, "point": _vec(self.point),
                "normal": _vec(self.normal), "material": self.material.to_dict()}


@dataclass
class BoxProxy:
    """Axis-aligned box (table, cabinet, ...)."""

    id: int
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    material: Material = field(default_factory=Material)

    def to_dict(self) -> dict:
        return {"type": "box", "id": self.id, "min": _vec(self.min), "max": _vec(self.max),
                "material": self.material.to_dict()}


@dataclass
class HullProxy(BoxProxy):
    """Shape fitted to a detected object (carved from its cutout and depth, or a convex hull): `vertices` (local,
    around `point`) and outward `faces`.
    min/max hold its world bounding box, so everything that handles boxes treats it as that box."""

    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertices: list = field(default_factory=list)    # [[x, y, z], ...] relative to point
    faces: list = field(default_factory=list)       # [[i, j, k], ...], counter-clockwise seen from outside

    def to_dict(self) -> dict:
        return {"type": "hull", "id": self.id, "min": _vec(self.min), "max": _vec(self.max), "point": _vec(self.point),
                "vertices": [_vec(v) for v in self.vertices], "faces": [[int(i) for i in f] for f in self.faces],
                "material": self.material.to_dict()}


Proxy = PlaneProxy | BoxProxy


def proxy_from_dict(d: dict) -> Proxy:
    if d["type"] == "plane":
        return PlaneProxy(int(d["id"]), tuple(d["point"]), tuple(d["normal"]), Material.from_dict(d.get("material")))
    if d["type"] == "box":
        return BoxProxy(int(d["id"]), tuple(d["min"]), tuple(d["max"]), Material.from_dict(d.get("material")))
    if d["type"] == "hull":
        return HullProxy(int(d["id"]), tuple(d["min"]), tuple(d["max"]), Material.from_dict(d.get("material")),
                         tuple(d.get("point", (0.0, 0.0, 0.0))), list(d.get("vertices", [])), list(d.get("faces", [])))
    raise ValueError(f"unknown proxy type {d['type']!r}")


@dataclass
class AreaLight:
    """Quad light given by 4 corners (in order around the edge) and emitted radiance."""

    corners: np.ndarray                     # (4, 3)
    radiance: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 3)

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def area(self) -> float:
        c = self.corners
        return 0.5 * (np.linalg.norm(np.cross(c[1] - c[0], c[3] - c[0])) +
                      np.linalg.norm(np.cross(c[1] - c[2], c[3] - c[2])))

    @property
    def normal(self) -> np.ndarray:
        c = self.corners
        n = np.cross(c[1] - c[0], c[3] - c[0])
        return n / np.linalg.norm(n)

    def samples(self, n: int = 4) -> np.ndarray:
        """n*n points spread over the quad (bilinear), for soft shadows/irradiance."""
        t = (np.arange(n) + 0.5) / n
        u, v = np.meshgrid(t, t)
        u, v = u.ravel()[:, None], v.ravel()[:, None]
        c = self.corners
        return (1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3]

    def to_dict(self) -> dict:
        return {"type": "area", "corners": [_vec(c) for c in self.corners], "radiance": _vec(self.radiance)}

    @classmethod
    def from_dict(cls, d: dict) -> "AreaLight":
        return cls(np.asarray(d["corners"]), tuple(d.get("radiance", (1.0, 1.0, 1.0))))


@dataclass
class Lighting:
    sh_order: int = 1                       # 1 -> 4 coefficients, 2 -> 9
    sh: np.ndarray | None = None            # (4 or 9, 3) radiance SH
    ambient: tuple[float, float, float] | None = None   # the engine's, from its light detection

    def to_dict(self) -> dict:
        d = {"sh_order": self.sh_order,
             "sh": [] if self.sh is None else [_vec(row) for row in np.asarray(self.sh)]}
        if self.ambient is not None:
            d["ambient"] = _vec(self.ambient)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "Lighting":
        d = d or {}
        sh = d.get("sh") or None
        ambient = tuple(d["ambient"]) if d.get("ambient") is not None else None
        return cls(int(d.get("sh_order", 1)), None if sh is None else np.asarray(sh, dtype=np.float64), ambient)


@dataclass
class ObjectPlacement:
    path: str = "object.glb"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_y_deg: float = 0.0
    scale: float = 1.0

    def to_dict(self) -> dict:
        return {"path": self.path, "transform": {"position": _vec(self.position),
                                                 "rotation_y_deg": float(self.rotation_y_deg),
                                                 "scale": float(self.scale)}}

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectPlacement":
        t = d.get("transform", {})
        return cls(d.get("path", "object.glb"), tuple(t.get("position", (0, 0, 0))),
                   float(t.get("rotation_y_deg", 0.0)), float(t.get("scale", 1.0)))


@dataclass
class Scene:
    camera: Camera
    photo: str = PHOTO_NAME
    scale_reference: dict = field(default_factory=lambda: {"kind": "camera_height", "metres": 1.5})
    objects: list[ObjectPlacement] = field(default_factory=list)   # every object placed in the room
    proxies: list[Proxy] = field(default_factory=list)
    lights: list[AreaLight] = field(default_factory=list)
    # the engine's point/directional lights (detected or placed by hand), kept exactly as written
    engine_lights: list[dict] = field(default_factory=list)
    lighting: Lighting = field(default_factory=Lighting)
    extra: dict = field(default_factory=dict)  # keys we don't own, kept as-is

    @property
    def image_size(self) -> tuple[int, int]:
        return self.camera.width, self.camera.height

    def proxy(self, proxy_id: int) -> Proxy:
        for p in self.proxies:
            if p.id == proxy_id:
                return p
        raise KeyError(proxy_id)

    def next_proxy_id(self) -> int:
        return max((p.id for p in self.proxies), default=-1) + 1

    def to_dict(self) -> dict:
        d = dict(self.extra)
        d.update({
            "version": SCHEMA_VERSION,
            "photo": self.photo,
            "image_size": [self.camera.width, self.camera.height],
            "camera": self.camera.to_dict(),
            "scale_reference": self.scale_reference,
            "proxies": [p.to_dict() for p in self.proxies],
            "lights": [l.to_dict() for l in self.lights] + [dict(l) for l in self.engine_lights],
            "lighting": self.lighting.to_dict(),
        })
        d.pop("object", None)                      # the older single object, replaced by the list
        d["objects"] = [o.to_dict() for o in self.objects]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        known = {"version", "photo", "image_size", "camera", "scale_reference", "object", "objects",
                 "proxies", "lights", "lighting"}
        return cls(
            camera=Camera.from_dict(d["camera"], d["image_size"]),
            photo=d.get("photo", PHOTO_NAME),
            scale_reference=d.get("scale_reference", {"kind": "camera_height", "metres": 1.5}),
            objects=[ObjectPlacement.from_dict(o) for o in scene_objects(dict(d)) if o.get("path")],
            proxies=[proxy_from_dict(p) for p in d.get("proxies", [])],
            lights=[AreaLight.from_dict(l) for l in d.get("lights", []) if l.get("type", "area") == "area"],
            engine_lights=[l for l in d.get("lights", []) if l.get("type", "area") != "area"],
            lighting=Lighting.from_dict(d.get("lighting")),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def save(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text()))


def write_json_atomic(path: str | Path, data: dict) -> None:
    """Write to a temp file in the same folder, then rename over the target."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def update_scene_file(path: str | Path, **sections) -> None:
    """Replace only the given top-level sections, leaving the engine's edits alone.

    e.g. update_scene_file(p, lighting={...}, lights=[...])
    """
    path = Path(path)
    data = json.loads(path.read_text())
    data.update(sections)
    scene_objects(data)                             # an older single "object" becomes the "objects" list
    write_json_atomic(path, data)


def scene_objects(data: dict) -> list[dict]:
    """The placed objects of a raw scene.json dict, edited in place: older files have one "object" (possibly null or
    with an empty path, meaning none), which moves into the "objects" list."""
    legacy = data.pop("object", None)
    objects = data.setdefault("objects", [])
    if legacy and legacy.get("path") and not objects:
        objects.append(legacy)
    return objects


# --- building proxies from clicks -------------------------------------------------

def floor_proxy(proxy_id: int = 0, material: Material | None = None) -> PlaneProxy:
    return PlaneProxy(proxy_id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), material or Material())


def wall_through(proxy_id: int, camera: Camera, uv_a, uv_b, material: Material | None = None) -> PlaneProxy:
    """Vertical wall through two clicked pixels where the wall meets the floor."""
    a, b = camera.pixel_on_floor(np.array([uv_a, uv_b], dtype=np.float64))
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("clicked points are not on the floor (above the horizon?)")
    n = np.cross(b - a, [0.0, 1.0, 0.0])
    n /= np.linalg.norm(n)
    if np.dot(camera.position - a, n) < 0:  # face the camera
        n = -n
    return PlaneProxy(proxy_id, tuple(float(v) for v in a), tuple(float(v) for v in n), material or Material())


def box_on_floor(proxy_id: int, camera: Camera, uv_corners, height: float,
                 material: Material | None = None) -> BoxProxy:
    """Axis-aligned box from clicked floor-contact pixels (any number >= 2) and a height."""
    pts = camera.pixel_on_floor(np.asarray(uv_corners, dtype=np.float64))
    if np.isnan(pts).any():
        raise ValueError("clicked points are not on the floor")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return BoxProxy(proxy_id, (float(lo[0]), 0.0, float(lo[2])), (float(hi[0]), float(height), float(hi[2])),
                    material or Material())


# --- ray casting against proxies -------------------------------------------------

def _hit_plane(o: np.ndarray, d: np.ndarray, p: PlaneProxy) -> tuple[np.ndarray, np.ndarray]:
    n = p.unit_normal()
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((np.asarray(p.point) - o) @ n) / denom
    t = np.where((denom < 0) & (t > 1e-6), t, np.inf)  # front faces only
    return t, np.broadcast_to(n, d.shape)


def _hit_box(o: np.ndarray, d: np.ndarray, b: BoxProxy) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(b.min, dtype=np.float64), np.asarray(b.max, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    tmin = np.nan_to_num(tmin, nan=-np.inf)
    tmax = np.nan_to_num(tmax, nan=np.inf)
    t_near = tmin.max(axis=1)
    t_far = tmax.min(axis=1)
    hit = (t_near <= t_far) & (t_near > 1e-6)
    t = np.where(hit, t_near, np.inf)
    axis = tmin.argmax(axis=1)
    normals = np.zeros_like(d)
    rows = np.arange(len(d))
    normals[rows, axis] = -np.sign(d[rows, axis])
    return t, normals


def cast(origins: np.ndarray, dirs: np.ndarray, proxies: list[Proxy]):
    """Nearest hit of each ray. Returns (t, normals, proxy_ids) with id -1 / t=inf for misses."""
    dirs = np.atleast_2d(dirs)
    o = np.broadcast_to(np.asarray(origins, dtype=np.float64), dirs.shape)
    best_t = np.full(len(dirs), np.inf)
    best_n = np.zeros_like(dirs)
    best_id = np.full(len(dirs), -1)
    for p in proxies:
        if isinstance(p, PlaneProxy):
            t, n = _hit_plane(o, dirs, p)
        else:
            t, n = _hit_box(o, dirs, p)
        closer = t < best_t
        best_t[closer] = t[closer]
        best_n[closer] = n[closer]
        best_id[closer] = p.id
    return best_t, best_n, best_id


def render_gbuffer(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python stand-in for the engine's gbuffer export.

    Returns normals (H, W, 3), ids (H, W) int with -1 = no proxy, depth (H, W)
    as distance along the ray in metres (inf = miss).
    """
    cam = scene.camera
    h, w = cam.height, cam.width
    ys, xs = np.mgrid[0:h, 0:w]
    uv = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    t, n, ids = cast(cam.position, cam.rays(uv), scene.proxies)
    return n.reshape(h, w, 3), ids.reshape(h, w), t.reshape(h, w)
