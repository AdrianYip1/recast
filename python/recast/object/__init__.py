"""3D object from one photo, without AI.

Input: one object photo, a few clicks, one real measurement ("it's 32 cm tall").
Output: object.glb in metres, +Y up, pivot at the base, textured from the photo,
plus object_meta.json, object_mask.jpg and object_check.jpg (mesh outline
reprojected on the photo). Photos in are .jpg (anything Pillow reads works).

Modes (recast.object.modes): card, revolution, box, extrude, and inflate (the
older rounded-silhouette mesh from recast.object3d).
"""

MODES = ("card", "revolution", "box", "extrude", "inflate")
