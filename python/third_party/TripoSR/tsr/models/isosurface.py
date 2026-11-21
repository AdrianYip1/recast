from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # recast: scikit-image's marching cubes instead of torchmcubes (which needs compiling)
        from skimage.measure import marching_cubes

        level = -level.view(self.resolution, self.resolution, self.resolution).detach().float().cpu().numpy()
        verts, faces, _, _ = marching_cubes(level, 0.0)
        # skimage already returns (x, y, z) in grid order; recast fixes the winding by the mesh's volume sign
        v_pos = torch.from_numpy(verts.astype(np.float32).copy())
        t_pos_idx = torch.from_numpy(faces.astype(np.int64).copy())
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos, t_pos_idx
