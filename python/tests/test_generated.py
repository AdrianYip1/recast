"""The parts of the TripoSR object mode that don't need the model."""

import numpy as np
import trimesh

from recast.object.modes import generated as g


def test_orient_outward_fixes_inside_out_meshes():
    inside_out = trimesh.creation.icosphere(subdivisions=2)
    inside_out = trimesh.Trimesh(inside_out.vertices, inside_out.faces[:, ::-1], process=False)
    fixed = g._orient_outward(inside_out)
    out = (fixed.vertex_normals * (fixed.vertices - fixed.centroid)).sum(1)
    assert (out > 0).all()


def test_upright_undoes_a_camera_tilt():
    bottle = trimesh.creation.cylinder(radius=0.04, height=0.25, sections=48)
    bottle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))   # cylinder axis -> +Y
    tilt = trimesh.transformations.rotation_matrix(np.radians(12), [1, 0, 0])[:3, :3]
    verts = trimesh.remesh.subdivide_to_size(bottle.vertices, bottle.faces, 0.01)[0] @ tilt.T
    R = g.upright(verts)
    straight = verts @ R.T
    assert np.ptp(straight[:, 0]) < 0.085 and np.ptp(straight[:, 2]) < 0.085


def test_delight_removes_light_from_one_side():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(5000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    albedo = np.array([0.6, 0.3, 0.2])
    lit = albedo * (0.6 + 0.3 * n[:, 1:2])                    # brighter on top
    out, coef = g.delight(lit, n)
    assert coef[2] > 0.1                                       # found the light from above
    assert np.std(out[:, 0]) < 0.25 * np.std(lit[:, 0])


def test_uv_bake_covers_the_chart_and_pads_the_rest():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    ids, bary = g._rasterize_uv(uv, np.array([[0, 1, 2]]), 64)
    assert (ids >= 0).sum() > 0.25 * 64 * 64 and np.allclose(bary.sum(1), 1.0)
    tex = np.zeros((64, 64, 3), np.float32)
    tex[ids >= 0] = [1.0, 0.5, 0.0]
    padded = g._pad(tex, ids >= 0)
    assert np.allclose(padded[63, 63], [1.0, 0.5, 0.0])


def test_old_vit_checkpoint_names_are_renamed():
    expected = {"image_tokenizer.model.layers.0.attention.q_proj.weight", "image_tokenizer.model.layers.0.mlp.fc2.bias",
                "decoder.layers.0.weight"}
    ckpt = {"image_tokenizer.model.encoder.layer.0.attention.attention.query.weight": 1,
            "image_tokenizer.model.encoder.layer.0.output.dense.bias": 2, "decoder.layers.0.weight": 3}
    assert set(g._match_checkpoint(ckpt, expected)) == expected
