import numpy as np

from recast.evaluate import change_mask, score


def test_identical_images():
    img = np.random.default_rng(0).random((40, 50, 3))
    s = score(img, img)
    assert s.psnr == float("inf") or s.psnr > 100
    assert np.isclose(s.ssim, 1.0)


def test_region_scores_focus_on_the_object():
    rng = np.random.default_rng(1)
    truth = rng.random((60, 80, 3)) * 0.2 + 0.4
    render = truth.copy()
    render[20:40, 30:50] += 0.2  # object rendered too bright
    region = np.zeros((60, 80), bool)
    region[20:40, 30:50] = True

    s = score(render, truth, region)
    assert s.region_pixels == 400
    assert s.region_psnr < s.psnr       # error is concentrated in the region
    assert np.isclose(s.region_psnr, 10 * np.log10(1 / 0.04), atol=0.01)
    assert s.region_ssim < 1.0


def test_change_mask_finds_object_and_shadow():
    empty = np.full((50, 50, 3), 0.5)
    render = empty.copy()
    render[10:20, 10:20] = 0.9   # object
    render[20:24, 10:20] = 0.4   # its shadow
    m = change_mask(render, empty, grow_px=0)
    assert m[10:24, 10:20].all()
    assert m.sum() == 140
