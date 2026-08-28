"""Image corruption pipelines stay seeded, sized, and identity at severity 0."""

import numpy as np
import pytest
from PIL import Image

from visioneval.robustness.corruptions import (
    apply_corruption,
    contrast_jitter,
    gaussian_noise,
    motion_blur,
    random_occlusion,
)


def _arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def test_severity_zero_is_identity(rgb_ramp) -> None:
    for fn in (gaussian_noise, motion_blur, contrast_jitter, random_occlusion):
        out = fn(rgb_ramp, 0.0, seed=0)
        assert out.size == rgb_ramp.size
        assert out.mode == "RGB"
        np.testing.assert_array_equal(_arr(out), _arr(rgb_ramp))


def test_gaussian_noise_is_seed_stable_and_changes_pixels(rgb_ramp) -> None:
    a = gaussian_noise(rgb_ramp, 0.5, seed=7)
    b = gaussian_noise(rgb_ramp, 0.5, seed=7)
    c = gaussian_noise(rgb_ramp, 0.5, seed=8)
    np.testing.assert_array_equal(_arr(a), _arr(b))
    assert not np.array_equal(_arr(a), _arr(c))
    assert not np.array_equal(_arr(a), _arr(rgb_ramp))


def test_motion_blur_preserves_shape_and_changes_pixels(rgb_ramp) -> None:
    out = motion_blur(rgb_ramp, 1.0, seed=0)
    assert out.size == rgb_ramp.size
    assert not np.array_equal(_arr(out), _arr(rgb_ramp))


def test_contrast_jitter_increases_dynamic_range(rgb_ramp) -> None:
    out = contrast_jitter(rgb_ramp, 1.0, seed=0)
    src = _arr(rgb_ramp).astype(np.float32)
    dst = _arr(out).astype(np.float32)
    assert dst.std() >= src.std() - 1e-6


def test_occlusion_paints_black_pixels(rgb_ramp) -> None:
    out = random_occlusion(rgb_ramp, 1.0, seed=1)
    assert (_arr(out) == 0).all(axis=2).any()
    a = random_occlusion(rgb_ramp, 0.8, seed=3)
    b = random_occlusion(rgb_ramp, 0.8, seed=3)
    np.testing.assert_array_equal(_arr(a), _arr(b))


def test_apply_corruption_dispatches_and_rejects_unknown(rgb_ramp) -> None:
    out = apply_corruption(rgb_ramp, "gaussian_noise", 0.2, seed=0)
    assert out.mode == "RGB"
    with pytest.raises(KeyError, match="unknown corruption"):
        apply_corruption(rgb_ramp, "not_a_real_corruption", 0.1)
