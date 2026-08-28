"""Deterministic image corruption pipelines for robustness stress tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from PIL import Image

CorruptionName = Literal["gaussian_noise", "motion_blur", "contrast_jitter", "occlusion"]


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _from_rgb_array(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)


def gaussian_noise(
    image: Image.Image,
    severity: float,
    *,
    seed: int | None = 0,
) -> Image.Image:
    """Add i.i.d. Gaussian noise. ``severity`` in ``[0, 1]`` scales sigma up to 80."""
    severity = float(min(1.0, max(0.0, severity)))
    array = _as_rgb_array(image)
    if severity == 0.0:
        return _from_rgb_array(array)
    rng = np.random.default_rng(seed)
    sigma = 80.0 * severity
    noise = rng.normal(0.0, sigma, size=array.shape)
    return _from_rgb_array(array + noise)


def motion_blur(
    image: Image.Image,
    severity: float,
    *,
    seed: int | None = 0,
) -> Image.Image:
    """Horizontal motion blur. Kernel width grows with severity (odd, 1..15)."""
    del seed
    severity = float(min(1.0, max(0.0, severity)))
    array = _as_rgb_array(image)
    if severity == 0.0:
        return _from_rgb_array(array)
    kernel_size = max(1, int(round(1 + severity * 14)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return _from_rgb_array(array)
    pad = kernel_size // 2
    padded = np.pad(array, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    blurred = np.zeros_like(array)
    weight = 1.0 / kernel_size
    for offset in range(kernel_size):
        blurred += padded[:, offset : offset + array.shape[1], :] * weight
    return _from_rgb_array(blurred)


def contrast_jitter(
    image: Image.Image,
    severity: float,
    *,
    seed: int | None = 0,
) -> Image.Image:
    """Scale contrast around the per-channel mean. ``severity`` 0 is identity.

    Positive direction (higher severity) *increases* contrast; the magnitude is
    ``1 + 1.5 * severity`` so tests can assert a measurable pixel change.
    """
    del seed
    severity = float(min(1.0, max(0.0, severity)))
    array = _as_rgb_array(image)
    if severity == 0.0:
        return _from_rgb_array(array)
    factor = 1.0 + 1.5 * severity
    mean = array.mean(axis=(0, 1), keepdims=True)
    return _from_rgb_array((array - mean) * factor + mean)


def random_occlusion(
    image: Image.Image,
    severity: float,
    *,
    seed: int | None = 0,
) -> Image.Image:
    """Paint a deterministic black rectangle. Area scales with ``severity``."""
    severity = float(min(1.0, max(0.0, severity)))
    array = _as_rgb_array(image)
    if severity == 0.0:
        return _from_rgb_array(array)
    height, width = array.shape[:2]
    rng = np.random.default_rng(seed)
    frac = 0.08 + 0.42 * severity
    box_h = max(1, int(round(height * frac)))
    box_w = max(1, int(round(width * frac)))
    box_h = min(box_h, height)
    box_w = min(box_w, width)
    top = int(rng.integers(0, max(1, height - box_h + 1)))
    left = int(rng.integers(0, max(1, width - box_w + 1)))
    array[top : top + box_h, left : left + box_w, :] = 0
    return _from_rgb_array(array)


CORRUPTION_REGISTRY: dict[str, Callable[..., Image.Image]] = {
    "gaussian_noise": gaussian_noise,
    "motion_blur": motion_blur,
    "contrast_jitter": contrast_jitter,
    "occlusion": random_occlusion,
}


def apply_corruption(
    image: Image.Image,
    name: str,
    severity: float,
    *,
    seed: int | None = 0,
) -> Image.Image:
    """Apply a named corruption. Unknown names raise ``KeyError``."""
    try:
        fn = CORRUPTION_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(CORRUPTION_REGISTRY))
        raise KeyError(f"unknown corruption {name!r}; expected one of: {known}") from exc
    return fn(image, severity, seed=seed)
