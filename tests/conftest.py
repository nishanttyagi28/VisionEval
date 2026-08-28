"""Shared tiny image fixtures. No downloads, no API keys."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visioneval.multimodal.fixtures import solid_scene


@pytest.fixture
def red_square() -> Image.Image:
    return solid_scene("red_square", size=32)


@pytest.fixture
def blue_circle() -> Image.Image:
    return solid_scene("blue_circle", size=32)


@pytest.fixture
def rgb_ramp() -> Image.Image:
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 16
    array[:, :, 1] = np.arange(16, dtype=np.uint8)[:, None] * 16
    array[:, :, 2] = 80
    return Image.fromarray(array)
