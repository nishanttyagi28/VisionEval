"""Tiny in-memory RGB fixtures so tests and the Streamlit demo never download data."""

from __future__ import annotations

from PIL import Image, ImageDraw


def solid_scene(color: str, size: int = 64) -> Image.Image:
    """Build a small RGB scene used by the demo/test fixtures.

    * ``red_square`` — red square on white
    * ``blue_circle`` — blue ellipse on white
    * ``green_split`` — green left half, yellow right half
    """
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = size // 5
    if color == "red_square":
        draw.rectangle((margin, margin, size - margin, size - margin), fill=(200, 24, 24))
    elif color == "blue_circle":
        draw.ellipse((margin, margin, size - margin, size - margin), fill=(24, 72, 200))
    elif color == "green_split":
        draw.rectangle((0, 0, size // 2, size), fill=(24, 160, 72))
        draw.rectangle((size // 2, 0, size, size), fill=(220, 200, 40))
    else:
        draw.rectangle((margin, margin, size - margin, size - margin), fill=(128, 128, 128))
    return image


def load_sample_image(image_path: str | None, color: str | None, root: str | None = None) -> Image.Image:
    """Load a sample image from disk, or synthesise one from ``color``."""
    if image_path:
        from pathlib import Path

        path = Path(image_path)
        if root and not path.is_absolute():
            path = Path(root) / path
        if path.exists():
            with Image.open(path) as handle:
                return handle.convert("RGB")
    if color:
        return solid_scene(color)
    return solid_scene("red_square")


DEMO_SAMPLES = [
    {
        "id": "red_square",
        "color": "red_square",
        "caption": "A red square on a white background.",
        "objects": ["square", "red square"],
        "absent_objects": ["cat", "car"],
        "spatial_notes": "square in the center",
    },
    {
        "id": "blue_circle",
        "color": "blue_circle",
        "caption": "A blue circle on a white background.",
        "objects": ["circle", "blue circle"],
        "absent_objects": ["tree", "person"],
        "spatial_notes": "circle in the center",
    },
]
