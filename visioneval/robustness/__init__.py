"""Image corruptions and clean-vs-corrupted degradation scores."""

from visioneval.robustness.corruptions import (
    CORRUPTION_REGISTRY,
    apply_corruption,
    contrast_jitter,
    gaussian_noise,
    motion_blur,
    random_occlusion,
)
from visioneval.robustness.degradation import (
    DegradationReport,
    drop_off,
    resilience_score,
    summarize_degradation,
)

__all__ = [
    "CORRUPTION_REGISTRY",
    "DegradationReport",
    "apply_corruption",
    "contrast_jitter",
    "drop_off",
    "gaussian_noise",
    "motion_blur",
    "random_occlusion",
    "resilience_score",
    "summarize_degradation",
]
