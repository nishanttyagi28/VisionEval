"""Loading and validating image-classification adapter callables."""

from collections.abc import Callable
from importlib import import_module

from visioneval.core.types import ClassificationPrediction, ClassificationSample

ClassificationAdapter = Callable[[ClassificationSample], ClassificationPrediction]


def load_adapter(import_path: str) -> ClassificationAdapter:
    """Load a ``module:callable`` adapter and validate its basic contract."""
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("adapter must use the format 'module:callable'")

    adapter = getattr(import_module(module_name), attribute_name)
    if not callable(adapter):
        raise TypeError("configured adapter must be callable")
    return adapter