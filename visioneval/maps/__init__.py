"""Black-box hallucination maps: where a multimodal model consistently fails."""

from visioneval.maps.hallucination import build_map, format_human, to_json
from visioneval.maps.types import FailureEvent, HallucinationMap

__all__ = [
    "FailureEvent",
    "HallucinationMap",
    "build_map",
    "format_human",
    "to_json",
]
