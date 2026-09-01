"""Black-box hallucination maps: where a multimodal model consistently fails."""

from visioneval.maps.hallucination import (
    FailureEvent,
    HallucinationMap,
    build_map,
    format_human,
    to_json,
)

__all__ = [
    "FailureEvent",
    "HallucinationMap",
    "build_map",
    "format_human",
    "to_json",
]
