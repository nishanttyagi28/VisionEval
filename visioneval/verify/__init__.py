"""TruthGraph-style claim verification for VisionEval multimodal outputs.

Vendored and adapted from https://github.com/nishanttyagi28/truthgraph
(core text analyzer + verifier only; no FastAPI).
"""

from __future__ import annotations

from visioneval.verify.adapter import (
    build_dossier,
    format_human,
    load_verify_input,
    to_json,
    verify_row,
)
from visioneval.verify.models import Claim, Evidence, VerificationResult
from visioneval.verify.verifier import verify_claim

__all__ = [
    "Claim",
    "Evidence",
    "VerificationResult",
    "verify_claim",
    "verify_row",
    "build_dossier",
    "load_verify_input",
    "format_human",
    "to_json",
]
