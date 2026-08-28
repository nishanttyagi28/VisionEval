"""Living VLM hallucination traps: harvest, replay, retire after consecutive passes."""

from visioneval.traps.baseline import (
    TrapRegression,
    compare_trap_baseline,
    load_trap_baseline,
    save_trap_baseline,
)
from visioneval.traps.generator import generate_hard_negative, mint_hard_negatives
from visioneval.traps.harvest import HarvestSummary, harvest_report
from visioneval.traps.runner import TrapRunResult, run_open_traps
from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord

__all__ = [
    "HarvestSummary",
    "TrapRecord",
    "TrapRegression",
    "TrapRunResult",
    "TrapStore",
    "compare_trap_baseline",
    "generate_hard_negative",
    "harvest_report",
    "load_trap_baseline",
    "mint_hard_negatives",
    "run_open_traps",
    "save_trap_baseline",
]
