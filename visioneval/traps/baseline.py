"""Git-trackable lockfile of open-trap ids/outcomes for regression checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord, utc_now


@dataclass(frozen=True)
class TrapRegression:
    """Machine-actionable trap gate result.

    Regressions:
    - ``new_open``: open traps that were not locked (and not previously retired)
    - ``reappeared``: traps that were retired in the lockfile but are open again
    - ``worse``: locked open traps whose fail_count grew or flipped pass→fail

    ``still_open`` lists every currently open trap id for CI consumers.
    ``recovered`` lists locked open traps that are now retired (not a regression).
    """

    is_regression: bool
    reappeared: tuple[str, ...]
    worse: tuple[str, ...]
    recovered: tuple[str, ...]
    new_open: tuple[str, ...] = ()
    still_open: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_regression": self.is_regression,
            "new_open": list(self.new_open),
            "reappeared": list(self.reappeared),
            "worse": list(self.worse),
            "recovered": list(self.recovered),
            "still_open": list(self.still_open),
        }


def trap_lockfile_payload(store: TrapStore) -> dict[str, Any]:
    """Serialize current trap outcomes for a git-trackable JSON lockfile."""
    traps = store.list_traps()
    open_traps = {
        trap.trap_id: {
            "last_outcome": trap.last_outcome,
            "fail_count": trap.fail_count,
            "consecutive_passes": trap.consecutive_passes,
            "retired": False,
            "probe_type": trap.probe_type,
            "model": trap.model,
            "sample_id": trap.sample_id,
        }
        for trap in traps
        if not trap.retired
    }
    retired_ids = tuple(sorted(trap.trap_id for trap in traps if trap.retired))
    return {
        "created_at": utc_now(),
        "open_traps": open_traps,
        "retired_ids": list(retired_ids),
    }


def save_trap_baseline(path: Path, store: TrapStore) -> dict[str, Any]:
    payload = trap_lockfile_payload(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_trap_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def compare_trap_baseline(
    locked: Mapping[str, Any],
    current: Mapping[str, TrapRecord] | TrapStore,
) -> TrapRegression:
    """Detect new open traps, reappearance of retired traps, or worse outcomes."""
    if isinstance(current, TrapStore):
        current = current.snapshot()
    locked_open = locked.get("open_traps") or {}
    retired_ids = set(locked.get("retired_ids") or ())
    known = set(locked_open) | retired_ids
    current_open = {trap_id: trap for trap_id, trap in current.items() if not trap.retired}
    still_open = tuple(sorted(current_open))

    reappeared = tuple(sorted(trap_id for trap_id in current_open if trap_id in retired_ids))
    new_open = tuple(sorted(trap_id for trap_id in current_open if trap_id not in known))
    recovered = tuple(
        sorted(trap_id for trap_id in locked_open if trap_id in current and current[trap_id].retired)
    )

    worse: list[str] = []
    for trap_id, snapshot in locked_open.items():
        trap = current.get(trap_id)
        if trap is None or trap.retired:
            continue
        locked_fails = int(snapshot.get("fail_count") or 0)
        locked_outcome = str(snapshot.get("last_outcome") or "fail")
        got_worse = trap.fail_count > locked_fails
        flipped = locked_outcome == "pass" and trap.last_outcome == "fail"
        if got_worse or flipped:
            worse.append(trap_id)
    worse_ids = tuple(sorted(worse))
    return TrapRegression(
        is_regression=bool(reappeared or worse_ids or new_open),
        reappeared=reappeared,
        worse=worse_ids,
        recovered=recovered,
        new_open=new_open,
        still_open=still_open,
    )


def format_trap_regression(regression: TrapRegression) -> str:
    """Human-readable gate summary (deterministic)."""
    status = "REGRESSION" if regression.is_regression else "PASS"
    lines = [
        f"Trap gate: {status}",
        f"  new_open:    {','.join(regression.new_open) or '-'}",
        f"  reappeared:  {','.join(regression.reappeared) or '-'}",
        f"  worse:       {','.join(regression.worse) or '-'}",
        f"  recovered:   {','.join(regression.recovered) or '-'}",
        f"  still_open:  {','.join(regression.still_open) or '-'}",
    ]
    return "\n".join(lines) + "\n"


def trap_regression_json(regression: TrapRegression) -> str:
    return json.dumps(regression.as_dict(), indent=2, sort_keys=True) + "\n"
