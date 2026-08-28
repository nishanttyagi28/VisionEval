"""Dataclasses for living VLM hallucination traps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

PROBE_TYPES = ("pope", "judge", "caption")
ProbeType = Literal["pope", "judge", "caption"]
OUTCOMES = ("fail", "pass")

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def utc_now() -> str:
    """UTC timestamp with second precision, suffixed Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str, *, limit: int = 48) -> str:
    """Make a short filesystem-safe token for trap ids."""
    cleaned = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return (cleaned or "x")[:limit]


def make_trap_id(model: str, sample_id: str, probe_type: str, discriminator: str) -> str:
    """Stable, readable trap identity (not a SQLite rowid)."""
    raw = f"{probe_type}:{slug(model, limit=32)}:{slug(sample_id, limit=32)}:{slug(discriminator, limit=64)}"
    return raw[:180]


def image_identity(sample_id: str, image_path: str | None = None, color: str | None = None) -> str:
    """Content hash of an image file, or a stand-in hash for fixture samples."""
    from pathlib import Path

    if image_path:
        path = Path(image_path)
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
    material = f"{sample_id}|{color or ''}|{image_path or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class TrapRecord:
    """One durable hallucination trap. Beaten only after consecutive passes."""

    trap_id: str
    model: str
    sample_id: str
    image_hash: str
    prompt: str
    expected_objects: tuple[str, ...]
    absent_objects: tuple[str, ...]
    probe_type: str
    last_outcome: str
    fail_count: int
    consecutive_passes: int
    created_at: str
    updated_at: str
    retired: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_objects"] = list(self.expected_objects)
        payload["absent_objects"] = list(self.absent_objects)
        return payload

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TrapRecord":
        expected = row.get("expected_objects") or []
        absent = row.get("absent_objects") or []
        extra = row.get("extra") or {}
        if isinstance(expected, str):
            expected = json.loads(expected)
        if isinstance(absent, str):
            absent = json.loads(absent)
        if isinstance(extra, str):
            extra = json.loads(extra)
        return cls(
            trap_id=str(row["trap_id"]),
            model=str(row["model"]),
            sample_id=str(row["sample_id"]),
            image_hash=str(row.get("image_hash") or ""),
            prompt=str(row.get("prompt") or ""),
            expected_objects=tuple(str(item) for item in expected),
            absent_objects=tuple(str(item) for item in absent),
            probe_type=str(row["probe_type"]),
            last_outcome=str(row.get("last_outcome") or "fail"),
            fail_count=int(row.get("fail_count") or 0),
            consecutive_passes=int(row.get("consecutive_passes") or 0),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
            retired=bool(int(row.get("retired") or 0)),
            extra=dict(extra) if isinstance(extra, dict) else {},
        )
