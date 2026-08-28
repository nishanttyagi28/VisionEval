"""SQLite WAL store for living VLM traps.

Table names are distinct from Phase 1 classification memory
(``sample_outcomes``, ``predictions``) so the two layers cannot clobber
each other even if they share a database file.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from visioneval.traps.types import TrapRecord, utc_now

_CREATE = """PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS vlm_traps (
    trap_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    image_hash TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    expected_objects TEXT NOT NULL DEFAULT '[]',
    absent_objects TEXT NOT NULL DEFAULT '[]',
    probe_type TEXT NOT NULL,
    last_outcome TEXT NOT NULL DEFAULT 'fail',
    fail_count INTEGER NOT NULL DEFAULT 0,
    consecutive_passes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retired INTEGER NOT NULL DEFAULT 0,
    extra TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_vlm_traps_open ON vlm_traps(retired, sample_id);
CREATE INDEX IF NOT EXISTS idx_vlm_traps_model ON vlm_traps(model, probe_type);
"""


class TrapStore:
    """Keep durable hallucination traps with concurrent-reader-friendly WAL mode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_CREATE)

    def get(self, trap_id: str) -> TrapRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM vlm_traps WHERE trap_id = ?", (trap_id,)).fetchone()
        return self._to_record(row) if row else None

    def list_traps(self, *, retired: bool | None = None, model: str | None = None) -> list[TrapRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if retired is not None:
            clauses.append("retired = ?")
            params.append(int(retired))
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        sql = "SELECT * FROM vlm_traps"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY retired ASC, fail_count DESC, trap_id ASC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._to_record(row) for row in rows]

    def list_open(self, *, model: str | None = None) -> list[TrapRecord]:
        return self.list_traps(retired=False, model=model)

    def count_open(self, *, model: str | None = None) -> int:
        return len(self.list_open(model=model))

    def upsert_failure(self, trap: TrapRecord) -> TrapRecord:
        """Record a hallucination. Re-opens a retired trap (consecutive_passes resets)."""
        now = utc_now()
        existing = self.get(trap.trap_id)
        if existing is None:
            record = TrapRecord(
                trap_id=trap.trap_id,
                model=trap.model,
                sample_id=trap.sample_id,
                image_hash=trap.image_hash,
                prompt=trap.prompt,
                expected_objects=trap.expected_objects,
                absent_objects=trap.absent_objects,
                probe_type=trap.probe_type,
                last_outcome="fail",
                fail_count=max(trap.fail_count, 1),
                consecutive_passes=0,
                created_at=trap.created_at or now,
                updated_at=now,
                retired=False,
                extra=dict(trap.extra),
            )
            self._insert(record)
            return record
        extra = dict(existing.extra)
        extra.update(trap.extra)
        updated = TrapRecord(
            trap_id=existing.trap_id,
            model=trap.model or existing.model,
            sample_id=existing.sample_id,
            image_hash=trap.image_hash or existing.image_hash,
            prompt=trap.prompt or existing.prompt,
            expected_objects=trap.expected_objects or existing.expected_objects,
            absent_objects=trap.absent_objects or existing.absent_objects,
            probe_type=existing.probe_type,
            last_outcome="fail",
            fail_count=existing.fail_count + 1,
            consecutive_passes=0,
            created_at=existing.created_at,
            updated_at=now,
            retired=False,
            extra=extra,
        )
        self._update(updated)
        return updated

    def record_outcome(self, trap_id: str, passed: bool, *, retire_after: int = 2) -> TrapRecord:
        """Apply one replay result. Two consecutive passes retire (default)."""
        existing = self.get(trap_id)
        if existing is None:
            raise KeyError(f"unknown trap_id {trap_id!r}")
        now = utc_now()
        if passed:
            consecutive = existing.consecutive_passes + 1
            retired = consecutive >= retire_after
            updated = TrapRecord(
                trap_id=existing.trap_id,
                model=existing.model,
                sample_id=existing.sample_id,
                image_hash=existing.image_hash,
                prompt=existing.prompt,
                expected_objects=existing.expected_objects,
                absent_objects=existing.absent_objects,
                probe_type=existing.probe_type,
                last_outcome="pass",
                fail_count=existing.fail_count,
                consecutive_passes=consecutive,
                created_at=existing.created_at,
                updated_at=now,
                retired=retired,
                extra=existing.extra,
            )
        else:
            extra = dict(existing.extra)
            updated = TrapRecord(
                trap_id=existing.trap_id,
                model=existing.model,
                sample_id=existing.sample_id,
                image_hash=existing.image_hash,
                prompt=existing.prompt,
                expected_objects=existing.expected_objects,
                absent_objects=existing.absent_objects,
                probe_type=existing.probe_type,
                last_outcome="fail",
                fail_count=existing.fail_count + 1,
                consecutive_passes=0,
                created_at=existing.created_at,
                updated_at=now,
                retired=False,
                extra=extra,
            )
        self._update(updated)
        return updated

    def snapshot(self) -> dict[str, TrapRecord]:
        return {trap.trap_id: trap for trap in self.list_traps()}

    def _insert(self, trap: TrapRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO vlm_traps(
                    trap_id, model, sample_id, image_hash, prompt, expected_objects, absent_objects,
                    probe_type, last_outcome, fail_count, consecutive_passes, created_at, updated_at,
                    retired, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._values(trap),
            )

    def _update(self, trap: TrapRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE vlm_traps SET
                    model=?, sample_id=?, image_hash=?, prompt=?, expected_objects=?, absent_objects=?,
                    probe_type=?, last_outcome=?, fail_count=?, consecutive_passes=?, created_at=?,
                    updated_at=?, retired=?, extra=?
                WHERE trap_id=?""",
                self._values(trap)[1:] + (trap.trap_id,),
            )

    @staticmethod
    def _values(trap: TrapRecord) -> tuple[object, ...]:
        return (
            trap.trap_id,
            trap.model,
            trap.sample_id,
            trap.image_hash,
            trap.prompt,
            json.dumps(list(trap.expected_objects)),
            json.dumps(list(trap.absent_objects)),
            trap.probe_type,
            trap.last_outcome,
            trap.fail_count,
            trap.consecutive_passes,
            trap.created_at,
            trap.updated_at,
            int(trap.retired),
            json.dumps(trap.extra),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> TrapRecord:
        return TrapRecord.from_row(dict(row))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.commit()
            connection.close()
