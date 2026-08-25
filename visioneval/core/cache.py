"""SQLite WAL cache for failure memory and content-addressable predictions."""

import hashlib
import importlib.util
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from visioneval.core.types import ClassificationPrediction, EvaluationRecord

_ATTENTION_RECOVERY_PASSES = 2


class SQLiteCache:
    """Keep local evaluation memory with concurrent-reader-friendly WAL mode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sample_outcomes (
                    sample_id TEXT PRIMARY KEY,
                    correct INTEGER NOT NULL,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    consecutive_passes INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS predictions (cache_key TEXT PRIMARY KEY, label TEXT NOT NULL, confidence REAL NOT NULL);""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(sample_outcomes)")}
            if "fail_count" not in columns:
                connection.execute("ALTER TABLE sample_outcomes ADD COLUMN fail_count INTEGER NOT NULL DEFAULT 0")
                connection.execute("ALTER TABLE sample_outcomes ADD COLUMN consecutive_passes INTEGER NOT NULL DEFAULT 0")
                connection.execute("UPDATE sample_outcomes SET fail_count = 1 WHERE correct = 0")
                connection.execute("UPDATE sample_outcomes SET consecutive_passes = 1 WHERE correct = 1")

    def record(self, records: tuple[EvaluationRecord, ...]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO sample_outcomes(sample_id, correct, fail_count, consecutive_passes) VALUES (?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    correct=excluded.correct,
                    fail_count=sample_outcomes.fail_count + excluded.fail_count,
                    consecutive_passes=CASE WHEN excluded.correct = 1 THEN sample_outcomes.consecutive_passes + 1 ELSE 0 END""",
                [(record.sample_id, int(record.correct), 0 if record.correct else 1, 1 if record.correct else 0) for record in records],
            )

    def previous_failure_ids(self) -> set[str]:
        with self._connect() as connection:
            return {row[0] for row in connection.execute("SELECT sample_id FROM sample_outcomes WHERE fail_count > 0 AND consecutive_passes < ?", (_ATTENTION_RECOVERY_PASSES,))}

    def get_prediction(self, cache_key: str) -> ClassificationPrediction | None:
        with self._connect() as connection:
            row = connection.execute("SELECT label, confidence FROM predictions WHERE cache_key = ?", (cache_key,)).fetchone()
        return ClassificationPrediction(row[0], row[1]) if row else None

    def put_prediction(self, cache_key: str, prediction: ClassificationPrediction) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO predictions(cache_key, label, confidence) VALUES (?, ?, ?)", (cache_key, prediction.label, prediction.confidence))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            yield connection
        finally:
            connection.commit()
            connection.close()


def cache_identities(adapter_spec: str, adapter: object) -> tuple[str, str]:
    """Derive model and preprocess hashes from adapter spec, weights, and code."""
    model_parts = [f"adapter:{adapter_spec}"]
    preprocess_parts: list[str] = []
    origin = _adapter_module_path(adapter_spec)
    if origin is not None:
        digest = f"module:{_hash_file(origin)}"
        model_parts.append(digest)
        preprocess_parts.append(digest)
    weights = getattr(adapter, "model_path", None)
    if isinstance(weights, str) and Path(weights).is_file():
        model_parts.append(f"weights:{_hash_file(Path(weights))}")
    for name in ("architecture", "weights"):
        value = getattr(adapter, name, None)
        if isinstance(value, str) and value:
            model_parts.append(f"{name}:{value}")
    labels = getattr(adapter, "labels", None)
    if isinstance(labels, (list, tuple)):
        model_parts.append("labels:" + ",".join(map(str, labels)))
    size = getattr(adapter, "input_size", None)
    if isinstance(size, int):
        preprocess_parts.append(f"input_size:{size}")
    if not preprocess_parts:
        preprocess_parts.append(f"adapter:{adapter_spec}")
    return _hash_text("|".join(model_parts)), _hash_text("|".join(preprocess_parts))


def prediction_cache_key(model_hash: str, preprocess_hash: str, image_path: str) -> str:
    """Build a reproducible key from model, preprocessing, and image contents."""
    return _hash_text("|".join((model_hash, preprocess_hash, _hash_file(Path(image_path)))))


def _adapter_module_path(adapter_spec: str) -> Path | None:
    module_name = adapter_spec.partition(":")[0]
    if not module_name:
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    path = Path(spec.origin)
    return path if path.is_file() else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()