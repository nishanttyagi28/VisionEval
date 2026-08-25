"""SQLite WAL cache for failure memory and content-addressable predictions."""

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from visioneval.core.types import ClassificationPrediction, EvaluationRecord


class SQLiteCache:
    """Keep local evaluation memory with concurrent-reader-friendly WAL mode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sample_outcomes (sample_id TEXT PRIMARY KEY, correct INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions (cache_key TEXT PRIMARY KEY, label TEXT NOT NULL, confidence REAL NOT NULL);""")

    def record(self, records: tuple[EvaluationRecord, ...]) -> None:
        with self._connect() as connection:
            connection.executemany("INSERT INTO sample_outcomes(sample_id, correct) VALUES (?, ?) ON CONFLICT(sample_id) DO UPDATE SET correct=excluded.correct", [(record.sample_id, int(record.correct)) for record in records])

    def previous_failure_ids(self) -> set[str]:
        with self._connect() as connection:
            return {row[0] for row in connection.execute("SELECT sample_id FROM sample_outcomes WHERE correct = 0")}

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


def prediction_cache_key(model_hash: str, preprocess_hash: str, image_path: str) -> str:
    """Build a reproducible key from model, preprocessing, and image contents."""
    return _hash_text("|".join((model_hash, preprocess_hash, _hash_file(Path(image_path)))))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()