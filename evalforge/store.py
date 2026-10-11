"""SQLite-backed store for normalised logs, categories, test cases, reviews and runs.

SQLite is deliberate: zero infrastructure, one file, transactional, and good for
millions of rows. The access layer is small enough to port to Postgres/ClickHouse
when traffic outgrows a single file (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from evalforge.schema import LogRecord


def encode_vector(vec: list[float]) -> str:
    """float32 little-endian, base64: compact and exact (JSON floats would be ~5x larger)."""
    return base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode("ascii")


def decode_vector(text: str) -> list[float]:
    return [float(v) for v in np.frombuffer(base64.b64decode(text), dtype="<f4")]


SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
  request_id TEXT PRIMARY KEY,
  fingerprint TEXT UNIQUE NOT NULL,
  day TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS logs_day ON logs(day);
CREATE TABLE IF NOT EXISTS ingested_files (sha256 TEXT PRIMARY KEY, path TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS categories (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  centroid TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  created_day TEXT
);
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  category_id INTEGER,
  category TEXT,
  difficulty TEXT,
  polarity TEXT,
  label_source TEXT,
  confidence REAL,
  data TEXT NOT NULL,
  embedding TEXT NOT NULL,
  first_seen TEXT,
  created_day TEXT
);
CREATE INDEX IF NOT EXISTS cases_status ON cases(status);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  case_id TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  decision TEXT NOT NULL,
  labels TEXT,
  note TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS pipeline_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS eval_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, model TEXT, created_at TEXT, report TEXT
);
"""

CASE_COLUMNS = (
    "id",
    "status",
    "category_id",
    "category",
    "difficulty",
    "polarity",
    "label_source",
    "confidence",
    "first_seen",
    "created_day",
)


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        self.con.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.con:
            yield self.con

    # ---- logs -------------------------------------------------------------------------
    def has_log(self, request_id: str, fingerprint: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM logs WHERE request_id = ? OR fingerprint = ? LIMIT 1", (request_id, fingerprint)
        ).fetchone()
        return row is not None

    def add_logs(self, records: Iterable[LogRecord]) -> int:
        n = 0
        with self.tx() as con:
            for r in records:
                cur = con.execute(
                    "INSERT OR IGNORE INTO logs VALUES (?,?,?,?,?,?,?)",
                    (
                        r.request_id,
                        r.metadata.get("fingerprint") or r.fingerprint(),
                        r.timestamp[:10],
                        r.timestamp,
                        r.feature,
                        r.model,
                        json.dumps(r.to_dict()),
                    ),
                )
                n += cur.rowcount
        return n

    def logs_for_day(self, day: str) -> list[LogRecord]:
        rows = self.con.execute("SELECT record FROM logs WHERE day = ? ORDER BY timestamp, request_id", (day,))
        return [LogRecord(**json.loads(r["record"])) for r in rows]

    def log_days(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT day FROM logs ORDER BY day")]

    def count_logs(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM logs").fetchone()[0])

    def file_seen(self, sha256: str) -> bool:
        return self.con.execute("SELECT 1 FROM ingested_files WHERE sha256 = ?", (sha256,)).fetchone() is not None

    def mark_file(self, sha256: str, path: str, stats: dict[str, Any]) -> None:
        with self.tx() as con:
            con.execute("INSERT OR REPLACE INTO ingested_files VALUES (?,?,?)", (sha256, path, json.dumps(stats)))

    # ---- categories -------------------------------------------------------------------
    def categories(self) -> list[dict[str, Any]]:
        rows = self.con.execute("SELECT * FROM categories ORDER BY id")
        return [{**dict(r), "centroid": json.loads(r["centroid"])} for r in rows]

    def upsert_category(self, cat_id: int | None, name: str, centroid: list[float], size: int, day: str) -> int:
        with self.tx() as con:
            if cat_id is None:
                cur = con.execute(
                    "INSERT INTO categories (name, centroid, size, created_day) VALUES (?,?,?,?)",
                    (name, json.dumps(centroid), size, day),
                )
                return int(cur.lastrowid or 0)
            con.execute(
                "UPDATE categories SET centroid = ?, size = ? WHERE id = ?", (json.dumps(centroid), size, cat_id)
            )
            return cat_id

    # ---- cases ------------------------------------------------------------------------
    def add_case(self, case: dict[str, Any], embedding: list[float]) -> None:
        with self.tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO cases (id, status, category_id, category, difficulty, polarity, label_source,"
                " confidence, data, embedding, first_seen, created_day) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    case["id"],
                    case["status"],
                    case.get("category_id"),
                    case.get("category"),
                    case.get("difficulty"),
                    case.get("polarity"),
                    case.get("label_source"),
                    case.get("confidence"),
                    json.dumps(case),
                    encode_vector(embedding),
                    case.get("first_seen"),
                    case.get("created_day"),
                ),
            )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        row = self.con.execute("SELECT data FROM cases WHERE id = ?", (case_id,)).fetchone()
        return None if row is None else dict(json.loads(row["data"]))

    def update_case(self, case: dict[str, Any]) -> None:
        row = self.con.execute("SELECT embedding FROM cases WHERE id = ?", (case["id"],)).fetchone()
        if row is None:
            raise KeyError(case["id"])
        self.add_case(case, decode_vector(row["embedding"]))

    def cases(self, status: str | Iterable[str] | None = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self.con.execute("SELECT data FROM cases ORDER BY created_day, id")
        else:
            statuses = [status] if isinstance(status, str) else list(status)
            marks = ",".join("?" * len(statuses))
            rows = self.con.execute(
                f"SELECT data FROM cases WHERE status IN ({marks}) ORDER BY created_day, id",  # noqa: S608
                statuses,
            )
        return [dict(json.loads(r["data"])) for r in rows]

    def case_embeddings(
        self, statuses: Iterable[str] = ("active", "pending_review")
    ) -> tuple[list[str], list[list[float]]]:
        st = list(statuses)
        marks = ",".join("?" * len(st))
        rows = self.con.execute(f"SELECT id, embedding FROM cases WHERE status IN ({marks}) ORDER BY id", st)  # noqa: S608
        ids, embs = [], []
        for r in rows:
            ids.append(r["id"])
            embs.append(decode_vector(r["embedding"]))
        return ids, embs

    # ---- reviews, runs ----------------------------------------------------------------
    def add_review(
        self,
        case_id: str,
        reviewer: str,
        decision: str,
        labels: dict[str, Any] | None,
        note: str | None,
        created_at: str,
    ) -> None:
        with self.tx() as con:
            con.execute(
                "INSERT INTO reviews (case_id, reviewer, decision, labels, note, created_at) VALUES (?,?,?,?,?,?)",
                (case_id, reviewer, decision, json.dumps(labels) if labels else None, note, created_at),
            )

    def reviews(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute("SELECT * FROM reviews ORDER BY id")]

    def add_pipeline_run(self, day: str, stats: dict[str, Any]) -> None:
        with self.tx() as con:
            con.execute("INSERT INTO pipeline_runs (day, stats) VALUES (?,?)", (day, json.dumps(stats)))

    def pipeline_runs(self) -> list[dict[str, Any]]:
        rows = self.con.execute("SELECT day, stats FROM pipeline_runs ORDER BY id")
        return [{"day": r["day"], **json.loads(r["stats"])} for r in rows]

    def add_eval_run(self, name: str, model: str, created_at: str, report: dict[str, Any]) -> int:
        with self.tx() as con:
            cur = con.execute(
                "INSERT INTO eval_runs (name, model, created_at, report) VALUES (?,?,?,?)",
                (name, model, created_at, json.dumps(report)),
            )
            return int(cur.lastrowid or 0)

    def eval_runs(self) -> list[dict[str, Any]]:
        rows = self.con.execute("SELECT * FROM eval_runs ORDER BY id")
        return [{**dict(r), "report": json.loads(r["report"])} for r in rows]
