"""Ingestion: adapter -> schema validation -> PII redaction -> dedup -> store."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalforge.adapters import read_any
from evalforge.redact import Redactor
from evalforge.schema import LogRecord, validate
from evalforge.store import Store


@dataclass
class IngestStats:
    read: int = 0
    invalid: int = 0
    duplicates: int = 0
    stored: int = 0
    errors: Counter[str] = field(default_factory=Counter)
    redactions: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "stored": self.stored,
            "errors": dict(self.errors),
            "redactions": dict(self.redactions),
        }

    def merge(self, other: IngestStats) -> None:
        self.read += other.read
        self.invalid += other.invalid
        self.duplicates += other.duplicates
        self.stored += other.stored
        self.errors.update(other.errors)
        self.redactions.update(other.redactions)


def redact_record(record: LogRecord, redactor: Redactor) -> LogRecord:
    record.user_prompt = redactor.redact(record.user_prompt)
    record.system_prompt = redactor.redact(record.system_prompt)
    record.response = redactor.redact(record.response)
    return record


def normalise(
    raws: Iterable[dict[str, Any]], store: Store, redactor: Redactor | None = None
) -> tuple[list[LogRecord], IngestStats]:
    """Validate, redact and dedup raw records. Returns the new records (not yet stored)."""
    redactor = redactor or Redactor()
    before = Counter(redactor.counts)
    stats = IngestStats()
    seen_ids: set[str] = set()
    seen_fp: set[str] = set()
    out: list[LogRecord] = []
    for raw in raws:
        stats.read += 1
        if "_parse_error" in raw:
            stats.invalid += 1
            stats.errors["unparseable"] += 1
            continue
        record, errors = validate(raw)
        if record is None:
            stats.invalid += 1
            stats.errors.update(errors)
            continue
        # Dedup on the raw content so redaction cannot merge two genuinely different calls.
        fp = record.fingerprint()
        if record.request_id in seen_ids or fp in seen_fp or store.has_log(record.request_id, fp):
            stats.duplicates += 1
            continue
        seen_ids.add(record.request_id)
        seen_fp.add(fp)
        record.metadata["fingerprint"] = fp
        out.append(redact_record(record, redactor))
    stats.redactions = redactor.counts - before
    return out, stats


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(
    path: str | Path, store: Store, redactor: Redactor | None = None, fmt: str | None = None, force: bool = False
) -> IngestStats:
    """Ingest one file. Files already ingested (same bytes) are skipped unless ``force``."""
    sha = file_sha256(path)
    if store.file_seen(sha) and not force:
        return IngestStats()
    records, stats = normalise(read_any(path, fmt), store, redactor)
    stats.stored = store.add_logs(records)
    store.mark_file(sha, str(path), stats.to_dict())
    return stats
