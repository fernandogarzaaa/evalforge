"""Source adapters: turn JSON logs, OpenTelemetry traces and database rows into raw dicts.

Adapters only reshape data. Validation, redaction and deduplication happen once, in
``evalforge.ingest``, so every source gets identical treatment.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """One JSON object per line. Unparseable lines are yielded as ``{"_parse_error": ...}``."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_parse_error": f"line {lineno}: {exc.msg}"}
                continue
            yield obj if isinstance(obj, dict) else {"_parse_error": f"line {lineno}: not an object"}


def _otel_value(value: dict[str, Any]) -> Any:
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        return int(value["intValue"])
    return None


def _nanos_to_iso(nanos: int) -> str:
    dt = datetime.fromtimestamp(nanos / 1e9, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def otel_span_to_raw(span: dict[str, Any], service: str | None = None) -> dict[str, Any] | None:
    """Map one OTLP/JSON span carrying GenAI semantic-convention attributes to a raw record.

    Supports ``gen_ai.request.model``/``gen_ai.response.model``, ``gen_ai.usage.*`` tokens and
    the OpenLLMetry-style indexed content attributes ``gen_ai.prompt.N.role|content`` and
    ``gen_ai.completion.0.content``. Spans without a GenAI model attribute are skipped.
    """
    attrs = {a["key"]: _otel_value(a.get("value", {})) for a in span.get("attributes", [])}
    model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
    if not model:
        return None
    system_parts: list[str] = []
    user_parts: list[str] = []
    i = 0
    while f"gen_ai.prompt.{i}.content" in attrs:
        role = attrs.get(f"gen_ai.prompt.{i}.role", "user")
        content = str(attrs[f"gen_ai.prompt.{i}.content"])
        (system_parts if role == "system" else user_parts).append(content)
        i += 1
    start = int(span.get("startTimeUnixNano", 0))
    end = int(span.get("endTimeUnixNano", start))
    feedback = attrs.get("app.feedback")
    return {
        "request_id": span.get("spanId", ""),
        "timestamp": _nanos_to_iso(start) if start > 0 else None,  # 0 = unset in OTLP
        "feature": attrs.get("app.feature") or service or "unknown",
        "model": model,
        "system_prompt": "\n".join(system_parts),
        "user_prompt": "\n".join(user_parts),
        "response": attrs.get("gen_ai.completion.0.content") or "",
        "latency_ms": max(0.0, (end - start) / 1e6),
        "prompt_tokens": attrs.get("gen_ai.usage.input_tokens", attrs.get("gen_ai.usage.prompt_tokens", 0)),
        "completion_tokens": attrs.get("gen_ai.usage.output_tokens", attrs.get("gen_ai.usage.completion_tokens", 0)),
        "session_id": attrs.get("app.session_id"),
        "feedback": feedback if feedback else None,
        "content_filtered": bool(attrs.get("app.content_filtered", False)),
        "confidence": attrs.get("app.confidence"),
        "metadata": {"trace_id": span.get("traceId", "")},
    }


def read_otel(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read an OTLP/JSON trace export (``{"resourceSpans": [...]}``)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for rs in doc.get("resourceSpans", []):
        res_attrs = {a["key"]: _otel_value(a.get("value", {})) for a in rs.get("resource", {}).get("attributes", [])}
        service = res_attrs.get("service.name")
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                raw = otel_span_to_raw(span, service)
                if raw is not None:
                    yield raw


def read_sqlite(
    path: str | Path, table: str = "llm_logs", column_map: dict[str, str] | None = None
) -> Iterator[dict[str, Any]]:
    """Read rows from an application database table, renaming columns via ``column_map``."""
    if not table.replace("_", "").isalnum():
        raise ValueError(f"unsafe table name: {table!r}")
    mapping = column_map or {}
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(f"SELECT * FROM {table}"):  # noqa: S608 - table name validated above
            raw = {mapping.get(k, k): row[k] for k in row.keys()}  # noqa: SIM118 - sqlite3.Row is not a dict
            for flag in ("edited", "content_filtered"):
                if flag in raw:
                    raw[flag] = bool(raw[flag])
            yield raw
    finally:
        con.close()


def detect_format(path: str | Path) -> str:
    """Guess the adapter for a file from its extension and first bytes."""
    p = Path(path)
    if p.suffix in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    if p.suffix == ".jsonl":
        return "jsonl"
    with open(p, encoding="utf-8") as fh:
        head = fh.read(4096)
    return "otel" if '"resourceSpans"' in head else "jsonl"


def read_any(path: str | Path, fmt: str | None = None, table: str = "llm_logs") -> Iterator[dict[str, Any]]:
    fmt = fmt or detect_format(path)
    if fmt == "jsonl":
        return read_jsonl(path)
    if fmt == "otel":
        return read_otel(path)
    if fmt == "sqlite":
        return read_sqlite(path, table)
    raise ValueError(f"unknown log format: {fmt}")
