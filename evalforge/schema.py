"""Unified production-log schema.

Every LLM call site in an application is normalised into one ``LogRecord`` so the
rest of the pipeline never has to care whether a row came from a JSON log line, an
OpenTelemetry span or a database table.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

FEEDBACK_VALUES = ("up", "down")
REQUIRED_TEXT = ("request_id", "timestamp", "feature", "model", "user_prompt")


@dataclass
class LogRecord:
    request_id: str
    timestamp: str  # ISO-8601, UTC
    feature: str
    model: str
    user_prompt: str
    response: str = ""
    system_prompt: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    session_id: str | None = None
    feedback: str | None = None  # "up" | "down" | None
    edited: bool = False
    content_filtered: bool = False
    confidence: float | None = None  # e.g. mean token probability, if the provider exposes it
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_error(self) -> bool:
        """An interaction counts as an error when it was filtered or produced no text."""
        return self.content_filtered or not self.response.strip()

    def fingerprint(self) -> str:
        """Content hash used to drop re-delivered copies that carry a fresh request id."""
        key = "\x1f".join(
            [self.feature, self.model, self.system_prompt, self.user_prompt, self.response, self.timestamp]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_count(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0 and int(value) == value


def validate(raw: dict[str, Any]) -> tuple[LogRecord | None, list[str]]:
    """Check one raw record against the schema.

    Returns ``(record, [])`` on success and ``(None, errors)`` when the record is not
    schema compliant. Unknown keys are kept in ``metadata`` rather than dropped.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["record is not an object"]

    for name in REQUIRED_TEXT:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} is required")

    if isinstance(raw.get("timestamp"), str) and raw["timestamp"].strip():
        try:
            ts = parse_timestamp(raw["timestamp"]).isoformat().replace("+00:00", "Z")
        except ValueError:
            errors.append("timestamp is not ISO-8601")
            ts = ""
    else:
        ts = ""

    for name in ("response", "system_prompt"):
        if name in raw and raw[name] is not None and not isinstance(raw[name], str):
            errors.append(f"{name} must be a string")

    latency = raw.get("latency_ms", 0.0)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        errors.append("latency_ms must be a non-negative number")
    for name in ("prompt_tokens", "completion_tokens"):
        if not _is_count(raw.get(name, 0)):
            errors.append(f"{name} must be a non-negative integer")

    feedback = raw.get("feedback")
    if feedback is not None and feedback not in FEEDBACK_VALUES:
        errors.append("feedback must be 'up', 'down' or null")

    confidence = raw.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0
    ):
        errors.append("confidence must be within [0, 1]")

    session = raw.get("session_id")
    if session is not None and not isinstance(session, str):
        errors.append("session_id must be a string")

    if errors:
        return None, errors

    known = set(LogRecord.__dataclass_fields__)
    metadata = dict(raw.get("metadata") or {})
    metadata.update({k: v for k, v in raw.items() if k not in known})
    record = LogRecord(
        request_id=raw["request_id"].strip(),
        timestamp=ts,
        feature=raw["feature"].strip(),
        model=raw["model"].strip(),
        user_prompt=raw["user_prompt"],
        response=raw.get("response") or "",
        system_prompt=raw.get("system_prompt") or "",
        latency_ms=float(latency),
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        session_id=session,
        feedback=feedback,
        edited=bool(raw.get("edited", False)),
        content_filtered=bool(raw.get("content_filtered", False)),
        confidence=None if confidence is None else float(confidence),
        metadata=metadata,
    )
    return record, []
