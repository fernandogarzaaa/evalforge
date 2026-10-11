from __future__ import annotations

from typing import Any

import pytest

from evalforge.schema import LogRecord
from evalforge.store import Store


def raw(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "request_id": "r1",
        "timestamp": "2026-09-14T10:00:00Z",
        "feature": "support_bot",
        "model": "gpt-4o-mini",
        "user_prompt": "How do I reset my password?",
        "response": "Open Settings > Security and click Reset password; the link expires after 30 minutes.",
        "system_prompt": "You are a support bot.",
        "latency_ms": 800.0,
        "prompt_tokens": 20,
        "completion_tokens": 18,
        "session_id": "s1",
    }
    base.update(over)
    return base


def rec(**over: Any) -> LogRecord:
    fields = raw(**over)
    return LogRecord(**{k: v for k, v in fields.items() if k in LogRecord.__dataclass_fields__})


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()
