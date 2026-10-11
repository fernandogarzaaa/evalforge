from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalforge.redact import Redactor, find_unredacted, luhn_valid


def test_luhn() -> None:
    assert luhn_valid("4111 1111 1111 1111")
    assert luhn_valid("5500-0000-0000-0004")
    assert luhn_valid("4242424242424242")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("1234")
    assert not luhn_valid("1" * 20)
    assert luhn_valid("0" * 13)
    assert not luhn_valid("0" * 12)
    assert luhn_valid("0" * 19)
    assert luhn_valid("79927398713") is False  # valid Luhn but only 11 digits
    assert luhn_valid("4012888888881881")
    assert not luhn_valid("4012888888881882")


@pytest.mark.parametrize(
    "text,expected,kind",
    [
        ("mail me at maria.santos12@gmail.com now", "mail me at [EMAIL] now", "email"),
        ("call +63 917 555 1234 today", "call [PHONE] today", "phone"),
        ("call (415) 555-0199 today", "call [PHONE] today", "phone"),
        ("call +44 20 7946 0958 today", "call [PHONE] today", "phone"),
        ("card 4111 1111 1111 1111 ok", "card [CARD] ok", "credit_card"),
        ("server 10.0.12.254 down", "server [IP] down", "ipv4"),
        ("key sk-abcdefghijklmnop1234 leaked", "key [API_KEY] leaked", "api_key"),
        ("aws AKIAABCDEFGHIJKLMNOP here", "aws [API_KEY] here", "api_key"),
    ],
)
def test_default_rules(text: str, expected: str, kind: str) -> None:
    r = Redactor()
    assert r.redact(text) == expected
    assert r.counts[kind] == 1
    assert sum(r.counts.values()) == 1


def test_non_pii_is_left_alone() -> None:
    r = Redactor()
    text = "order 1234 5678 9012 3456 failed on 2024-05-01; version 1.2.3; 600 requests per minute"
    assert r.redact(text) == text
    assert sum(r.counts.values()) == 0


def test_counts_accumulate() -> None:
    r = Redactor()
    r.redact("a@b.co and c@d.io")
    r.redact("e@f.org")
    assert r.counts["email"] == 3


def test_config_adds_and_disables_rules(tmp_path: Path) -> None:
    cfg = tmp_path / "rules.json"
    cfg.write_text(
        json.dumps(
            {
                "rules": [
                    {"name": "employee_id", "pattern": r"EMP-\d{5}"},
                    {"name": "ticket", "pattern": r"T#\d+", "placeholder": "<T>"},
                ],
                "disable": ["ipv4"],
            }
        )
    )
    r = Redactor.from_config(cfg)
    assert r.redact("EMP-12345 at 10.0.0.1 re T#77 a@b.co") == "[EMPLOYEE_ID] at 10.0.0.1 re <T> [EMAIL]"
    only = Redactor.from_config(cfg, include_defaults=False)
    assert only.redact("x@y.com EMP-00001") == "x@y.com [EMPLOYEE_ID]"
    assert [rule.name for rule in only.rules] == ["employee_id", "ticket"]
    assert Redactor.from_config(None).rules == Redactor().rules


def test_find_unredacted() -> None:
    assert find_unredacted("left a@b.co here", ["a@b.co", "", "zz"]) == ["a@b.co"]
