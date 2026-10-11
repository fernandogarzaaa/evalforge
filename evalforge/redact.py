"""Configurable PII redaction applied before anything is written to the log store."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path


def luhn_valid(digits: str) -> bool:
    """Luhn checksum, used so that order numbers are not mistaken for card numbers."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            n = n // 10 + n % 10  # digit sum of the doubled value
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    placeholder: str
    check: Callable[[str], bool] | None = None


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    Rule("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b"), "[API_KEY]"),
    Rule("credit_card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "[CARD]", luhn_valid),
    Rule("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "[IP]"),
    Rule(
        "phone",
        re.compile(
            r"(?<![\w-])(?<!\d[ .-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}"
            r"(?![\w-])(?![ .]\d)"
        ),
        "[PHONE]",
    ),
)


@dataclass
class Redactor:
    rules: tuple[Rule, ...] = DEFAULT_RULES
    counts: Counter[str] = field(default_factory=Counter)

    def redact(self, text: str) -> str:
        for rule in self.rules:
            text = self._apply(rule, text)
        return text

    def _apply(self, rule: Rule, text: str) -> str:
        def sub(match: re.Match[str]) -> str:
            if rule.check is not None and not rule.check(match.group(0)):
                return match.group(0)
            self.counts[rule.name] += 1
            return rule.placeholder

        return rule.pattern.sub(sub, text)

    @classmethod
    def from_config(cls, path: str | Path | None, include_defaults: bool = True) -> Redactor:
        """Load extra rules from JSON: ``{"rules": [{"name", "pattern", "placeholder"}], "disable": [names]}``."""
        if path is None:
            return cls()
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        disabled = set(cfg.get("disable", []))
        base = [r for r in DEFAULT_RULES if r.name not in disabled] if include_defaults else []
        extra = [
            Rule(r["name"], re.compile(r["pattern"]), r.get("placeholder", f"[{r['name'].upper()}]"))
            for r in cfg.get("rules", [])
        ]
        return cls(rules=tuple(base + extra))


def find_unredacted(text: str, secrets: Iterable[str]) -> list[str]:
    """Return the secrets that still appear verbatim in ``text`` (used to measure recall)."""
    return [s for s in secrets if s and s in text]
