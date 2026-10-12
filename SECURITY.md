# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **Report a vulnerability** button (Security tab → Advisories) on this repository, rather than in a public issue. Include steps to reproduce and the affected version or commit. You should get a first response within a few days.

## Scope notes

evalforge processes production LLM logs, which can contain personal data and secrets. Redaction runs before anything is stored or sent to a judge model, but it is pattern-based: review `evalforge/redact.py` against your own data before pointing it at real traffic, and keep exported datasets access-controlled.
