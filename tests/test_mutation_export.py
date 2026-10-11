"""Targeted tests for evalforge.export (mutation-testing survivors)."""

from __future__ import annotations

import json
from pathlib import Path

from evalforge.export import load_golden, write_golden
from tests.test_export_eval import case


def test_write_golden_creates_nested_dirs_and_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "g.json"
    write_golden([case()], target, created_at="t")
    write_golden([case(), case(id="c2")], target, created_at="t")  # same dir again: no error
    assert len(load_golden(target)) == 2
    jl = tmp_path / "x" / "y" / "g.jsonl"
    write_golden([case()], jl)
    write_golden([case()], jl)
    assert len(load_golden(jl)) == 1


def test_write_golden_keeps_unicode_and_two_space_indent(tmp_path: Path) -> None:
    c = case(input={"system": "sys", "prompt": "¿Dónde está mi café?"})
    p = write_golden([c], tmp_path / "g.json", created_at="t")
    text = p.read_text(encoding="utf-8")
    assert "¿Dónde está mi café?" in text and "\\u00bf" not in text
    assert text.endswith("}\n")
    assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"
    pl = write_golden([c], tmp_path / "g.jsonl")
    line = pl.read_text(encoding="utf-8")
    assert "café" in line and "\\u00e9" not in line and line.endswith("\n") and line.count("\n") == 1
