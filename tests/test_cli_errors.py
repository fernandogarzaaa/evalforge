from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.cli import main


def test_missing_ingest_file_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(tmp_path / "nope.jsonl"), "--db", str(tmp_path / "x.db")]) == 2
    err = capsys.readouterr().err
    assert "evalforge ingest: error:" in err and "nope.jsonl" in err


def test_missing_compare_reports_are_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2
    assert "evalforge compare: error:" in capsys.readouterr().err


def test_invalid_json_report_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "a.json"
    bad.write_text("not json")
    assert main(["compare", str(bad), str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
