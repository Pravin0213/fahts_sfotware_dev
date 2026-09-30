"""fahts-run: headless vessel-case runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fahts.cli import main

pytest.importorskip("CoolProp")
DECK = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases" / "M07-0004"


def test_deck_to_case_and_csv(tmp_path, capsys):
    rc = main([str(DECK), "--t-end", "60", "--out", str(tmp_path / "s.csv"),
               "--save-case", str(tmp_path / "c.vcase.json"), "-q"])
    out = capsys.readouterr()
    assert rc == 0 and "Peak pressure" in out.out and "material" in out.err
    assert len(pd.read_csv(tmp_path / "s.csv")) == 7
    rc = main([str(tmp_path / "c.vcase.json"), "--t-end", "30", "--out", str(tmp_path / "r.xlsx"),
               "-q"])
    assert rc == 0 and "time series" in pd.ExcelFile(tmp_path / "r.xlsx").sheet_names


def test_bad_input_returns_2(tmp_path, capsys):
    assert main([str(tmp_path / "missing.vcase.json"), "-q"]) == 2
    (tmp_path / "bad.vcase.json").write_text('{"schema": "x"}')
    assert main([str(tmp_path / "bad.vcase.json"), "-q"]) == 2
    assert "cannot read" in capsys.readouterr().err
