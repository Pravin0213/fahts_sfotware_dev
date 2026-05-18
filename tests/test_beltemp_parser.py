"""Tests for BELTEMP file parser and cumulative temperature calculator."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fahts.core.io.beltemp_parser import (
    BeltempRecord,
    cumulative_temperatures,
    element_temperature_at,
    parse_beltemp,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_beltemp(content: str) -> Path:
    """Write content to a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fem", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


_SIMPLE_CONTENT = """\
'  BELTEMP test file
'
 LCASETIM          1         1.0
'              load case   element     Mean
 BELTEMP           1           1      10.000    0.000    0.000
 BELTEMP           1           2      20.000    0.000    0.000
 LCASETIM          2         2.0
 BELTEMP           2           1      15.000    1.000   -1.000
 BELTEMP           2           2      25.000   -1.000    1.000
"""


class TestParseBeltemp:
    def test_parses_correct_number_of_records(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        assert len(records) == 4

    def test_record_fields(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        r = records[0]
        assert r.load_case == 1
        assert r.time_min == 1.0
        assert r.element_id == 1
        assert abs(r.dT_mean - 10.0) < 1e-6

    def test_gradient_fields(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        r = records[2]  # second time step, element 1
        assert abs(r.dT_grad_y - 1.0) < 1e-6
        assert abs(r.dT_grad_z - -1.0) < 1e-6

    def test_skip_comments(self):
        content = "' comment\n' another\n LCASETIM 1 1.0\n BELTEMP 1 5 3.0 0.0 0.0\n"
        path = _write_beltemp(content)
        records = parse_beltemp(path)
        assert len(records) == 1
        assert records[0].element_id == 5

    def test_empty_file_returns_empty(self):
        path = _write_beltemp("' empty\n")
        assert parse_beltemp(path) == []

    def test_returns_list_of_beltemp_records(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        for r in records:
            assert isinstance(r, BeltempRecord)


class TestCumulativeTemperatures:
    def test_accumulates_increments(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records, T_initial=20.0)

        # element 1: dT = +10, +15 → T = 20, 30, 45
        times, T_mean, _, _ = cum[1]
        assert abs(T_mean[0] - 20.0) < 1e-9   # t=0 initial
        assert abs(T_mean[1] - 30.0) < 1e-9   # t=1 min
        assert abs(T_mean[2] - 45.0) < 1e-9   # t=2 min

    def test_times_include_zero(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records)
        times, _, _, _ = cum[1]
        assert times[0] == 0.0

    def test_gradient_accumulates(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records)
        _, _, T_gy, T_gz = cum[1]
        assert T_gy[0] == 0.0           # initial gradient
        assert abs(T_gy[1] - 0.0) < 1e-9  # after step 1: dT_grad_y=0
        assert abs(T_gy[2] - 1.0) < 1e-9  # after step 2: dT_grad_y=1

    def test_two_elements_independent(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records, T_initial=20.0)
        _, T1, _, _ = cum[1]
        _, T2, _, _ = cum[2]
        assert T1[-1] != T2[-1]

    def test_default_T_initial_20(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records)
        _, T_mean, _, _ = cum[1]
        assert T_mean[0] == 20.0


class TestElementTemperatureAt:
    def test_exact_time(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records, T_initial=20.0)
        T = element_temperature_at(cum, 1, 1.0)
        assert abs(T - 30.0) < 1e-6

    def test_interpolation(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records, T_initial=20.0)
        T = element_temperature_at(cum, 1, 1.5)
        assert 30.0 < T < 45.0

    def test_missing_element_raises(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records)
        with pytest.raises(KeyError):
            element_temperature_at(cum, 999, 1.0)

    def test_before_start_returns_initial(self):
        path = _write_beltemp(_SIMPLE_CONTENT)
        records = parse_beltemp(path)
        cum = cumulative_temperatures(records, T_initial=20.0)
        T = element_temperature_at(cum, 1, 0.0)
        assert abs(T - 20.0) < 1e-6


class TestRealBeltempFile:
    """Integration test with the actual USFOS benchmark file."""
    _PATH = Path(__file__).parents[1] / "usfos_verification_results" / "fahts_beltemp.fem"

    def test_file_parses_without_error(self):
        if not self._PATH.exists():
            pytest.skip("Benchmark BELTEMP file not found")
        records = parse_beltemp(self._PATH)
        assert len(records) > 0

    def test_15_time_steps(self):
        if not self._PATH.exists():
            pytest.skip("Benchmark BELTEMP file not found")
        records = parse_beltemp(self._PATH)
        cum = cumulative_temperatures(records)
        # Pick a common element and check 15 output steps + t=0
        eid = list(cum.keys())[0]
        times, _, _, _ = cum[eid]
        assert len(times) == 16  # 15 steps + initial t=0

    def test_temperatures_rise_over_time(self):
        if not self._PATH.exists():
            pytest.skip("Benchmark BELTEMP file not found")
        records = parse_beltemp(self._PATH)
        cum = cumulative_temperatures(records)
        eid = list(cum.keys())[0]
        _, T_mean, _, _ = cum[eid]
        assert T_mean[-1] > T_mean[0]
