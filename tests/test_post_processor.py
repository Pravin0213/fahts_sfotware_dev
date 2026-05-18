"""
Unit tests for Task 3.9 — PostProcessor.

Uses a synthetic TemperatureField (no solver needed for most tests).
Integration tests run the full analysis_runner → PostProcessor pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fahts.core.results.post_processor import ElementSummary, PostProcessor, T_CRIT_DEFAULT
from fahts.core.results.temperature_field import TemperatureField


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_tf(
    *,
    eids: list[int] | None = None,
    T_centroid: np.ndarray | None = None,
    times: np.ndarray | None = None,
    with_section: bool = False,
) -> TemperatureField:
    """
    Build a synthetic TemperatureField.

    Default: two elements (10, 20), three time steps (0, 60, 120 s).
    elem 10: 20 → 500 → 700 °C  (exceeds 600 °C)
    elem 20: 20 → 200 → 400 °C  (never reaches 600 °C)
    """
    if eids is None:
        eids = [10, 20]
    if times is None:
        times = np.array([0.0, 60.0, 120.0])
    if T_centroid is None:
        T_centroid = np.array([
            [20.0,  20.0],
            [500.0, 200.0],
            [700.0, 400.0],
        ])

    T_section: dict[int, np.ndarray] = {}
    if with_section:
        n_nodes = 8
        for eid in eids:
            idx = eids.index(eid)
            T_section[eid] = T_centroid[:, idx : idx + 1] * np.ones((1, n_nodes))

    return TemperatureField(
        times=times,
        element_ids=eids,
        T_centroid=T_centroid,
        T_section=T_section,
    )


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_wraps_result(self):
        tf = _make_tf()
        pp = PostProcessor(tf)
        assert pp.result is tf

    def test_n_elements(self):
        assert PostProcessor(_make_tf()).n_elements == 2

    def test_n_steps(self):
        assert PostProcessor(_make_tf()).n_steps == 3

    def test_duration_s(self):
        assert PostProcessor(_make_tf()).duration_s == pytest.approx(120.0)

    def test_duration_min(self):
        assert PostProcessor(_make_tf()).duration_min == pytest.approx(2.0)


# ── ElementSummary ────────────────────────────────────────────────────────────

class TestElementSummary:
    def test_eid(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.eid == 10

    def test_T_initial(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.T_initial == pytest.approx(20.0)

    def test_T_peak_centroid_exceeds_600(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.T_peak_centroid == pytest.approx(700.0)

    def test_T_peak_centroid_below_600(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(20)
        assert s.T_peak_centroid == pytest.approx(400.0)

    def test_T_peak_nodal_none_without_section(self):
        pp = PostProcessor(_make_tf(with_section=False))
        assert pp.element_summary(10).T_peak_nodal is None

    def test_T_peak_nodal_set_with_section(self):
        pp = PostProcessor(_make_tf(with_section=True))
        s = pp.element_summary(10)
        assert s.T_peak_nodal is not None
        assert s.T_peak_nodal == pytest.approx(700.0, rel=1e-3)

    def test_t_crit_500_interpolated(self):
        # elem 10: 20 at t=0, 500 at t=60 → exactly at t=60
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.t_crit_500 == pytest.approx(60.0, abs=1e-6)

    def test_t_crit_500_none_for_cool_element(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(20)
        assert s.t_crit_500 is None

    def test_t_crit_600_interpolated(self):
        # elem 10: 500 at t=60, 700 at t=120 → 600 at t=90
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.t_crit_600 == pytest.approx(90.0, abs=1e-6)

    def test_t_crit_600_none(self):
        pp = PostProcessor(_make_tf())
        assert pp.element_summary(20).t_crit_600 is None

    def test_fire_resistance_min(self):
        pp = PostProcessor(_make_tf())
        s = pp.element_summary(10)
        assert s.fire_resistance_min == pytest.approx(1.5)   # 90 s = 1.5 min

    def test_fire_resistance_min_none(self):
        pp = PostProcessor(_make_tf())
        assert pp.element_summary(20).fire_resistance_min is None

    def test_passes_600_true(self):
        assert PostProcessor(_make_tf()).element_summary(20).passes_600 is True

    def test_passes_600_false(self):
        assert PostProcessor(_make_tf()).element_summary(10).passes_600 is False

    def test_element_summaries_all(self):
        pp = PostProcessor(_make_tf())
        summaries = pp.element_summaries()
        assert len(summaries) == 2
        assert all(isinstance(s, ElementSummary) for s in summaries)
        assert [s.eid for s in summaries] == [10, 20]


# ── Bulk queries ──────────────────────────────────────────────────────────────

class TestBulkQueries:
    def test_critical_elements(self):
        pp = PostProcessor(_make_tf())
        assert pp.critical_elements() == [10]

    def test_passing_elements(self):
        pp = PostProcessor(_make_tf())
        assert pp.passing_elements() == [20]

    def test_critical_elements_custom_threshold(self):
        pp = PostProcessor(_make_tf())
        # At 450°C threshold, elem 10 (peak=700) is critical; elem 20 (peak=400) passes
        assert pp.critical_elements(T_crit=450.0) == [10]

    def test_critical_elements_all_pass(self):
        pp = PostProcessor(_make_tf())
        assert pp.critical_elements(T_crit=800.0) == []

    def test_peak_temperatures(self):
        pp = PostProcessor(_make_tf())
        peaks = pp.peak_temperatures()
        assert peaks[10] == pytest.approx(700.0)
        assert peaks[20] == pytest.approx(400.0)

    def test_fire_resistance_minutes(self):
        pp = PostProcessor(_make_tf())
        fr = pp.fire_resistance_minutes(10)
        assert fr == pytest.approx(1.5)

    def test_fire_resistance_minutes_none(self):
        pp = PostProcessor(_make_tf())
        assert pp.fire_resistance_minutes(20) is None

    def test_minimum_fire_resistance(self):
        # Only elem 10 fails → min = 1.5 min
        pp = PostProcessor(_make_tf())
        assert pp.minimum_fire_resistance() == pytest.approx(1.5)

    def test_minimum_fire_resistance_none_when_all_pass(self):
        # Low-temperature run — no element reaches 600°C
        tf = TemperatureField(
            times=np.array([0.0, 60.0]),
            element_ids=[1],
            T_centroid=np.array([[20.0], [300.0]]),
        )
        pp = PostProcessor(tf)
        assert pp.minimum_fire_resistance() is None


# ── Time-history tables ───────────────────────────────────────────────────────

class TestHistoryTables:
    def test_centroid_history_keys(self):
        pp = PostProcessor(_make_tf())
        h = pp.centroid_history(10)
        assert "times_s" in h and "times_min" in h and "T_centroid_C" in h

    def test_centroid_history_shape(self):
        pp = PostProcessor(_make_tf())
        h = pp.centroid_history(10)
        assert len(h["times_s"]) == 3
        assert len(h["T_centroid_C"]) == 3

    def test_centroid_history_values(self):
        pp = PostProcessor(_make_tf())
        h = pp.centroid_history(10)
        np.testing.assert_allclose(h["T_centroid_C"], [20.0, 500.0, 700.0])

    def test_centroid_history_times_min(self):
        pp = PostProcessor(_make_tf())
        h = pp.centroid_history(10)
        np.testing.assert_allclose(h["times_min"], [0.0, 1.0, 2.0])

    def test_all_centroid_history_keys(self):
        pp = PostProcessor(_make_tf())
        h = pp.all_centroid_history()
        assert "T_eid_10" in h and "T_eid_20" in h

    def test_all_centroid_history_shape(self):
        pp = PostProcessor(_make_tf())
        h = pp.all_centroid_history()
        assert h["T_eid_10"].shape == (3,)
        assert h["T_eid_20"].shape == (3,)


# ── Summary text ──────────────────────────────────────────────────────────────

class TestSummaryText:
    def test_contains_element_ids(self):
        text = PostProcessor(_make_tf()).summary_text()
        assert "10" in text and "20" in text

    def test_contains_pass_fail(self):
        text = PostProcessor(_make_tf()).summary_text()
        assert "PASS" in text and "FAIL" in text

    def test_contains_duration(self):
        text = PostProcessor(_make_tf()).summary_text()
        assert "2.0 min" in text

    def test_contains_critical_count(self):
        text = PostProcessor(_make_tf()).summary_text()
        assert "1 / 2" in text


# ── CSV export ────────────────────────────────────────────────────────────────

class TestCSVExport:
    def test_to_csv_creates_file(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pp.to_csv(path)
            assert path.exists()

    def test_to_csv_has_header(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pp.to_csv(path)
            lines = path.read_text().splitlines()
            assert lines[0].startswith("element_id")

    def test_to_csv_row_count(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pp.to_csv(path)
            lines = path.read_text().splitlines()
            assert len(lines) == 3   # header + 2 elements

    def test_to_csv_element_ids_present(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pp.to_csv(path)
            content = path.read_text()
            assert "10" in content and "20" in content

    def test_to_temperature_csv_creates_file(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            pp.to_temperature_csv(path)
            assert path.exists()

    def test_to_temperature_csv_row_count(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            pp.to_temperature_csv(path)
            lines = path.read_text().splitlines()
            assert len(lines) == 4   # header + 3 time steps

    def test_to_temperature_csv_columns(self):
        pp = PostProcessor(_make_tf())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            pp.to_temperature_csv(path)
            header = path.read_text().splitlines()[0]
            assert "T_eid_10" in header and "T_eid_20" in header


# ── Pandas DataFrame export ───────────────────────────────────────────────────

class TestDataFrameExport:
    pytest.importorskip("pandas")

    def test_to_dataframe_shape(self):
        import pandas as pd
        pp = PostProcessor(_make_tf())
        df = pp.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 9)

    def test_to_dataframe_index(self):
        pp = PostProcessor(_make_tf())
        df = pp.to_dataframe()
        assert list(df.index) == [10, 20]

    def test_to_dataframe_T_peak_column(self):
        pp = PostProcessor(_make_tf())
        df = pp.to_dataframe()
        assert df.loc[10, "T_peak_centroid_C"] == pytest.approx(700.0)
        assert df.loc[20, "T_peak_centroid_C"] == pytest.approx(400.0)

    def test_to_dataframe_passes_600(self):
        pp = PostProcessor(_make_tf())
        df = pp.to_dataframe()
        assert df.loc[10, "passes_600"] == False   # noqa: E712  (numpy.bool_ compatible)
        assert df.loc[20, "passes_600"] == True    # noqa: E712

    def test_to_history_dataframe_shape(self):
        import pandas as pd
        pp = PostProcessor(_make_tf())
        df = pp.to_history_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)   # time_min + T_eid_10 + T_eid_20

    def test_to_history_dataframe_index(self):
        pp = PostProcessor(_make_tf())
        df = pp.to_history_dataframe()
        np.testing.assert_allclose(df.index, [0.0, 60.0, 120.0])

    def test_to_history_dataframe_values(self):
        pp = PostProcessor(_make_tf())
        df = pp.to_history_dataframe()
        np.testing.assert_allclose(df["T_eid_10"].values, [20.0, 500.0, 700.0])


# ── Integration: real solver → PostProcessor ──────────────────────────────────

class TestIntegration:
    def test_real_solver_pipeline(self):
        """Run the full analysis_runner → PostProcessor pipeline."""
        from pathlib import Path

        import numpy as np

        from fahts.core.heat.solver.analysis_runner import run_analysis
        from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
        from fahts.core.model.element import BeamElement
        from fahts.core.model.fem_model import FEMModel
        from fahts.core.model.material import SteelMaterial
        from fahts.core.model.node import Node
        from fahts.core.model.section import BoxSection
        from fahts.core.results.analysis_config import AnalysisConfig

        n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(nid=2, x=1.0, y=0.0, z=0.0)
        sec = BoxSection(sid=10, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
        mat = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)
        elem = BeamElement(
            eid=101, n1=1, n2=2, mat_id=1, geom_id=10, lcoor_id=0,
            length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            local_z=np.array([0.0, 0.0, 1.0]),
        )
        model = FEMModel(
            nodes={1: n1, 2: n2},
            elements={101: elem},
            sections={10: sec},
            materials={1: mat},
            groups={}, unitvecs={},
            source_file=Path("synthetic.fem"),
        )
        zone = FireZone(
            name="Z", center=np.array([0.5, 0.0, 0.0]),
            dims=np.array([2.0, 2.0, 2.0]),
            curve=FireCurve(FireCurveType.ISO_834),
            epsilon_fire=1.0, h_conv=25.0, active=True,
        )
        cfg = AnalysisConfig(t_end=120.0, dt=60.0, output_dt=120.0, n_layers=1)
        tf = run_analysis(model, [zone], cfg)

        pp = PostProcessor(tf)
        assert pp.n_elements == 1
        s = pp.element_summary(101)
        assert s.T_peak_centroid > 20.0
        assert s.T_initial == pytest.approx(20.0, abs=0.1)
        # Summary text should not raise
        assert "101" in pp.summary_text()
