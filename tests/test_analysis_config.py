"""
Unit tests for Task 3.7 — AnalysisConfig dataclass.

Tests cover default construction, validation rules, n_output_steps, and summary.
The Qt dialog (RunAnalysisDialog) is smoke-tested for importability only.
"""
import pytest
from fahts.core.results.analysis_config import AnalysisConfig


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_minimal_required_fields(self):
        cfg = AnalysisConfig(t_end=7200.0, dt=30.0, output_dt=60.0)
        assert cfg.t_end == 7200.0
        assert cfg.dt == 30.0
        assert cfg.output_dt == 60.0

    def test_defaults(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0)
        assert cfg.n_layers == 1
        assert cfg.elem_size is None
        assert cfg.element_ids == []

    def test_explicit_n_layers(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, n_layers=2)
        assert cfg.n_layers == 2

    def test_explicit_elem_size(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, elem_size=0.02)
        assert cfg.elem_size == pytest.approx(0.02)

    def test_element_ids(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, element_ids=[1, 2, 3])
        assert cfg.element_ids == [1, 2, 3]

    def test_element_ids_default_not_shared(self):
        """Each instance gets its own list (no mutable default sharing)."""
        a = AnalysisConfig(t_end=100.0, dt=10.0, output_dt=10.0)
        b = AnalysisConfig(t_end=100.0, dt=10.0, output_dt=10.0)
        a.element_ids.append(99)
        assert b.element_ids == []


# ── validate() ───────────────────────────────────────────────────────────────

class TestValidate:
    def _ok(self, **kwargs) -> AnalysisConfig:
        defaults = dict(t_end=3600.0, dt=30.0, output_dt=60.0)
        defaults.update(kwargs)
        cfg = AnalysisConfig(**defaults)
        cfg.validate()   # must not raise
        return cfg

    def test_valid_typical(self):
        self._ok()   # just must not raise

    def test_t_end_zero_raises(self):
        cfg = AnalysisConfig(t_end=0.0, dt=30.0, output_dt=60.0)
        with pytest.raises(ValueError, match="t_end"):
            cfg.validate()

    def test_t_end_negative_raises(self):
        cfg = AnalysisConfig(t_end=-60.0, dt=30.0, output_dt=60.0)
        with pytest.raises(ValueError, match="t_end"):
            cfg.validate()

    def test_dt_zero_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=0.0, output_dt=60.0)
        with pytest.raises(ValueError, match="dt"):
            cfg.validate()

    def test_dt_negative_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=-1.0, output_dt=60.0)
        with pytest.raises(ValueError, match="dt"):
            cfg.validate()

    def test_dt_exceeds_t_end_raises(self):
        cfg = AnalysisConfig(t_end=60.0, dt=120.0, output_dt=60.0)
        with pytest.raises(ValueError, match="dt"):
            cfg.validate()

    def test_dt_equals_t_end_ok(self):
        # Single step is valid
        self._ok(t_end=60.0, dt=60.0, output_dt=60.0)

    def test_output_dt_less_than_dt_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=60.0, output_dt=30.0)
        with pytest.raises(ValueError, match="output_dt"):
            cfg.validate()

    def test_output_dt_equals_dt_ok(self):
        self._ok(dt=60.0, output_dt=60.0)

    def test_output_dt_exceeds_t_end_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=7200.0)
        with pytest.raises(ValueError, match="output_dt"):
            cfg.validate()

    def test_output_dt_equals_t_end_ok(self):
        self._ok(t_end=3600.0, dt=30.0, output_dt=3600.0)

    def test_n_layers_zero_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, n_layers=0)
        with pytest.raises(ValueError, match="n_layers"):
            cfg.validate()

    def test_n_layers_one_ok(self):
        self._ok(n_layers=1)

    def test_n_layers_two_ok(self):
        self._ok(n_layers=2)

    def test_elem_size_zero_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, elem_size=0.0)
        with pytest.raises(ValueError, match="elem_size"):
            cfg.validate()

    def test_elem_size_negative_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, elem_size=-0.01)
        with pytest.raises(ValueError, match="elem_size"):
            cfg.validate()

    def test_elem_size_none_ok(self):
        self._ok(elem_size=None)

    def test_elem_size_positive_ok(self):
        self._ok(elem_size=0.04)


# ── n_output_steps ────────────────────────────────────────────────────────────

class TestNOutputSteps:
    def test_every_step(self):
        # t_end=300, output_dt=60 → 5 steps + t=0 = 6
        cfg = AnalysisConfig(t_end=300.0, dt=60.0, output_dt=60.0)
        assert cfg.n_output_steps == 6

    def test_single_step(self):
        cfg = AnalysisConfig(t_end=60.0, dt=60.0, output_dt=60.0)
        assert cfg.n_output_steps == 2  # t=0 and t=60

    def test_infrequent_output(self):
        # t_end=7200, output_dt=3600 → 2 steps + t=0 = 3
        cfg = AnalysisConfig(t_end=7200.0, dt=60.0, output_dt=3600.0)
        assert cfg.n_output_steps == 3


# ── summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_contains_duration(self):
        cfg = AnalysisConfig(t_end=7200.0, dt=30.0, output_dt=60.0)
        assert "120" in cfg.summary()   # 7200 s = 120 min

    def test_contains_dt(self):
        cfg = AnalysisConfig(t_end=7200.0, dt=30.0, output_dt=60.0)
        assert "30" in cfg.summary()

    def test_contains_layers(self):
        cfg = AnalysisConfig(t_end=7200.0, dt=30.0, output_dt=60.0, n_layers=2)
        assert "2" in cfg.summary()

    def test_no_elements_says_all(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0)
        assert "all" in cfg.summary().lower()

    def test_with_element_ids(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0, element_ids=[1, 2, 3])
        # Should mention the count or the list
        assert "3" in cfg.summary()


# ── Dialog importability ──────────────────────────────────────────────────────

def test_dialog_importable():
    """The RunAnalysisDialog module must import without raising."""
    from fahts.gui.dialogs.run_analysis_dialog import RunAnalysisDialog  # noqa: F401
    assert RunAnalysisDialog is not None
