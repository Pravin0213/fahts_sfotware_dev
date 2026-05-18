"""
Unit tests for Task 3.6 — TemperatureField results container.

Covers: construction, per-element queries, bulk queries, factory methods,
and merge logic.
"""
import pytest
import numpy as np
from fahts.core.results.temperature_field import TemperatureField
from fahts.core.model.section import BoxSection
from fahts.core.heat.section_mesh import BoxMesher
from fahts.core.model.material import SteelMaterial
from fahts.core.heat.solver.time_integrator import TransientSolver


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _simple_field() -> TemperatureField:
    """Two elements, three time steps."""
    times = np.array([0.0, 60.0, 120.0])
    # elem 10: 20 → 200 → 400 °C; elem 20: 20 → 100 → 200 °C
    T_centroid = np.array([
        [20.0,  20.0],
        [200.0, 100.0],
        [400.0, 200.0],
    ])
    return TemperatureField(
        times=times,
        element_ids=[10, 20],
        T_centroid=T_centroid,
    )


def _mesh():
    sec = BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
    return BoxMesher(sec, elem_size=0.04, n_layers=1).build()


def _material() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


# ── Construction and properties ───────────────────────────────────────────────

class TestConstruction:
    def test_n_steps(self):
        tf = _simple_field()
        assert tf.n_steps == 3

    def test_n_elements(self):
        tf = _simple_field()
        assert tf.n_elements == 2

    def test_element_ids_preserved(self):
        tf = _simple_field()
        assert tf.element_ids == [10, 20]

    def test_times_preserved(self):
        tf = _simple_field()
        np.testing.assert_array_equal(tf.times, [0.0, 60.0, 120.0])

    def test_T_centroid_shape(self):
        tf = _simple_field()
        assert tf.T_centroid.shape == (3, 2)

    def test_T_section_default_empty(self):
        tf = _simple_field()
        assert tf.T_section == {}

    def test_eid_to_idx_built(self):
        tf = _simple_field()
        assert tf._eid_to_idx == {10: 0, 20: 1}


# ── centroid_temperature ──────────────────────────────────────────────────────

class TestCentroidTemperature:
    def test_last_step_default(self):
        tf = _simple_field()
        assert tf.centroid_temperature(10) == pytest.approx(400.0)

    def test_last_step_explicit(self):
        tf = _simple_field()
        assert tf.centroid_temperature(10, t_idx=-1) == pytest.approx(400.0)

    def test_first_step(self):
        tf = _simple_field()
        assert tf.centroid_temperature(20, t_idx=0) == pytest.approx(20.0)

    def test_middle_step(self):
        tf = _simple_field()
        assert tf.centroid_temperature(10, t_idx=1) == pytest.approx(200.0)

    def test_second_element(self):
        tf = _simple_field()
        assert tf.centroid_temperature(20, t_idx=2) == pytest.approx(200.0)

    def test_invalid_eid_raises(self):
        tf = _simple_field()
        with pytest.raises(KeyError):
            tf.centroid_temperature(999)


# ── section_temperatures ──────────────────────────────────────────────────────

class TestSectionTemperatures:
    def test_returns_nodal_array(self):
        n_nodes = 8
        T_nodes = np.linspace(20.0, 400.0, n_nodes)
        T_sec = T_nodes[np.newaxis, :]  # (1, n_nodes)
        tf = TemperatureField(
            times=np.array([60.0]),
            element_ids=[5],
            T_centroid=np.array([[200.0]]),
            T_section={5: T_sec},
        )
        result = tf.section_temperatures(5, t_idx=0)
        np.testing.assert_array_equal(result, T_nodes)

    def test_last_step_default(self):
        n_nodes = 6
        T_hist = np.array([[100.0] * n_nodes, [300.0] * n_nodes])
        tf = TemperatureField(
            times=np.array([0.0, 60.0]),
            element_ids=[7],
            T_centroid=np.array([[100.0], [300.0]]),
            T_section={7: T_hist},
        )
        result = tf.section_temperatures(7)
        np.testing.assert_array_equal(result, np.full(n_nodes, 300.0))

    def test_missing_eid_raises(self):
        tf = _simple_field()  # no T_section entries
        with pytest.raises(KeyError):
            tf.section_temperatures(10)


# ── peak_centroid_temperature ─────────────────────────────────────────────────

class TestPeakCentroidTemperature:
    def test_peak_for_elem_10(self):
        assert _simple_field().peak_centroid_temperature(10) == pytest.approx(400.0)

    def test_peak_for_elem_20(self):
        assert _simple_field().peak_centroid_temperature(20) == pytest.approx(200.0)

    def test_single_step_is_peak(self):
        tf = TemperatureField(
            times=np.array([0.0]),
            element_ids=[3],
            T_centroid=np.array([[750.0]]),
        )
        assert tf.peak_centroid_temperature(3) == pytest.approx(750.0)


# ── time_to_critical ──────────────────────────────────────────────────────────

class TestTimeToCritical:
    def test_exact_crossing(self):
        """T reaches exactly T_crit at a stored step."""
        times = np.array([0.0, 60.0, 120.0])
        T = np.array([[20.0], [600.0], [700.0]])
        tf = TemperatureField(times=times, element_ids=[1], T_centroid=T)
        t_crit = tf.time_to_critical(1, T_crit=600.0)
        assert t_crit == pytest.approx(60.0, abs=1e-6)

    def test_linear_interpolation(self):
        """T crosses T_crit between two steps — should interpolate."""
        times = np.array([0.0, 100.0, 200.0])
        T = np.array([[20.0], [500.0], [700.0]])
        tf = TemperatureField(times=times, element_ids=[1], T_centroid=T)
        # Crossing between t=100 (500°C) and t=200 (700°C) at T_crit=600°C
        # t = 100 + (600-500)/(700-500) * 100 = 100 + 50 = 150
        t_crit = tf.time_to_critical(1, T_crit=600.0)
        assert t_crit == pytest.approx(150.0, abs=1e-6)

    def test_never_reached_returns_none(self):
        times = np.array([0.0, 60.0, 120.0])
        T = np.array([[20.0], [200.0], [400.0]])
        tf = TemperatureField(times=times, element_ids=[1], T_centroid=T)
        assert tf.time_to_critical(1, T_crit=600.0) is None

    def test_already_critical_at_t0(self):
        times = np.array([0.0, 60.0])
        T = np.array([[700.0], [800.0]])
        tf = TemperatureField(times=times, element_ids=[1], T_centroid=T)
        t_crit = tf.time_to_critical(1, T_crit=600.0)
        assert t_crit == pytest.approx(0.0, abs=1e-9)

    def test_custom_T_crit(self):
        """Works with a threshold other than 600°C."""
        times = np.array([0.0, 100.0])
        T = np.array([[20.0], [420.0]])
        tf = TemperatureField(times=times, element_ids=[1], T_centroid=T)
        # T_crit=220: halfway between 20 and 420 → t=50
        t_crit = tf.time_to_critical(1, T_crit=220.0)
        assert t_crit == pytest.approx(50.0, abs=1e-6)


# ── peak_temperatures property ────────────────────────────────────────────────

class TestPeakTemperatures:
    def test_returns_dict_all_elements(self):
        tf = _simple_field()
        peaks = tf.peak_temperatures
        assert set(peaks.keys()) == {10, 20}

    def test_values_correct(self):
        tf = _simple_field()
        peaks = tf.peak_temperatures
        assert peaks[10] == pytest.approx(400.0)
        assert peaks[20] == pytest.approx(200.0)


# ── critical_elements ─────────────────────────────────────────────────────────

class TestCriticalElements:
    def test_none_critical(self):
        tf = _simple_field()
        assert tf.critical_elements(T_crit=500.0) == []

    def test_one_critical(self):
        tf = _simple_field()
        assert tf.critical_elements(T_crit=350.0) == [10]

    def test_both_critical(self):
        tf = _simple_field()
        assert tf.critical_elements(T_crit=150.0) == [10, 20]

    def test_exact_threshold_included(self):
        tf = _simple_field()
        # elem 10 peaks at exactly 400.0
        assert 10 in tf.critical_elements(T_crit=400.0)

    def test_order_preserved(self):
        """Elements appear in element_ids order, not sorted by temperature."""
        times = np.array([0.0, 60.0])
        T = np.array([[20.0, 20.0, 20.0], [700.0, 650.0, 800.0]])
        tf = TemperatureField(
            times=times, element_ids=[30, 20, 10], T_centroid=T
        )
        assert tf.critical_elements(T_crit=600.0) == [30, 20, 10]


# ── from_solver_run factory ───────────────────────────────────────────────────

class TestFromSolverRun:
    def setup_method(self):
        self.mesh = _mesh()
        self.n_nodes = self.mesh.n_nodes

    def test_returns_temperature_field(self):
        times = np.array([0.0, 60.0])
        T_hist = np.full((2, self.n_nodes), 20.0)
        tf = TemperatureField.from_solver_run(eid=42, times=times, T_history=T_hist, mesh=self.mesh)
        assert isinstance(tf, TemperatureField)

    def test_single_element_id(self):
        times = np.array([0.0])
        T_hist = np.full((1, self.n_nodes), 20.0)
        tf = TemperatureField.from_solver_run(eid=42, times=times, T_history=T_hist, mesh=self.mesh)
        assert tf.element_ids == [42]

    def test_T_centroid_shape(self):
        n_steps = 5
        times = np.linspace(0, 300, n_steps)
        T_hist = np.random.uniform(20, 500, (n_steps, self.n_nodes))
        tf = TemperatureField.from_solver_run(eid=1, times=times, T_history=T_hist, mesh=self.mesh)
        assert tf.T_centroid.shape == (n_steps, 1)

    def test_T_section_stored(self):
        times = np.array([0.0, 60.0])
        T_hist = np.full((2, self.n_nodes), 100.0)
        tf = TemperatureField.from_solver_run(eid=7, times=times, T_history=T_hist, mesh=self.mesh)
        assert 7 in tf.T_section
        assert tf.T_section[7].shape == (2, self.n_nodes)

    def test_uniform_temperature_centroid_equals_nodal(self):
        """When all nodes are at the same T, centroid = that T."""
        times = np.array([0.0, 60.0, 120.0])
        T_val = 300.0
        T_hist = np.full((3, self.n_nodes), T_val)
        tf = TemperatureField.from_solver_run(eid=1, times=times, T_history=T_hist, mesh=self.mesh)
        np.testing.assert_allclose(tf.T_centroid[:, 0], T_val, atol=1e-10)

    def test_area_weighted_centroid_gradient(self):
        """Centroid temperature lies strictly between min and max nodal temperatures."""
        times = np.array([0.0])
        T_hist = np.linspace(100.0, 400.0, self.n_nodes).reshape(1, -1)
        tf = TemperatureField.from_solver_run(eid=1, times=times, T_history=T_hist, mesh=self.mesh)
        T_cen = tf.T_centroid[0, 0]
        assert 100.0 < T_cen < 400.0

    def test_times_preserved(self):
        times = np.array([0.0, 30.0, 60.0])
        T_hist = np.full((3, self.n_nodes), 20.0)
        tf = TemperatureField.from_solver_run(eid=1, times=times, T_history=T_hist, mesh=self.mesh)
        np.testing.assert_array_equal(tf.times, times)


# ── from_solver_run via real solver ──────────────────────────────────────────

class TestFromSolverRunIntegration:
    def test_round_trip_with_transient_solver(self):
        """Run the solver and wrap output in TemperatureField — shape and range checks."""
        mesh = _mesh()
        mat = _material()
        solver = TransientSolver(
            mesh=mesh, material=mat,
            fire_temp=lambda t: 900.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
        )
        times, T_hist = solver.run(t_end=120.0, dt=60.0)
        tf = TemperatureField.from_solver_run(eid=99, times=times, T_history=T_hist, mesh=mesh)

        assert tf.n_elements == 1
        assert tf.n_steps == len(times)
        assert tf.T_centroid.shape == (len(times), 1)
        # Temperature should increase from 20°C (fire at 900°C)
        assert tf.T_centroid[-1, 0] > tf.T_centroid[0, 0]


# ── merge ─────────────────────────────────────────────────────────────────────

class TestMerge:
    def _make_field(self, eid: int, T_val: float) -> TemperatureField:
        times = np.array([0.0, 60.0, 120.0])
        T = np.full((3, 1), T_val)
        T[0, 0] = 20.0
        return TemperatureField(
            times=times,
            element_ids=[eid],
            T_centroid=T,
        )

    def test_merged_element_ids(self):
        f1 = self._make_field(1, 400.0)
        f2 = self._make_field(2, 300.0)
        merged = TemperatureField.merge([f1, f2])
        assert merged.element_ids == [1, 2]

    def test_merged_T_centroid_shape(self):
        f1 = self._make_field(1, 400.0)
        f2 = self._make_field(2, 300.0)
        merged = TemperatureField.merge([f1, f2])
        assert merged.T_centroid.shape == (3, 2)

    def test_merged_values_correct(self):
        f1 = self._make_field(1, 400.0)
        f2 = self._make_field(2, 300.0)
        merged = TemperatureField.merge([f1, f2])
        assert merged.centroid_temperature(1) == pytest.approx(400.0)
        assert merged.centroid_temperature(2) == pytest.approx(300.0)

    def test_merged_times_from_first(self):
        f1 = self._make_field(1, 400.0)
        f2 = self._make_field(2, 300.0)
        merged = TemperatureField.merge([f1, f2])
        np.testing.assert_array_equal(merged.times, f1.times)

    def test_merge_single_field(self):
        f1 = self._make_field(5, 350.0)
        merged = TemperatureField.merge([f1])
        assert merged.element_ids == [5]
        assert merged.T_centroid.shape == (3, 1)

    def test_merge_three_fields(self):
        fields = [self._make_field(i, 100.0 * i) for i in range(1, 4)]
        merged = TemperatureField.merge(fields)
        assert merged.n_elements == 3
        assert merged.T_centroid.shape == (3, 3)

    def test_merge_propagates_T_section(self):
        mesh = _mesh()
        mat = _material()
        solver = TransientSolver(
            mesh=mesh, material=mat,
            fire_temp=lambda t: 900.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
        )
        times, T_hist = solver.run(t_end=60.0, dt=60.0)
        f1 = TemperatureField.from_solver_run(1, times, T_hist, mesh)
        f2 = TemperatureField.from_solver_run(2, times, T_hist, mesh)
        merged = TemperatureField.merge([f1, f2])
        assert 1 in merged.T_section
        assert 2 in merged.T_section

    def test_merge_empty_list_raises(self):
        with pytest.raises(ValueError):
            TemperatureField.merge([])

    def test_merged_eid_index_correct(self):
        """After merge, _eid_to_idx must be rebuilt correctly."""
        f1 = self._make_field(10, 400.0)
        f2 = self._make_field(20, 300.0)
        merged = TemperatureField.merge([f1, f2])
        assert merged._eid_to_idx == {10: 0, 20: 1}
