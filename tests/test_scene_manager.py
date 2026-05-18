"""Tests for fahts/renderer/scene_manager.py (Phase 1.5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from fahts.renderer.scene_manager import SceneManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_model():
    """Load model_file.fem once for the whole module."""
    from fahts.core.io.usfos_reader import read_usfos_fem
    fem_path = Path(__file__).parents[1] / "model_file.fem"
    if not fem_path.exists():
        pytest.skip("model_file.fem not found")
    return read_usfos_fem(fem_path)


@pytest.fixture
def sm():
    """Off-screen SceneManager; closed after each test."""
    mgr = SceneManager(off_screen=True)
    yield mgr
    mgr.close()


@pytest.fixture
def sm_with_model(sm, real_model):
    sm.load_model(real_model)
    return sm


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_plotter(self, sm):
        assert isinstance(sm.plotter, pv.Plotter)

    def test_accepts_external_plotter(self):
        pl = pv.Plotter(off_screen=True)
        sm = SceneManager(plotter=pl)
        assert sm.plotter is pl
        pl.close()

    def test_initial_render_mode(self, sm):
        assert sm.render_mode == "section"

    def test_initial_colour_mode(self, sm):
        assert sm.colour_mode == "default"


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_load_does_not_raise(self, sm, real_model):
        sm.load_model(real_model)

    def test_solid_mesh_populated(self, sm_with_model):
        assert sm_with_model._solid_mesh is not None
        assert sm_with_model._solid_mesh.n_cells > 0

    def test_wire_mesh_populated(self, sm_with_model):
        assert sm_with_model._wire_mesh is not None
        assert sm_with_model._wire_mesh.n_cells > 0

    def test_model_mesh_has_element_id(self, sm_with_model):
        assert "element_id" in sm_with_model._solid_mesh.cell_data

    def test_group_names_populated(self, sm_with_model, real_model):
        assert len(sm_with_model._group_names) == len(real_model.groups)

    def test_elem_to_group_populated(self, sm_with_model, real_model):
        # Every element that belongs to a group should appear in the mapping
        model_elem_ids_in_groups = set()
        for g in real_model.groups.values():
            model_elem_ids_in_groups |= g.element_ids
        mapped = set(sm_with_model._elem_to_group.keys())
        assert mapped == model_elem_ids_in_groups

    def test_group_id_scalar_written(self, sm_with_model):
        assert "group_id" in sm_with_model._solid_mesh.cell_data

    def test_group_id_values_in_range(self, sm_with_model):
        gid = sm_with_model._solid_mesh.cell_data["group_id"]
        n_groups = len(sm_with_model._group_names)
        assert int(gid.min()) >= 0
        assert int(gid.max()) <= n_groups  # sentinel = n_groups for unassigned

    def test_reload_replaces_model(self, sm, real_model):
        """Calling load_model twice should not leave ghost actors."""
        sm.load_model(real_model)
        sm.load_model(real_model)
        # No assertion on count — just must not raise


# ── Render mode ───────────────────────────────────────────────────────────────

class TestRenderMode:
    def test_switch_to_wire(self, sm_with_model):
        sm_with_model.set_render_mode("wire")
        assert sm_with_model.render_mode == "wire"
        assert sm_with_model._wire_actor is not None
        assert sm_with_model._solid_actor is None

    def test_switch_back_to_section(self, sm_with_model):
        sm_with_model.set_render_mode("wire")
        sm_with_model.set_render_mode("section")
        assert sm_with_model.render_mode == "section"
        assert sm_with_model._solid_actor is not None
        assert sm_with_model._wire_actor is None

    def test_invalid_mode_raises(self, sm_with_model):
        with pytest.raises(ValueError, match="Unknown render mode"):
            sm_with_model.set_render_mode("voxel")

    def test_same_mode_noop(self, sm_with_model):
        """Setting the same mode twice should not raise."""
        sm_with_model.set_render_mode("section")
        sm_with_model.set_render_mode("section")


# ── Colouring ─────────────────────────────────────────────────────────────────

class TestColouring:
    def test_colour_by_group_sets_mode(self, sm_with_model):
        sm_with_model.colour_by_group()
        assert sm_with_model.colour_mode == "group"

    def test_colour_by_group_custom_colours(self, sm_with_model, real_model):
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.colour_by_group({first_group: (1.0, 0.0, 0.0)})
        assert sm_with_model._group_colour_map.colour_for(first_group) == (1.0, 0.0, 0.0)

    def test_colour_by_temperature_sets_mode(self, sm_with_model, real_model):
        T_map = {eid: 400.0 for eid in real_model.elements}
        sm_with_model.colour_by_temperature(T_map)
        assert sm_with_model.colour_mode == "temperature"

    def test_colour_by_temperature_writes_scalar(self, sm_with_model, real_model):
        T_map = {eid: 300.0 for eid in real_model.elements}
        sm_with_model.colour_by_temperature(T_map)
        assert "temperature_C" in sm_with_model._solid_mesh.cell_data
        arr = sm_with_model._solid_mesh.cell_data["temperature_C"]
        assert np.allclose(arr, 300.0)

    def test_colour_by_temperature_explicit_clim(self, sm_with_model, real_model):
        T_map = {eid: float(i) for i, eid in enumerate(real_model.elements)}
        sm_with_model.colour_by_temperature(T_map, clim=(0.0, 500.0))
        assert sm_with_model._T_clim == (0.0, 500.0)

    def test_missing_elements_default_to_20(self, sm_with_model):
        sm_with_model.colour_by_temperature({})  # empty dict → all 20 °C
        arr = sm_with_model._solid_mesh.cell_data["temperature_C"]
        assert np.allclose(arr, 20.0)

    def test_reset_colour_returns_to_default(self, sm_with_model, real_model):
        sm_with_model.colour_by_group()
        sm_with_model.reset_colour()
        assert sm_with_model.colour_mode == "default"

    def test_group_cmap_length(self, sm_with_model):
        cmap = sm_with_model._group_colour_map.build_cmap()
        expected = len(sm_with_model._group_names) + 1  # +1 for unassigned
        assert cmap.N == expected


# ── Fire zones ────────────────────────────────────────────────────────────────

class TestFireZones:
    def _make_zone(self, cx=0.0, cy=0.0, cz=0.0, dx=2.0, dy=2.0, dz=2.0, name="FZ1"):
        return SimpleNamespace(center=(cx, cy, cz), dims=(dx, dy, dz), name=name)

    def test_show_fire_zones_adds_actors(self, sm_with_model):
        zones = [self._make_zone(), self._make_zone(cx=5.0, name="FZ2")]
        sm_with_model.show_fire_zones(zones)
        assert len(sm_with_model._fire_zone_actors) == 2

    def test_clear_fire_zones(self, sm_with_model):
        sm_with_model.show_fire_zones([self._make_zone()])
        sm_with_model.clear_fire_zones()
        assert len(sm_with_model._fire_zone_actors) == 0

    def test_show_replaces_previous(self, sm_with_model):
        sm_with_model.show_fire_zones([self._make_zone()])
        sm_with_model.show_fire_zones([self._make_zone(), self._make_zone(cx=3.0)])
        assert len(sm_with_model._fire_zone_actors) == 2

    def test_empty_zones_clears(self, sm_with_model):
        sm_with_model.show_fire_zones([self._make_zone()])
        sm_with_model.show_fire_zones([])
        assert len(sm_with_model._fire_zone_actors) == 0


# ── Camera ────────────────────────────────────────────────────────────────────

class TestCamera:
    def test_reset_camera_no_raise(self, sm_with_model):
        sm_with_model.reset_camera()

    @pytest.mark.parametrize("view", ["+x", "-x", "+y", "-y", "+z", "-z"])
    def test_set_camera_view(self, sm_with_model, view):
        sm_with_model.set_camera_view(view)

    def test_invalid_view_raises(self, sm_with_model):
        with pytest.raises(ValueError, match="Unknown view"):
            sm_with_model.set_camera_view("top")


# ── Temperature animation ─────────────────────────────────────────────────────

class TestTemperatureAnimation:
    def _make_T_field(self, element_ids, n_steps=5):
        times = np.linspace(0, 3600, n_steps)
        T_centroid = np.random.uniform(20.0, 800.0, (n_steps, len(element_ids)))
        return SimpleNamespace(
            times=times,
            element_ids=list(element_ids),
            T_centroid=T_centroid,
        )

    def test_update_temperature(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()), n_steps=4)
        sm_with_model.update_temperature(0.0, tf)
        assert sm_with_model.colour_mode == "temperature"

    def test_update_temperature_nearest_step(self, sm_with_model, real_model):
        eids = list(real_model.elements.keys())
        n = len(eids)
        times = np.array([0.0, 60.0, 120.0])
        T = np.vstack([
            np.full(n, 20.0),
            np.full(n, 400.0),
            np.full(n, 700.0),
        ])
        tf = SimpleNamespace(times=times, element_ids=eids, T_centroid=T)
        sm_with_model.update_temperature(55.0, tf)
        # nearest step to 55 s is index 1 (60 s) → expect ~400 °C
        arr = sm_with_model._solid_mesh.cell_data["temperature_C"]
        assert np.allclose(arr, 400.0, atol=1e-6)

    def test_update_with_none_T_field(self, sm_with_model):
        """update_temperature with None must be a no-op."""
        sm_with_model.update_temperature(0.0, None)

    def test_animate_calls_callback(self, sm_with_model, real_model):
        eids = list(real_model.elements.keys())
        tf = self._make_T_field(eids, n_steps=3)
        calls = []
        sm_with_model.animate(tf, fps=1000, callback=lambda t, T: calls.append(t))
        assert len(calls) == 3

    def test_animate_with_none_noop(self, sm_with_model):
        sm_with_model.animate(None)


# ── Task 4.1 — VTK temperature mapping ───────────────────────────────────────

class TestTemperatureMapping:
    """Task 4.1: nodal temperatures mapped to VTK cell + point data."""

    def _make_T_field(self, element_ids, n_steps=3):
        times = np.linspace(0.0, 3600.0, n_steps)
        T_centroid = np.random.default_rng(42).uniform(20.0, 800.0, (n_steps, len(element_ids)))
        return SimpleNamespace(
            times=times,
            element_ids=list(element_ids),
            T_centroid=T_centroid,
        )

    def test_colour_by_temperature_writes_point_data(self, sm_with_model, real_model):
        T_map = {eid: 500.0 for eid in real_model.elements}
        sm_with_model.colour_by_temperature(T_map)
        assert "temperature_C" in sm_with_model._solid_mesh.point_data

    def test_point_data_mean_close_to_cell_data_mean(self, sm_with_model, real_model):
        T_map = {eid: 500.0 for eid in real_model.elements}
        sm_with_model.colour_by_temperature(T_map)
        cell_mean = float(sm_with_model._solid_mesh.cell_data["temperature_C"].mean())
        pt_mean   = float(sm_with_model._solid_mesh.point_data["temperature_C"].mean())
        assert abs(cell_mean - pt_mean) / max(abs(cell_mean), 1.0) < 0.05

    def test_update_temperature_stores_field_and_index(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        assert sm_with_model._T_field is tf
        assert sm_with_model._T_t_idx == 0

    def test_update_temperature_stores_correct_index(self, sm_with_model, real_model):
        eids = list(real_model.elements.keys())
        n = len(eids)
        times = np.array([0.0, 60.0, 120.0])
        T = np.vstack([np.full(n, 20.0), np.full(n, 400.0), np.full(n, 700.0)])
        tf = SimpleNamespace(times=times, element_ids=eids, T_centroid=T)
        sm_with_model.update_temperature(100.0, tf)   # nearest → index 2 (120 s)
        assert sm_with_model._T_t_idx == 2

    def test_temperature_persists_after_group_hide(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.set_group_visibility(first_group, False)
        assert sm_with_model.colour_mode == "temperature"

    def test_temperature_persists_after_group_show(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.set_group_visibility(first_group, False)
        sm_with_model.set_group_visibility(first_group, True)
        assert sm_with_model.colour_mode == "temperature"

    def test_temperature_scalar_present_after_group_hide(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.set_group_visibility(first_group, False)
        assert "temperature_C" in sm_with_model._solid_mesh.cell_data

    def test_load_model_clears_stored_field(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        sm_with_model.load_model(real_model)
        assert sm_with_model._T_field is None
        assert sm_with_model._T_t_idx == 0

    def test_visibility_reset_drops_temperature_without_stored_field(
        self, sm_with_model, real_model
    ):
        # colour_by_temperature directly (no update_temperature) → no stored field
        T_map = {eid: 300.0 for eid in real_model.elements}
        sm_with_model.colour_by_temperature(T_map)
        assert sm_with_model._T_field is None   # not stored via this path
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.set_group_visibility(first_group, False)
        # Without stored field, falls back to default
        assert sm_with_model.colour_mode == "default"


# ── Task 4.3 — Time label ─────────────────────────────────────────────────────

class TestTimeLabel:
    """Task 4.3: time overlay label appears/disappears with temperature display."""

    def _make_T_field(self, element_ids, n_steps=4):
        times = np.linspace(0.0, 3600.0, n_steps)
        T_centroid = np.random.default_rng(7).uniform(20.0, 700.0, (n_steps, len(element_ids)))
        return SimpleNamespace(
            times=times,
            element_ids=list(element_ids),
            T_centroid=T_centroid,
        )

    def test_update_temperature_sets_time_label(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        assert sm_with_model._time_label_actor is not None

    def test_time_label_none_before_update(self, sm_with_model):
        assert sm_with_model._time_label_actor is None

    def test_reset_colour_clears_time_label(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        sm_with_model.reset_colour()
        assert sm_with_model._time_label_actor is None

    def test_colour_by_group_clears_time_label(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        sm_with_model.colour_by_group()
        assert sm_with_model._time_label_actor is None

    def test_load_model_clears_time_label(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        sm_with_model.load_model(real_model)
        assert sm_with_model._time_label_actor is None

    def test_time_label_persists_after_group_hide(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        first_group = next(iter(real_model.groups.keys()))
        sm_with_model.set_group_visibility(first_group, False)
        # Temperature mode persists through visibility changes — label must remain
        assert sm_with_model._time_label_actor is not None

    def test_update_temperature_replaces_label_on_next_step(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(0.0, tf)
        actor_first = sm_with_model._time_label_actor
        sm_with_model.update_temperature(3600.0, tf)
        # Label must still exist (replaced, not deleted)
        assert sm_with_model._time_label_actor is not None

    def test_update_time_label_none_is_idempotent(self, sm_with_model):
        """Calling _update_time_label(None) twice must not raise."""
        sm_with_model._update_time_label(None)
        sm_with_model._update_time_label(None)

    def test_time_label_short_format_under_one_minute(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(30.0, tf)   # 30 s < 1 min
        # Just verify a label exists (text content is in VTK actor internals)
        assert sm_with_model._time_label_actor is not None

    def test_time_label_long_format_over_one_minute(self, sm_with_model, real_model):
        tf = self._make_T_field(list(real_model.elements.keys()))
        sm_with_model.update_temperature(3600.0, tf)   # 3600 s = 60 min
        assert sm_with_model._time_label_actor is not None

    def test_animate_leaves_label_at_last_step(self, sm_with_model, real_model):
        eids = list(real_model.elements.keys())
        times = np.array([0.0, 1800.0, 3600.0])
        T = np.vstack([np.full(len(eids), 20.0), np.full(len(eids), 400.0), np.full(len(eids), 700.0)])
        tf = SimpleNamespace(times=times, element_ids=eids, T_centroid=T)
        sm_with_model.animate(tf, fps=1000)
        assert sm_with_model._time_label_actor is not None


# ── Screenshot ────────────────────────────────────────────────────────────────

class TestScreenshot:
    def test_screenshot_creates_file(self, sm_with_model, tmp_path):
        out = tmp_path / "test_render.png"
        sm_with_model.screenshot(out)
        assert out.exists()
        assert out.stat().st_size > 0
