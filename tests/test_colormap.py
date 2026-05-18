"""Tests for fahts/renderer/colormap.py (Phase 1.10 / 4.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from fahts.renderer.colormap import (
    CRITICAL_TEMPERATURE_STEEL,
    DEFAULT_PALETTE,
    THRESHOLD_COLOUR,
    UNASSIGNED_COLOUR,
    GroupColourMap,
    TemperatureColourMap,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mesh_with_eids(element_ids: list[int]) -> pv.PolyData:
    """Build a minimal PolyData with one triangle per element ID."""
    pts_list = []
    faces_list = []
    offset = 0
    for _ in element_ids:
        pts_list.append([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces_list.extend([3, offset, offset + 1, offset + 2])
        offset += 3
    pts = np.array(pts_list, dtype=float).reshape(-1, 3)
    faces = np.array(faces_list, dtype=int)
    mesh = pv.PolyData(pts, faces)
    mesh.cell_data["element_id"] = np.array(element_ids, dtype=np.int32)
    return mesh


# ── DEFAULT_PALETTE / UNASSIGNED_COLOUR ──────────────────────────────────────

class TestConstants:
    def test_palette_is_non_empty(self):
        assert len(DEFAULT_PALETTE) > 0

    def test_palette_entries_are_rgb_triples(self):
        for colour in DEFAULT_PALETTE:
            assert len(colour) == 3
            assert all(0.0 <= c <= 1.0 for c in colour)

    def test_unassigned_colour_is_rgb_triple(self):
        assert len(UNASSIGNED_COLOUR) == 3
        assert all(0.0 <= c <= 1.0 for c in UNASSIGNED_COLOUR)


# ── GroupColourMap construction ───────────────────────────────────────────────

class TestConstruction:
    def test_empty_group_names(self):
        gcm = GroupColourMap([])
        assert gcm.group_names == []
        assert gcm.colours == {}

    def test_group_names_stored(self):
        names = ["ELEV_A", "MAIN_STEEL"]
        gcm = GroupColourMap(names)
        assert gcm.group_names == names

    def test_default_palette_assigned(self):
        gcm = GroupColourMap(["G0", "G1", "G2"])
        for i, name in enumerate(["G0", "G1", "G2"]):
            assert gcm.colour_for(name) == DEFAULT_PALETTE[i]

    def test_palette_cycles_for_many_groups(self):
        n = len(DEFAULT_PALETTE) * 2 + 1
        names = [f"G{i}" for i in range(n)]
        gcm = GroupColourMap(names)
        for i, name in enumerate(names):
            assert gcm.colour_for(name) == DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]

    def test_overrides_applied_at_construction(self):
        gcm = GroupColourMap(["A", "B"], overrides={"A": (1.0, 0.0, 0.0)})
        assert gcm.colour_for("A") == (1.0, 0.0, 0.0)
        assert gcm.colour_for("B") == DEFAULT_PALETTE[1]

    def test_unknown_group_returns_unassigned_colour(self):
        gcm = GroupColourMap(["A"])
        assert gcm.colour_for("NOPE") == UNASSIGNED_COLOUR

    def test_colours_property_is_copy(self):
        gcm = GroupColourMap(["A"])
        d = gcm.colours
        d["A"] = (0.0, 0.0, 0.0)
        assert gcm.colour_for("A") != (0.0, 0.0, 0.0)  # original unchanged


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_update_changes_colour(self):
        gcm = GroupColourMap(["A", "B"])
        gcm.update({"A": (0.5, 0.5, 0.5)})
        assert gcm.colour_for("A") == (0.5, 0.5, 0.5)

    def test_update_leaves_other_groups_unchanged(self):
        gcm = GroupColourMap(["A", "B"])
        original_b = gcm.colour_for("B")
        gcm.update({"A": (0.1, 0.2, 0.3)})
        assert gcm.colour_for("B") == original_b

    def test_update_is_cumulative(self):
        gcm = GroupColourMap(["A", "B"])
        gcm.update({"A": (1.0, 0.0, 0.0)})
        gcm.update({"B": (0.0, 1.0, 0.0)})
        assert gcm.colour_for("A") == (1.0, 0.0, 0.0)
        assert gcm.colour_for("B") == (0.0, 1.0, 0.0)


# ── build_cmap ────────────────────────────────────────────────────────────────

class TestBuildCmap:
    def test_cmap_has_n_plus_1_entries(self):
        gcm = GroupColourMap(["A", "B", "C"])
        cmap = gcm.build_cmap()
        assert cmap.N == 4   # 3 groups + 1 sentinel

    def test_cmap_empty_groups_has_one_entry(self):
        gcm = GroupColourMap([])
        cmap = gcm.build_cmap()
        assert cmap.N == 1   # only the sentinel colour

    def test_last_entry_is_unassigned_colour(self):
        gcm = GroupColourMap(["A"])
        cmap = gcm.build_cmap()
        # index 1 = sentinel
        sentinel_rgba = cmap(1)[:3]   # (r, g, b, a) → (r, g, b)
        assert np.allclose(sentinel_rgba, UNASSIGNED_COLOUR, atol=0.02)

    def test_cmap_reflects_overrides(self):
        gcm = GroupColourMap(["A"])
        gcm.update({"A": (1.0, 0.0, 0.0)})
        cmap = gcm.build_cmap()
        entry_rgba = cmap(0)[:3]
        assert np.allclose(entry_rgba, (1.0, 0.0, 0.0), atol=0.02)


# ── write_scalars ─────────────────────────────────────────────────────────────

class TestWriteScalars:
    def test_writes_group_id_array(self):
        gcm = GroupColourMap(["ELEV_A", "MAIN"])
        mesh = _make_mesh_with_eids([1, 2, 3])
        elem_to_group = {1: "ELEV_A", 2: "MAIN", 3: "ELEV_A"}
        gcm.write_scalars(mesh, elem_to_group)
        assert "group_id" in mesh.cell_data

    def test_correct_group_ids_assigned(self):
        gcm = GroupColourMap(["ELEV_A", "MAIN"])
        mesh = _make_mesh_with_eids([1, 2])
        elem_to_group = {1: "ELEV_A", 2: "MAIN"}
        gcm.write_scalars(mesh, elem_to_group)
        gids = mesh.cell_data["group_id"]
        assert gids[0] == 0   # ELEV_A is index 0 (sorted)
        assert gids[1] == 1   # MAIN is index 1

    def test_unassigned_element_gets_sentinel(self):
        gcm = GroupColourMap(["ELEV_A"])
        mesh = _make_mesh_with_eids([1, 99])
        elem_to_group = {1: "ELEV_A"}   # 99 not in any group
        gcm.write_scalars(mesh, elem_to_group)
        gids = mesh.cell_data["group_id"]
        assert gids[1] == 1   # sentinel = n_groups = 1

    def test_multiple_cells_same_element(self):
        # A beam mesh has multiple faces per beam; all should get the same gid
        gcm = GroupColourMap(["ELEV_A", "MAIN"])
        mesh = _make_mesh_with_eids([5, 5, 5, 6])
        elem_to_group = {5: "MAIN", 6: "ELEV_A"}
        gcm.write_scalars(mesh, elem_to_group)
        gids = mesh.cell_data["group_id"]
        assert all(gids[:3] == 1)   # MAIN → index 1
        assert gids[3] == 0         # ELEV_A → index 0

    def test_write_scalars_with_real_model(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        from fahts.renderer.beam_geometry import build_model_mesh
        model = read_usfos_fem(p)
        mesh = build_model_mesh(model)

        group_names = sorted(model.groups.keys())
        elem_to_group: dict[int, str] = {}
        for name, group in model.groups.items():
            for eid in group.element_ids:
                elem_to_group[eid] = name

        gcm = GroupColourMap(group_names)
        gcm.write_scalars(mesh, elem_to_group)

        gids = mesh.cell_data["group_id"]
        n_groups = len(group_names)
        assert gids.min() >= 0
        assert gids.max() <= n_groups   # max is sentinel


# ── add_mesh_kwargs ───────────────────────────────────────────────────────────

class TestAddMeshKwargs:
    def test_required_keys_present(self):
        gcm = GroupColourMap(["A", "B"])
        kwargs = gcm.add_mesh_kwargs()
        assert "scalars" in kwargs
        assert "cmap" in kwargs
        assert "clim" in kwargs
        assert "n_colors" in kwargs

    def test_scalars_key_is_group_id(self):
        gcm = GroupColourMap(["A"])
        assert gcm.add_mesh_kwargs()["scalars"] == "group_id"

    def test_n_colors_is_n_plus_1(self):
        gcm = GroupColourMap(["A", "B", "C"])
        assert gcm.add_mesh_kwargs()["n_colors"] == 4

    def test_clim_spans_all_groups(self):
        gcm = GroupColourMap(["A", "B"])
        clim = gcm.add_mesh_kwargs()["clim"]
        assert clim[0] < 0       # lower bound below 0
        assert clim[1] >= 2      # upper bound covers n_groups

    def test_scalar_bar_hidden(self):
        gcm = GroupColourMap(["A"])
        assert gcm.add_mesh_kwargs()["show_scalar_bar"] is False


# ── SceneManager integration ──────────────────────────────────────────────────

class TestSceneManagerIntegration:
    def test_colour_map_created_after_load(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        from fahts.renderer.scene_manager import SceneManager

        model = read_usfos_fem(p)
        scene = SceneManager(off_screen=True)
        scene.load_model(model)

        assert scene._group_colour_map is not None
        assert isinstance(scene._group_colour_map, GroupColourMap)
        scene.close()

    def test_group_id_scalars_written_on_load(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        from fahts.renderer.scene_manager import SceneManager

        model = read_usfos_fem(p)
        scene = SceneManager(off_screen=True)
        scene.load_model(model)
        assert "group_id" in scene._full_solid_mesh.cell_data
        scene.close()

    def test_colour_by_group_override_reflected(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        from fahts.renderer.scene_manager import SceneManager

        model = read_usfos_fem(p)
        scene = SceneManager(off_screen=True)
        scene.load_model(model)

        first_group = sorted(model.groups.keys())[0]
        scene.colour_by_group({first_group: (1.0, 0.0, 0.0)})
        assert scene._group_colour_map.colour_for(first_group) == (1.0, 0.0, 0.0)
        scene.close()


# ── MainWindow colour-mode persistence ────────────────────────────────────────

class TestColourPersistence:
    """Colour mode must survive a reload when the user has selected group colour."""

    @pytest.fixture(scope="module")
    def qapp(self):
        import sys
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def window(self, qapp):
        pytest.importorskip("pyvistaqt")
        from fahts.gui.main_window import MainWindow
        w = MainWindow()
        w._recent_files.clear()
        w._rebuild_recent_menu()
        yield w
        w._plotter.close()
        w.close()

    def test_default_colour_after_load(self, window):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        window.open_file(p)
        assert window._scene.colour_mode == "default"

    def test_group_colour_persists_across_reload(self, window):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        window.open_file(p)
        # Select group colour
        window._action_colour_group.setChecked(True)
        window._on_colour_mode_changed(window._action_colour_group)
        assert window._scene.colour_mode == "group"
        # Reload: colour mode must be re-applied
        window.open_file(p)
        assert window._scene.colour_mode == "group"

    def test_default_colour_not_reapplied_when_not_selected(self, window):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        window.open_file(p)
        # User did NOT switch to group colour
        assert window._action_colour_default.isChecked()
        window.open_file(p)
        assert window._scene.colour_mode == "default"


# ── TemperatureColourMap ──────────────────────────────────────────────────────


def _make_mesh_with_eids_temp(element_ids: list[int]) -> pv.PolyData:
    pts_list, faces_list = [], []
    offset = 0
    for _ in element_ids:
        pts_list.append([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces_list.extend([3, offset, offset + 1, offset + 2])
        offset += 3
    pts = np.array(pts_list, dtype=float).reshape(-1, 3)
    mesh = pv.PolyData(pts, np.array(faces_list, dtype=int))
    mesh.cell_data["element_id"] = np.array(element_ids, dtype=np.int32)
    return mesh


class TestTemperatureColourMap:

    # ── Construction ──────────────────────────────────────────────────────────

    def test_default_cmap_is_inferno(self):
        tcm = TemperatureColourMap()
        assert tcm.cmap == "inferno"

    def test_clim_initially_none(self):
        assert TemperatureColourMap().clim is None

    def test_n_colors_default(self):
        assert TemperatureColourMap().n_colors == 256

    def test_invalid_cmap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            TemperatureColourMap(cmap="viridis")

    def test_supported_cmaps_non_empty(self):
        assert len(TemperatureColourMap.SUPPORTED_CMAPS) >= 4

    def test_all_supported_cmaps_construct(self):
        for name in TemperatureColourMap.SUPPORTED_CMAPS:
            tcm = TemperatureColourMap(cmap=name)
            assert tcm.cmap == name

    def test_clim_stored_at_construction(self):
        tcm = TemperatureColourMap(clim=(100.0, 800.0))
        assert tcm.clim == (100.0, 800.0)

    # ── set_cmap ──────────────────────────────────────────────────────────────

    def test_set_cmap_valid(self):
        tcm = TemperatureColourMap()
        tcm.set_cmap("jet")
        assert tcm.cmap == "jet"

    def test_set_cmap_invalid_raises(self):
        tcm = TemperatureColourMap()
        with pytest.raises(ValueError, match="Unknown colormap"):
            tcm.set_cmap("magma")

    # ── set_clim / clear_clim ─────────────────────────────────────────────────

    def test_set_clim(self):
        tcm = TemperatureColourMap()
        tcm.set_clim(50.0, 700.0)
        assert tcm.clim == (50.0, 700.0)

    def test_set_clim_coerces_to_float(self):
        tcm = TemperatureColourMap()
        tcm.set_clim(0, 500)
        lo, hi = tcm.clim
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_clear_clim(self):
        tcm = TemperatureColourMap(clim=(0.0, 100.0))
        tcm.clear_clim()
        assert tcm.clim is None

    # ── auto_clim ─────────────────────────────────────────────────────────────

    def test_auto_clim_basic(self):
        tcm = TemperatureColourMap()
        lo, hi = tcm.auto_clim([100.0, 300.0, 600.0])
        assert lo == pytest.approx(100.0)
        assert hi == pytest.approx(600.0)

    def test_auto_clim_adds_pad_when_range_tiny(self):
        tcm = TemperatureColourMap()
        lo, hi = tcm.auto_clim([500.0, 500.0])   # zero range
        assert hi - lo == pytest.approx(1.0)

    def test_auto_clim_no_pad_when_range_large(self):
        tcm = TemperatureColourMap()
        lo, hi = tcm.auto_clim([20.0, 800.0])
        assert hi == pytest.approx(800.0)

    def test_auto_clim_stores_result(self):
        tcm = TemperatureColourMap()
        tcm.auto_clim([20.0, 400.0])
        assert tcm.clim is not None
        assert tcm.clim[0] == pytest.approx(20.0)

    def test_auto_clim_works_with_ndarray(self):
        tcm = TemperatureColourMap()
        arr = np.linspace(20.0, 600.0, 50)
        lo, hi = tcm.auto_clim(arr)
        assert lo == pytest.approx(20.0)
        assert hi == pytest.approx(600.0)

    # ── write_scalars ─────────────────────────────────────────────────────────

    def test_write_scalars_writes_cell_data(self):
        mesh = _make_mesh_with_eids_temp([1, 2, 3])
        tcm = TemperatureColourMap()
        tcm.write_scalars(mesh, {1: 100.0, 2: 200.0, 3: 300.0})
        assert "temperature_C" in mesh.cell_data

    def test_write_scalars_correct_values(self):
        mesh = _make_mesh_with_eids_temp([10, 20])
        tcm = TemperatureColourMap()
        tcm.write_scalars(mesh, {10: 400.0, 20: 600.0})
        arr = mesh.cell_data["temperature_C"]
        assert arr[0] == pytest.approx(400.0)
        assert arr[1] == pytest.approx(600.0)

    def test_write_scalars_default_for_missing(self):
        mesh = _make_mesh_with_eids_temp([1, 99])
        tcm = TemperatureColourMap()
        tcm.write_scalars(mesh, {1: 500.0})  # eid 99 absent
        arr = mesh.cell_data["temperature_C"]
        assert arr[1] == pytest.approx(20.0)  # default_T

    def test_write_scalars_custom_default(self):
        mesh = _make_mesh_with_eids_temp([5])
        tcm = TemperatureColourMap()
        tcm.write_scalars(mesh, {}, default_T=100.0)
        assert mesh.cell_data["temperature_C"][0] == pytest.approx(100.0)

    def test_write_scalars_returns_array(self):
        mesh = _make_mesh_with_eids_temp([1])
        tcm = TemperatureColourMap()
        result = tcm.write_scalars(mesh, {1: 350.0})
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(350.0)

    def test_write_scalars_also_writes_point_data(self):
        mesh = _make_mesh_with_eids_temp([1, 2])
        tcm = TemperatureColourMap()
        tcm.write_scalars(mesh, {1: 200.0, 2: 400.0})
        assert "temperature_C" in mesh.point_data

    # ── add_mesh_kwargs ───────────────────────────────────────────────────────

    def test_add_mesh_kwargs_raises_without_clim(self):
        tcm = TemperatureColourMap()
        with pytest.raises(ValueError, match="Colour limits not set"):
            tcm.add_mesh_kwargs()

    def test_add_mesh_kwargs_has_required_keys(self):
        tcm = TemperatureColourMap(clim=(20.0, 600.0))
        kw = tcm.add_mesh_kwargs()
        for key in ("scalars", "cmap", "clim", "n_colors", "show_scalar_bar"):
            assert key in kw

    def test_add_mesh_kwargs_scalars_is_temperature_c(self):
        tcm = TemperatureColourMap(clim=(0.0, 800.0))
        assert tcm.add_mesh_kwargs()["scalars"] == "temperature_C"

    def test_add_mesh_kwargs_uses_cmap(self):
        tcm = TemperatureColourMap(cmap="jet", clim=(0.0, 500.0))
        assert tcm.add_mesh_kwargs()["cmap"] == "jet"

    def test_add_mesh_kwargs_uses_clim(self):
        tcm = TemperatureColourMap(clim=(50.0, 700.0))
        kw = tcm.add_mesh_kwargs()
        assert kw["clim"][0] == pytest.approx(50.0)
        assert kw["clim"][1] == pytest.approx(700.0)

    def test_add_mesh_kwargs_shows_scalar_bar(self):
        tcm = TemperatureColourMap(clim=(0.0, 600.0))
        assert tcm.add_mesh_kwargs()["show_scalar_bar"] is True

    def test_add_mesh_kwargs_after_auto_clim(self):
        tcm = TemperatureColourMap()
        tcm.auto_clim([20.0, 800.0])
        kw = tcm.add_mesh_kwargs()
        assert kw["clim"][0] == pytest.approx(20.0)
        assert kw["clim"][1] == pytest.approx(800.0)

    def test_cmap_change_reflected_in_kwargs(self):
        tcm = TemperatureColourMap(cmap="inferno", clim=(0.0, 500.0))
        tcm.set_cmap("plasma")
        assert tcm.add_mesh_kwargs()["cmap"] == "plasma"


# ── SceneManager — set_temperature_cmap integration ──────────────────────────


class TestSceneManagerTemperatureCmap:
    @pytest.fixture(scope="module")
    def real_model(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        return read_usfos_fem(p)

    @pytest.fixture
    def sm(self, real_model):
        from fahts.renderer.scene_manager import SceneManager
        mgr = SceneManager(off_screen=True)
        mgr.load_model(real_model)
        yield mgr
        mgr.close()

    def test_default_temperature_cmap_is_inferno(self, sm):
        assert sm.temperature_cmap == "inferno"

    def test_set_temperature_cmap_changes_name(self, sm):
        sm.set_temperature_cmap("jet")
        assert sm.temperature_cmap == "jet"

    def test_set_temperature_cmap_invalid_raises(self, sm):
        with pytest.raises(ValueError):
            sm.set_temperature_cmap("viridis")

    def test_set_temperature_cmap_reflected_in_kwargs(self, sm, real_model):
        T_map = {eid: 400.0 for eid in real_model.elements}
        sm.colour_by_temperature(T_map)
        sm.set_temperature_cmap("plasma")
        assert sm.temperature_cmap == "plasma"

    def test_T_clim_property_after_colour_by_temperature(self, sm, real_model):
        T_map = {eid: 400.0 for eid in real_model.elements}
        sm.colour_by_temperature(T_map, clim=(0.0, 800.0))
        assert sm._T_clim == (0.0, 800.0)

    def test_T_clim_none_after_load(self, sm):
        assert sm._T_clim is None


# ── Task 4.9 — Threshold overlay constants ───────────────────────────────────


class TestThresholdConstants:
    def test_critical_temperature_steel_is_660(self):
        assert CRITICAL_TEMPERATURE_STEEL == pytest.approx(660.0)

    def test_threshold_colour_is_rgba_tuple(self):
        assert len(THRESHOLD_COLOUR) == 4
        assert all(0.0 <= c <= 1.0 for c in THRESHOLD_COLOUR)


# ── Task 4.9 — TemperatureColourMap threshold overlay ────────────────────────


class TestThresholdOverlay:

    # ── Defaults ──────────────────────────────────────────────────────────────

    def test_threshold_overlay_defaults_false(self):
        tcm = TemperatureColourMap()
        assert tcm.threshold_overlay is False

    def test_T_crit_defaults_to_critical_temperature_steel(self):
        tcm = TemperatureColourMap()
        assert tcm.T_crit == pytest.approx(CRITICAL_TEMPERATURE_STEEL)

    def test_threshold_overlay_set_at_construction(self):
        tcm = TemperatureColourMap(threshold_overlay=True)
        assert tcm.threshold_overlay is True

    def test_T_crit_set_at_construction(self):
        tcm = TemperatureColourMap(T_crit=500.0)
        assert tcm.T_crit == pytest.approx(500.0)

    # ── set_threshold_overlay ─────────────────────────────────────────────────

    def test_set_threshold_overlay_enables(self):
        tcm = TemperatureColourMap()
        tcm.set_threshold_overlay(True)
        assert tcm.threshold_overlay is True

    def test_set_threshold_overlay_disables(self):
        tcm = TemperatureColourMap(threshold_overlay=True)
        tcm.set_threshold_overlay(False)
        assert tcm.threshold_overlay is False

    def test_set_threshold_overlay_updates_T_crit(self):
        tcm = TemperatureColourMap()
        tcm.set_threshold_overlay(True, T_crit=500.0)
        assert tcm.T_crit == pytest.approx(500.0)

    def test_set_threshold_overlay_none_T_crit_unchanged(self):
        tcm = TemperatureColourMap(T_crit=550.0)
        tcm.set_threshold_overlay(True, T_crit=None)
        assert tcm.T_crit == pytest.approx(550.0)

    # ── _build_threshold_lut ──────────────────────────────────────────────────

    def test_lut_has_correct_number_of_colours(self):
        tcm = TemperatureColourMap(n_colors=64, clim=(20.0, 800.0), threshold_overlay=True)
        lut = tcm._build_threshold_lut()
        assert lut.N == 64

    def test_lut_colours_above_threshold_are_alarm_colour(self):
        # clim 20–800, T_crit=660 → roughly top 17 % of the LUT is alarm colour
        tcm = TemperatureColourMap(n_colors=256, clim=(20.0, 800.0))
        tcm.set_threshold_overlay(True, T_crit=660.0)
        lut = tcm._build_threshold_lut()
        # Sample a position clearly above T_crit (e.g. 750 °C → fraction ~0.94)
        high_frac = (750.0 - 20.0) / (800.0 - 20.0)
        colour = lut(high_frac)
        expected = THRESHOLD_COLOUR
        assert np.allclose(colour[:3], expected[:3], atol=0.02)

    def test_lut_colours_below_threshold_differ_from_alarm(self):
        tcm = TemperatureColourMap(n_colors=256, clim=(20.0, 800.0))
        tcm.set_threshold_overlay(True, T_crit=660.0)
        lut = tcm._build_threshold_lut()
        # 200 °C is well below T_crit
        low_frac = (200.0 - 20.0) / (800.0 - 20.0)
        colour = lut(low_frac)
        assert not np.allclose(colour[:3], THRESHOLD_COLOUR[:3], atol=0.02)

    def test_lut_all_alarm_when_T_crit_below_lo(self):
        # T_crit below clim[0] → entire LUT is alarm colour
        tcm = TemperatureColourMap(n_colors=64, clim=(200.0, 800.0))
        tcm.set_threshold_overlay(True, T_crit=100.0)
        lut = tcm._build_threshold_lut()
        sample = lut(0.5)
        assert np.allclose(sample[:3], THRESHOLD_COLOUR[:3], atol=0.02)

    def test_lut_no_alarm_when_T_crit_above_hi(self):
        # T_crit above clim[1] → no cell gets alarm colour
        tcm = TemperatureColourMap(n_colors=64, clim=(20.0, 500.0))
        tcm.set_threshold_overlay(True, T_crit=660.0)
        lut = tcm._build_threshold_lut()
        # Check every sample in [0,1]: none should be alarm colour
        for frac in np.linspace(0.0, 1.0, 32):
            colour = lut(frac)
            assert not np.allclose(colour[:3], THRESHOLD_COLOUR[:3], atol=0.02)

    # ── add_mesh_kwargs with overlay ──────────────────────────────────────────

    def test_add_mesh_kwargs_returns_listed_colormap_when_overlay_on(self):
        from matplotlib.colors import ListedColormap
        tcm = TemperatureColourMap(clim=(20.0, 800.0), threshold_overlay=True)
        kw = tcm.add_mesh_kwargs()
        assert isinstance(kw["cmap"], ListedColormap)

    def test_add_mesh_kwargs_returns_string_cmap_when_overlay_off(self):
        tcm = TemperatureColourMap(clim=(20.0, 800.0), threshold_overlay=False)
        kw = tcm.add_mesh_kwargs()
        assert isinstance(kw["cmap"], str)

    def test_add_mesh_kwargs_title_includes_threshold_when_overlay_on(self):
        tcm = TemperatureColourMap(clim=(20.0, 800.0), threshold_overlay=True, T_crit=660.0)
        kw = tcm.add_mesh_kwargs()
        title = kw["scalar_bar_args"]["title"]
        assert "660" in title

    def test_add_mesh_kwargs_title_plain_when_overlay_off(self):
        tcm = TemperatureColourMap(clim=(20.0, 800.0), threshold_overlay=False)
        kw = tcm.add_mesh_kwargs()
        title = kw["scalar_bar_args"]["title"]
        assert "660" not in title and "⚠" not in title


# ── Task 4.9 — SceneManager threshold overlay integration ────────────────────


class TestSceneManagerThresholdOverlay:
    @pytest.fixture(scope="module")
    def real_model(self):
        p = Path(__file__).parents[1] / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        return read_usfos_fem(p)

    @pytest.fixture
    def sm(self, real_model):
        from fahts.renderer.scene_manager import SceneManager
        mgr = SceneManager(off_screen=True)
        mgr.load_model(real_model)
        yield mgr
        mgr.close()

    def test_set_threshold_overlay_toggles_colour_map_flag(self, sm):
        sm.set_threshold_overlay(True)
        assert sm._T_colour_map.threshold_overlay is True
        sm.set_threshold_overlay(False)
        assert sm._T_colour_map.threshold_overlay is False

    def test_set_threshold_overlay_updates_T_crit(self, sm):
        sm.set_threshold_overlay(True, T_crit=500.0)
        assert sm._T_colour_map.T_crit == pytest.approx(500.0)

    def test_set_threshold_overlay_refreshes_when_in_temperature_mode(
        self, sm, real_model
    ):
        T_map = {eid: 400.0 for eid in real_model.elements}
        sm.colour_by_temperature(T_map, clim=(20.0, 800.0))
        assert sm.colour_mode == "temperature"
        # Should not raise and scene should still be in temperature mode
        sm.set_threshold_overlay(True)
        assert sm.colour_mode == "temperature"
