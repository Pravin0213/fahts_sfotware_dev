# GUI Domain — FAHTS

## Two workspaces (tabs of the main window)

- **Structure (3-D)** — the original USFOS / fire-zone / 3-D heat solver view (below).
- **Process vessel** — `gui/process/` (vessel in fire; see section at the end).

## Main Window — main_window.py (1251 lines)

Central controller. Owns: `FEMModel`, `TemperatureField`, `list[FireZone|RadiationBall]`,
`SceneManager`, all panels and dialogs.

Key state:
- `_model: FEMModel | None`
- `_T_field: TemperatureField | None`
- `_fire_sources: list`
- `_selected_eid: int | None`

Key slots:
- `open_file()` → `UsfosReader.read()` → `SceneManager.load_model()`
- `_on_analysis_config_accepted(cfg)` → starts `AnalysisWorker` + `QProgressDialog`
- `_on_element_picked(eid)` → `ResultsPanel.show_section()`
- `_anim_go_to(idx)` → `SceneManager.update_temperature()` + `ResultsPanel.update_time()`

---

## Panel Architecture

All panels live in `gui/panels/`. They receive data via method calls, not signals from
the main window (except HeatSourcePanel which emits `sources_changed`).

| Panel | File | Role |
|-------|------|------|
| ModelTreePanel | `model_tree_panel.py` | Element/group tree; emits `element_selected(eid)` |
| PropertiesPanel | `properties_panel.py` | Shows selected element properties |
| HeatSourcePanel | `heat_source_panel.py` | FireZone + RadiationBall list; emits `sources_changed` |
| ResultsPanel | `results_panel.py` | 2-D cross-section contour + T-t graph (Matplotlib embedded) |
| MeshInspectorPanel | `mesh_inspector_panel.py` | Quad connectivity inspector; `show_quad_info(beam_eid, quad_idx, quad_nodes, beam_quads)` |

---

## ResultsPanel Detail

Two Matplotlib canvases stacked vertically:
1. **Cross-section contour** (`_canvas`): `mtri.Triangulation` from mesh quads (each Quad4 → 2 triangles); `tripcolor(shading="gouraud")`. Hidden for non-BOX sections.
2. **T-t graph** (`_tt_canvas`): centroid T vs time; 600°C dashed line; `_tt_vline` moved via `set_xdata()` on animation ticks (no full replot).

Key methods:
- `show_section(eid, model, result, t_idx)` — full rebuild
- `update_time(t_idx)` — move marker only, no mesh rebuild
- `clear()` — hides both canvases, shows placeholders

---

## Dialogs

| Dialog | File | Trigger |
|--------|------|---------|
| FireZoneDialog | `dialogs/fire_zone_dialog.py` | Non-modal; viewport point-picking for centre |
| RadBallDialog | `dialogs/rad_ball_dialog.py` | Non-modal; viewport picking for centre |
| RunAnalysisDialog | `dialogs/run_analysis_dialog.py` | Modal (F5); emits `config_accepted(AnalysisConfig)` |

**RunAnalysisDialog** has four Default/Custom mesh groups: BOX (n_top/n_side/n_length),
IHPROFIL (n_top_i/n_side_i/n_bottom_i/n_length_i), PIPE (c_circ/n_length_p), Shell (mesh_12/mesh_14).

---

## Animation Toolbar

Shown after analysis, hidden on new model load (in `main_window.py`).
Buttons: ⏮⏪▶/⏸⏩⏭ + FPS combo + stretching QSlider + time label.
Keyboard: Home/Left/Space/Right/End.
`_anim_go_to(idx)` is the single dispatch point for all frame changes.

---

## 3D Viewport — SceneManager (renderer/scene_manager.py, 1137 lines)

Wraps `pyvistaqt.BackgroundPlotter`.

Key methods:
- `load_model(model)` — builds beam geometry, clears temperature state
- `colour_by_temperature(T_per_element, clim)` — writes cell_data + point_data for smooth interpolation
- `update_temperature(t, T_field)` — stores `_T_field`/`_T_t_idx`, restores after mesh rebuilds
- `show_fire_zones(sources)` — semi-transparent orange boxes
- `highlight_exposed_beams(bcs)` — grey→fire-orange per-cell scalar
- `save_animation(T_field, path, fps, progress_callback) → int` — GIF/MP4
- `set_threshold_overlay(enabled, T_crit)` — delegates to `TemperatureColourMap`

**Axis marker:** All geometry at origin (0,0,0). 1 ms QTimer (`_on_scene_tick`) → `actor.SetPosition(camera.focal_point)`. Do NOT use `add_on_render_callback` — it fires on interaction start/end, not per frame.

---

## Colormap — renderer/colormap.py

`TemperatureColourMap` — jet/inferno/plasma/coolwarm LUT.
`set_threshold_overlay(enabled, T_crit)` — replaces LUT entries ≥ T_crit with vivid red.
`CRITICAL_TEMPERATURE_STEEL = 660.0`.

---

## Key Gotchas

- Non-modal dialogs for FireZone/RadBall — user can interact with viewport while dialog is open
- `_rebuild_axis_marker()` must be called after any plotter reset
- Animation GIF: pass `duration` in ms to imageio, not `fps` (avoids deprecation warning)
- MP4 requires `imageio-ffmpeg` — raise `ImportError` with install hint if missing
- `_apply_visibility_filter` must reapply temperature when `_T_field` is set (post-rebuild restore)
- `_mesh_inspector_data` is cached; invalidated when mesh config changes (set to None in `_on_mesh_preview_accepted`)
- Inspector picking coexists with element picking — both modes active simultaneously; routing by cell data key

---

## Process vessel workspace — `gui/process/`

Edits a `fahts.coupling.VesselCase`, runs it (`fahts.coupling.run_case`) and shows results.
Created before the menus (`MainWindow._process_ws`); menu **Process** (new / open / save /
import VessFire deck / check / run F6 / stop); `MainWindow.open_process_case(path)` (case JSON or
deck folder; also `python -m fahts <case.vcase.json | deck folder>`).

| File | Role |
|---|---|
| `workspace.py` | `ProcessWorkspace`: section list + forms, file actions, validation list, Run/Stop + progress, results area; `current_case()`, `load_case()`, `start_run()` |
| `forms.py` | `_Form` contract (`set_case(case)` / `apply(case)` / `changed`); Case, Vessel & material, Relief valves, Surroundings, Stress & run, Model options |
| `contents_form.py` | initial conditions, liquid depths (fill %), composition + pseudo-components, presets, **Check initial state** (builds the real model) |
| `fire_form.py` | heat-load table, quick fill, peak (jet) zone, flux + end/side-view sketches |
| `run_worker.py` | `CaseRunWorker` QThread (progress / finished_ok / error / cancelled) |
| `results_view.py` | summary, plots (pressure, temperatures, inventory, release, stress), failure-time table, CSV / Excel export |
| `fields.py` | `dspin` → `ExactDoubleSpinBox` (keeps the exact loaded value unless the user edits it) |
| `vessel_view.py` | `VesselView`: 3-D vessel (tab "3-D view" next to "Results"): solid steel wall (thickness ×1/3/5/10 for display), cut-away none / quarter / half; wall regions before a run; after it wall temperature through the thickness (model node temperatures at each depth) or through-wall mean, with a time slider; VTK widget created lazily on first show |

Gotchas:
- A form only touches its own part of the case; `current_case()` applies all forms to a copy.
- Add a section: write a `_Form`, then `ProcessWorkspace.add_form(title, form)`.
- Wall regions without area are not plotted (`fahts.coupling.report.regions_with_area`).
- Numbers are shown in the user's locale (comma decimal separator on this machine).
- Tests: `tests/gui/` run offscreen (no VTK); `tests/test_main_window_process.py` and
  `tests/test_vessel_view.py` need a display.
- Qt binding: `fahts/gui/__init__.py` sets `QT_API=pyqt6` (qtpy would otherwise default to
  PyQt5, which is installed here, and pyvistaqt widgets could not be parented to ours).
- Hidden matplotlib canvases have zero size: `ResultsView` draws when first shown.
- `python -m fahts` installs an exception hook: an exception in a slot shows an error dialog
  instead of PyQt6 aborting the application.
- Screenshots of VTK views: use `plotter.screenshot()`; `QWidget.grab()` cannot read OpenGL.

