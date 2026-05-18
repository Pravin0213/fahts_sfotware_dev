# GUI Domain — FAHTS

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
