# Renderer Domain — FAHTS

## Files

| File | Role |
|------|------|
| `beam_geometry.py` | Build PyVista meshes from FEMModel (centreline + extruded sections) |
| `colormap.py` | Colour lookup tables for group coloring and temperature mapping |
| `scene_manager.py` | Drives the PyVista plotter; all 3-D scene operations |

> **SceneManager** is large (1137 lines). Full API docs live in `fahts/gui/CLAUDE.md`
> (the SceneManager section). This file covers geometry and colormap specifics.

---

## beam_geometry.py

Two public entry points consumed by `SceneManager.load_model()`:

```python
build_centreline_mesh(model: FEMModel) → pv.PolyData
build_model_mesh(model: FEMModel) → pv.PolyData
```

- `build_centreline_mesh`: line segments between node pairs — fast overview, no cross-section.
- `build_model_mesh`: extruded cross-sections at true scale. Supports **BOX**, **PIPE**
  (N-gon approximation with `_PIPE_SIDES=16`), **IHPROFIL**, and flat-polygon shells
  (QUADSHEL/TRISHELL use raw node positions).

### Local frame convention

```python
local_x = normalize(n2 - n1)            # axial
local_z = UNITVEC from .fem file         # cross-section height axis
local_y = normalize(cross(local_x, local_z))   # cross-section width axis
```

Same convention as the solver — do not invert or swap axes.

### BOX corner layout (local y–z plane)

```
index 0: (+H/2, -W/2)  top-left
index 1: (+H/2, +W/2)  top-right
index 2: (-H/2, +W/2)  bottom-right
index 3: (-H/2, -W/2)  bottom-left
```

---

## colormap.py

### GroupColourMap

Assigns one stable colour per named group (discrete LUT). Groups not in the map
get `UNASSIGNED_COLOUR = (0.56, 0.57, 0.59)` (steel grey).

### TemperatureColourMap

Continuous LUT for temperature visualisation.

```python
TemperatureColourMap(lut_name: str = "inferno")
```

Available LUT names: `"jet"`, `"inferno"`, `"plasma"`, `"coolwarm"`.

```python
tcm.set_threshold_overlay(enabled: bool, T_crit: float)
```

When enabled, replaces all LUT entries ≥ `T_crit` with vivid red to flag critical elements.
`CRITICAL_TEMPERATURE_STEEL = 660.0` °C is the default threshold.

---

## scene_manager.py — Quick Reference

Full docs: `fahts/gui/CLAUDE.md` → "3D Viewport" section.

Critical gotchas specific to the renderer layer:
- All geometry is placed at origin (0, 0, 0) — model coords are X≈350m in real space, but
  PyVista works relative to the camera focal point.
- Axis marker position is updated via a 1 ms `QTimer` (`_on_scene_tick`), not a render callback.
- `_rebuild_axis_marker()` must be called after any plotter reset.
- Temperature colour is applied to **both** `cell_data` and `point_data` for smooth interpolation.
