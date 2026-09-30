# Renderer Domain — FAHTS

## Files

| File | Role |
|------|------|
| `beam_geometry.py` | Build PyVista meshes from FEMModel (centreline + extruded sections) |
| `colormap.py` | Colour lookup tables for group coloring and temperature mapping |
| `scene_manager.py` | Drives the PyVista plotter; all 3-D scene operations |
| `vessel_geometry.py` | Process vessel for the Process tab's 3-D view: shell with per-cell wall-region ids (dry / wet / jet zone dry / wet) at a liquid level, heads, liquid body; `paint()` from region temperatures. No Qt. |

> **SceneManager** is large (1137 lines). Full API docs live in `fahts/gui/CLAUDE.md`
> (the SceneManager section). This file covers geometry and colormap specifics.

---

## beam_geometry.py

Public entry points:

```python
build_centreline_mesh(model: FEMModel) → pv.PolyData
build_model_mesh(model: FEMModel) → pv.PolyData
build_analysis_mesh_overlay(model, solved_eids, config, centroid) → pv.PolyData
build_mesh_inspector_data(model, centroid, config=None) → MeshInspectorData
```

- `build_centreline_mesh`: line segments between node pairs — fast overview, no cross-section.
- `build_model_mesh`: **USFOS-style thin mid-surface panels** (no wall thickness rendered).
  Supports **BOX** (4 lateral quads), **PIPE** (outer-ring quads only), **IHPROFIL**
  (3 flat panels: top flange / web / bottom flange), and flat-polygon shells.
  Mesh nodes from the FEM solver sit on the same surfaces — no occlusion.
- `build_mesh_inspector_data`: builds `MeshInspectorData` for interactive connectivity
  inspection. Stores per-element quad connectivity so clicking a quad reveals K-matrix
  neighbours (quads sharing a node = shared DOF = coupled in K).

### MeshInspectorData

```python
@dataclass
class MeshInspectorData:
    mesh: pv.PolyData              # global scene-coord quads for picking
    cell_beam_eid: np.ndarray      # (n_cells,) element ID per cell
    cell_quad_idx: np.ndarray      # (n_cells,) local quad index
    beam_quads: dict[int, np.ndarray]   # eid → (n_quads, 4) local node indices
    beam_cell_offset: dict[int, int]    # eid → first cell index in mesh
    beam_node_offset: dict[int, int]    # eid → first global point index in mesh
```

Cell data keys: `inspector_beam_eid`, `inspector_quad_idx` (distinguish from element mesh).

**Node label placement gotcha:** `extract_cells([cell_idx])` returns points sorted by
ascending global point index, NOT in the quad's connectivity order. Always look up label
positions via `mesh.points[node_idx + beam_node_offset[eid]]` — never use
`highlighted.points` for label coordinates.

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
- `_on_cell_picked` routes on cell data keys: `inspector_beam_eid` → inspector callback;
  `element_id` → element-pick callback. Both coexist — inspector mode does not disable element picking.

---

## vessel_geometry.py

`VesselGeometry3D(case)`: axis along x, z up, angles from the top (heat-load convention).
Grid lines include the jet-zone edges so the zone is exact; the 0/360 deg seam is merged
(`clean()`), otherwise the zone outline shows a false edge where the zone crosses the top.
`wall_surface(cutaway)` is the solid wall (built on first use: x × theta × radial hexahedra
in that axis order - (theta, x, r) turns every cell inside out; seam merged); `paint_wall()`
takes a temperature or a through-thickness profile `(x_nodes, T_nodes)` per region
(`region_profiles`). `thickness_scale` exaggerates the drawn thickness only; depths are mapped
back to real metres before interpolating the model's node temperatures.
`set_level(level)` recomputes region ids; `region_fractions()` equals the model's
`VesselFireModel.frac` to < 0.3 % (tests/unit/renderer) - keep it that way: the view must show
what the model computes.

