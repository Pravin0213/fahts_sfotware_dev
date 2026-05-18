# FAHTS — Completed Phase History

This file is an archive of completed phase implementation details.
It is NOT auto-loaded — read it only when debugging past decisions or understanding
why something was built a specific way.

---

## Phase 1 — Foundation ✅ COMPLETE

All 10 tasks done. See ROADMAP.md for full details.
Key deliverables: FEMModel dataclasses, USFOS reader, PyQt6 + PyVista app shell,
model tree panel, 3D beam geometry rendering.

---

## Phase 2 — Heat Source Engine ✅ COMPLETE

| Task | Module | What it does |
|------|--------|-------------|
| 2.1 | `core/heat/sources/fire_zone.py` | FireCurve: ISO 834, hydrocarbon, user-defined piecewise linear |
| 2.2 | `core/heat/sources/fire_zone.py` | FireZone: rectangular box with curve, emissivity, h_conv, active flag |
| 2.3 | `gui/dialogs/fire_zone_dialog.py` | Qt dialog: name, centre X/Y/Z, dims, curve type, user pts, ε, h_conv |
| 2.4 | `renderer/scene_manager.py` | `show_fire_zones()` — semi-transparent orange box actors |
| 2.5 | `core/heat/bc/view_factor.py` | Binary midpoint-in-box exposure; `exposed_element_ids()`; `compute_all_bcs()` |
| 2.6 | `core/heat/bc/net_flux.py` | `net_heat_flux()`, `radiative_flux()`, `convective_flux()` per EN 1993-1-2 §3.1 |
| 2.7 | `gui/panels/heat_source_panel.py` | HeatSourcePanel: mixed FireZone+RadiationBall list; Add Zone/Add Ball/Edit/Delete |
| 2.8 | `renderer/scene_manager.py` | `highlight_exposed_beams()` — grey→fire-orange per-cell scalar |
| 2.9 | `core/io/results_writer.py` | `export_bc_summary_csv()`, `export_bc_summary_multi_time()` |

**Post-completion GUI enhancements:**
- Non-modal fire zone dialog; viewport point-picking for fire zone centre
- KFXView-style axis marker (3 bidirectional coloured arrows + sphere, pinned to camera focal point via 1 ms QTimer in `_on_scene_tick`)
- Live world-coordinate display in status bar

**Axis marker architecture (critical detail):**
- All axis geometry built at origin `(0,0,0)` in `_rebuild_axis_marker()`.
- 1 ms QTimer fires `_on_scene_tick()` → `actor.SetPosition(camera.focal_point)` on every axis actor.
- `add_on_render_callback` was NOT used — it fires on interaction start/end, not per frame.
- `set_axis_origin_world(xyz)` pans camera so focal point moves to the world position.

---

## Phase 3 — Heat Transfer Solver ✅ COMPLETE

### Tasks 3.1–3.9

| Task | Module | Status |
|------|--------|--------|
| 3.1 | `core/model/material.py` — EN 1993-1-2 Annex C k(T), cp(T) | ✅ Done |
| 3.2 | `core/heat/section_mesh/box_mesher.py` — BOX 2-D Quad4 mesh | ✅ Done |
| 3.3 | `core/heat/solver/fem_2d_section.py` — Quad4 K and C matrices | ✅ Done |
| 3.4 | `core/heat/solver/fem_2d_section.py` — global assembly + Robin BC | ✅ Done |
| 3.5 | `core/heat/solver/time_integrator.py` — TransientSolver (legacy, kept) | ✅ Done |
| 3.6 | `core/results/temperature_field.py` — TemperatureField results container | ✅ Done |
| 3.7 | `gui/dialogs/run_analysis_dialog.py` + `core/results/analysis_config.py` | ✅ Done |
| 3.8 | `core/heat/solver/analysis_runner.py` + `gui/analysis_worker.py` | ✅ Done |
| 3.9 | `core/results/post_processor.py` — PostProcessor + ElementSummary | ✅ Done |

### Key module signatures (Tasks 3.2–3.9)

**SectionMesh / BoxMesher (3.2):**
- `SectionMesh`: nodes (n,2) [y,z coords], quads (n,4) CCW, outer_edge_pairs, inner_edge_pairs
- `BoxMesher(section, elem_size=None, n_layers=1).build() → SectionMesh`
- Wall-by-wall: bottom/top span full width; left/right webs span inner height only
- Node deduplication via coordinate snapping (1e-10 m tolerance)

**FEM 2-D (3.3 + 3.4) — fem_2d_section.py:**
- `quad4_conductivity_matrix(coords, k)` — 2×2 Gauss quadrature, (4,4) symmetric
- `quad4_capacity_matrix(coords, rho, cp, lumped=True)` — row-sum lumped: (4,) diagonal
- `assemble_K(mesh, k) → csr_matrix`, `assemble_C_lumped(mesh, rho, cp) → ndarray`
- `add_robin_bc(A_lil, b, edge_pairs, nodes, T_fire, T_prev, epsilon_m, h_conv)`:
  - Convection: semi-implicit → added to A
  - Radiation: `ε·σ·(T_fire_K⁴ − T_K⁴)` explicit → added to b; `_SIGMA = 5.67e-8`

**AnalysisConfig (3.7):**
- `validate()` raises ValueError on physically inconsistent params
- `n_output_steps` property, `summary()` string

**RunAnalysisDialog (3.7):**
- Modal; shows exposed element count, duration/dt/output_dt spinboxes
- `config_accepted(AnalysisConfig)` signal on OK
- Heat menu → "Run Analysis…" (F5)

**analysis_runner / AnalysisWorker (3.8):**
- `run_analysis(model, fire_zones, config, progress_cb, cancel_check) → TemperatureField`
- Multi-zone: `fire_temp(t) = max(zone.temperature(t) for covering_zones)`
- RadiationBall takes precedence over FireZone when both cover an element
- `epsilon_m = ref_zone.epsilon_fire × 0.7`; ref_zone = hottest FireZone at t_end
- `AnalysisWorker(QThread)` signals: `progress(cur,tot,eid)`, `finished(tf)`, `error(str)`, `cancelled()`

**PostProcessor (3.9):**
- `ElementSummary(eid, T_initial, T_peak_centroid, T_peak_nodal, t_crit_500, t_crit_600)`
- `fire_resistance_minutes(eid)`, `minimum_fire_resistance()`, `summary_text()`
- `to_csv(path)`, `to_dataframe()`, `to_history_dataframe()`

### Phase 3E — Multi-section solver ✅ COMPLETE (2026-05-17)

| Task | Module | Notes |
|------|--------|-------|
| 3E.1 | `heat/section_mesh/ihprofil_mesher.py` | 3-patch Quad4 mesh for I/H-sections |
| 3E.2 | `heat/section_mesh/pipe_mesher.py` | Polar annular Quad4 mesh for PIPE |
| 3E.3 | `heat/section_mesh/shell_mesh.py` | 1-D through-thickness mesh (ShellMesh1D) |
| 3E.4 | `heat/solver/shell_1d_solver.py` | 1-D backward Euler rod FEM for shells |
| 3E.5 | `heat/sources/rad_ball.py`, `gui/dialogs/rad_ball_dialog.py` | RadiationBall (USERFLUX type 0) |
| 3E.6 | `heat/solver/analysis_runner.py` | Extended to all section types + RadiationBall |
| 3E.7 | `core/io/beltemp_parser.py` + `results_writer.export_beltemp()` | USFOS BELTEMP I/O |

**IProfileMesher (3E.1):** 3 rectangular patches (top flange, web, bottom flange); shared nodes at junctions. `inner_edge_pairs` always empty (solid).

**PipeMesher (3E.2):** Polar annular mesh: `(n_radial+1) × n_arc` nodes. `outer_edge_pairs` at r=R_out; `inner_edge_pairs` at r=R_in (adiabatic).

**ShellMesh1D (3E.3):** 1-D through-thickness. `outer_node=0` (fire), `inner_node=n_layers` (adiabatic).

**Shell1DSolver (3E.4):** Backward Euler 1-D rod FEM. Same interface as TransientSolver. `TemperatureField.from_shell_solver_run()` for results.

**RadiationBall (3E.5):** Fields: `center (3,)`, `r1`, `flux1`, `r2`, `flux2`. `flux_at(distance)` returns flux1/flux2/0 by zone. `epsilon_m=0` in solver.

### Phase 3F — Crank-Nicolson + Heat Accumulation ✅ COMPLETE (2026-05-18)

**Step 1 ✅:** Crank-Nicolson (θ=1/2) in `time_integrator.py` and `shell_1d_solver.py`.

CN incremental form (SINTEF Eq. 3.2.30):
```
A = Ki + (2/Δt)·Mi
B = Qi - K_{i-1}·T_{i-1} + M_{i-1}·Ṫ_{i-1}
ΔTi = A^{-1}·B;  Ti = T_{i-1} + ΔTi;  Ṫi = (2/Δt)·ΔTi - Ṫ_{i-1}
Initial rate: Ṫ0 = M0^{-1}·(Q0 - K0·T0)
```

**Step 2 ✅:** Heat accumulation element in `analysis_runner.py`. Hollow profiles (BOX, PIPE) get lumped mass `m_i = A_inner·L·ρc_air/n_inner_nodes` (air ρc≈1200 J/m³K) added to M at inner nodes.

**Step 3 ✅:** Temperature gradient (βy, βz) in `results_writer.export_beltemp()` and `TemperatureField.section_gradient()`. SINTEF §3.4.2: `βz = Σ(T_k·y_k·A_k)/Iz`, `βy = Σ(T_k·z_k·A_k)/Iy`.

### Full Surface Mesh Refactor ✅ COMPLETE (2026-05-18)

All section types converted from cross-section 2-D to FAHTS axial × hoop surface-shell.

| Mesher | Defaults | Key detail |
|--------|----------|-----------|
| `BoxSurfaceMesher` | n_top=2, n_side=3, n_length=4 | 4 outer faces |
| `IProfileSurfaceMesher` | n_top=4, n_side=2, n_bottom=2, n_length=2 | 8 faces incl. overhang undersides |
| `PipeSurfaceMesher` | c_circ=8, n_length=4 | Unrolled cylinder → `local_coords_2d` |
| `PlateSurfaceMesher` | mesh_12=4, mesh_14=2 | Bilinear grid from 4 corner positions |

`BeamSurfaceMesh` gained optional `local_coords_2d` field for curved/tilted geometries.
`SurfaceTransientSolver` calls `element_coords_2d()` uniformly for all types.
`RunAnalysisDialog` has four Default/Custom mesh groups (BOX, IHPROFIL, PIPE, Shell).

---

## Phase 4 — Results Visualisation ✅ COMPLETE

| Task | Module | Status |
|------|--------|--------|
| 4.1 | `renderer/scene_manager.py` — VTK cell + point data; temperature persistence across visibility changes | ✅ |
| 4.2 | `renderer/colormap.py` — `TemperatureColourMap`; jet/inferno/plasma/coolwarm LUT | ✅ |
| 4.3 | `renderer/scene_manager.py` — Colorbar styling; time overlay label | ✅ |
| 4.4 | `gui/main_window.py` — Animation toolbar ⏮⏪▶/⏸⏩⏭; FPS combo; QSlider; keyboard shortcuts | ✅ |
| 4.5 | `gui/panels/results_panel.py` — 2-D cross-section contour (inferno, gouraud) | ✅ |
| 4.6 | `gui/panels/results_panel.py` — Centroid T-t graph; 600°C threshold; animated marker | ✅ |
| 4.7 | `core/io/results_writer.py` — CSV/VTK/Excel export; Heat → Export Results menu | ✅ |
| 4.8 | `renderer/scene_manager.py` — `save_animation()` GIF/MP4; File → Save Screenshot | ✅ |
| 4.9 | `renderer/colormap.py` — Critical threshold overlay (660°C, vivid red) | ✅ |

**Key Phase 4.1:** `colour_by_temperature()` writes both `cell_data` and `point_data["temperature_C"]` (via `cell_data_to_point_data()`) for smooth interpolation. `update_temperature(t, T_field)` stores `_T_field`/`_T_t_idx` for post-rebuild restore.

**Key Phase 4.5:** `ResultsPanel` uses `mtri.Triangulation` (each Quad4 → 2 triangles) + `tripcolor(shading="gouraud")`. Non-BOX sections → `_triang=None` → canvas hidden.

**Key Phase 4.6:** `_tt_vline` is an `axvline` Line2D moved via `set_xdata()` + `draw_idle()` on each animation tick — no full replot.

**Key Phase 4.8:** GIF uses `duration` in ms (not fps) to avoid imageio deprecation. MP4 requires `imageio-ffmpeg`. Frame capture via `self._pl.screenshot(return_img=True)`.

**Key Phase 4.9:** `CRITICAL_TEMPERATURE_STEEL = 660.0`. `_build_threshold_lut()` replaces LUT entries at/above T_crit's fraction in [clim_lo, clim_hi] with `THRESHOLD_COLOUR`.
