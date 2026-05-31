# Solver Domain — FAHTS

## Active Solver Architecture

**Primary path (all beam sections):** `SurfaceTransientSolver` in `surface_solver.py`
**Fallback (TRISHELL only):** `Shell1DSolver` in `shell_1d_solver.py`
**Legacy (kept, not primary):** `TransientSolver` in `time_integrator.py`

### SurfaceTransientSolver
- Crank-Nicolson θ=1/2 (SINTEF Eq. 3.2.30)
- Accepts a `BeamSurfaceMesh`; calls `element_coords_2d()` for all geometry
- Heat accumulation element: lumped mass added at inner nodes for BOX/PIPE

### Analysis Dispatch — analysis_runner.py
`run_analysis(model, fire_zones, config, progress_cb, cancel_check) → TemperatureField`

Section type → mesher mapping:
| Section | Mesher | Solver |
|---------|--------|--------|
| BOX | `BoxSurfaceMesher(n_top, n_side, n_length)` | `SurfaceTransientSolver` |
| IHPROFIL | `IProfileSurfaceMesher(n_top_i, n_side_i, n_bottom_i, n_length_i)` | `SurfaceTransientSolver` |
| PIPE | `PipeSurfaceMesher(c_circ, n_length_p)` | `SurfaceTransientSolver` |
| QUADSHEL | `PlateSurfaceMesher(mesh_12, mesh_14)` | `SurfaceTransientSolver` |
| TRISHELL | `ShellMesher` | `Shell1DSolver` |

BC logic:
- FireZone: `fire_temp(t) = max(zone.temperature(t) for covering zones)`
- RadiationBall: overrides FireZone when both cover element; `epsilon_m=0`, `h_conv=0`
- `epsilon_m = ref_zone.epsilon_fire × 0.7`; ref_zone = hottest FireZone at t_end

---

## Crank-Nicolson Theory (SINTEF Eq. 3.2.30)

```
A = Ki + (2/Δt)·Mi
B = Qi - K_{i-1}·T_{i-1} + M_{i-1}·Ṫ_{i-1}
ΔTi = A^{-1}·B
Ti   = T_{i-1} + ΔTi
Ṫi   = (2/Δt)·ΔTi - Ṫ_{i-1}
```
Initial rate: `Ṫ0 = M0^{-1}·(Q0 - K0·T0)`
Between steps: save `K_prev`, `M_prev`, `T_dot_prev`.

---

## Heat Accumulation Element (BOX/PIPE hollow sections)

Trapped air has thermal mass. Lumped at inner surface nodes:
```
m_i = A_inner · L_beam · ρc_air / n_inner_nodes   [J/K]
```
- Air: ρc ≈ 1200 J/(m³·K)
- BOX: `A_inner = (W - 2·T_side) × (H - T_top - T_bot)`
- PIPE: `A_inner = π · R_inner²`
- Inner node indices from `mesh.inner_node_indices`

---

## Surface Mesh Classes

**BeamSurfaceMesh** — `section_mesh/beam_surface_mesh.py`
Key fields: `nodes_3d`, `quads`, `node_areas`, `outer_face_indices`, `inner_node_indices`,
`local_coords_2d` (optional, for curved/tilted geometries like PIPE/plate).

**BoxSurfaceMesher** — 4 outer faces (bottom, top, left, right). Defaults: n_top=2, n_side=3, n_length=4.

**IProfileSurfaceMesher** — 3 faces: top flange inner face (z=z_top_in), web left face (y=−tw/2), bottom flange inner face (z=z_bot_in). Web corner nodes are shared with inner-flange nodes via `_flange_ys` → connected K matrix (web-to-flange heat conduction). Web at y=−tw/2, not y=0, to avoid zero normal in `_quad_outward_normal_local`. Defaults: n_top=4, n_side=2, n_bottom=2, n_length=2.

**PipeSurfaceMesher** — Unrolled cylindrical surface → `local_coords_2d`. Defaults: c_circ=8, n_length=4.

**PlateSurfaceMesher** — Bilinear grid from 4 corner positions → `local_coords_2d`. Defaults: mesh_12=4, mesh_14=2.

---

## FEM 2-D Matrices — fem_2d_section.py (legacy cross-section, still used by BoxMesher tests)

- `quad4_conductivity_matrix(coords, k)` — 2×2 Gauss, (4,4)
- `quad4_capacity_matrix(coords, rho, cp, lumped=True)` — row-sum lumped: (4,)
- `assemble_K(mesh, k) → csr_matrix`
- `assemble_C_lumped(mesh, rho, cp) → ndarray`
- `add_robin_bc(A_lil, b, edge_pairs, nodes, T_fire, T_prev, epsilon_m, h_conv)`
  - Convection semi-implicit (added to A); radiation explicit (added to b)
  - `_SIGMA = 5.67e-8` W/(m²·K⁴)

---

## Steel Properties — EN 1993-1-2 Annex C

`core/model/material.py` — piecewise-linear tables for k(T) and cp(T) over 20–1200°C.
Critical point: cp peaks at 735°C (phase transformation).

Fire curves:
- ISO 834: `T = 20 + 345·log₁₀(8t+1)` [t in minutes]
- Hydrocarbon: `T = 20 + 1080·(1−0.325·e^{−0.167t}−0.675·e^{−2.5t})`

Heat flux (EN 1993-1-2 §3.1):
```
q_net = ε_m·σ·(T_fire_K⁴ − T_steel_K⁴) + h_conv·(T_fire − T_steel)
```

---

## Full theory reference

`docs/3D_FEM_heat_transfer_theory.txt` and `FAHTS_theory/Fahts_Theory_Manual.pdf` Ch. 3.
