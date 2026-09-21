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

**Directional-source I-beam blind spot (fixed 2026-08-25):** I-beam plates (top
flange/web/bottom flange) are meshed with one fixed normal each but are physically
exposed on both faces (§3.4.1). Directional sources (RadiationBall/ConcentratedSource/
LineSource) used to silently zero out a whole plate whenever the source was on the side
the mesh's stored normal didn't point toward. Fixed via a `double_sided` flag threaded
through `_rad_ball_per_quad_flux`/`_concentrated_source_per_quad_flux`/
`_line_source_per_quad_flux` and the QUADSHEL block: evaluate both the stored normal and
its mirror (per source, then sum across sources — never sum both sides of one source,
since at most one is ever lit for a single point). See `section_mesh/CLAUDE.md`.
Known remaining gap: re-radiation for these plates still only emits from the "stored
normal" side (`rerad_scale` unaffected by `double_sided`) — a genuinely two-sided-exposed
plate should probably re-radiate from both faces too; not yet addressed.

**RadiationBall model (2026-08-24, breaking change):** replaced the old two-zone
`r1/flux1/r2/flux2` piecewise-linear calibration with a single-zone exact point-to-sphere
model — `(radius, flux)`. `q = flux` when engulfed (`d <= radius`, applied to every face, no
cos weighting — this is the exact cavity-immersion limit, not a special case), else
`q = flux * (radius/d)**2 * cos(θ)` (exact "differential area to sphere" configuration
factor). Per-element ball selection (`_bc_for_element` in `analysis_runner.py`) now uses a
single distance cutoff (`_MIN_BALL_FLUX`, direction-agnostic upper bound) instead of the old
covering/falloff-beyond-r2 split — `_bc_falloff_ball` was removed entirely. Multiple
overlapping balls still use "pick the single strongest by midpoint" (not true superposition)
— left as a follow-up per user decision. See `heat/sources/CLAUDE.md` and `rad_ball.py`
module docstring for the full derivation.

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

## ⚠️ KNOWN LIMITATION — No Cross-Element Thermal Coupling (needs fix)

**Status:** confirmed gap, not yet scheduled. See ROADMAP.md 5.11.

**What the real SINTEF FAHTS does** (`docs/3D_FEM_heat_transfer_theory.txt` §3.2.18/§17,
§15.2): builds ONE global system `M·Ṫ + K·T = Q(T)` across the whole structure. Each
beam's 2-D cross-section mesh is generated independently, but nodes from *different*
beam elements that land within a coincidence tolerance of each other (e.g. at a shared
structural joint/node) are **merged into one shared global mesh node**. This couples
their element K/M contributions into a single assembled matrix, so heat genuinely
conducts from one member into its neighbor through shared joints. Confirmed empirically
by the user running the real SINTEF solver: applying heat to one element and refining
the mesh causes heat to visibly flow into adjacent elements through shared nodes.

**What this codebase currently does** (verified in `analysis_runner.py`): each beam
element gets its own independent `BeamSurfaceMesh` and its own `SurfaceTransientSolver`
instance (`analysis_runner.py:461`, `:582`). The main time-stepping loop does
`for eid in all_eids:` and solves each beam's local Crank-Nicolson system in complete
isolation — K, M, Q are sized only to that one beam's own mesh. There is no
node-coincidence/merge logic across separate beam elements anywhere in
`fahts/core/heat/` (only the intra-section wall-corner dedup for a single BOX,
1e-10 m tolerance — that's within one element, not across elements). Beams currently
only "interact" indirectly, via shared fire-zone/radiation-ball boundary conditions
(same environment temperature), never via real conduction through a shared joint.

**Fix needed (rough shape, to be scoped properly before work starts):**
1. After per-element surface meshes are built, find mesh nodes from different beam
   elements that coincide (within a tolerance) at shared structural `Node` positions
   (model joints) — analogous to the existing wall-corner dedup, but *across* elements.
2. Merge coincident nodes into shared global DOFs (union-find / global node numbering
   scheme, or a mapping table from local (eid, local_node_idx) → global_dof).
3. Assemble one global K, M, Q per time step from all element contributions (scatter-add
   into the global sparse matrix at shared DOFs), replacing the current per-element
   `SurfaceTransientSolver` loop with a single global Crank-Nicolson solve.
4. Update `analysis_runner.py` dispatch, `TemperatureField` storage (currently keyed
   per-eid — needs to map global DOF solution back to per-element `T_section` arrays),
   and re-validate against the USFOS benchmark (`usfos_verification_results/`) since
   this changes solved temperatures, not just internal structure.
5. Consider whether this should be opt-in (perf cost of one big sparse solve vs many
   small independent ones, though the current per-element loop is already parallelized
   with a `ThreadPoolExecutor`-style pattern — see `analysis_runner.py:160`).

Do not attempt this as a quick patch — it changes the fundamental solve architecture
(per-element solvers → one global assembled system) and touches mesh generation,
solver dispatch, and results storage. Scope it as its own task.

---

## Full theory reference

`docs/3D_FEM_heat_transfer_theory.txt` and `FAHTS_theory/Fahts_Theory_Manual.pdf` Ch. 3.
