# Solver Domain — FAHTS

## Active Solver Architecture

**Primary path (all beam sections):** `SurfaceTransientSolver` in `surface_solver.py`
**Fallback (TRISHELL only):** `Shell1DSolver` in `shell_1d_solver.py`
**Legacy (kept, not primary):** `TransientSolver` in `time_integrator.py`

### SurfaceTransientSolver
- Crank-Nicolson θ=1/2 (current-iterate history terms — see CN section)
- Accepts a `BeamSurfaceMesh`; calls `element_coords_2d()` for all geometry
- Heat accumulation element: lumped mass added at inner nodes for BOX/PIPE

### 3-D solid path (default, `config.solver_dim="3d"`, 2026-09-27)
- BOX/IHPROFIL/PIPE → `*SolidMesher(n_layers=config.n_layers_3d)` (PIPE uses
  `config.c_circ_3d = max(c_circ, 12)`); QUADSHEL → `PlateSolidMesher`; TRISHELL → 1-D fallback.
- Solver: `SolidTransientSolver` (`solid_solver.py`). Per-face BC data (exposure, view
  factors, ball/point/line flux, enclosed-gas `M_extra`) built in `solid_integration.py` from
  FACE_OUTER faces in global coords (origin = n1 + ecc1). No `double_sided` hacks.
- DOF map: `solid_integration.build_solid_dof_map` (tol = min(1 mm, 0.2·h_min), never unions
  two nodes of one member) + `build_joint_links` → `joint_ties.build_joint_ties`: every free
  beam-end FACE_END node / plate edge node is tied to the CLOSEST POINT on the partner member's
  surface (triangle barycentric interpolation), G = 45·A_trib/max(d, h_min), K += G·w·wᵀ with
  w = e_p − Σλ_k e_k → symmetric, zero row sums, PSD. Through-ends (already merged collinear
  continuations) don't tie unless the joint has no free end. Couples T/K/X joints (brace end
  inside a hollow chord), plate edges along beams, and angled plates with odd n_layers.
- Prescribed nodes: `GlobalThermalSolver` maps each member solver's `_prescribed_bcs` to global
  DOFs and solves only the free block (symmetric elimination, CG-safe); pinned Ṫ = 0.
- Point/line sources: `bc/face_flux.TimeVaryingFaceFlux` = static + Σ E_s(t)·unit-power pattern;
  both solvers accept it as `q_per_face`/`q_per_quad` and re-evaluate every assembly. Members
  are kept when any face CAN be lit (E(0)=0 ramps no longer skipped).
- `GlobalThermalSolver(linear_solver="cg"|"direct")` — performance design (2026-09-27):
  * batchable solid members (lumped mass, no insulation) assembled together by
    `batch_assembly.SolidBatch`: K_data = S_K·params, M = S_M·ρc, Q = S_Q·loads with precomputed
    sparse operators (params = k per hex, h per face, radiation tangent per face Gauss point,
    h_in); other members (2-D, 1-D, insulated, consistent mass) assembled one by one into the
    same global CSR pattern (built once). Single-threaded (thread pool measured slower — GIL).
  * `linear_solve.SPDSolver`: Jacobi-PCG, warm-started, rtol 1e-8 in the global solver
    (1e-10 in standalone `SolidTransientSolver`), direct fallback. numba parallel PCG kernel
    when numba is importable and N ≥ 20 000 (else scipy CG). Dirichlet by symmetric elimination.
  * Radiation −εσT⁴ is Newton-linearised (h_r = 4εσT³ into K) in the 3-D solver.
  * model_file.fem (517 members, 64k DOFs, 20 steps): stepping 17 s → ~5.5 s; single 8.4k-node
    member 60 steps: 28 s → ~1 s. Batch ≡ per-member to round-off (tests/test_batch_assembly.py).
  `solver_dim="2d"` keeps the legacy surface path (per-member assembly).

### Monotone 3-D discretisation (default since 2026-09-27)
Found on model_t1.fem with a 350 kW/m² RadiationBall: nodes at −69 °C / +1577 °C
(radiative equilibrium is 1450 °C). Causes and fixes (config defaults):
- `conduction_3d="monotone"`: two-point edge stencil `fem_3d.hex8_conductivity_twopoint_base`
  (M-matrix; metric det J·J⁻ᵀJ⁻¹ at the hex centre). The consistent trilinear K has POSITIVE
  in-plane couplings on thin walls (c ≪ a, b) → over/undershoot at sharp lit/shadow edges.
  `"consistent"` still available (verification tests use it).
- Monotone mode lumps boundary terms (nodal quadrature for convection/radiation/inner Robin).
- `axial_aspect_3d=2.0`: axial elements ≤ 2× cross-section element size (`axial_divisions`);
  configured n_length* are minima. Coarse axial meshes on long members smeared the ball load.
- BOX solid mesh is an orthogonal tensor ring (rectangular corner blocks) — the old diagonal-
  trapezoid corners were skewed and made the two-point stencil inconsistent there.
- Joint ties are star links p–v_k with G·λ_k (graph Laplacian, no positive couplings).
- Re-radiation (prescribed-flux mode) is to the ambient/initial temperature, not 0 K.
Result: ball case bounded 20.3–1451.5 °C; OpenFOAM comparison BOX now PASS (0.19 %).
Tests: tests/test_monotone_3d.py.

### Radiation geometry — shielding + member-to-member exchange (3-D, 2026-09-27)
Package `fahts/core/heat/radiation/`; set up in `analysis_runner._setup_radiation_geometry`
after the DOF map. Config: `shielding=True`, `radiation_exchange=True`, `rad_patch_size=0.5`,
`rad_rays_per_patch=256` (Run Analysis dialog check boxes).
- `raycast.TriangleScene`: numba uniform-grid ray caster (Möller–Trumbore, double-sided);
  verified against brute force; ~10⁷ rays/s.
- `shielding.build_scene`: all members' boundary faces → triangles (owner = global face id).
  NOTE the beam frame (x, y = x×z, z) is LEFT-handed → local→global mirrors; scene normals are
  flipped when the global hex volume is negative (bug found: normals pointed inward).
- Shielding (`apply_source_shielding`, in place on member solvers before GlobalThermalSolver):
  RadiationBall → visibility = unblocked fraction of 19 rays to the ball's visible FRONT CAP
  (steel inside the ball doesn't block it), only cap points above the face's tangent plane;
  engulfed faces keep full flux. Point source: 1 ray. LineSource: per sub-source rays.
- Exchange (`exchange.RadiationExchange`): patches (member × normal bin × spatial bin), Monte
  Carlo cosine-weighted view factors (obstruction included), A·F symmetrised, rows ≤ 1.
  q_i = exp_i Σ_j F_ij [ε_i J_j − a_i G_i], J_j = ε_j σT_j⁴ + (1−ε_j)G_j; background G/a:
  ambient (σT_amb⁴, ε), engulfed in ball (flux, 1), FireZone (σT_f⁴ or σFε_fT_f⁴, ε).
  Explicit load added in `GlobalThermalSolver._assemble_global`. Isothermal ambient → 0.
- Tests: tests/test_radiation_geometry.py (analytic parallel-plate F within 3 %, shielding of
  parallel_plates.fem, fire-zone boundedness).

### Free edges / free ends and unlit elements (3-D, 2026-09-27)
- `_model_topology(model)` → node and edge use counts (beams count as an edge).
- Plate edges used by only one shell (and no beam) → `PlateSolidMesher(exposed_edges=…)`
  makes their thickness faces FACE_OUTER; beam end caps at nodes used by only that member
  → FACE_OUTER (`_expose_free_beam_ends`). Connected edges/ends stay FACE_END (adiabatic).
- 3-D no longer skips elements whose faces get zero DIRECT ball / point / line flux: they
  are heated by conduction and radiation exchange (found with an 80 cm plate level with a
  ball: whole plate skipped). Tests: tests/test_free_edges.py.

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

## Crank-Nicolson Theory

**Changed 2026-09-27:** the history terms use the CURRENT Picard iterate's K_i, M_i (not
K_{i-1}, M_{i-1} as in SINTEF Eq. 3.2.30). The lagged form left an O(Δt) error whenever
k(T)/c(T) vary (validation case 7: order 1.07 → 1.95 after the fix). Applied in all solvers.
With prescribed nodes, Ṫ0 is solved on the free block only (consistent-mass correctness).

```
A = Ki + (2/Δt)·Mi
B = Qi - Ki·T_{i-1} + Mi·Ṫ_{i-1}
ΔTi = A^{-1}·B
Ti   = T_{i-1} + ΔTi
Ṫi   = (2/Δt)·ΔTi - Ṫ_{i-1}
```
Initial rate: `Ṫ0 = M0^{-1}·(Q0 - K0·T0)`
Between steps only `T_dot_prev` is needed (`K_prev`/`M_prev` just mark the state as initialised).

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

## Cross-element thermal coupling — RESOLVED

Previously listed here as a known limitation. Now: one global system in `GlobalThermalSolver`;
co-located nodes merged into shared DOFs (2-D: `_build_global_dof_map`, 3-D:
`build_solid_dof_map`), plus 3-D joint surface ties (`joint_ties.py`) for non-coincident
joint geometry. Tests: `tests/test_solid_integration.py::TestJointCoupling`,
`tests/test_open_issue_fixes.py::TestJointTies`.

---

## Full theory reference

`docs/3D_FEM_heat_transfer_theory.txt` and `FAHTS_theory/Fahts_Theory_Manual.pdf` Ch. 3.
