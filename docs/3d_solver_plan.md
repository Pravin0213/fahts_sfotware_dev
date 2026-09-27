# 3-D Solid Heat Solver — Implementation Plan (2026-09-27)

## Goal & context
Project pivot: simulate heat transfer into **process equipment** (vessels, piping, supports)
that will later contain hydrocarbons / other contents. Step one: a true **3-D solid** transient
conduction solver. The old 2-D "surface" solver (`SurfaceTransientSolver`, wall thickness as a
scalar, no through-thickness gradient) stays in the code as an alternative/cross-check, but the
3-D solver becomes the default path.

Dropped constraints: BELTEMP mean/gradient output, SINTEF FAHTS conformance, the USFOS benchmark,
the "DO NOT CHANGE" table in CLAUDE.md. Do not spend effort on them. Don't break them gratuitously
either — leave 2-D code paths working.

Reference studied: `3DHeatTransfer-main/` (C++ voxel FD, 7-point stencil, implicit Euler,
Jacobi/CG, analytical sin·sin·sin test). We keep its *ideas* (sparse CSR assembly, CG iterative
solve for large systems, analytical verification cases) but use **Hex8 FEM** instead of voxels —
thin steel walls and curved pipes/vessels need body-fitted elements.

## Physics
    ρ c(T) ∂T/∂t = ∇·(k(T) ∇T)                  in Ω (steel volume)
    −k ∂T/∂n = h (T − T_gas) + ε σ (T⁴ − T_r⁴) − q_src   on Γ_outer (fire side)
    −k ∂T/∂n = 0                                  on Γ_inner (cavity/wetted — contents hook), Γ_end
Galerkin Hex8, 2×2×2 Gauss; boundary integrals on Quad4 faces, 2×2 Gauss.
Semi-discrete: M(T) Ṫ + K(T) T = Q(T). Time: Crank–Nicolson (θ = ½) incremental form (same as
`GlobalThermalSolver` / surface solver) with bounded Picard iteration for nonlinearity.
k(T), c(T): evaluated **per hex** at the element mean temperature (improvement over the 2-D
solver's single member-mean value).

## Shared contract — DO NOT change without telling the lead
`fahts/core/heat/solid_mesh/solid_mesh.py` → `SolidMesh(nodes, hexes, faces, face_group, section_kind)`,
constants `FACE_OUTER / FACE_INNER / FACE_END`. Hex8 = VTK ordering, det J > 0. Boundary faces CCW
seen from outside (outward normal = (p2−p0)×(p3−p1)). Beam members in beam-local coords
(x axial 0..L, y width, z height, same as `BeamSurfaceMesh`); plates in global coords.

## Work packages

### WP-A  Solid meshers  (`fahts/core/heat/solid_mesh/`)
- `extrude.py`: `extrude_section(section_mesh: SectionMesh, length, n_length) -> SolidMesh`
  — take a legacy 2-D cross-section mesh (y,z) and sweep along x. Cross-section quad (CCW in y-z)
  × axial segment → Hex8 with positive Jacobian. `outer_edge_pairs` × segments → FACE_OUTER,
  `inner_edge_pairs` → FACE_INNER, the two cross-section end caps → FACE_END. Orient every face
  outward (verify with centroid test / geometry, not assumptions).
- `box_solid_mesher.py` (`BoxSolidMesher(section, length, n_top, n_side, n_length, n_layers)`),
  `iprofile_solid_mesher.py`, `pipe_solid_mesher.py` (`c_circ`, `n_layers` radial),
  `plate_solid_mesher.py` (`PlateSolidMesher(section, corners(4,3), mesh_12, mesh_14, n_layers)`,
  mid-surface ± t/2, both big faces FACE_OUTER, edges FACE_END), and `block_solid_mesher.py`
  (`BlockSolidMesher(lx, ly, lz, nx, ny, nz)` — rectangular solid, all faces FACE_OUTER; used for
  analytical validation). Reuse legacy `BoxMesher`/`IProfileMesher`/`PipeMesher` where they give
  a good cross-section; otherwise generate directly. Respect hoop/axial counts from config
  (n_top/n_side/n_length etc.) and `n_layers` through thickness. Watch out for the hoop-count
  semantics: BoxMesher uses elem_size, so you may need your own structured ring generator.
- Tests `tests/test_solid_meshers.py`: volumes equal analytic steel volume (box ring, I-profile,
  pipe annulus — pipe converges with c_circ), outer area equals analytic outer area, all det J>0,
  all faces outward (divergence theorem: Σ n·centroid·A = 3V), no duplicate nodes, watertight
  boundary (every boundary edge shared by exactly two boundary faces).

### WP-B  Hex8 FEM kernel + solver  (`fahts/core/heat/solver/`)
- `fem_3d.py`: vectorised, batch-over-elements routines:
  `hex8_shape(ξ,η,ζ)`, `hex8_conductivity_base(X (n,8,3)) -> (n,8,8)` (k=1),
  `hex8_capacity_base(X, lumped)`, face routines `quad3d_face_mass_base(P (n,4,3)) -> (n,4,4)`,
  `quad3d_face_gauss(P) -> N_gp (4,4), detJ (n,4)`.
- `solid_solver.py`: `SolidTransientSolver` — same public surface as `SurfaceTransientSolver`
  (`_assemble_step(T, t) -> (K csr, M (vector or csr), Q)`, `_init_rate`, `step`, `run`) so
  `GlobalThermalSolver` can drive it. Constructor mirrors the surface solver but per boundary
  FACE instead of per quad: `mesh: SolidMesh, material, fire_temp, epsilon_m, h_conv, T0,
  q_prescribed_fn, epsilon_steel, epsilon_fire, view_factors (n_outer_faces,),
  q_per_face (n_outer_faces,), face_exposure (n_outer_faces,), M_extra (n_nodes,), mass_matrix,
  nonlinear_max_iter, nonlinear_tol, insulation, prescribed_node_bcs, inner_bc`.
  Fire BC only on FACE_OUTER. `inner_bc`: optional object with `h(t)`/`T_fluid(t)` (Robin) —
  default None = adiabatic. Define a tiny `InnerRobinBC` dataclass for it (contents hook).
  Re-radiation for prescribed-flux mode as in the surface solver. No `n_exposed_sides` /
  `double_sided` hacks — a solid has real faces on both sides with true outward normals.
  Per-hex k(T), c(T) at element mean temperature.
- Tests `tests/test_solid_solver.py` on `BlockSolidMesher`-style meshes (build tiny ones inline
  if WP-A not ready): patch test (linear T field → zero residual K·T with matching flux),
  constant-property energy balance, uniform heating lumped limit, symmetric K, M row-sum = ρcV.

### WP-C  Integration
- `AnalysisConfig`: `solver_dim: str = "3d"` ("2d"|"3d"), `linear_solver: str = "direct"`
  ("direct"|"cg"), validate(); `n_layers` already exists (through-thickness layers, default 2
  for 3-D is fine — keep field default 1 for back-compat and use `max(n_layers, 1)`).
- `analysis_runner.run_analysis`: when `solver_dim == "3d"` build `SolidMesh` per member and
  `SolidTransientSolver`. Per-face flux/exposure/view-factor helpers must work from
  `SolidMesh.face_centroids()/face_normals()` of FACE_OUTER faces transformed to global coords
  (beam local→global rotation already exists: `_beam_local_to_global`). Heat accumulation
  (air in hollow BOX/PIPE) on inner nodes via `M_extra`. TRISHELL → keep 1-D fallback.
- `GlobalThermalSolver`: stop densifying element matrices (`toarray()`) — scatter COO triplets
  via gdof map (solid members have hundreds–thousands of nodes). Add optional CG solve
  (scipy `cg` with Jacobi or ILU preconditioner, warm-started from T_dot·dt) when
  `linear_solver == "cg"`; direct `spsolve`/`splu` otherwise. Node merging (1 mm) unchanged —
  it works on any node set.
- `TemperatureField.from_solid_solver_run(eid, times, T_history, mesh, nodes_global)`:
  T_centroid = volume-weighted mean; `T_section` = all solid nodes; `nodal_geometry` =
  (nodes_global, boundary faces) so the renderer draws the true 3-D surface.
  Make other consumers (post_processor, results panel, section_gradient) not crash on solid
  meshes (graceful fallback is fine).
- GUI: Run Analysis dialog — 2-D/3-D selector + layers-through-thickness spinbox.

### WP-D  Validation  (`tests/test_solid_validation.py` + `validation/validate_3d.py` report)
1. 3-D sin·sin·sin decay in a unit cube, Dirichlet T=0 on all faces (prescribed nodes):
   T = sin πx sin πy sin πz · exp(−3π²αt) — spatial convergence ≈ 2nd order.
2. Semi-infinite slab, sudden surface temperature (Dirichlet) or constant flux —
   erfc solution; check early-time profile through the thickness.
3. Steady 1-D wall with convection both sides (or fixed T both sides) — linear profile exact.
4. Steady radial conduction through a pipe wall (T_in, T_out prescribed) — ln(r) profile;
   checks curved PIPE mesh.
5. Lumped-capacitance / thin-wall limit: thin, highly conductive BOX under convection+radiation
   ≈ 2-D surface solver and EN 1993-1-2 lumped formula.
6. Energy conservation: ∫ρc ΔT dV equals ∫∫ q_net dA dt within tolerance.
7. Time-convergence of CN (≈ 2nd order in dt).
8. End-to-end: `model_file.fem` / `model_t1.fem` run in 3-D with a FireZone; finite temps,
   monotone heating, 3-D vs 2-D member means agree within engineering tolerance for thin walls;
   through-thickness gradient present for thick walls.
Report: `validation/validate_3d.py` writes `validation/report_3d.md` with tables/plots (PNG).

## Test baseline (2026-09-27, before 3-D work)
Non-GUI: 694 passed, **8 pre-existing failures** (test_beltemp_linearization,
test_insulation::very_high_conductivity, test_prescribed_node_bc ×2, test_shadow_detection ×3,
test_theory_manual_equations::section_3_4_2). 14 GUI/VTK test files abort headless (no display)
— run non-GUI tests with the ignore list in the lead's scratchpad or:
`grep -lE "pyvista|scene_manager|pyvistaqt|QApplication|qtbot|fahts.gui|fahts.renderer" tests/*.py`.
New work must not add failures beyond these 8.

## Rules for agents
- Python 3.11, type hints, dataclasses, `logging` not print, ≤100 char lines, pure physics fns.
- Don't modify `legacy/` or `3DHeatTransfer-main/`. Don't commit to git.
- Stay inside your WP's files; if you must touch another WP's file, keep it minimal and report it.

## Status (2026-09-27, end of day)
All WPs done. Validation report: all cases PASS (`python -m validation.validate_3d`).
Fixed after validation: CN history terms (all solvers), consistent-mass Ṫ0 with Dirichlet,
GlobalThermalSolver prescribed nodes, time-varying point/line source power, joint surface ties
(`joint_ties.py`). Non-GUI tests: 917 passed, 6 pre-existing failures (beltemp_linearization,
insulation very_high_conductivity, shadow_detection ×3, theory_manual section_3_4_2).
Timing (600 s, 20 steps, HC fire zone): model_file.fem 517 members / 64k DOFs: direct 82 s,
CG 23 s; model_t1.fem 202 members / 27k DOFs: direct 21 s, CG 10 s.
Stale .pyc files (compiled in another checkout) were untracked from git (`git rm --cached`);
`.gitignore` already covers `__pycache__/`. Default linear solver is now "cg".

## Performance + OpenFOAM verification (2026-09-27, later)
- Speed-ups: PCG instead of SuperLU, batched sparse-operator assembly, numba PCG for N ≥ 20k,
  Newton-linearised radiation. Results unchanged (OpenFOAM report identical; validate_3d all 30 PASS).
- Code-to-code vs OpenFOAM v2412 solidFoam: `python -m validation.openfoam.compare_openfoam`
  → validation/openfoam/report_openfoam.md. 4/6 PASS; I-beam converging (104→56→28 K);
  BOX corner max difference (≈6 K at t=5 min, 0.56 % of rise) grows with refinement — open.
- Found: OpenFOAM's externalWallHeatFluxTemperature (mixed) is converted to enthalpy with
  h(Ta) → wall flux too high by c̄p/cp (+17–22 % for EN 1993 steel); comparison uses a codedMixed
  flux-form BC instead.
- 2-D check of the CN history fix on model_file.fem: Δt=30 s gives 944 °C (fixed) vs 987 °C (old);
  both converge to ≈941 °C at Δt=5 s → the old formula overpredicted by ~46 K.

## Monotone discretisation fix (2026-09-27, evening)
User saw blue (cold) spots next to a 350 kW/m² RadiationBall on model_t1.fem. Reproduced:
−69 °C / +1577 °C (equilibrium 1450 °C). Not the time scheme (CN/BE, Δt 30/5 s identical) —
spatial: non-M-matrix consistent K on thin elongated hexes + consistent boundary mass +
interpolated ties + too few axial elements + re-radiation to 0 K. Fixed with conduction_3d=
"monotone", lumped boundary terms, axial_aspect_3d=2, orthogonal BOX ring, star ties, rerad to
ambient. Now bounded 20.3–1451.5 °C. OpenFOAM: 4/6 PASS (BOX fixed); I-beam and hot-spot plate
still converging (3.9 % / 2.1 % of rise, halving per refinement).

## Radiation shielding + member-to-member exchange (2026-09-27, night)
New package fahts/core/heat/radiation/ (numba ray caster, source shielding, Monte Carlo
surface-to-surface exchange). parallel_plates.fem with a ball below: far plate 134 °C → 20 °C
with shielding, +2–25 K from exchange. model_t1.fem ball case: mean member T 510 → 339 °C
(shielding) → 377 °C (+exchange), bounded 20–1449 °C; set-up 1 s (shielding) / 8 s (exchange,
49k patches, 12.6M rays). FireZone + exchange bounded after exposure weighting.
Only 3-D solid members take part (2-D surface / 1-D shell members are not in the scene).
