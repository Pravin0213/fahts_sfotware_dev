# FAHTS Theory Manual Alignment Todo

This checklist tracks work needed to align the Python solver with `FAHTS_theory/Fahts_Theory_Manual.pdf`.

## Priority 1 - Solver Physics

- [x] Add manual-equation regression tests for Eq. 3.2.13, 3.2.15, 3.2.30, §3.2.4, §3.3.4, §3.3.5, and §3.4.2.
- [x] Implement nonlinear iteration per time step so `K_i`, `M_i`, and `Q_i` converge at time `t_i`.
- [x] Add a consistent mass-matrix mode for theory-aligned runs.
- [x] Replace lumped surface boundary loads/stiffness with shape-function-consistent integration.
- [x] Enable heat accumulation elements in the active BOX/PIPE surface-solver path.
- [x] Assemble a global thermal system instead of solving structural elements independently.

## Priority 2 - Element And Boundary Modeling

- [x] Model inside/outside energy exchange exactly for hollow, open, and shell profiles.
- [x] Apply QUADSHEL/TRISHELL exposure to both outsides as specified in §3.4.1.
- [x] Correct FireZone environmental-temperature radiation semantics; gas emissivity is 1.0 for ISO/HC/environmental fires.
- [x] Replace midpoint/binary exposure with heat-transfer-element-level exposure.
- [x] Implement full §3.3.4 view-factor double-area integration.
- [x] Add obstruction/shadow detection for radiative exchange.

## Priority 3 - Missing Manual Features

- [x] Implement insulation model §3.3.3, starting with massless temperature-dependent type 1 insulation.
- [x] Implement prescribed nodal boundary temperature §3.5.2.
- [x] Implement manual concentrated source §3.5.4: `q_i = E cos(theta)/(4*pi*r_i^2)`.
- [x] Implement manual line source §3.5.5.
- [x] Implement KAMELEON/FIREINT-compatible radiation/convection behavior or document it as explicitly unsupported.

## Priority 4 - Output And Modes

- [x] Make BELTEMP linearization consistently use the §3.4.2 equivalent thermal-expansion formulation.
- [x] Make material-property source explicit in analysis configuration and tests.
- [x] Audit heat-flux sign conventions across all source and boundary paths.
- [x] Add a selectable `theory_mode`/`engineering_mode` distinction for strict manual conformance vs faster approximations.

## Fix Log

- 2026-05-23: Selectable theory/engineering analysis mode. Added `analysis_mode: str =
  "engineering"` field to `AnalysisConfig` (valid values: `"engineering"`, `"theory"`).
  `validate()` rejects unknown strings with `ValueError`. In `run_analysis()` a local
  variable `effective_mass_matrix` is derived from the mode before any solver is
  constructed: `"theory"` forces `"consistent"`; `"engineering"` passes
  `config.mass_matrix` unchanged. The config object is never mutated. An `INFO` log
  message is emitted when theory mode is active. All three `SurfaceTransientSolver`
  instantiations (BOX/I/PIPE beams, QUADSHEL shells) and the `Shell1DSolver` (TRISHELL)
  now use `effective_mass_matrix` instead of `config.mass_matrix` directly.
  Both modes use the same CN θ=1/2 integration and physical BC formulas.
  16 new tests in `tests/test_analysis_mode.py` cover: default value, both valid modes
  accepted, invalid mode rejected, per-instance independence, engineering passes
  config.mass_matrix through, theory overrides to consistent regardless of config field,
  config not mutated, and end-to-end runs for both modes.

- 2026-05-23: Heat-flux sign convention audit. Audited all source and boundary paths
  against FAHTS Theory Manual §3.2.4–§3.2.5 and §3.5.1–§3.5.5. The convention is
  uniformly **positive flux = heat INTO the steel** throughout the codebase, which
  matches the FEM load vector Q in M·Ṫ + K·T = Q (Eq. 3.2.14/3.2.18). This is the
  negation of the manual's E_s/E_c surface-energy notation (positive = out of steel).
  No sign bugs were found. Confirmed correct in:
  `net_flux.py` (q_rad/q_conv formulas + docstring already stated convention),
  `view_factor.py` (FireZone BC assembly, cos_i/cos_j visibility angles),
  `surface_solver.py` (q_rad_gp, Q_conv, view-factor path, re-radiation negative),
  `shell_1d_solver.py` (outer/inner node fire BC),
  `fem_2d_section.py` / `time_integrator.py` (legacy Robin BC),
  `rad_ball.py` (positive prescribed irradiance; eps_rerad path handles re-radiation),
  `concentrated_source.py` / `line_source.py` (cos(θ) = −dot(n_hat, r_hat) correctly
  selects faces pointing toward the source for positive flux).
  Added comprehensive module-level sign-convention note to `fahts/core/heat/bc/net_flux.py`
  documenting the convention, its relation to the manual's notation, and its application
  in every BC path. No runtime code was changed; all existing tests continue to pass.

- 2026-05-23: Material-property source made explicit. Added module-level constant
  `MATERIAL_STANDARD = "EN1993-1-2:2005 Annex C"` to `fahts/core/model/material.py` to
  label the EN 1993-1-2:2005 Annex C standard from which k(T), cp(T), and ρ formulas
  are taken. Added `material_standard: str` field to `AnalysisConfig` (default via
  `default_factory` so it always equals the module constant and each instance is
  independent); `analysis_config.py` imports `MATERIAL_STANDARD` directly (not under
  `TYPE_CHECKING`) so the default is always the live constant rather than a copied
  string. Updated `tests/test_material.py` to import `MATERIAL_STANDARD` and added
  `TestMaterialStandard` class (8 tests) verifying the constant value, EC3 formula
  values at 20°C (k=53.334 W/mK, cp≈439.83 J/kgK, ρ=7850 kg/m³). Added
  `TestMaterialStandard` to `tests/test_analysis_config.py` (5 tests) verifying default
  value, match with module constant, override, instance independence, and type.

- 2026-05-23: §3.4.2 BELTEMP linearization now consistently uses the equivalent
  thermal-expansion formulation for both mesh types. `TemperatureField.section_gradient()`
  now dispatches on mesh type: for the legacy `SectionMesh` (2-D cross-section, nodes
  [y, z]) the existing `_section_gradient_cross` path is unchanged; for the active
  `BeamSurfaceMesh` (3-D axial × hoop, nodes [x, y, z]) the new
  `_section_gradient_surface` path extracts y from column 1 and z from column 2 and
  computes A_k using the full 3-D cross-product area (equivalent to integrating ΔT·y
  and ΔT·z over the full beam surface per §3.4.2 Eq. 3-32). Previously passing a
  `BeamSurfaceMesh` to `section_gradient()` silently used column 0 (axial x) as y and
  column 1 (cross-section y) as z — producing wrong gradients. The `export_beltemp()`
  docstring was updated to state it accepts either mesh type. 7 new tests added: 3 in
  `test_theory_manual_equations.py` (pure §3.4.2 gradient recovery for `BeamSurfaceMesh`,
  uniform-T zero-gradient, and dispatch routing) and 7 in `test_beltemp_linearization.py`
  (zero-gradient without mesh, mean-temperature round-trip, linear T(y)/T(z)/T(y,z)
  gradient export, incremental accumulation, missing-mesh zero-fill, 3-D area weighting).

- 2026-05-23: §3.5.6 KAMELEON/FIREINT interface documented as explicitly unsupported.
  Added a comprehensive module-level docstring to `fahts/core/heat/sources/fire_zone.py`
  explaining: (1) the three components required — KAMELEON/FIREINT CFD field files
  (spatial grid of gas T, velocity, and absorption coefficients), Discrete Transfer
  Method (DTM) ray-casting radiation (Shah & Lockwood, nθ=3·nAccur/nΦ=4·nAccur), and
  local Nusselt-number convection (h_c = N_u·K_l/L, Pr=0.707, L=0.2 m); (2) why these
  are not implemented (proprietary 1990s SINTEF codes, non-trivial DTM solver, CFD
  velocity data not available); (3) which existing source types to use instead, and how
  a future `KameleonFireSource` class should be structured if CFD coupling is needed.
  No runtime code was changed; all existing tests continue to pass.

- 2026-05-23: §3.5.5 time-dependent line source. Added `LineSource` dataclass to
  `fahts/core/heat/sources/line_source.py`: fields `name`, `start` (3-array), `end`
  (3-array), `power` (float or callable(t)), `active`. The line is divided into
  `n_segments` discrete sub-sources (default 10) placed uniformly from start to end;
  end sub-sources emit 50% of the interior energy per the FAHTS manual. `flux_at(point,
  normal, t, n_segments)` sums §3.5.4 contributions from each sub-source (cos(θ) clamped
  to 0 for back-facing quads); `per_quad_flux` is the vectorised batch form. Added
  `line_sources: list[LineSource]` to `AnalysisConfig` (TYPE_CHECKING import). Wired
  through `analysis_runner.run_analysis`: `active_lsrcs` extracted from config; included
  in `_exposed_beam_ids`/`_exposed_shell_ids` (all elements conservatively); `_bc_for_element`
  returns ambient BC for elements covered only by line sources; `_line_source_per_quad_flux`
  helper accumulates per-quad flux; flux added to `q_per_quad` after ConcentratedSource
  accumulation (skipped when RadiationBall prescribes uniform flux); QUADSHEL shells also
  receive LineSource per-quad flux. 22 new tests in `tests/test_line_source.py`.

- 2026-05-23: §3.5.4 concentrated point source. Added `ConcentratedSource` dataclass to
  `fahts/core/heat/sources/concentrated_source.py`: fields `name`, `center` (3-array),
  `power` (float or callable(t)), `active`. `flux_at(point, normal, t)` computes
  `q = E(t)·cos(θ)/(4π·r²)` (clamped to 0 for back-facing quads); `per_quad_flux`
  vectorised form. Added `concentrated_sources: list[ConcentratedSource]` to `AnalysisConfig`.
  `analysis_runner.run_analysis` collects `active_csrcs`, calls
  `_concentrated_source_per_quad_flux` to accumulate per-quad flux, and adds it to
  `q_per_quad` for `SurfaceTransientSolver`. 14 new tests in `tests/test_concentrated_source.py`.

- 2026-05-23: §3.5.2 prescribed nodal boundary temperature. Added `PrescribedNodeBC`
  dataclass to `fahts/core/heat/bc/prescribed_node_bc.py`: holds `node_indices: list[int]`
  and `temperature: float | Callable[[float], float]` with `eval(t)` helper. Added
  `prescribed_node_bcs: list[PrescribedNodeBC]` field (default `[]`) to `AnalysisConfig`.
  Modified `SurfaceTransientSolver.__init__` to accept the list, validate indices against
  mesh size, and zero prescribed-DOF rates in `_init_rate`. Added `_apply_dirichlet`
  method: converts A to LIL, zeros row i, sets A[i,i]=1, sets B[i]=T_presc−T_prev[i]
  (correct increment form for the ΔT CN system), returns modified (A_csr, B). In `step`,
  calls `_apply_dirichlet` before each Picard solve, then clamps T_new[i]=T_presc and
  zeroes T_dot[i] so the predictor does not perturb prescribed DOFs at the next step.
  `analysis_runner.run_analysis` forwards `config.prescribed_node_bcs` to both beam and
  QUADSHEL shell solvers. 17 new tests in `tests/test_prescribed_node_bc.py`.

- 2026-05-23: Ray-based obstruction/shadow detection for radiative exchange. Added
  `ray_segment_intersects_beam(ray_origin, ray_target, beam_start, beam_end, radius) → bool`
  to `view_factor.py`: implements closest-point-between-two-line-segments capsule test
  (segment-vs-segment formulation), with an epsilon guard at s=0 and s=1 to prevent
  self-shadowing at the quad surface or fire-patch target. Added
  `compute_shadow_mask(quad_centroids, fire_patch_centroids, beam_starts, beam_ends,
  beam_radii) → (n_quads, n_patches) bool` to `view_factor.py`: iterates all
  (quad, patch, beam) triples, marks pairs False where any beam capsule blocks the ray.
  Extended `geometric_view_factor_double_area` with an optional `shadow_beams` parameter
  (list of (start, end, radius) tuples, default None): when provided, each (steel
  sub-patch centroid → fire-patch centroid) ray is tested against all beam capsules before
  contributing to the view-factor sum; blocked pairs contribute zero. Passing
  `shadow_beams=None` or `shadow_beams=[]` preserves the original behaviour exactly
  (backward compatible). 8 new tests in `tests/test_shadow_detection.py`.

- 2026-05-22: §3.3.4 full double-area view-factor integration. Added `geometric_view_factor_double_area(quad_corners, quad_normal, zone_patches, n_steel_sub=2)` to `view_factor.py`: subdivides the steel quad into n_steel_sub² sub-patches via bilinear interpolation and applies `F_12 = (1/A1)·ΣΣ cosθi·cosθj/(π·r²)·Ai·Aj`. The old `geometric_view_factor` (centroid point form) is retained for tests and backward compatibility. `_element_view_factors` in `analysis_runner.py` now passes all 4 quad corner nodes (global) instead of a single centroid, and calls the double-area form with n_steel_sub=2. 5 new tests in `test_theory_manual_equations.py` verify: n_steel_sub=1 equals simplified form, zero for back-facing quads, convergence with finer subdivision, and equivalence with area-weighted mean of per-sub-patch simplified VFs.

- 2026-05-22: Heat-transfer-element-level exposure for FireZone sources. `exposure_flags()` and `exposed_element_ids()` in `view_factor.py` now check the beam's n1, n2, and midpoint (instead of midpoint only) so that elements straddling a zone boundary are included. Added `element_quad_exposure_flags(mesh, elem, nodes, fire_zones)` to compute per-quad 0/1 flags from quad centroid global positions. `SurfaceTransientSolver` gained a `quad_exposure` parameter: convective stiffness, convection load, radiation, and prescribed-flux BC terms are each scaled per-quad by this array; `q_per_quad` (RadiationBall directional) and re-radiation are unaffected. `analysis_runner.run_analysis` passes endpoint positions to `_bc_for_element` for zone discovery, builds the exposure flag array after meshing, and passes it to the solver (None when all quads are 1.0 to preserve existing behaviour). 15 new tests in `tests/test_element_level_exposure.py`.


- 2026-05-22: FireZone environmental-fire gas emissivity. Added `effective_epsilon_fire` property to `FireZone`: returns 1.0 for `ISO_834` and `HYDROCARBON` curve types per EN 1991-1-2 §3.3.2 (optically thick flames); returns user-supplied `epsilon_fire` for `USER_DEFINED` only. Updated `analysis_runner._bc_for_element` and `view_factor.compute_element_bc` to use `effective_epsilon_fire` when computing `epsilon_m`. 5 new tests in `TestEffectiveEpsilonFire` in `test_fire_zone.py`.

- 2026-05-22: §3.4.1 double-sided shell exposure. Added `n_exposed_sides: int = 1` to `SurfaceTransientSolver`: when set to 2 (QUADSHEL), convection stiffness, convection load, radiation, and uniform prescribed-flux BC terms are scaled by 2; directional per-quad RadiationBall falloff flux is intentionally not scaled. Added `fire_temp_inner: Callable | None = None` to `Shell1DSolver`: when set (TRISHELL), inner node receives the same convection + radiation fire BC as the outer node. `analysis_runner.py` now passes `n_exposed_sides=2` for QUADSHEL (unless using directional q_per_quad) and `fire_temp_inner=fire_temp` for TRISHELL. `BeamSurfaceMesh` gained two optional fields `outer_face_indices` and `inner_node_indices` (default empty, backward-compatible). 11 new tests in `tests/test_inside_outside_exchange.py` verify double-sided heating rate (≈2× at small dt), inner-node heating for TRISHELL, and BOX default remaining single-sided.

- 2026-05-22: Added `GlobalThermalSolver` to `analysis_runner.py`. Assembles all element K_e, M_e, Q_e blocks into a single block-diagonal sparse system (one `spsolve` per Picard iterate per time step) instead of N independent element solves. Per-element convergence checks ensure exact numerical equivalence with independent solving. `run_analysis()` now uses `GlobalThermalSolver.step()` in its time loop; element-local temperatures are scattered back from the global vector after each step. 7 new tests in `tests/test_global_thermal_solver.py` verify DOF layout, block-diagonal equivalence (rtol=1e-6), and physical monotonicity.

- 2026-05-22: Replaced lumped A_e/4 surface BC with shape-function-consistent integration in `SurfaceTransientSolver`. K_conv is now the full 4×4 matrix h·∫∫NᵢNⱼdA (combined into K_e in one COO pass); Q_rad uses 2×2 Gauss-point integration of ε·σ·(T_fire_K⁴−T(ξ,η)_K⁴); prescribed/re-radiation loads use consistent row sums. All 730 core tests pass.

- 2026-05-20: Added hollow-profile heat accumulation to `SurfaceTransientSolver` and `run_analysis`. Current surface meshes have no explicit inner nodes, so the §3.3.5 enclosed-fluid capacitance is distributed over surface thermal DOFs by tributary area until explicit inside/outside DOFs are implemented.
- 2026-05-21: Added bounded Picard iteration to `TransientSolver`, `SurfaceTransientSolver`, and `Shell1DSolver`. Current-step matrices and loads are now reassembled from the latest temperature iterate and stored at the accepted `T_i`, with regression tests covering all three transient solver paths.
- 2026-05-21: Added focused manual-equation regression tests for Eq. 3.2.13, 3.2.15, 3.2.30, §3.2.4, §3.3.4, §3.3.5, and §3.4.2.
- 2026-05-21: Added `mass_matrix="consistent"` as an opt-in analysis mode for 2-D section, beam-surface, and shell solvers while keeping lumped mass as the default engineering mode. Hollow-air heat accumulation remains diagonal per §3.3.5.
