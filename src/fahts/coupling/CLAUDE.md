# Process model (thermo / fire / wall.fem_3d / relief / process / rupture / coupling)

Ported 2026-09-30 from `Test/vfpy` (frozen reference: `legacy/vfpy/`, see its README for the
file → package map). The port was verified bit for bit: every package against legacy
(`tests/unit/*/test_equivalence_legacy.py`) and the whole model against 18 golden runs. Since
then deliberate physics fixes change results; the goldens track `fahts`
(`python -m pytest tests/regression --golden -q`) and record why they changed.
Compare against VessFire with `python -m validation.vessfire.compare_cases <study>`.

## Layers (imports only go down)

| Package | Content | May import |
|---|---|---|
| `common` | constants, compat | numpy |
| `thermo` | PR EOS, flashes (PT/UV/PH/PS), free water, pseudo-components, saturation, transport | common |
| `materials` | `SteelTable` (k, cp, rho, strength factors vs T), EN 1993-1-2 E(T), thermal strain | common |
| `fire` | outer-surface exposure: `GuidelineFire`, `FireBC`, `AmbientBC`/`AmbientAuto` | common |
| `wall.fem_3d` | vessel shell as a Hex8 solid: `VesselShellMesh`, `ShellConduction3D` (kernel: `core.heat.solver.fem_3d`) | materials, core |
| `relief` | blowdown flow (real-gas nozzle + Borda-Carnot + Fanno line), ideal-gas fallback, PSV opening | thermo, common |
| `process` | geometry, fluid property dicts, zones + fast flashes, `inner_ht` (convection, boiling, condensation, interface, internal radiation) | thermo, common |
| `rupture` | stress solutions, axisymmetric FE, failure times | materials, common |
| `coupling` | `VesselFireModel` — the only place that combines the above | all |

## Wall (3-D only)

The steel wall is one Hex8 solid shell (`wall.fem_3d`), coupled by
`coupling.wall3d_coupling.Wall3DCoupling` (the 1-D radial wall columns were removed
2026-09-30). Heat conducts through the thickness, around and along the shell, so a jet-fire hot
spot spreads into the surrounding steel. Mesh: options `wall3d_n_theta` / `wall3d_n_length` /
`wall3d_n_radial` (default 72 x 40 x 6; converged to ~1 K / 10 s of rupture time,
`validation/vessfire/reports/wall3d_mesh.md`). Per step: fire flux per outer node (background
or peak zone, `peak_zone=False` = background only); inner heat transfer (gas convection +
radiation, liquid boiling / convection) as curves of wall temperature sampled at quantiles of
the nodes that use them (`sampled_curve`, PCHIP); one implicit step (backward Euler, secant heat
capacity: exact energy balance; CG with a radial line preconditioner, 2-3 iterations).

Output regions (`region_profiles`): `background` / `wet` / `peak` / `peak_wet` area averages
(`self.frac` = their area fractions), and `hot` = the through-thickness profile at the hottest
point, which drives the membrane rupture check (`T_mean_hot_C`). `meta["wall3d"]` holds the mesh
and the node-temperature history (for the 3-D view). Old case files with the 1-D options
(`wall_model`, `wall_cells`, `wall_nodes`) load with a warning; the options are ignored.

vs VessFire (140 cases, `validation/vessfire/reports/wall3d.md`, 1-D -> 3-D medians): hottest
wall 16.3 -> 15.3 K, wetted wall 20.1 -> 14.2 K, pressure 4.96 -> 4.85 %, Tresca rupture 139.5 ->
138.5 s; ~10x run time of the 1-D wall.

## `VesselFireModel.step()` stages (debug one at a time)

`_refresh_saturation` → `_valve_flows` → `wall3d.step` (fire → shell → gas and liquid) →
`_interface` → `_zone_balances` → `_solve_pressure` → `_phase_transfer` →
`_merge_vanishing_zones` → `wall3d.set_level` → `_check_finite`; values pass between stages in
a `StepContext`. Module: `coupling/vessel_fire_model.py` (was `vessel_fire_1d.py`).

## Rules

- Physics packages never import each other sideways; only `coupling` combines them.
- Nothing VessFire-derived in `src/`: the VessFire deck reader and material DB loader live in
  `validation/vessfire/`. Steel tables are passed in (`mat` argument).
- Known issues are fixed as separate, deliberate changes that regenerate the goldens:
  `docs/process_model_known_issues.md`. Never loosen `RTOL` or regenerate goldens to make a
  refactor pass.
