# Process model (thermo / fire / wall.column_1d / relief / process / rupture / coupling)

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
| `wall.column_1d` | radial 1-D conduction `WallColumn` | materials |
| `relief` | blowdown flow (real-gas nozzle + Borda-Carnot + Fanno line), ideal-gas fallback, PSV opening | thermo, common |
| `process` | geometry, fluid property dicts, zones + fast flashes, `inner_ht` (convection, boiling, condensation, interface, internal radiation) | thermo, common |
| `rupture` | stress solutions, axisymmetric FE, failure times | materials, common |
| `coupling` | `VesselFireModel` — the only place that combines the above | all |

## Wall regions

One 1-D `WallColumn` per region, weighted by area fraction (`self.frac`, sums to 1):
`dry` / `wet` (background fire, above / below the liquid) and, when the heat load has a local
peak flux different from the background (jet fire), `peak_dry` / `peak_wet`
(`wall_regions.region_fractions`: peak zone = length `xi_start..xi_end` x arc
`attack_deg +- circ_deg/2`, angle from the top). Level changes re-weight each dry/wet pair with
energy-conserving mixing. Regions with zero area are still stepped (legacy behaviour) but never
contribute to heat sums, energy or the hot-spot/rupture check (`T_mean_hot_C` = hottest region
with area); their temperatures in the output are meaningless. Option `peak_zone=False` restores
background-only fire.

## `VesselFireModel.step()` stages (debug one at a time)

`_refresh_saturation` → `_valve_flows` → `_dry_wall` → `_wet_wall` → `_interface` →
`_zone_balances` → `_solve_pressure` → `_phase_transfer` → `_merge_vanishing_zones` →
`_reweight_wall` → `_check_finite`; values pass between stages in a `StepContext`.

## Rules

- Physics packages never import each other sideways; only `coupling` combines them.
- Nothing VessFire-derived in `src/`: the VessFire deck reader and material DB loader live in
  `validation/vessfire/`. Steel tables are passed in (`mat` argument).
- Known issues are fixed as separate, deliberate changes that regenerate the goldens:
  `docs/process_model_known_issues.md`. Never loosen `RTOL` or regenerate goldens to make a
  refactor pass.
