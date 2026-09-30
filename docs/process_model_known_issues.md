# Process model — known issues found during the port

The port of `legacy/vfpy` into `src/fahts/` is behaviour-preserving: the golden tests require
the same numbers as the frozen reference. Issues found while porting are therefore **recorded
here, not fixed in place**. Each fix is a separate, deliberate change: fix in `src/fahts/`,
regenerate the goldens from the fixed code, and explain the change in the commit.

| # | Issue | Where | Effect | Status |
|---|---|---|---|---|
| 1 | ~~The 1-D wall always used the 105 mm node layout, whatever the vessel's wall thickness.~~ The wall grid now follows the case thickness (`wall.column_1d.radial_nodes`, 10 cells; `wall_cells` / `wall_nodes` options). | `coupling.vessel_fire_1d` | vs VessFire, 74 cases (`validation/vessfire/reports/wall_grid_fix.md`), medians before -> after: pressure RMS 10.2 -> 4.6 %, gas T 120 -> 16 K, dry wall 120 -> 6 K, wet wall 59 -> 12 K, liquid 16 -> 6.5 K, Tresca rupture time |error| 897 -> 147 s (17 of 19 VessFire ruptures reproduced, was 13); energy balance error 10 % -> 0 %. 20 cells change results < 2 %. The vfpy calibration choices (`MODEL_CHOICES.md`) were made before this fix. | **fixed 2026-09-30** |
| 2 | The vessel geometry is always horizontal with flat heads; `#Vessel_Orientation` and insulation are ignored (`process.geometry` already supports vertical vessels and heads). | `coupling.vessel_fire_1d` | Vertical vessels and insulated cases are not modelled (golden set excludes them). | open — feature |
| 3 | Material properties come from VessFire's database (`vessfire.db`). | `materials` | The product cannot ship these tables; own EN 1993-1-2 / EN 10028 / certificate data needed. | open — before release |
| 4 | The blowdown flow solver keeps process-wide caches (`relief.blowdown._WARM`, `_MODELS`). The warm start makes the line-limited flow depend slightly on call history, including earlier cases in the same process. | `relief.blowdown` | Results within the solver tolerance (TOL = 3e-4 in ln L) but not strictly reproducible run-to-run; global state. | open — move the cache into the model instance |
| 5 | Two gravity values: 9.81 (gas/liquid wall correlations in the model, ambient) and 9.80665 (`process.inner_ht` correlations). | `coupling`, `fire.ambient`, `process.inner_ht` | < 0.03 % in Ra; cosmetic. | open — unify on 9.80665 |
| 6 | `WallColumn` evaluates cp(T) and k(T) at the start of each step (lagged). The energy balance then has a first-order time error: ~0.07 % after 300 s at dt = 1 s for the 60 mm H2 fire case (EN-type steel), 4x smaller at dt = 0.25 s. | `wall.column_1d.WallColumn.step` | Small; grows with steep cp(T) (e.g. the ~735 C carbon-steel peak). | open — Picard iteration or enthalpy formulation |
| 7 | Liquid-full initial state is rejected ("initial state has no vapour phase"). | `coupling.vessel_fire_1d` | Cases M01-0027, M14-0157 cannot run. | open — feature |
| 8 | Benzene (`BNZ`) is missing from `thermo.component_data`. | `thermo` | Case M16-0033 cannot run. | open |
| 9 | Intermittent `ValueError: math domain error` for gas + free water, fire + BDV (M12-0010); depends on the wall grid (10 cells fails, 20 passes). | not located | One validation case fails. | open — reproduce and locate |
