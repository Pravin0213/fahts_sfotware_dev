# Process model — known issues found during the port

The port of `legacy/vfpy` into `src/fahts/` is behaviour-preserving: the golden tests require
the same numbers as the frozen reference. Issues found while porting are therefore **recorded
here, not fixed in place**. Each fix is a separate, deliberate change: fix in `src/fahts/`,
regenerate the goldens from the fixed code, and explain the change in the commit.

| # | Issue | Where | Effect | Status |
|---|---|---|---|---|
| 1 | The 1-D wall always uses the 105 mm node layout (`REFERENCE_NODES_105MM`), whatever the vessel's wall thickness. The stress check uses the real thickness `t` and rescales the 105 mm node temperatures onto it. | `coupling.vessel_fire_1d` (wall columns), `rupture.failure.stress_series` (x_nodes) | For the 60 mm validation vessel the thermal wall has ~75 % too much heat capacity and conduction length: wall heats too slowly, fluid heat input is delayed, and rupture times are affected. The calibration studies (`MODEL_CHOICES.md`) on the validation set were made with this. | open — fix after the port |
| 2 | The vessel geometry is always horizontal with flat heads; `#Vessel_Orientation` and insulation are ignored (`process.geometry` already supports vertical vessels and heads). | `coupling.vessel_fire_1d` | Vertical vessels and insulated cases are not modelled (golden set excludes them). | open — feature |
| 3 | Material properties come from VessFire's database (`vessfire.db`). | `materials` | The product cannot ship these tables; own EN 1993-1-2 / EN 10028 / certificate data needed. | open — before release |
| 4 | The blowdown flow solver keeps process-wide caches (`relief.blowdown._WARM`, `_MODELS`). The warm start makes the line-limited flow depend slightly on call history, including earlier cases in the same process. | `relief.blowdown` | Results within the solver tolerance (TOL = 3e-4 in ln L) but not strictly reproducible run-to-run; global state. | open — move the cache into the model instance |
| 5 | Two gravity values: 9.81 (gas/liquid wall correlations in the model, ambient) and 9.80665 (`process.inner_ht` correlations). | `coupling`, `fire.ambient`, `process.inner_ht` | < 0.03 % in Ra; cosmetic. | open — unify on 9.80665 |

