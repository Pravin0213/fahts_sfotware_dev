# legacy/vfpy — frozen reference of the process (vessel) model

Verbatim snapshot (2026-09-29) of the core of `Test/vfpy`, the Python vessel-in-fire model
calibrated against VessFire (see `MODEL_CHOICES.md`). **Do not edit.** `MD5SUMS` records the
file hashes; `tests/regression/process/golden/manifest.json` records them too.

It is the reference that the port into `src/fahts/` (thermo, relief, process, wall/column_1d,
rupture) is checked against: `python -m pytest tests/regression --golden -q`.

| File | Role | Ported to (2026-09-30) |
|---|---|---|
| `thermo_pr.py`, `thermo_data.py` | Peng-Robinson EOS, flash, pseudo-components | `thermo/` (split into core + mixins) |
| `fluid.py` | CoolProp pure-fluid wrapper for the v1 model | not ported (only used by v1 `vessel.simulate`) |
| `heat_transfer.py` | steel `Material`, fire/ambient BCs, 1-D radial `WallColumn` | `materials/steel_table.py`, `fire/`, `wall/column_1d/`; DB reader → `validation/vessfire/material_db.py` |
| `twophase_physics.py` §1-4 | geometry, free convection, boiling, condensation, interface | `process/geometry.py`, `process/inner_ht/` |
| `twophase_physics.py` §5 | generic TwoZoneVessel / HomogeneousVessel framework | not ported (unused by `vessel2`) |
| `valves.py`, `valves2.py` | orifice, blowdown line (Fanno), HEM | `relief/ideal_gas.py`, `relief/{friction,vapour_state,nozzle,fanno_line,blowdown}.py` |
| `vessel.py` | v1 single-phase model; `PSV`, `AmbientAuto`, `H_CORRELATIONS`, `stresses` | `relief/psv.py`, `fire/ambient.py`, `process/inner_ht/natural_convection.py`, `rupture/stress_solutions.py`; v1 `simulate` not ported |
| `vessel2.py` | two-zone model `simulate2`, property helpers, zones | `coupling/vessel_fire_1d.py` (`VesselFireModel`, one method per stage), `coupling/options.py`, `process/{fluid_properties,zones}.py`, `process/inner_ht/{nucleate_only,internal_radiation}.py` |
| `stress.py` | stresses, GPS FE, failure times | `materials/en1993_mechanical.py`, `rupture/` |

Not included (scratch, duplicates or VessFire tooling, still in `Test/vfpy`): `vessel2_strat.py`
(stratified fork — merge as an option later), `gas_side_models.py`, `*_proto.py`, `build_*`,
`run_*`, `compare.py`, `vf_pairs.py`, `vessfire_io.py`, `.bak` files.

Runtime quirks, handled by `tests/regression/process/harness.py` without editing the files:
- `heat_transfer.Material.from_vessfire_db` defaults to a Windows VessFire install path; the
  harness points it at `data/reference/vessfire/vessfire.db` (local only, never committed).
- `stress.py` uses `np.trapezoid` (NumPy ≥ 2); the harness aliases it on NumPy 1.x.
