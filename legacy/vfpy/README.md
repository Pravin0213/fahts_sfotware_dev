# legacy/vfpy — frozen reference of the process (vessel) model

Verbatim snapshot (2026-09-29) of the core of `Test/vfpy`, the Python vessel-in-fire model
calibrated against VessFire (see `MODEL_CHOICES.md`). **Do not edit.** `MD5SUMS` records the
file hashes; `tests/regression/process/golden/manifest.json` records them too.

It is the reference that the port into `src/fahts/` (thermo, relief, process, wall/column_1d,
rupture) is checked against: `python -m pytest tests/regression --golden -q`.

| File | Role | Target in `src/fahts/` |
|---|---|---|
| `thermo_pr.py`, `thermo_data.py` | Peng-Robinson EOS, flash, pseudo-components | `thermo/` |
| `fluid.py` | CoolProp pure-fluid wrapper | `thermo/` |
| `heat_transfer.py` | steel `Material`, fire/ambient BCs, 1-D radial `WallColumn` | `materials/`, `fire/`, `wall/column_1d/` |
| `twophase_physics.py` | geometry, free convection, boiling curve, condensation, interface | `process/geometry.py`, `process/inner_ht/` |
| `valves.py`, `valves2.py` | BDV/PSV orifice, blowdown line (Fanno), HEM | `relief/` |
| `vessel.py` | v1 single-phase model; still supplies `PSV`, `AmbientAuto`, `H_CORRELATIONS` | merged into `process/`, `relief/` |
| `vessel2.py` | two-zone non-equilibrium model, `simulate2` | `process/model.py`, `process/zones.py`, `coupling/` |
| `stress.py` | membrane / Lamé / thermal stress, failure times | `rupture/` |

Not included (scratch, duplicates or VessFire tooling, still in `Test/vfpy`): `vessel2_strat.py`
(stratified fork — merge as an option later), `gas_side_models.py`, `*_proto.py`, `build_*`,
`run_*`, `compare.py`, `vf_pairs.py`, `vessfire_io.py`, `.bak` files.

Runtime quirks, handled by `tests/regression/process/harness.py` without editing the files:
- `heat_transfer.Material.from_vessfire_db` defaults to a Windows VessFire install path; the
  harness points it at `data/reference/vessfire/vessfire.db` (local only, never committed).
- `stress.py` uses `np.trapezoid` (NumPy ≥ 2); the harness aliases it on NumPy 1.x.
