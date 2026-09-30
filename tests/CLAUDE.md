# Testing Domain — FAHTS

## Current State

Non-GUI (2026-09-30, after the 1-D wall removal): **1269 passing, 6 pre-existing failures**
(list in `docs/3d_solver_plan.md`),
22 golden tests skipped unless `--golden`.
GUI/VTK: 506 passing, 13 pre-existing failures (run with `QT_QPA_PLATFORM=xcb` on a display).
Process-vessel GUI widgets: `tests/gui/` run offscreen (no VTK) and are part of the default run;
`tests/test_main_window_process.py` needs a display like the other main-window tests.

```bash
cd /home/oslprb/vessfire_heatsolver
python -m pytest tests/ -q                        # all tests (golden tests skipped)
python -m pytest tests/test_foo.py                # single file
python -m pytest tests/regression --golden -q     # process-model goldens (3-D wall, ~5 min)
```

Run tests before declaring any task done. Both `model_file.fem` and `model_t1.fem`
must parse cleanly — smoke tests for both exist.

---

## Test File Naming

| Pattern | What it covers |
|---------|---------------|
| `test_phase3*.py` | Solver phases (FEM, CN integrator, section meshes) |
| `test_phase3e*.py` | Multi-section extensions (IHPROFIL, PIPE, shells, RadiationBall, BELTEMP) |
| `test_phase3f*.py` | Crank-Nicolson, heat accumulation, temperature gradients |
| `test_phase4*.py` | Results visualisation, colormap, animation |
| `test_*_surface_mesher.py` | Surface mesh geometry tests |
| `test_scene_manager_phase4.py` | SceneManager / VTK rendering (14 tests) |
| `test_usfos_reader.py` | Parser smoke tests for both model files |
| `test_beltemp*.py` | BELTEMP parse + export round-trip |

---

## Conventions

- One test file per module (mirror `src/fahts/` path structure)
- Physics tests: use small hand-crafted meshes, not the full model files
- Numeric tolerance: `np.testing.assert_allclose(rtol=1e-5)` unless physics dictates looser
- Do NOT mock the filesystem for parser tests — use real fixture files in `tests/fixtures/` or the project model files
- Temperature history values changed when CN replaced backward Euler — tolerance updates were made; do not revert to Euler to fix tests
- `model_file.fem` and `model_t1.fem` are the smoke-test inputs; both must parse without error

---

## Process-model unit tests (`tests/unit/`)

- `tests/unit/<package>/test_equivalence_legacy.py` — each ported package is bit-identical with
  `legacy/vfpy` (via `tests/legacy_ref.py`); synthetic steel table fixture in
  `tests/unit/conftest.py` (no proprietary data needed). The 1-D wall package and its test were
  removed 2026-09-30; the 3-D wall is checked against an independent 1-D radial finite-volume
  reference in `tests/unit/wall3d/`.
- `tests/unit/test_layering.py` — physics packages only import the layers below them.

## Golden Regression Tests — process model (`tests/regression/process/`)

Freeze the results of the product process model (`fahts.coupling`) so every change in numbers
is visible and deliberate. **Run `--golden` after any change to the process-model packages.**

- `cases/<id>/` — 15 input decks (Admin/Segment/Scenario.brl, heatload.scn), one per physics
  branch (H2/CH4/LPG/pseudo/free water/retrograde; fire, BDV, PSV, ambient, rupture;
  jet-fire peak zone bottom / top / side)
- `golden/` — 22 runs: time series + rupture table per (case, profile, duration);
  `manifest.json`: library versions, git commit, and a `history` of why goldens changed
- `harness.py` — `IMPLEMENTATIONS`: `fahts` (checked) and `legacy` (frozen vfpy; diagnostics
  and before/after studies only). History: goldens came from legacy until the port was verified
  bit for bit (2026-09-30), then were regenerated from `fahts` for the wall-grid fix (issue #1)
  and for the switch to the 3-D wall (coarse mesh 36x20x4, `harness.WALL_MESH`).
- Tolerance: `RTOL = 1e-6` of each column's magnitude. Never loosen it. A deliberate physics
  change regenerates goldens in the same commit:
  `python -m tests.regression.process.generate --reason "<why>"`.
- Needs CoolProp and the local material DB `data/reference/vessfire/vessfire.db` (gitignored,
  VessFire proprietary); skipped without them.
- `tests/integration/test_energy_balance.py` — the coupled model conserves energy (first-order
  convergence in dt); runs without the DB.

## USFOS Benchmark Test (Phase 3E.8 — NEXT TASK)

Goal: automated comparison of FAHTS output against `validation/usfos/reference/fahts_beltemp.fem`.

Reference: `validation/usfos/reference/fahts_beltemp.fem` — 2056 elements, 15 time steps.
Heat source: RadiationBall, center=(343,484,64)m, r1=5m/flux1=350kW, r2=100m/flux2=1.5kW.
Model: `examples/models/model_t1.fem`.

Expected test structure:
1. Build RadiationBall from benchmark config
2. Run analysis on model_t1.fem with that source
3. Export BELTEMP
4. Parse both reference and generated BELTEMP
5. Compare cumulative temperatures element-by-element with loose tolerance (material tables differ)

See `fahts/core/io/CLAUDE.md` for BELTEMP format details and material difference note.
