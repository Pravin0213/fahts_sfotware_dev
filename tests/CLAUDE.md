# Testing Domain — FAHTS

## Current State

**1008 passing, 1 skipped** (as of 2026-05-18)

```bash
cd /home/oslprb/vessfire_heatsolver
python -m pytest tests/ -q          # all tests
python -m pytest tests/test_foo.py  # single file
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

## USFOS Benchmark Test (Phase 3E.8 — NEXT TASK)

Goal: automated comparison of FAHTS output against `validation/usfos/reference/fahts_beltemp.fem`.

Reference: `validation/usfos/reference/fahts_beltemp.fem` — 2056 elements, 15 time steps.
Heat source: RadiationBall, center=(343,484,64)m, r1=5m/flux1=350kW, r2=100m/flux2=1.5kW.
Model: `model_t1.fem`.

Expected test structure:
1. Build RadiationBall from benchmark config
2. Run analysis on model_t1.fem with that source
3. Export BELTEMP
4. Parse both reference and generated BELTEMP
5. Compare cumulative temperatures element-by-element with loose tolerance (material tables differ)

See `fahts/core/io/CLAUDE.md` for BELTEMP format details and material difference note.
