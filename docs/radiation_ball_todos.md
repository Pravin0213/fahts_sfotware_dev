# RadiationBall Bug Tracker

**Goal:** Match USFOS `fahts_beltemp.fem` reference output.
**Benchmark config:** `usfos_verification_results/fahts.fem` → `USERFLUX 0 1 346.593 485.144 65.673 5 350000 70 1500`
**Structural model:** `model_t1.fem` (1644 beams + 412 shells); entire structure within 20.6 m of ball.
**Reference numbers:** MaxFlux 595,000 W/m², MaxTemp 1860 °C, 15-min run.

Supersedes `radiation_ball_fix_todo.md` and `radiation_ball_investigation_2026-05-23.md`.

---

## Status summary

| ID  | Description                                   | Status              | Impact   |
|-----|-----------------------------------------------|---------------------|----------|
| C1  | flux_at was inverse-square → fixed to linear  | ✅ Done 2026-05-23  | Dominant |
| C3b | flux_at at per-quad centroid, not midpoint    | ✅ Done 2026-06-01  | Small    |
| C3a | Near-field (r < R1) rule: MaxFlux > Flux1     | ⬜ Open             | High     |
| C2  | PIPE over-heating (+138 °C bias)              | ⬜ Open             | High     |
| C5  | Re-radiation applied to all faces, flux only to facing faces | ⬜ Open | Medium |
| C4  | GUI never sets usfos_benchmark_mode           | ⬜ Open             | Medium   |
| C6  | Inside-member (enclosed fluid) mass accuracy  | ⬜ Open             | Low–Med  |
| C7  | Test uses wrong benchmark ball; docs stale    | ⬜ Open             | Low      |

---

## Open bugs

### C3a — Near-field (r < R1) under-heating

**Symptom:** Engulfed elements (r < R1 = 5 m) reach ~1150 °C in USFOS but only ~620 °C
in our solver. USFOS reports MaxFlux = 595,000 W/m² > Flux1 = 350,000 W/m². Our code
clamps `flux_at` at Flux1 for r < R1.

**Evidence:** `usfos_verification_results/fahts.out`:
```
Maximum Heat Flux (MaxFlux) : 595000.00
```
At the nearest element (~3.84 m): `350000 × (5/3.84)² ≈ 595000` — this matches inverse-square
extrapolation **below** R1, even though the USFOS User Manual figure shows a flat Flux1 plateau.
Either USFOS's near-field implementation differs from the manual's figure, or there is a
per-quad vs per-element distance effect at work.

**Files:**
- `fahts/core/heat/sources/rad_ball.py` — `flux_at()`: currently returns `self.flux1` for
  `d < r1`; the USFOS rule below R1 is unconfirmed.

**To investigate:** compare `flux_at(d)` for `d < R1` against `fahts.out` per-element
heat-flux values to find the actual law.

---

### C2 — PIPE sections over-heated (+138 °C mean bias)

**Symptom:** After C1, I-sections have near-zero bias (−30 °C) but PIPE sections are still
~+138 °C too hot.

**Likely causes (in order of suspicion):**

1. **Inside-member thermal mass**: USFOS reports 1866 `inside-member` elements — enclosed
   fluid mass for hollow tubes. Our `_compute_M_extra` adds a coarse lumped mass of
   `ρ_air × c_air = 1200 J/(m³·K)` distributed over mesh nodes. USFOS's inside-member
   property specifies its own thermpar; its effective heat capacity per metre may be much
   larger.

2. **Mesh count vs USFOS MESHTUBE**: `fahts.out` reports USFOS uses 2 axial × 8 hoop = 16
   quads per pipe. Our default is `c_circ=8`, `n_length=4` = 32 quads. Different quad areas
   change the BC integration weight per node.

3. **Exposed area**: check whether USFOS's `MESHTUBE` faces are outer-surface only, and
   whether our `PipeSurfaceMesher` produces the same exposed area per element.

**Files:**
- `fahts/core/heat/solver/analysis_runner.py` — `_compute_M_extra()`: PIPE branch.
- `fahts/core/heat/section_mesh/pipe_surface_mesher.py` — mesh geometry.
- `fahts/core/results/analysis_config.py` — `c_circ`, `n_length_p` defaults.

---

### C5 — Re-radiation applied to all quads; incoming flux only to facing quads

**Symptom:** `surface_solver._assemble_step` applies `q_rerad = -ε·σ·T⁴` to **all** quads,
while `q_per_quad` (RadiationBall directional flux) is only non-zero on facing quads
(`cos θ > 0`). Back-facing quads therefore cool via re-radiation but receive no incoming
flux, creating a net-negative energy contribution on cold faces.

**Impact:** Uncertain — could be measured by comparing runs with re-radiation disabled on
non-facing quads. Back-facing faces on a 15-minute run are still near 20 °C so `T⁴` is
small, but worth quantifying.

**Files:**
- `fahts/core/heat/solver/surface_solver.py` — `_assemble_step()` lines ~589–595: the
  `rerad_scale = 1 if q_per_quad else n_exposed_sides` block.
- Possible fix: pass a `facing_mask` array alongside `q_per_quad` and only apply re-radiation
  to those quads.

---

### C4 — GUI never enables `usfos_benchmark_mode`

**Symptom:** The `RunAnalysisDialog.get_config()` always returns
`usfos_benchmark_mode=False`. Production runs therefore use EN 1993-1-2 Annex C k/cp
tables and `ε_rerad = 0.7`, while the USFOS reference used `thermpar` tables with `emiss = 0.85`.
This is a ~10–15% secondary accuracy difference for the screenshot comparison.

**Additional detail:** USFOS `thermpar` 1 and 6 have `k = 0` (zero conductivity) per
`fahts.out`. Our `usfos_mode` uses a single `k_ref = 50` table for all materials; per-material
thermpar handling would be needed for strict parity.

**Files:**
- `fahts/gui/dialogs/run_analysis_dialog.py` — `get_config()`: add a "USFOS benchmark mode"
  checkbox.
- `fahts/core/model/material.py` — `SteelMaterial.usfos_mode`: confirm it switches to
  USFOS thermpar tables; add per-material k=0 support.

---

### C6 — Inside-member element capacity (hollow sections)

**Symptom:** USFOS reports 1866 inside-member elements in `fahts.out`. Our proxy
(`_compute_M_extra`, ρ_air × c_air = 1200 J/(m³·K)) is a single fixed value that ignores
material assignment. If USFOS's inside-member uses steel-like thermpar (ρc ≈ 3.8 × 10⁶)
the enclosed mass would be ~3000× larger.

**To investigate:** Find the `inside-member` property line in `fahts.out` and identify the
thermpar number; compare effective ρc against our 1200 J/(m³·K) value.

**Files:**
- `fahts/core/heat/solver/analysis_runner.py` — `_compute_M_extra()`.

---

### C7 — Test uses wrong benchmark ball; docs inconsistent

**Tests:**
- `tests/test_usfos_benchmark.py` constructs `RadiationBall` with `center=(343, 484, 64)`,
  `r2=100`. The actual `fahts.fem` benchmark is `center=(346.593, 485.144, 65.673)`, `r2=70`.
  Fix: update test fixture to use the real `fahts.fem` parameters.

**Documentation (all describe an outdated or wrong model):**
- `fahts/core/heat/sources/rad_ball.py` — module docstring still says "piecewise-linear"
  and describes the old two-zone step model; `flux_at` docstring example uses wrong r2=100.
- `fahts/core/heat/sources/CLAUDE.md` — benchmark config line shows wrong `r2=100`.
- `docs/3D_FEM_heat_transfer_theory.txt` §15.8 — describes USERFLUX as two-zone step.
- `fahts/core/heat/solver/CLAUDE.md` — references old `rad_ball.py` inverse-square law.

**Files:** all listed above.

---

## Measured baseline (current code, usfos_benchmark_mode=True, 60-element sample)

| Variant                  | Mean err | |Err| mean | PIPE bias | ISec bias |
|--------------------------|----------|------------|-----------|-----------|
| Linear + cos θ (current) | +62 °C   | 198 °C     | +138 °C   | −30 °C    |

Target: mean |err| < 50 °C with no section-type bias.

## Excluded from scope

- r2 cutoff / falloff path: entire structure < 21 m, r2 = 70 m irrelevant.
- Crank-Nicolson / FEM assembly math: verified correct.
- Renderer / colormap: cosmetic; both codes use same 20–800 °C range.
