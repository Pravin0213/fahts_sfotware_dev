# RadiationBall Benchmark Mismatch — Fix Brief

**Status:** Open. Investigation complete; no code changed yet.
**Audience:** The next agent who will implement the fixes.
**Date:** 2026-05-23

---

## 1. The problem (what you're fixing)

We compared our solver against the USFOS reference for a single **RadiationBall**
(USFOS `USERFLUX` type 0) heat source on the same `.fem` model.

- **USFOS (reference):** a strong spatial gradient — a few members very close to the
  ball get extremely hot, most of the structure stays cool (blue/green).
- **Our solver:** almost every element saturates near the top of the scale (uniformly
  red). We heat far too many elements far too much.

The visual difference is large and is **not** a rendering bug. The temperatures our
solver computes genuinely are too high. The cause is the flux-vs-distance model used
for the RadiationBall.

### Benchmark input (the exact source under test)

From `usfos_verification_results/fahts.fem` line 22:

```
USERFLUX  0  1   346.593  485.144  65.673   5  350000   70   1500
```

→ `center=(346.593, 485.144, 65.673) m`, `r1=5 m`, `flux1=350000 W/m²`,
`r2=70 m`, `flux2=1500 W/m²`.

### USFOS reference results (ground truth to match)

From `usfos_verification_results/fahts.out`:

- `Maximum Heat Flux (MaxFlux) : 595000.00` W/m²  ← **higher than flux1 (350000)**
- `Maximum Temperature (MaxTemp) : 1860.62` °C
- `Minimum Temperature (MinTemp) : -273.00` °C  (= 0 K → unexposed elements reported at 0 K)

---

## 2. Root cause: the flux model is inverse-square, not linear

**USFOS treats the ball as a true inverse-square point source.** The two `(radius, flux)`
pairs are just two samples on the curve `q(d) = flux1 · (r1/d)²`.

Evidence:

1. **`MaxFlux = 595000 > flux1 = 350000`.** Impossible for a step model or a linear ramp
   (both are bounded by flux1). It only occurs if the inverse-square law is continued
   *inside* r1: `350000 · (5/3.84)² ≈ 595000` → the closest element sits ~3.84 m from
   the centre. So there is **no clamp at flux1** — the law extends below r1.
2. **Consistency of the two sample points:** `flux1 · (r1/r2)² = 350000 · (5/70)² = 1786`,
   which is ≈ `flux2 = 1500`. Solving for the exact exponent through both points gives
   `n ≈ 2.07` — i.e. inverse-square.
3. **`MaxTemp = 1860 °C` with most elements cold** is the signature of a steep `1/r²`
   falloff: only the few elements within a couple of metres get blasted.

### What our code does instead (the bug)

`fahts/core/heat/sources/rad_ball.py`, method `flux_at()` (around lines 66–78):

```python
def flux_at(self, distance: float) -> float:
    if distance <= self.r1:
        return self.flux1
    t = (distance - self.r1) / (self.r2 - self.r1)
    return max(0.0, self.flux1 + t * (self.flux2 - self.flux1))   # LINEAR — WRONG
```

This is **linear interpolation** between flux1 and flux2. Numerical impact at the
benchmark values:

| distance | linear (current) | inverse-square (USFOS) | over-flux |
|----------|------------------|------------------------|-----------|
| 20 m     | ~270,000 W/m²    | ~21,900 W/m²           | ~12×      |
| 35 m     | ~178,000 W/m²    | ~7,100 W/m²            | ~25×      |
| 70 m     | 1,500 W/m²       | 1,786 W/m²             | —         |

So nearly every element within 70 m receives 10–25× too much flux and saturates. **This
is the dominant bug.** Fixing it should recover most of the USFOS behaviour.

> ⚠️ **The docs lie.** Both `rad_ball.py`'s module docstring AND
> `fahts/core/heat/sources/CLAUDE.md` AND `docs/3D_FEM_heat_transfer_theory.txt` §15.8
> describe a **two-zone step** (`flux1` inside r1, `flux2` between r1–r2). The code
> implements **linear interpolation**. The real USFOS is **inverse-square**. All three
> disagree — trust the USFOS `.out` evidence above, not the prose.

---

## 3. Secondary issue: no directionality (cos θ) inside r2

Within r2, the flux is applied **uniformly to every quad on every face** (front, back,
sides). USFOS (and the §15.6 concentrated-source formula `q = E·cos(θ)/(4π·r²)`) applies
a `cos(θ)` factor so faces pointing *away* from the ball receive ~0.

The relevant code paths:

- `fahts/core/heat/solver/analysis_runner.py`, `_bc_for_element()` (lines ~764–775):
  when a ball covers the element midpoint it returns a single uniform
  `q_fn = lambda t: ref_flux`, with `covering_zones=[]`, `ref_zone=None`. That uniform
  flux is later applied to all quads in `SurfaceTransientSolver._assemble_step` via the
  `q_presc` branch (`surface_solver.py` ~579–582).
- **The correct directional model already exists** for the beyond-r2 case:
  `_rad_ball_per_quad_flux()` in `analysis_runner.py` (lines ~833–886) computes
  `q_face = base_flux · cos(θ_face)` per quad, where `base_flux = flux2·(r2/r)²` and
  faces pointing away get 0. This per-quad array is passed as `q_per_quad` to the solver
  (`surface_solver.py` ~584–587).

So the two paths are inconsistent: **beyond r2 is directional + inverse-square (correct),
inside r2 is uniform + linear (wrong).** They should use the *same* per-quad,
inverse-square, cos(θ) model — the only difference being which reference point calibrates
the curve.

---

## 4. The fixes (in priority order)

### Fix 1 — Inverse-square flux law (highest impact)

In `fahts/core/heat/sources/rad_ball.py`, replace the linear `flux_at()` with an
inverse-square / log-log power law passing through both `(r1, flux1)` and `(r2, flux2)`:

- For all `d > 0`: `q(d) = flux1 · (r1 / d)^n` where
  `n = ln(flux1/flux2) / ln(r2/r1)` (≈ 2.07 for the benchmark; effectively inverse-square).
  Equivalently, log-log linear interpolation between the two points.
- **Do NOT clamp at flux1 for `d < r1`** — extrapolate the same law (this reproduces
  `MaxFlux = 595000`).
- For `d > r2`: USFOS reports `q = 0` beyond r2 in the theory text, but the `.out` peak
  behaviour is dominated by near-ball elements, so far-field is negligible either way.
  Decide explicitly: simplest is `q = 0` for `d > r2` (matches §15.8) — but confirm this
  doesn't regress the existing falloff path (see Fix 2). Document whichever you pick.

Update `falloff_element_ids()` and the docstring in the same file to match the new law.

### Fix 2 — Unify the within-r2 path with the directional per-quad model

Make the RadiationBall coverage inside r2 use the same per-quad, cos(θ), inverse-square
flux as `_rad_ball_per_quad_flux()` (in `analysis_runner.py`), instead of the single
uniform `q_fn`. Concretely:

- In `_bc_for_element()` (analysis_runner.py ~764–775), stop returning a uniform
  `q_fn = ref_flux` for ball coverage. Instead route ball-covered elements through the
  same `q_per_quad` assembly used by the falloff branch, but using the element's actual
  distance `d` (the inverse-square law from Fix 1) rather than the `flux2·(r2/r)²` form.
- `_rad_ball_per_quad_flux()` currently hard-codes `base_flux = ball.flux2*(ball.r2/r)²`.
  Generalise it to use `ball.flux_at(r)` (the corrected law) so the inner and outer
  regions are one continuous model.
- Keep the `cos(θ)` per-face logic — faces pointing away from the ball get 0.
- Re-radiation handling (`eps_rerad` in `surface_solver.py` ~359–367, ~589–595) already
  fires when `q_per_quad is not None` and `epsilon_m == 0`, so it should keep working;
  verify it still triggers after the refactor.

### Fix 3 — Verify against the benchmark

After Fixes 1–2, run the benchmark and check:
- Peak element temperature approaches USFOS's **1860 °C** on near-ball elements.
- Far elements stay cool — the spatial gradient returns (not uniformly red).
- Peak applied flux reaches ~**595,000 W/m²** on the closest element.

Run the test suite first: `python -m pytest tests/ -q` (baseline: 1008 passing, 1 skipped
per CLAUDE.md). Then run the GUI / benchmark on the model and compare visually + against
`usfos_verification_results/fahts_beltemp.fem`.

> There is a `benchmark-runner` agent available that runs the solver against the USFOS
> benchmark and diffs against the BELTEMP reference — use it for the numeric comparison.
> There is also a `physics-reviewer` agent for checking solver changes against the theory.

### Fix 4 — Material mode for benchmark parity (minor, ~5–15%)

`SteelMaterial.usfos_mode` (in `fahts/core/model/material.py`) switches k(T)/cp(T) from
EN 1993-1-2 Annex C to USFOS's thermpar tables, but **nothing in the runtime path sets
it** — production runs use Annex C. For a true apples-to-apples benchmark match against
USFOS, enable `usfos_mode=True` on the materials *for the benchmark run only* (do not
change production defaults). This is a secondary accuracy effect, not the cause of the
gross mismatch.

### Fix 5 — Correct the contradictory documentation

Once Fix 1 lands, rewrite the RadiationBall flux description in all three places to the
inverse-square model:
- `fahts/core/heat/sources/rad_ball.py` module + method docstrings
- `fahts/core/heat/sources/CLAUDE.md` (RadiationBall section)
- `docs/3D_FEM_heat_transfer_theory.txt` §15.8

### Fix 6 — (Optional) Reporting parity

USFOS reports *all* elements (unexposed → −273 °C / 0 K); we only solve exposed elements
and render the rest at 20 °C. This is cosmetic and explains the legend min-temp
difference. Only address if exact output parity is required.

---

## 5. Things that are NOT the problem (don't waste time here)

- **Rendering / colormap.** `fahts/renderer/colormap.py` is a standard clamped LUT; both
  images use the same 20–800 °C range. The red is real heat, not a display artifact.
- **Crank-Nicolson time integration / FEM assembly.** The solver math is fine; the bug is
  purely in the boundary-condition flux fed into it.

---

## 6. Key files & line references (as of 2026-05-23)

| File | What's there |
|------|--------------|
| `fahts/core/heat/sources/rad_ball.py` | `flux_at()` (~66–78) — the linear-interp bug; `falloff_element_ids()` (~100–120) |
| `fahts/core/heat/solver/analysis_runner.py` | `_bc_for_element()` (~740–805) ball→uniform flux; `_rad_ball_per_quad_flux()` (~833–886) correct directional model; `_bc_falloff_ball()` (~808–830) |
| `fahts/core/heat/solver/surface_solver.py` | `_assemble_step()` (~480–597) — `q_presc` uniform branch (~579–582), `q_per_quad` branch (~584–587), re-radiation (~589–595, ~359–367) |
| `fahts/core/model/material.py` | `usfos_mode` flag + thermpar tables |
| `usfos_verification_results/fahts.fem` | benchmark input (USERFLUX line 22) |
| `usfos_verification_results/fahts.out` | reference results (MaxFlux 595000, MaxTemp 1860, line ~5301) |
| `usfos_verification_results/fahts_beltemp.fem` | BELTEMP reference for numeric diff |

Read `CLAUDE.md` and the relevant subdirectory `CLAUDE.md` files first. Note: those docs
describe the RadiationBall as a two-zone step — **that description is wrong** (see §2).
