# RadiationBall vs USFOS — Investigation & Fix TODO (2026-05-23)

**Author:** investigation pass against `usfos_verification_results/fahts_beltemp.fem`.
**Supersedes the diagnosis in** `docs/radiation_ball_fix_todo.md` (that file describes the
*old* linear-interp bug, which is already fixed; the problem has since flipped to
*under*-heating).

> ⚠️ The MD/docstring descriptions of the RadiationBall flux model are NOT reliable
> (CLAUDE.md, rad_ball.py docstring, 3D_FEM_heat_transfer_theory.txt §15.8 all describe
> the model differently and all describe **inverse-square**, which is WRONG). USERFLUX is
> not in `Fahts_Theory_Manual.pdf`, but the **USFOS User Manual `USERFLUX` page**
> (provided 2026-05-23) is authoritative — see §1b.

## 1b. AUTHORITATIVE USFOS USERFLUX model (from USFOS User Manual)

`USERFLUX  TimeHist  Type  x y z  R1 Flux1  R2 Flux2 ...`  (max 7 points; record may repeat,
fluxes added). Our benchmark `USERFLUX 0 1 346.593 485.144 65.673 5 350000 70 1500` means
**TimeHist=0** (no time scaling), **Type=1** (flux defined by discrete points) — NOT "set".
*(Our code/docs mislabel the 2nd field as "set".)*

**Flux-vs-distance law (the manual's figure):**
- `r < R1`  : **constant = Flux1** ("assuming engulfed structure")
- `R1 ≤ r ≤ R2` : **LINEAR interpolation** between (R1,Flux1) and (R2,Flux2)
- `r > R2`  : **linear extrapolation** of that line (dashed; clamp at 0 once negative)

→ The law is **LINEAR, not inverse-square.** The current `rad_ball.py` implements an
inverse-square power law — this is the primary bug (see C1).

**Directionality:** the manual shows a tubular member with surface normal `v`, ray angle
`α`, and **`I = intensity · cos(α)`**. So per-face `cos θ` IS correct (faces pointing away
get 0). Our `cos θ` form is right; only the flux magnitude law was wrong.

**Unresolved:** the `.out` reports MaxFlux = 595000 > Flux1 = 350000, which neither a
Flux1-clamp nor linear extrapolation below R1 (~377000 max) reproduces. Near-field (r<R1)
elements reach ~1150 °C in BELTEMP, more than a clamped 350000 W/m² + cos + re-radiation
yields (~620 °C). USFOS evidently applies >Flux1 to engulfed elements by some rule we don't
yet have. (Per-quad distance evaluation helps a little but not enough — see C3/Question.)

---

## 1. Benchmark setup (the real one)

`usfos_verification_results/FAHTS.sh` shows the USFOS reference run was:

- Structural model: **`model_t1.fem`** (1644 beams + 412 shells = 2056 elements)
- Thermal control: `usfos_verification_results/fahts.fem`
  - `TimeUnit Min`, `TempSim 15 300 1` → 15 **minutes**, 300 steps (dt = 3 s), BELTEMP every 1 min
  - `USERFLUX 0 1 346.593 485.144 65.673  5 350000  70 1500`
  - `thermpar`: rho=7850, c_ref=510, **emiss=0.85**; k_ref=50 for most materials but
    **k=0 for thermpar 1 and 6** (zero conductivity!).
  - `tempdepy 100` (cp factors, spike 9.804 at 731 °C) and `tempdepy 200` (k factors).
- Reported: MaxFlux 595000 W/m², MaxTemp 1860 °C, SHAPEFACT (internal view factors) **Off**.

Geometry: the **entire structure is within 20.6 m of the ball** (bbox 15×21×18 m,
dist min 2.8 / mean 9.8 / max 20.6). → **The r2 = 70 m cutoff is irrelevant here**; do not
chase the falloff/cutoff path for this case.

BELTEMP stores **mean temperature increment per element** + linearized Y/Z gradients.
The gradients are large (e.g. −1600 °C/m), i.e. USFOS heating is strongly **one-sided**
(directional), and the section is NOT isothermal. (Hoop-conduction time constant
≈ L²/α ≈ 0.25/1e-5 ≈ 6 h ≫ 15 min, so each face heats almost independently — both codes
should agree the section stays non-uniform.)

---

## 2. Measured discrepancy (current code, `usfos_benchmark_mode=True`)

Stratified 60-element sample across distance, 15-min run, compared to BELTEMP mean T:

| model variant | mean err | colder/hotter | mean \|err\| | PIPE bias | ISec bias |
|---|---|---|---|---|---|
| **current: inv-sq + cos θ** | **−163 °C** | 46 / 14 | 222 | −66 | **−282** |
| binary (cos=1 if facing) | +13 °C | 28 / 32 | 188 | +155 | −160 |
| uniform (all faces, no dir.) | very hot | — | — | — | — |
| **LINEAR + cos θ (manual)** | **+62 °C** | 16 / 44 | **198** | +138 | **−30** |

The **LINEAR + cos θ** row uses the manual's correct law. It cuts the I-section error from
−282 → **−30 °C** and removes the gross net-cold bias — confirming the inverse-square law
was the dominant bug. Residuals after this fix: near-field (r<R1) too cold (Flux1 clamp too
low vs USFOS), and PIPE sections too hot (+138, pipe-specific — likely thermal mass).

**Conclusions:**

1. **Confirmed: current solver runs net COLD** (matches the screenshot). Median −193 °C.
2. **I-sections (IHPROFIL) are the worst** — under-heated by ~280 °C on average, up to
   −480 °C (e.g. eid 53: FAHTS 184 vs USFOS 652; eid 717: 655 vs 1136). I-beams have a
   small projected (absorbing) area but a large re-radiating surface, so the per-area
   `cos θ` reduction starves them.
3. **The directional flux model is the dominant lever** and the least certain part of the
   code. cos θ → too cold; full-on-all-faces → far too hot; binary "facing" → net-neutral
   but PIPE overshoots and ISec still undershoots.
4. **A section-type split survives every directional variant** (PIPE vs ISec move in
   opposite directions). That points to a *second*, independent error in how each section
   type is meshed / how flux area & mass are computed vs USFOS (MESHBOX/MESHIPRO/MESHTUBE).

---

## 3. Root-cause candidates (ranked by measured impact)

### C1 — Flux-vs-distance law is inverse-square; should be LINEAR (HIGHEST — CONFIRMED)
`rad_ball.py::flux_at` uses `q = flux1·(r1/d)^n` (inverse-square). The USFOS manual (§1b)
specifies **linear interpolation** between (R1,Flux1) and (R2,Flux2), constant=Flux1 below
R1, linear extrapolation beyond R2. At 13 m the laws differ ~6.5× (linear ~307k vs inv-sq
~47k W/m²), which is exactly the I-section starvation. **Fix:** replace `flux_at` with the
linear law; the `cos θ` per-quad path in `analysis_runner` already matches the manual's
`I = intensity·cos α`. Measured: I-section error −282 → −30 °C. Directionality is NOT the
bug — the flux magnitude was.

### C2 — Per-section meshing & thermal mass vs USFOS (HIGH)
After C1, PIPE sections are still ~+138 °C too hot while I-sections are accurate. USFOS adds
**"inside-member" elements** (1866 of them) — enclosed thermal mass for tubes — which our
coarse `_compute_M_extra` (ρc_air=1200, distributed by area) likely under-represents, so our
tubes heat too fast. USFOS PIPE mesh = 2 axial × 8 hoop (`fahts.out`); our `PipeSurfaceMesher`
produced 32 quads here. Audit pipe exposed area, wall mass, enclosed-fluid/inside-member
capacity, and quad count against USFOS MESHTUBE. (I-profile/box accounting looks OK post-C1.)

### C3 — Near-field (r<R1) under-heated + single beam-midpoint distance (MEDIUM)
Engulfed elements (r<R1) reach ~1150 °C in USFOS but only ~620 °C with a Flux1 clamp + cos +
re-radiation. USFOS applies >Flux1 near the source (MaxFlux 595000). Two sub-items: (a) figure
out USFOS's near-field rule (not in the manual figure — *Question/needs USFOS source*); (b)
evaluate `flux_at` at each quad-centroid distance instead of one beam-midpoint distance, so
the steep near-field variation across a section is captured. (b) is cheap and physically
sound regardless of (a).

### C4 — Production runs don't use USFOS material/emissivity (MEDIUM, for the screenshot)
The GUI (`run_analysis_dialog.get_config`) never sets `usfos_benchmark_mode`, so a normal
GUI run uses **EN 1993-1-2 Annex C** k/cp (cp(20°)=440 vs USFOS 404 → slower heating) and
**ε_rerad = 0.7** instead of USFOS's 0.85. Secondary (<~15%) but matters for the screenshot
comparison. **DECISION (2026-05-23): add a UI toggle** in the Run Analysis dialog to enable
`usfos_benchmark_mode` (USFOS thermpar materials + ε=0.85), so the user can switch between
production (EN 1993-1-2) and USFOS-parity runs. Also note USFOS thermpar 1 & 6 have **k=0**
(zero conductivity) per-material — our `usfos_mode` uses a single k_ref=50 table for all
materials; per-material thermpar handling may be needed for strict parity.

### C5 — Steel re-radiation area accounting (MEDIUM)
Re-radiation `−ε·σ·T⁴` is integrated over **all** quads, while incoming flux is limited to
facing quads (cos θ). If USFOS only re-radiates from / heats the same faces, our energy
balance loses too much from cold back faces. Revisit once C1 is settled.

### C6 — `inside-member` elements / enclosed radiation (LOW–MED)
USFOS reports 1866 "inside-member" elements and an internal-member property. Our hollow-
section heat accumulation (`_compute_M_extra`, ρc_air=1200) is a coarse lumped-mass proxy.
Likely minor for peak temps but listed for completeness.

### C7 — Documentation cleanup (LOW)
Once C1 is resolved, correct rad_ball.py docstring, sources/CLAUDE.md, solver/CLAUDE.md,
and 3D_FEM_heat_transfer_theory.txt §15.8 to match the implemented model. Fix
`tests/test_usfos_benchmark.py` which uses the WRONG ball (center 343/484/64, r2=100) vs
the real fahts.fem (346.593/485.144/65.673, r2=70).

### NOT the problem
- r2 cutoff / far-field falloff (whole structure < 21 m).
- Crank-Nicolson / FEM assembly math.
- Renderer colormap (both use 20–800 °C; BELTEMP mean ≈ our T_centroid, apples-to-apples).

---

## 4. Proposed fix order (goal: EXACT USFOS parity)
1. **[DONE 2026-05-23] C1 — replaced inverse-square `flux_at` with the manual's LINEAR law**
   (constant<R1, linear R1–R2, linear extrap >R2, clamp 0); kept the `cos θ` per-quad path;
   fixed the `TimeHist`/`Type` field labels in docstrings/CLAUDE.md/§15.8; updated
   `test_rad_ball.py` assertions. Verified: ISec bias −282 → −30 °C, net bias −163 → +62 °C,
   33 rad-ball/benchmark tests pass.
2. **[DONE 2026-06-01] C3b — evaluate `flux_at` per quad-centroid distance** (cheap, physical)
   to sharpen the near-field gradient. (`analysis_runner._rad_ball_per_quad_flux` now computes
   `r_vec` and `r_hat` per quad centroid in global coords instead of once at the element midpoint.)
3. **C2 — audit PIPE thermal mass / inside-member capacity & mesh** to remove the +138 °C
   pipe overshoot.
4. **C4 — add the `usfos_benchmark_mode` UI toggle** (USFOS thermpar materials + ε=0.85);
   consider per-material thermpar (k=0 for mats 1 & 6).
5. **C3a — near-field (r<R1) rule:** needs USFOS source to explain MaxFlux 595000 / ~1150 °C
   engulfed temps. Investigate separately.
6. Re-validate against BELTEMP (target: mean |err| ≪ 198, no section-type bias). Then C5
   (re-radiation area review), C6 (inside-member), C7 (docs + fix the wrong test ball).
</content>
