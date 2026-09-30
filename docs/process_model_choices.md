# Process model — modelling choices (re-checked after the wall-grid fix)

The vfpy calibration (`legacy/vfpy/MODEL_CHOICES.md`, frozen) was done while the 1-D wall used a
105 mm grid for every vessel (known issue #1). After the fix (2026-09-30) every choice was
re-checked against VessFire. Rules unchanged: published physics only, choose between defined
options by agreement with VessFire outputs, no constants fitted to VessFire.

Evidence (reproducible, `python -m validation.vessfire.compare_cases <study>`; reports in
`validation/vessfire/reports/`):
- `calibration_recheck` — 77 validation-set cases (4 random per module, seed 0, + the 11 golden
  cases); each option changed alone from the defaults, run on the cases it can affect.
- `fire_boundary` — the 45 fire cases, with signed biases and absorbed fire heat vs VessFire.
- `gas_side_cc.csv` — the 17 gas-only fire cases with Churchill-Chu on the gas side.

Baseline (current defaults) over 73 runnable cases, medians: pressure RMS 4.6 %, gas T 16 K,
dry wall 6 K, wetted wall 12 K, liquid 6.5 K.

## Decisions

| Option | Default (kept) | Alternative tested | Result on affected cases (median, baseline → alternative) | Decision |
|---|---|---|---|---|
| `wet_above_crit` | `boiling` | `single-phase`, `supercritical` | 43 liquid cases: P 7.0 → 8.6 %, wet wall 11.7 → 14.8 K, liquid 6.5 → 8.1 K, Tresca time error 142 → 267-282 s | keep (confirmed) |
| `wet_boiling` | `nucleate_only` | `full` (CHF, film) | P 7.0 → 8.5 %, wet wall 11.7 → 13.7 K, Tresca error 142 → 221 s | keep (confirmed) |
| `liq_grashof` | `drho` | `beta` | no effect (1 case > 0.5 %-pt) | keep; option is inert on this set |
| `interface_mass` | `False` | `True` | P 7.0 → 6.5 % but liquid 6.5 → 7.2 K, Tresca error 142 → 266 s; 5 better / 6 worse | keep |
| `h_corr` (gas side) | `evans_stefany` | `churchill_chu` | fire: P 7.9 → 10.0 %, 13 better / 26 worse; cold blowdown: P 8.6 → 12.1 %, gas T 17.6 → 31.3 K | keep — but see open point A |
| `rad_internal` | `True` | `False` | P 7.9 → 8.3 %, dry wall 22.5 → 13.7 K, wet wall 20.2 → 27.1 K | keep (pressure first); dry-wall gain noted |
| `water_mode` | `vf` | `physical`, `sink` | 3-4 cases only; `physical`: liquid 1.4 → 9.6 K and one solver failure; `sink`: P 3.6 → 3.1 % | keep (too few cases to change) |
| `psv_liquid` | `liquid` | `gas` | 19 PSV cases: P 7.0 → 11.3 % | keep (confirmed) |
| `flux` | `blackbody` | `balance` | P 7.9 → 4.6 % but absorbed heat +1.7 → +9.6 % vs VessFire, dry wall bias −2 → +36 K, wet wall −7 → +30 K | **keep** — pressure improves only because 10 % too much heat is absorbed (right answer, wrong reason) |
| `eps_surf_fire` | 0.7 | 0.85 | same pattern: absorbed heat +9.6 %, walls +43 / +50 K | keep |
| `line_diameter` | `outer` | `inner` | 31 line cases: P 8.6 → 9.1 %, 1 better / 1 worse | keep |

With the wall fixed, the default fire boundary absorbs the right heat (median +1.7 % vs VessFire)
and the wall temperatures are nearly unbiased (dry −2 K, wetted −7 K).

## Open points (need a decision)

**A. Gas-side heat transfer in gas-only fire cases.** The gas runs cold with the default:
median gas bias −31 K (−25 to −103 K; worst for H2 at 250 kW/m2), pressure −1.7 %.
Churchill-Chu brackets VessFire from the other side: gas +28 K, pressure RMS 4.6 → 9.4 % for
methane-type gases. For **hydrogen** Churchill-Chu is clearly closer (pressure RMS M06-0003
7.1 → 1.0 %, M10-0021 8.3 → 2.2 %, M10-0024 11.3 → 4.8 %), as the vfpy study also noted. Options:
(1) keep Evans-Stefany everywhere; (2) select Churchill-Chu for hydrogen-dominated gas (a
defined-option choice, no fitting; evidence from 3 cases — widen the H2 sample first);
(3) look for a published correlation for heated gas in closed horizontal cylinders.
Note that VessFire's `TS_max` is the hottest-location wall temperature while the model's dry
wall is an area average, so "dry wall unbiased" may mean the model's wall holds slightly too
much heat — consistent with a gas side that is too weak.

**B. Local (jet) fire zone ignored** — known issue #10. Decisive for rupture in jet fires.
