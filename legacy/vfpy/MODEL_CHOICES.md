# Model choices: matching VessFire vs. physics-preferred

Purpose: the Python model (vfpy) is first tuned to reproduce VessFire as closely as
possible, by *selecting between defined modelling options* on the basis of agreement
with VessFire's outputs. Where the VessFire-matching choice differs from what the
physics suggests, both are kept as options and documented here, so the model can later
be switched to the physics-preferred setting and the difference argued.

Rules followed throughout: only published physics; VessFire outputs used as benchmark
data; no VessFire binaries inspected; no free constants fitted to VessFire outputs
(licence clause 3.3 prohibits determining VessFire's methodology).

Evidence references: `D:\Test\studies\...` (study folders), metrics are RMS errors vs
VessFire over 0-60 min unless stated.

---

## 1. Wetted-wall heat transfer above the cricondenbar  *(largest fire-case difference)*

**What the cricondenbar is.** For a mixture, the phase envelope (bubble + dew curves)
has a maximum pressure, the *cricondenbar*. Above it no pressure-temperature state is
two-phase for that composition: a heated liquid cannot form vapour bubbles, it just
becomes a hotter, less dense single-phase (compressed / supercritical) fluid. For a
pure fluid the cricondenbar is the critical pressure (propane 42.5 bar, n-butane 38.0
bar). Our LPG (C3 0.6 / nC4 0.4) crosses it at ~40-45 bar, rich-gas and condensate
liquids at somewhat higher pressures depending on their (changing) composition. In a
closed or slowly relieved vessel in fire the pressure passes these values within
10-20 min.

**What VessFire's outputs imply.** At the liquid-wetted peak-load location VessFire's
inner wall stays only 3-22 K above the liquid temperature for the whole hour
(e.g. rich gas at 30 min: wall 428 C, liquid 424 C, ~150 kW/m2 through the wall ->
effective h ~35 kW/m2K), i.e. boiling-level heat transfer continues well above the
cricondenbar (liquid at 400+ C, 100-300 bar).

**What the physics says.** With the pool above its cricondenbar there is no nucleate
boiling. Heat transfer is single-phase natural convection to a dense fluid, typically
0.3-1 kW/m2K, enhanced near the pseudo-critical line where cp peaks (Jackson & Hall
1979; Pioro & Duffey 2005). Our model with this physics runs the wet wall 100-180 K
above the liquid; the liquid heats more slowly, fire-case pressure is lower and LPG
reaches liquid-full later than in VessFire.

**Options** (`vessel2.Options2.wet_above_crit`):
| value | meaning | status |
|---|---|---|
| `"single-phase"` | natural convection (Churchill-Chu, film properties, density-difference Grashof) | physics-preferred |
| `"supercritical"` | as above with bulk-to-wall integrated cp (pseudo-critical enhancement) | physics-preferred alternative |
| `"boiling"` | keep nucleate-boiling correlation (Cooper) with the pool at saturation, last valid saturation properties | VessFire-matching |

Evidence (`studies/17_wet_wall_crit`, internal radiation on, Churchill-Chu gas side),
fire cases excluding methane+water, pressure RMS % median / liquid-temperature RMS K median:
| wet_above_crit | pressure | liquid T | wet wall |
|---|---|---|---|
| single-phase | 13.6 % | 84 K | 32 K |
| supercritical | 13.4 % | 84 K | 33 K |
| **boiling (default, VessFire-matching)** | **9.1 %** | **51 K** | **22 K** |
Per case (single-phase -> boiling): pseudo+BDV 14.9 -> 4.8 %, LPG+BDV 37.0 -> 23.3 %,
LPG fire 100 kW/m2 + BDV 12.2 -> 6.5 %, rich gas 80 bar + BDV 11.2 -> 6.4 %, condensate
40 C + BDV 16.9 -> 12.1 %, LPG closed 24.4 -> 19.5 %. The pseudo-critical cp enhancement
("supercritical") has negligible effect at these conditions.
Remaining LPG fire + PSV gap: VessFire's liquid fills the vessel and starts dense relief
~5 min earlier (between 15 and 20 min). Up to 15 min the models agree (P 25.4 / 28.7
barg, liquid 94 / 101 C, liquid mass 40.5 / 39.6 t); the timing is governed by liquid
thermal expansion near LPG's critical point, i.e. PR liquid density there.

**Wet-wall boiling curve (study 25).** Even with boiling kept above the cricondenbar, the
full boiling curve (nucleate -> CHF -> transition -> film) puts our wet wall into film
boiling within 5-15 min: near the critical point CHF collapses because h_fg and sigma
go to zero. The wall then runs 60-100 K above the liquid, while VessFire's stays 0.5-4 K
above it all hour (study 21). Option `wet_boiling`:
`"full"` (physics) or `"nucleate_only"` (Cooper 1984 blended with free convection, no
CHF cap; VessFire-matching). Result: fire liquid-T RMS median 51 -> 13 K, LPG closed
fire 19.5 -> 7.9 %, C40 12.1 -> 5.4 %.

**Current default profile ("VessFire-matching", N5, study 25):**
`wet_above_crit="boiling"`, `wet_boiling="nucleate_only"`, `h_corr="evans_stefany"`,
`h_corr_liq="churchill_chu"`, `liq_grashof="drho"`, `rad_internal=True`,
`water_mode="vf"`, `psv_liquid="liquid"`, blackbody fire (eps 0.7, h 25),
`valve_model="b1"`, `line_diameter="outer"`, `interface_mass=False`, `strat=False`.
Fire pressure RMS median 5.2 % (mean 6.6 %), cold blowdown <= 2.4 %
(`studies/11_twophase/RESULTS_twophase_final.csv`).
**Physics-preferred alternatives:** `wet_above_crit="single-phase"`/`"supercritical"`,
`wet_boiling="full"`, `water_mode="physical"`, `psv_liquid="gas"` (or a two-phase
HEM/omega nozzle), `strat=True` for LPG.

Detection in the model: saturation properties are refreshed every 10 s; if no bubble
point exists at the current pressure, or the latent heat is < 30 kJ/kg, or the bubble
point is > 50 K below the liquid temperature, the pool is treated as unable to boil.

---

## 2. Fire boundary (outer surface)

- VessFire: flux split "based on initial conditions" (published description); its
  absorbed flux vs. surface temperature lies 3-8 % above our black-body model at the
  same temperature (verified at the correct location, see `memory/vessfire-output-columns`).
- Model: incident flux converted to a black-body fire temperature, sigma*T_air^4 = q;
  absorbed = eps_s sigma (T_air^4 - T_s^4) + h (T_air - T_s), eps_s = 0.7 (EN 1993-1-2
  carbon steel), h = 25 W/m2K (EN 1991-1-2). Options: `flux` = blackbody | balance |
  prescribed (VessFire's absorbed flux as input, diagnostic only).
- VessFire input `#Vessel_Outside_Conditions 293.15 -0.5 0.85` probably holds the outer
  surface emissivity (0.85). Using eps 0.85 in our black-body formulation over-absorbs:
  fire pressure median 5.2 -> 6.8 %, liquid-T 13 -> 33 K (study 25, N6). Kept at 0.7.
  VessFire's flame model differs (agent study 24: its absorbed flux is a single function
  of surface temperature per load).

## 3. Gas-side (dry wall -> gas) natural convection

- Churchill-Chu (horizontal cylinder) vs Evans-Stefany (closed-container heating).
  With the nucleate-only wet wall (study 25), Evans-Stefany is closer in fire too (gas-T
  RMS median 57 -> 26 K, R-closed-fire 1.6 %) and is now the default. The results below
  used the older full boiling curve.
- Cold blowdown (two-phase and single-phase dense gases): Evans-Stefany closer
  (0.4-2.4 % vs 1.6-3.7 % pressure). Fire with internal radiation: Churchill-Chu closer
  (median fire pressure 14.9 % vs 18.4 %). Hydrogen single-phase: Churchill-Chu.
- Option `h_corr`. VessFire describes its gas-side convection as "a function of flow".

## 4. Internal radiation (dry wall - liquid surface - grey gas)

- VessFire's published description includes radiation + convection from the inner
  wall to the contents. Model: grey radiosity network, eps_wall 0.8, eps_liq 0.95,
  gas transparent by default (high-pressure gas emissivity correlations not available).
  Insensitive to emissivities (+-1 % pressure). Option `rad_internal` (on for matching).

## 5. Interface (gas <-> liquid free surface)

- Sensible exchange only (natural convection both sides, in series); phase change via
  each zone's equilibrium flash (BLOWDOWN / Haque et al. 1992 practice). Interface
  evaporation (T_i = T_bubble) available (`interface_mass`) but diverges near the
  critical point where h_fg -> 0.

## 6. Liquid thermal stratification

- Published (Birk & Cunningham 1996, Aydemir et al. 1988) but does not bring the model
  closer to VessFire (it lowers the mean liquid temperature). `vessel2_strat.py`,
  option `strat`, off for matching. Physics-preferred for LPG in fire.

## 7. Free-water layer in fire

- VessFire (study 23, 8 diagnostic cases, outputs only): when the vessel holds no
  hydrocarbon liquid, the free-water layer does not change: Mass_water is constant and
  its temperature stays at the initial value. This holds at 0.3 and 1.0 m water, at
  5 bara (T_sat 152 C), with closed fire to 190 barg, and when starting at 80 C. Its
  temperature follows T(P, initial enthalpy) within 0.7 K. The water still cools the
  wetted wall (inner wall 24-35 C at 205 kW/m2 through the wall). That heat, 29 GJ at
  0.3 m and 58 GJ at 1.0 m in 1 h (about 30 % of the fire input), is not conserved.
  With about 2 kg of hydrocarbon liquid present the water does heat and boil (357 C,
  7.3 t evaporated).
- Physics: the water heats by about 17 K/min and boils within the hour (IAPWS pool).
- Option `water_mode`: `"physical"` (default, physics-preferred); `"sink"` (the pool
  keeps wall contact, but the heat it receives is removed and booked as `Q_sink_MJ` in
  the energy balance); `"vf"` (sink only while the pool holds < 2 kg of hydrocarbon
  liquid, VessFire-matching). W-fire-bdv pressure RMS: physical 136 %, sink 9.3 %,
  sink + Evans-Stefany 5.3 %.
- The remaining early-time gap (0-10 min, before any boiling) comes from the gas side:
  Churchill-Chu heats dense methane faster than VessFire (T_gas 91 vs 51 C at 10 min).
  Evans-Stefany gives 1-2.3 %.
- Known limitation: the physical mode fails once a pure-water pool passes the water
  critical pressure (221 bar).

## 8. Equation of state

- VessFire's methane / lean gas / N2 inventories equal plain Peng-Robinson exactly;
  its H2 inventory is closest to the reference EOS (not PR). Model: PR for
  hydrocarbon systems (own `thermo_pr.py`), reference EOS (CoolProp HEOS) for
  single-phase H2 (`eos="auto"`).

## 9. Relief / blowdown valves

- API 520 gas equations with k = cp/cv at vessel conditions (matches VessFire within
  1-2 %); the rigorous isentropic exponent n = rho c^2 / P is 1-6 % lower.
- Blowdown line: Borda-Carnot expansion + real-gas Fanno line (`valves2.py`). Line
  diameter input read as OUTER diameter (bore = d - 2t): manual ambiguous; d - 2t gives
  6-9 % vs VessFire on LPG, d gives 20-37 %. Option `line_diameter`.
- Dense-phase / liquid relief (study 25, `dense_relief.csv`). VessFire vents liquid
  only once the vessel is liquid-full (gas < ~1 kg); there is no entrainment or level
  swell before that. Its PSV rate from the liquid zone equals the API 520 liquid equation
  Cd A sqrt(2 rho dP) within 0.3 %, even for a gas-like supercritical fluid (Z ~ 1,
  rho 10-60 kg/m3), where the vapour equation gives 2.2x less. Option `psv_liquid`:
  `"liquid"` (VessFire-matching) or `"gas"`. L-fire-psv 165 -> 11.8 %, R-fire-psv
  22 -> 2.0 %. Physics: a supercritical/flashing fluid chokes (HEM/HNE-omega, API 520
  Annex C); the liquid equation over-predicts it.
  VessFire's dense-phase BDV rate through the line is 0.30-0.35 of the orifice-only
  liquid value and matches no published form tested (gas, HEM, liquid, liquid + line
  friction). Our model (vapour equation + Fanno fallback) is within ~25 %. Open.
- PSV types 0/1/2 = trapezoidal / triangular / square (manual listing order);
  BDV delay = instant opening at the delay time.

## 10. Stress / failure

- von Mises failure times within ~40 s of VessFire (membrane stress vs 295 MPa x
  F_UTS(T_mean)). VessFire's Tresca failure is earlier than any published variant
  tested (`studies/04_stress`); not matched.

## 11. VessFire output columns (interpretation)

Verified by energy balances on VessFire's own outputs (`studies/20_audit_outputs`); this
supersedes an earlier "peak-zone" reading.
- `Energy` = absorbed flux at the outer surface at the COLDEST location.
- `RadIn_max + Conv_max` = AREA-AVERAGED absorbed flux (x A_out closes VessFire's energy
  balance within 0.3-2.5 %).
- `T_inner`, `T_o_wall`, `T_outer`, `dT_rad` = at the location of largest through-wall
  temperature difference (usually wetted). `TS_max`/`TS_min` and the two Tn node sets =
  hottest / coldest location.
- `ESV_rate` and BDVRate.txt include vented LIQUID; gas-only release =
  BDV1_Gas + BDV2-10_Gas + PSV1-10_Gas.
- `Mass_oil` excludes free water; `T_oil` is the liquid (incl. free water) temperature.
- Pairing helper: `vf_pairs.add_derived` (used by `compare.py`).
