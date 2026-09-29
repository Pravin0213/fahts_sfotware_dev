"""
Peng-Robinson (PR-78) thermodynamics engine for fire-exposed /
depressurising process vessels (vapour + hydrocarbon liquid + free water).

Written from published methods only (sources cited next to each method).  No
parameter in this file was fitted to VessFire output.

Units: SI throughout - T [K], P [Pa], molar quantities per mol (J/mol, J/mol/K,
m3/mol), molar mass kg/mol.  Mass-based values are available on Phase objects.

Quick use
---------
    m = PRMixture({"C1": 0.6, "C3": 0.1, "PSEU1": 0.3},
                  pseudo={"PSEU1": {"sg": 0.78, "Tb": 450.0}})
    r = m.flash_PT(50e5, 293.15)               # FlashResult
    r.vapour.rho, r.liquid.x, r.beta_V, r.h     # phase / total properties
    r2 = m.flash_UV(U, V, n, init=r)            # U [J], V [m3], n [mol] (array/dict)
    r3 = m.flash_PH(P, h, z) ; m.flash_PS(P, s, z)
    m.bubble_P(T, x), m.dew_P(T, y), m.bubble_T(P, x), m.dew_T(P, y)
    m.transport(r)                              # adds mu, k to phases, r.sigma

Methods and sources
-------------------
EOS      Peng & Robinson (1976) Ind. Eng. Chem. Fundam. 15, 59; m(omega) of the
         1978 revision for omega > 0.491 (Robinson & Peng 1978, GPA RR-28).
         Classical van der Waals one-fluid mixing, a_ij = sqrt(a_i a_j)(1-k_ij).
         Optional Peneloux volume translation (off by default), PR form
         c_i = 0.50033 (0.25969 - Z_RA) R Tc/Pc with Z_RA = 0.29056 - 0.08775 omega
         (Peneloux, Rauzy & Freze 1982; PR coefficients as in Pedersen,
         Christensen & Shaikh, "Phase Behavior of Petroleum Reservoir Fluids",
         2nd ed. 2015, Ch. 4; Z_RA of Spencer & Danner 1972).
Residual Analytic departure functions of the PR EOS (e.g. Michelsen & Mollerup,
props    "Thermodynamic Models: Fundamentals & Computational Aspects", 2nd ed.
         2007, Ch. 2-3).  Fugacity-coefficient composition derivatives from the
         reduced residual Helmholtz energy F(n,T,V) (same reference).
Ideal gas cp0 of the reference EOS in CoolProp (see thermo_data.py);
         reference state: ideal gas, h = 0 and s = 0 at T0 = 298.15 K, P0 = 1 bar
         for every pure component; ideal mixing entropy -R sum x ln x.
Pseudo   Riazi & Daubert (1987) Ind. Eng. Chem. Res. 26, 755 (M, Tc, Pc from Tb, SG;
components  coefficients as given in Riazi, "Characterization and Properties of
         Petroleum Fractions", ASTM MNL50, 2005, Table 2.5, Tb in K, Pc in bar);
         acentric factor Lee & Kesler (1975) Hydrocarbon Process. 54(3) 153 for
         Tb/Tc < 0.8, Kesler & Lee (1976) Hydrocarbon Process. 55(3) 153 otherwise;
         Zc = 0.2905 - 0.085 omega (Pitzer) for Vc; ideal-gas cp Kesler & Lee (1976)
         (SI form in Riazi 2005, Eq. 7.37); parachor Firoozabadi et al. (1988)
         SPE Res. Eng. 3, 265: P = -11.4 + 3.23 M - 0.0022 M^2.
         VessFire input convention: (mole fraction, SG, Tb[K]); SG > 1.5 is API.
k_ij     see _KIJ below (Knapp et al. 1982 regressions as tabulated in common PR
         data sets; Soreide & Whitson 1992 water-gas; Chueh & Prausnitz 1967 form
         with A=0.18, B=6 for C1 - C7+).
Flash    Michelsen (1982) Fluid Phase Equilib. 9, 1 (tangent-plane stability);
         Rachford & Rice (1952) with bounded Newton (negative flash window,
         Whitson & Michelsen 1989); successive substitution then Newton on vapour
         mole numbers with analytic Jacobian (Michelsen 1982b, FPE 9, 21).
Water    free-water approximation: aqueous phase = pure water (PR); water content
         of the hydrocarbon phases from equality of the water fugacity with that
         of pure water at (T,P).  Hydrocarbon solubility in water is neglected.
         (Free-water flash as in Michelsen & Mollerup 2007 Ch. 11 / GPSA "free
         water" option.)  Without water, or free_water=False, a standard 2-phase
         flash is used.
UV/PH/PS  UV: single-phase explicit (T,v) Newton + stability test; otherwise
         Newton in (T, lnP) on PT flashes (Broyden), fallback nested bracketing
         (outer T, inner P) that also handles single-component (discontinuous)
         two-phase states by lever-rule blending.  PH/PS: bracketed secant /
         Illinois in T, with lever-rule blending across discontinuities.
Transport  Chung, Ajlan, Lee & Starling (1988) Ind. Eng. Chem. Res. 27, 671:
         dense-fluid viscosity and thermal conductivity with their mixing rules
         (as given in Poling, Prausnitz & O'Connell, "The Properties of Gases and
         Liquids", 5th ed. 2001, Eqs. 9-5.x, 9-6.18..21, 10-5.5..6).  Alternative
         viscosity: Lohrenz, Bray & Clark (1964) JPT 16, 1171 with Stiel & Thodos
         (1961) dilute gas.  Surface tension: Macleod (1923)-Sugden (1924) parachor
         method, parachors of Weinaug & Katz (1943) / Pedersen et al. (2015).
"""

from __future__ import annotations

from fahts.thermo.flash_pt import FlashPTMixin
from fahts.thermo.flash_state import StateFlashMixin
from fahts.thermo.free_water import FreeWaterMixin
from fahts.thermo.pr_core import PRCore
from fahts.thermo.saturation import SaturationMixin
from fahts.thermo.stability import StabilityMixin
from fahts.thermo.transport import TransportMixin


class PRMixture(FlashPTMixin, StateFlashMixin, FreeWaterMixin, StabilityMixin,
                SaturationMixin, TransportMixin, PRCore):
    """Peng-Robinson mixture model.

    composition : dict name -> mole fraction (VessFire names; pseudo names allowed)
                  or a list of names (then z must be given to every call).
    pseudo      : dict name -> {"sg": SG or API, "Tb": K} (or (sg, Tb) tuple).
    kij         : dict {(name_i, name_j): value} overriding the defaults.
    volume_shift: Peneloux volume translation (default False = plain PR).
    free_water  : free-water treatment when H2O is present (default True).
    water_model : properties of the free (pure) aqueous phase: "iapws" (default; IAPWS-95
                  reference EOS of Wagner & Pruss 2002 as implemented in CoolProp, residual
                  part added to this module's ideal-gas water so h, s, u stay on the same
                  reference state; its fugacity sets the water content of the HC phases)
                  or "pr" (Peng-Robinson pure water; liquid density ~15 % low).
    """

