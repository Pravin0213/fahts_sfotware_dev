"""
thermo_pr.py - Peng-Robinson (PR-78) thermodynamics engine for fire-exposed /
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

import math

import numpy as np

try:
    from .thermo_data import COMPONENTS, THETA
except ImportError:  # flat import (vfpy on sys.path)
    from thermo_data import COMPONENTS, THETA

R = 8.314462618
SQ2 = math.sqrt(2.0)
D1 = 1.0 + SQ2
D2 = 1.0 - SQ2
T_REF = 298.15
P_REF = 1.0e5
ZFLOOR = 1e-30          # floor on mole fractions inside the flash algorithms
B_VC_PR = 0.25308       # b/Vc at the PR critical point (0.07780/0.3074)

ALIASES = {"NC4": "C4", "NC5": "C5", "NC6": "C6", "NC7": "C7", "NC8": "C8",
           "METHANE": "C1", "ETHANE": "C2", "PROPANE": "C3", "WATER": "H2O",
           "ARGON": "AR", "HELIUM": "HE", "OXYGEN": "O2", "NITROGEN": "N2"}

# Dipole moments [debye], Poling et al. (2001) Appendix A; others zero.
DIPOLE = {"H2O": 1.8, "H2S": 0.9}
# Chung et al. (1988) association factor (Poling 2001 Table 9-1)
KAPPA = {"H2O": 0.075908}
# Parachors: Weinaug & Katz (1943) / Pedersen et al. (2015) Table 10.? (hydrocarbons,
# N2, CO2, H2S); water 52.0 (Firoozabadi & Ramey 1988); H2, He, Ar, O2 from Sugden's
# atomic constants (approximate - these components are rarely in a liquid).
PARACHOR = {"N2": 41.0, "CO2": 78.0, "H2S": 80.1, "C1": 77.0, "C2": 108.0, "C3": 150.3,
            "IC4": 181.5, "C4": 189.9, "IC5": 225.0, "C5": 231.5, "C6": 271.0,
            "C7": 312.5, "C8": 351.5, "H2O": 52.0, "H2": 34.0, "HE": 20.5, "AR": 54.0,
            "O2": 54.0}

HEAVY_HC = ("C7", "C8")          # treated with the "C7+" k_ij column
HC = ("C1", "C2", "C3", "IC4", "C4", "IC5", "C5", "C6", "C7", "C8")

# Binary interaction parameters for PR.
# - N2, CO2, H2S with hydrocarbons and among themselves: Knapp, Doring, Oellrich,
#   Plocker & Prausnitz (1982) "Vapor-liquid equilibria for mixtures of low boiling
#   substances", DECHEMA Chem. Data Ser. VI (PR regressions, as reproduced in the
#   common PR data sets, e.g. ChemSep/DWSIM "pr_ip" table).  "C7+" values are used
#   for C7, C8 and pseudo-components.  Transcribed values; differences between
#   published tables are typically <= 0.02.
# - H2O with gases: Soreide & Whitson (1992) Fluid Phase Equilib. 77, 217
#   (k_ij for the non-aqueous phase), C5+ and other gases 0.5 (common default for
#   the classical mixing rule, Pedersen et al. 2015 Ch. 16).
# - hydrocarbon-hydrocarbon 0, except C1 with C7+ (Chueh-Prausnitz, below);
#   H2, He, Ar, O2 pairs 0 unless listed.
_KIJ = {
    ("N2", "C1"): 0.0311, ("N2", "C2"): 0.0515, ("N2", "C3"): 0.0852, ("N2", "IC4"): 0.1033,
    ("N2", "C4"): 0.0800, ("N2", "IC5"): 0.0922, ("N2", "C5"): 0.1000, ("N2", "C6"): 0.1496,
    ("N2", "C7+"): 0.1441,
    ("CO2", "C1"): 0.0919, ("CO2", "C2"): 0.1322, ("CO2", "C3"): 0.1241, ("CO2", "IC4"): 0.1200,
    ("CO2", "C4"): 0.1333, ("CO2", "IC5"): 0.1219, ("CO2", "C5"): 0.1222, ("CO2", "C6"): 0.1100,
    ("CO2", "C7+"): 0.1100,
    ("H2S", "C1"): 0.0888, ("H2S", "C2"): 0.0862, ("H2S", "C3"): 0.0925, ("H2S", "IC4"): 0.0474,
    ("H2S", "C4"): 0.0633, ("H2S", "IC5"): 0.0600, ("H2S", "C5"): 0.0633, ("H2S", "C6"): 0.0500,
    ("H2S", "C7+"): 0.0500,
    ("N2", "CO2"): -0.0170, ("N2", "H2S"): 0.1767, ("CO2", "H2S"): 0.0974,
    ("H2O", "C1"): 0.4850, ("H2O", "C2"): 0.4920, ("H2O", "C3"): 0.5525, ("H2O", "IC4"): 0.5091,
    ("H2O", "C4"): 0.5091, ("H2O", "N2"): 0.4778, ("H2O", "CO2"): 0.1896, ("H2O", "H2S"): 0.19031,
}


# ----------------------------------------------------------------------------- helpers
def characterise_pseudo(sg: float, Tb: float) -> dict:
    """Pseudo-component (SG or API, Tb [K]) -> constants.  See module docstring."""
    if sg > 1.5:                          # API gravity
        sg = 141.5 / (sg + 131.5)
    # Riazi-Daubert (1987): theta = a exp(b Tb + c SG + d Tb SG) Tb^e SG^f
    M = 42.965 * math.exp(2.097e-4 * Tb - 7.78712 * sg + 2.08476e-3 * Tb * sg) \
        * Tb ** 1.26007 * sg ** 4.98308                       # g/mol
    Tc = 9.5233 * math.exp(-9.314e-4 * Tb - 0.544442 * sg + 6.4791e-4 * Tb * sg) \
        * Tb ** 0.81067 * sg ** 0.53691                       # K
    Pc_bar = 3.1958e5 * math.exp(-8.505e-3 * Tb - 4.8014 * sg + 5.749e-3 * Tb * sg) \
        * Tb ** -0.4844 * sg ** 4.0846                        # bar
    Tbr = Tb / Tc
    Kw = (1.8 * Tb) ** (1.0 / 3.0) / sg                       # Watson K (Tb in R)
    if Tbr < 0.8:   # Lee-Kesler (1975)
        num = -math.log(Pc_bar / 1.01325) - 5.92714 + 6.09648 / Tbr + 1.28862 * math.log(Tbr) \
            - 0.169347 * Tbr ** 6
        den = 15.2518 - 15.6875 / Tbr - 13.4721 * math.log(Tbr) + 0.43577 * Tbr ** 6
        omega = num / den
    else:           # Kesler-Lee (1976)
        omega = -7.904 + 0.1352 * Kw - 0.007465 * Kw ** 2 + 8.359 * Tbr \
            + (1.408 - 0.01063 * Kw) / Tbr
    Pc = Pc_bar * 1e5
    Zc = 0.2905 - 0.085 * omega
    Vc = Zc * R * Tc / Pc
    # Kesler-Lee (1976) ideal-gas cp [kJ/kg/K], T in K (Riazi 2005 Eq. 7.37)
    A0 = -1.41779 + 0.11828 * Kw
    A1 = -(6.99724 - 8.69326 * Kw + 0.27715 * Kw ** 2) * 1e-4
    A2 = -2.2582e-6
    B0 = 1.09223 - 2.48245 * omega
    B1 = -(3.434 - 7.14 * omega) * 1e-3
    B2 = -(7.2661 - 9.2561 * omega) * 1e-7
    C = ((12.8 - Kw) * (10.0 - Kw) / (10.0 * omega)) ** 2 if 10.0 < Kw < 12.8 else 0.0
    p = np.array([A0 - C * B0, A1 - C * B1, A2 - C * B2]) * M   # J/mol/K polynomial in T
    Th = 1200.0
    if p[2] < 0:
        Th = min(Th, -p[1] / (2 * p[2]))
    return dict(Tc=Tc, Pc=Pc, omega=omega, M=M / 1000.0, Vc=Vc, Tb=Tb, sg=sg, Kw=Kw,
                poly=p, Tl=100.0, Th=Th, parachor=-11.4 + 3.23 * M - 0.0022 * M * M,
                pseudo=True)


def parse_vessfire_fluid(lines) -> tuple[dict, dict]:
    """Parse VessFire '#Fluid' lines ('NAME frac' or 'NAME frac SG Tb')."""
    comp, pseudo = {}, {}
    for ln in lines:
        t = ln.split("%")[0].split()
        if len(t) < 2 or t[0].startswith("#"):
            continue
        comp[t[0].upper()] = float(t[1])
        if len(t) >= 4:
            pseudo[t[0].upper()] = {"sg": float(t[2]), "Tb": float(t[3])}
    return comp, pseudo


def _cubic_roots(A: float, B: float) -> list:
    """Real roots Z > B of the PR cubic, sorted ascending (Cardano/trig + Newton polish)."""
    c2 = B - 1.0
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B * B * B)
    q = (c2 * c2 - 3.0 * c1) / 9.0
    r = (2.0 * c2 ** 3 - 9.0 * c2 * c1 + 27.0 * c0) / 54.0
    q3 = q * q * q
    if r * r < q3:
        th = math.acos(max(-1.0, min(1.0, r / math.sqrt(q3))))
        s = -2.0 * math.sqrt(q)
        roots = [s * math.cos(th / 3.0) - c2 / 3.0,
                 s * math.cos((th + 2 * math.pi) / 3.0) - c2 / 3.0,
                 s * math.cos((th - 2 * math.pi) / 3.0) - c2 / 3.0]
    else:
        a_ = -math.copysign((abs(r) + math.sqrt(r * r - q3)) ** (1.0 / 3.0), r)
        b_ = q / a_ if a_ != 0.0 else 0.0
        roots = [a_ + b_ - c2 / 3.0]
    out = []
    for Z in roots:
        for _ in range(2):
            f = ((Z + c2) * Z + c1) * Z + c0
            df = (3.0 * Z + 2.0 * c2) * Z + c1
            if df != 0.0:
                Z -= f / df
        if Z > B:
            out.append(Z)
    if not out:
        out = [max(max(roots), B * (1 + 1e-12) + 1e-300)]
    out.sort()
    return out


def _rr(z, K, beta0=None):
    """Rachford-Rice with negative-flash window; returns beta (may be <0 or >1)."""
    Km1 = K - 1.0
    kmax = K.max()
    kmin = K.min()
    if kmax <= 1.0:
        return -1.0
    if kmin >= 1.0:
        return 2.0
    lo = 1.0 / (1.0 - kmax)
    hi = 1.0 / (1.0 - kmin)
    b = 0.5 if (beta0 is None or not (lo < beta0 < hi)) else beta0
    for _ in range(200):
        den = 1.0 + b * Km1
        t = z * Km1 / den
        f = t.sum()
        df = -(t * Km1 / den).sum()
        if abs(f) < 1e-15:
            return b
        if f > 0:
            lo = b
        else:
            hi = b
        bn = b - f / df
        if not (lo <= bn <= hi):
            bn = 0.5 * (lo + hi)
        if abs(bn - b) < 1e-14:
            return bn
        b = bn
    return b


def _solve1d(f, x0, dx0, xmin, xmax, incr, xtol, ftol, maxit=80):
    """Monotonic 1-D root finder: secant bracketing then Illinois.
    f(x) -> (value, obj).  Returns (x, fx, obj, (xa, fa, oa), (xb, fb, ob))."""
    fa, oa = f(x0)
    xa = x0
    if abs(fa) < ftol:
        return xa, fa, oa, (xa, fa, oa), (xa, fa, oa)
    xb = min(max(x0 + dx0, xmin), xmax)
    if xb == xa:
        xb = min(max(x0 - dx0, xmin), xmax)
    fb, ob = f(xb)
    it = 0
    while fa * fb > 0:
        if abs(fb) < ftol:
            return xb, fb, ob, (xb, fb, ob), (xb, fb, ob)
        it += 1
        if it > 60:
            raise RuntimeError("could not bracket root")
        d = -1.0 if (fb > 0) == incr else 1.0
        prev = abs(xb - xa)
        sl = (fb - fa) / (xb - xa) if xb != xa else 0.0
        if sl != 0.0 and (sl > 0) == incr:
            mag = min(max(abs(fb / sl) * 1.3, 0.5 * prev), 4.0 * prev)
        else:
            mag = 2.0 * prev
        xn = min(max(xb + d * mag, xmin), xmax)
        if xn == xb:
            raise RuntimeError("root outside bounds")
        xa, fa, oa = xb, fb, ob
        xb = xn
        fb, ob = f(xb)
    # Illinois
    fa_eff = fa
    fa_true, oa_true = fa, oa
    best = (xb, fb, ob) if abs(fb) < abs(fa) else (xa, fa, oa)
    for _ in range(maxit):
        if abs(best[1]) < ftol or abs(xb - xa) < xtol:
            break
        xn = xb - fb * (xb - xa) / (fb - fa_eff)
        lo_, hi_ = min(xa, xb), max(xa, xb)
        if not (lo_ < xn < hi_):
            xn = 0.5 * (xa + xb)
        fn, on = f(xn)
        if abs(fn) < abs(best[1]):
            best = (xn, fn, on)
        if fn * fb < 0:
            xa, fa_eff, fa_true, oa_true = xb, fb, fb, ob
        else:
            fa_eff *= 0.5
        xb, fb, ob = xn, fn, on
    return best[0], best[1], best[2], (xa, fa_true, oa_true), (xb, fb, ob)


# ----------------------------------------------------------------------------- results
class Phase:
    """One equilibrium phase.  Molar properties per mol of this phase."""
    __slots__ = ("name", "beta", "x", "T", "P", "Z", "v", "M", "h", "u", "s", "cp", "cv",
                 "w", "dPdT_v", "dPdv_T", "mu", "k")

    def __init__(self, **kw):
        self.mu = None
        self.k = None
        for k_, v_ in kw.items():
            setattr(self, k_, v_)

    def copy(self, beta=None):
        p = Phase(**{s: getattr(self, s) for s in self.__slots__})
        if beta is not None:
            p.beta = beta
        return p

    @property
    def rho(self):          # kg/m3
        return self.M / self.v

    @property
    def rho_mol(self):      # mol/m3
        return 1.0 / self.v

    @property
    def h_mass(self):
        return self.h / self.M

    @property
    def u_mass(self):
        return self.u / self.M

    @property
    def s_mass(self):
        return self.s / self.M

    @property
    def cp_mass(self):
        return self.cp / self.M

    @property
    def cv_mass(self):
        return self.cv / self.M

    def __repr__(self):
        return (f"<Phase {self.name} beta={self.beta:.6g} rho={self.rho:.4g} kg/m3 "
                f"M={self.M * 1e3:.4g} g/mol Z={self.Z:.5g}>")


class FlashResult:
    """Result of a flash.  beta = phase mole fractions of the feed; totals per mol feed."""

    def __init__(self, T, P, z, phases, kind="", K=None, names=None):
        self.T = T
        self.P = P
        self.z = z
        self.phases = phases
        self.kind = kind
        self.K = K
        self.names = names
        self.N = 1.0
        self.sigma = None
        self.info = {}

    def _get(self, name):
        for p in self.phases:
            if p.name == name:
                return p
        return None

    @property
    def vapour(self):
        return self._get("vapour")

    @property
    def liquid(self):
        return self._get("liquid")

    @property
    def aqueous(self):
        return self._get("aqueous")

    def beta_of(self, name):
        return sum(p.beta for p in self.phases if p.name == name)

    @property
    def beta_V(self):
        return self.beta_of("vapour")

    @property
    def beta_L(self):
        return self.beta_of("liquid")

    @property
    def beta_W(self):
        return self.beta_of("aqueous")

    def _tot(self, attr):
        return sum(p.beta * getattr(p, attr) for p in self.phases)

    @property
    def v(self):
        return self._tot("v")

    @property
    def h(self):
        return self._tot("h")

    @property
    def u(self):
        return self._tot("u")

    @property
    def s(self):
        return self._tot("s")

    @property
    def M(self):
        return self._tot("M")

    @property
    def rho(self):
        return self.M / self.v

    @property
    def phase_names(self):
        return tuple(p.name for p in self.phases)

    def mass_fractions(self):
        m = {p.name: p.beta * p.M for p in self.phases}
        tot = sum(m.values())
        return {k: v / tot for k, v in m.items()}

    def volume_fractions(self):
        vv = {p.name: p.beta * p.v for p in self.phases}
        tot = sum(vv.values())
        return {k: v / tot for k, v in vv.items()}

    def summary(self):
        s = [f"T={self.T:.3f} K  P={self.P / 1e5:.5g} bar  phases={self.phase_names}"]
        for p in self.phases:
            comp = " ".join(f"{n}={x:.4g}" for n, x in zip(self.names, p.x) if x > 1e-8)
            s.append(f"  {p.name:8s} beta={p.beta:.5f} rho={p.rho:9.3f} M={p.M * 1e3:8.3f} "
                     f"Z={p.Z:.4f}  {comp}")
        return "\n".join(s)

    def __repr__(self):
        return f"<FlashResult {self.phase_names} T={self.T:.2f} P={self.P:.5g}>"


# ----------------------------------------------------------------------------- model
class PRMixture:
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

    def __init__(self, composition, pseudo=None, kij=None, volume_shift=False,
                 free_water=True, water_model="iapws"):
        pseudo = {k.upper(): v for k, v in (pseudo or {}).items()}
        if isinstance(composition, dict):
            items = [(k.upper(), float(v)) for k, v in composition.items()]
        else:
            items = [(k.upper(), 0.0) for k in composition]
        names, zs, data = [], [], []
        for nm, x in items:
            nm = ALIASES.get(nm, nm)
            if nm in pseudo:
                ps = pseudo[nm]
                sg, Tb = (ps["sg"], ps["Tb"]) if isinstance(ps, dict) else ps
                d = characterise_pseudo(sg, Tb)
            elif nm in COMPONENTS:
                d = dict(COMPONENTS[nm])
                d["pseudo"] = False
                d["parachor"] = PARACHOR.get(nm, 0.0)
            else:
                raise ValueError(f"unknown component {nm} (give pseudo properties)")
            names.append(nm)
            zs.append(x)
            data.append(d)
        self.names = names
        self.data = data
        self.nc = n = len(names)
        z = np.array(zs)
        self.z = z / z.sum() if z.sum() > 0 else z
        g = lambda key: np.array([d[key] for d in data], dtype=float)
        self.Tc, self.Pc, self.omega, self.Mw, self.Vc = g("Tc"), g("Pc"), g("omega"), g("M"), g("Vc")
        self.Tb = g("Tb")
        self.is_pseudo = np.array([d["pseudo"] for d in data])
        w = self.omega
        self._m = np.where(w <= 0.491, 0.37464 + 1.54226 * w - 0.26992 * w * w,
                           0.379642 + 1.48503 * w - 0.164423 * w * w + 0.016666 * w ** 3)
        self._ac = 0.45724 * (R * self.Tc) ** 2 / self.Pc
        self._sac = np.sqrt(self._ac)
        self._sTc = np.sqrt(self.Tc)
        self._b = 0.07780 * R * self.Tc / self.Pc
        # k_ij
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = self._default_kij(i, j)
        if kij:
            for (a, b_), val in kij.items():
                i = names.index(ALIASES.get(a.upper(), a.upper()))
                j = names.index(ALIASES.get(b_.upper(), b_.upper()))
                K[i, j] = K[j, i] = val
        self.kij = K
        self._omk = 1.0 - K
        # volume translation
        self.volume_shift = volume_shift
        zra = 0.29056 - 0.08775 * w
        self._cpen = 0.50033 * (0.25969 - zra) * R * self.Tc / self.Pc
        self._c = self._cpen.copy() if volume_shift else np.zeros(n)
        # ideal gas data
        th = np.array(THETA)
        self._theta = th
        self._c0 = np.array([d.get("c0", 0.0) for d in data])
        self._V = np.array([d.get("v", [0.0] * len(th)) for d in data])
        self._poly = np.array([d.get("poly", np.zeros(3)) for d in data])
        self._Tl = np.array([d.get("Tl", 100.0) for d in data])
        self._Th = np.array([d.get("Th", 1200.0) for d in data])
        self._h_off = np.zeros(n)
        self._s_off = np.zeros(n)
        _, h0, s0 = self._ig_raw(T_REF)
        self._h_off, self._s_off = h0, s0
        self._igT = None
        self._cT = None
        # water
        self.iw = names.index("H2O") if "H2O" in names else None
        self.free_water = free_water
        self.water_model = water_model.lower()
        self._ASw = None
        self._wcache = (None, None, None)
        if self.iw is not None and self.water_model == "iapws":
            try:
                import CoolProp.CoolProp as _CP
                self._CP = _CP
                self._ASw = _CP.AbstractState("HEOS", "Water")
            except Exception:       # CoolProp not available -> PR water
                self.water_model = "pr"
        self._nw = np.ones(n, bool)
        if self.iw is not None:
            self._nw[self.iw] = False
            self._ew = np.zeros(n)
            self._ew[self.iw] = 1.0
        # transport data
        self._dip = np.array([DIPOLE.get(nm, 0.0) for nm in names])
        self._kap = np.array([KAPPA.get(nm, 0.0) for nm in names])
        self._par = np.array([d["parachor"] for d in data])
        Vc_cm3 = self.Vc * 1e6
        self._csig = 0.809 * Vc_cm3 ** (1.0 / 3.0)
        self._ceps = self.Tc / 1.2593
        self._cM = self.Mw * 1e3

    # ------------------------------------------------------------- data helpers
    def _group(self, i):
        nm = self.names[i]
        if self.is_pseudo[i] or nm in HEAVY_HC:
            return "C7+"
        return nm

    def _default_kij(self, i, j):
        gi, gj = self._group(i), self._group(j)
        for key in ((gi, gj), (gj, gi)):
            if key in _KIJ:
                return _KIJ[key]
        pair = {gi, gj}
        if "H2O" in pair:
            other = (pair - {"H2O"}) or {"H2O"}
            other = other.pop()
            return 0.5 if other != "H2O" else 0.0
        if "C1" in pair and "C7+" in pair:
            # Chueh & Prausnitz (1967) form, A = 0.18, B = 6 (Whitson & Brule 2000 Eq. 4.?)
            vi, vj = self.Vc[i] * 1e6, self.Vc[j] * 1e6
            t = 2.0 * (vi * vj) ** (1 / 6) / (vi ** (1 / 3) + vj ** (1 / 3))
            return 0.18 * (1.0 - t ** 6)
        return 0.0

    def _z(self, z):
        if z is None:
            z = self.z
        elif isinstance(z, dict):
            zz = np.zeros(self.nc)
            for k_, v_ in z.items():
                zz[self.names.index(ALIASES.get(k_.upper(), k_.upper()))] = v_
            z = zz
        z = np.asarray(z, float)
        return z / z.sum()

    def _wilson(self, T, P):
        return self.Pc / P * np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))

    # ------------------------------------------------------------- ideal gas
    def _ig_raw(self, T):
        Tc_ = max(T, 20.0)
        x = self._theta / Tc_
        em1 = np.expm1(x)
        E = x * x * (em1 + 1.0) / (em1 * em1)
        cp = R * (self._c0 + self._V @ E)
        h = R * (self._c0 * Tc_ + self._V @ (self._theta / em1))
        s = R * (self._c0 * math.log(Tc_) + self._V @ (x / em1 - np.log(-np.expm1(-x))))
        if self.is_pseudo.any():
            p0, p1, p2 = self._poly.T
            Tl, Th = self._Tl, self._Th
            Tq = np.clip(T, Tl, Th)
            cpl = p0 + p1 * Tl + p2 * Tl * Tl
            cph = p0 + p1 * Th + p2 * Th * Th
            cp = cp + p0 + p1 * Tq + p2 * Tq * Tq
            h = h + p0 * Tq + p1 * Tq ** 2 / 2 + p2 * Tq ** 3 / 3 \
                + cpl * (np.minimum(T, Tl) - Tl) + cph * (np.maximum(T, Th) - Th)
            s = s + p0 * np.log(Tq) + p1 * Tq + p2 * Tq ** 2 / 2 \
                + cpl * np.log(np.minimum(T, Tl) / Tl) + cph * np.log(np.maximum(T, Th) / Th)
        return cp, h - self._h_off, s - self._s_off

    def _ig(self, T):
        if T != self._igT:
            self._igC = self._ig_raw(T)
            self._igT = T
        return self._igC

    def ideal_gas(self, T):
        """Pure-component ideal-gas cp0, h0, s0(P0) arrays [J/mol(/K)]."""
        return self._ig(T)

    # ------------------------------------------------------------- EOS core
    def _eos_T(self, T):
        if T != self._cT:
            sT = math.sqrt(T)
            sa = self._sac * (1.0 + self._m * (1.0 - sT / self._sTc))
            dsa = -self._sac * self._m / (2.0 * sT * self._sTc)
            d2sa = self._sac * self._m / (4.0 * T * sT * self._sTc)
            omk = self._omk
            aij = omk * np.outer(sa, sa)
            t = np.outer(sa, dsa)
            daij = omk * (t + t.T)
            t2 = np.outer(sa, d2sa)
            d2aij = omk * (t2 + t2.T + 2.0 * np.outer(dsa, dsa))
            self._cA = (aij, daij, d2aij)
            self._cT = T
        return self._cA

    @staticmethod
    def _zsel(A, B, phase):
        roots = _cubic_roots(A, B)
        if len(roots) == 1:
            return roots[0]
        if phase == "L":
            return roots[0]
        if phase == "V":
            return roots[-1]
        best, gbest = None, None
        for Z in (roots[0], roots[-1]):
            g = Z - 1.0 - math.log(Z - B) - A / (2 * SQ2 * B) * math.log((Z + D1 * B) / (Z + D2 * B))
            if gbest is None or g < gbest:
                best, gbest = Z, g
        return best

    def _lnphi(self, x, T, P, phase=None, Z=None):
        aij = self._eos_T(T)[0]
        ais = aij @ x
        a = x @ ais
        b = x @ self._b
        RT = R * T
        A = a * P / (RT * RT)
        B = b * P / RT
        if Z is None:
            Z = self._zsel(A, B, phase)
        L = math.log((Z + D1 * B) / (Z + D2 * B))
        bb = self._b / b
        lnphi = bb * (Z - 1.0) - math.log(Z - B) - A / (2 * SQ2 * B) * (2.0 * ais / a - bb) * L
        return lnphi, Z

    def lnphi(self, x, T, P, phase=None):
        """ln fugacity coefficients and Z (phase None = min-Gibbs root, 'L' or 'V')."""
        return self._lnphi(self._z(x), T, P, phase)

    def _lnphi_jac(self, x, T, P, phase=None, Z=None):
        """ln phi and J_ij = n (d ln phi_i / d n_j)_{T,P} (n = 1).  Michelsen & Mollerup
        (2007) Ch. 2-3, reduced residual Helmholtz energy F = -n g - D(T) f / T."""
        aij = self._eos_T(T)[0]
        bi = self._b
        ais = aij @ x
        a = x @ ais
        b = x @ bi
        RT = R * T
        A = a * P / (RT * RT)
        B = b * P / RT
        if Z is None:
            Z = self._zsel(A, B, phase)
        v = Z * RT / P
        vmb = v - b
        vd1 = v + D1 * b
        vd2 = v + D2 * b
        g = math.log(1.0 - b / v)
        gB = -1.0 / vmb
        gV = b / (v * vmb)
        gBB = -1.0 / vmb ** 2
        gBV = 1.0 / vmb ** 2
        gVV = -1.0 / vmb ** 2 + 1.0 / v ** 2
        f = math.log(vd1 / vd2) / (R * b * (D1 - D2))
        fV = -1.0 / (R * vd1 * vd2)
        fB = -(f + v * fV) / b
        fVV = (1.0 / vd1 + 1.0 / vd2) / (R * vd1 * vd2)
        fBV = -(2.0 * fV + v * fVV) / b
        fBB = -(2.0 * fB + v * fBV) / b
        D = a
        FB = -gB - D * fB / T
        FD = -f / T
        Di = 2.0 * ais
        Fi = -g + FB * bi + FD * Di
        lnphi = Fi - math.log(Z)
        FnB = -gB
        FBD = -fB / T
        FBB = -gBB - D * fBB / T
        ob = np.outer(bi, Di)
        Fij = FnB * (bi[:, None] + bi[None, :]) + FBD * (ob + ob.T) + FBB * np.outer(bi, bi) \
            + FD * 2.0 * aij
        FiV = -gV + (-gBV - D * fBV / T) * bi + (-fV / T) * Di
        FVV = -gVV - D * fVV / T
        dPdV = -RT * FVV - RT / v ** 2
        dPdn = -RT * FiV + RT / v
        J = Fij + 1.0 + np.outer(dPdn, dPdn) / (RT * dPdV)
        return lnphi, J, Z

    def _label(self, x, v_eos):
        b = x @ self._b
        if v_eos > b / B_VC_PR:
            return "vapour"
        if self.iw is not None and x[self.iw] > 0.5:
            return "aqueous"
        return "liquid"

    def _phase(self, x, T, P, phase=None, Z=None, name=None, beta=1.0):
        aij, daij, d2aij = self._eos_T(T)
        ais = aij @ x
        a = x @ ais
        da = x @ daij @ x
        d2a = x @ d2aij @ x
        b = x @ self._b
        RT = R * T
        if Z is None:
            Z = self._zsel(a * P / (RT * RT), b * P / RT, phase)
        v = Z * RT / P
        vd1 = v + D1 * b
        vd2 = v + D2 * b
        L = math.log(vd1 / vd2)
        k2 = 1.0 / (2.0 * SQ2 * b)
        ures = (T * da - a) * k2 * L
        hres = ures + P * v - RT
        sres = (R * math.log(Z * (v - b) / v) if Z > 0 else float("nan")) + da * k2 * L
        cvres = T * d2a * k2 * L
        dPdT = R / (v - b) - da / (vd1 * vd2)
        dPdv = -RT / (v - b) ** 2 + 2.0 * a * (v + b) / (vd1 * vd2) ** 2
        cp0_i, h0_i, s0_i = self._ig(T)
        xp = x[x > 0]
        s_ig = x @ s0_i - R * math.log(abs(P) / P_REF) - R * float(np.sum(xp * np.log(xp)))
        cv = x @ cp0_i - R + cvres
        cp = cv - T * dPdT ** 2 / dPdv
        h = x @ h0_i + hres
        s = s_ig + sres
        c = x @ self._c
        vt = v - c
        h -= P * c
        u = h - P * vt
        M = x @ self.Mw
        w2 = -vt * vt / M * cp / cv * dPdv
        if name is None:
            name = self._label(x, v)
        return Phase(name=name, beta=beta, x=x, T=T, P=P, Z=Z, v=vt, M=M, h=h, u=u, s=s,
                     cp=cp, cv=cv, w=math.sqrt(w2) if w2 > 0 else float("nan"),
                     dPdT_v=dPdT, dPdv_T=dPdv)

    def phase_props(self, x, T, P, phase=None):
        """Single-phase properties at (T,P) for composition x (root: None/'L'/'V')."""
        return self._phase(self._z(x), T, P, phase)

    def state_TV(self, x, T, v):
        """Single-phase properties at (T, true molar volume v)."""
        x = self._z(x)
        veos = v + x @ self._c
        aij = self._eos_T(T)[0]
        a = x @ aij @ x
        b = x @ self._b
        P = R * T / (veos - b) - a / ((veos + D1 * b) * (veos + D2 * b))
        return self._phase(x, T, P, Z=P * veos / (R * T))

    # ------------------------------------------------------------- stability
    def stability(self, z, T, P, lnphi_z=None, extra_trials=(), tol=1e-9, skip_aqueous=False):
        """Michelsen tangent-plane stability.  Returns (stable, w, tm, Zw).
        skip_aqueous: ignore trial phases that converge to a water-rich (x_w > 0.5)
        phase - used in free-water mode, where that phase is the (pure) aqueous phase."""
        zf = np.maximum(np.asarray(z, float), ZFLOOR)
        if lnphi_z is None:
            lnphi_z, _ = self._lnphi(zf, T, P)
        d = np.log(zf) + lnphi_z
        Kw = self._wilson(T, P)
        trials = [zf * Kw, zf / Kw]
        if skip_aqueous:      # water-free hydrocarbon trials (liquid-like first)
            for t in (zf / Kw, zf * Kw):
                t = t.copy()
                t[self.iw] = ZFLOOR
                trials.append(t)
        trials += [np.maximum(np.asarray(t, float), ZFLOOR) for t in extra_trials]
        best = (0.0, None, None)
        for W0 in trials:
            tm, w, Zw = self._stab_trial(d, np.log(W0), T, P, zf, tol)
            if skip_aqueous and w is not None and w[self.iw] > 0.5:
                continue
            if w is not None and tm < best[0]:
                best = (tm, w, Zw)
                if tm < -1e-3:
                    break
        tm, w, Zw = best
        return (w is None or tm > -1e-10), w, tm, Zw

    def _stab_trial(self, d, lnW, T, P, zf, tol):
        lnz = np.log(zf)
        Zw = None
        G_prev = None
        for it in range(80):
            W = np.exp(lnW)
            SW = W.sum()
            w = W / SW
            if it >= 6:
                lnphiw, J, Zw = self._lnphi_jac(w, T, P)
            else:
                lnphiw, Zw = self._lnphi(w, T, P)
            G = lnW + lnphiw - d
            err = np.abs(G).max()
            if err < tol:
                break
            if it >= 6 and G_prev is not None and err > 0.5 * G_prev:
                lnW = d - lnphiw            # Newton not contracting: plain SS step
                G_prev = err
                continue
            G_prev = err
            if it > 2 and np.abs(np.log(w) - lnz).max() < 1e-4:
                return 0.0, None, None               # trivial solution
            if it >= 4 and math.log(SW) > 0.1:
                break                                # clearly unstable, good enough for K init
            if it >= 6:
                sW = np.sqrt(W)
                g = sW * G
                H = np.diag(1.0 + 0.5 * G) + np.outer(sW, sW) * J / SW
                try:
                    dal = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    lnW = d - lnphiw
                    continue
                al = 2.0 * sW
                aln = al + dal
                aln = np.where(aln <= 0.0, 0.1 * al, aln)
                lnW_n = 2.0 * np.log(0.5 * aln)
                # guard: fall back to SS when Newton step is wild
                if np.abs(lnW_n - lnW).max() > 5.0:
                    lnW_n = d - lnphiw
                lnW = lnW_n
            else:
                lnW = d - lnphiw
        W = np.exp(lnW)
        SW = W.sum()
        w = W / SW
        lnphiw, Zw = self._lnphi(w, T, P)
        if np.abs(np.log(w) - lnz).max() < 1e-4:
            return 0.0, None, None
        tm = 1.0 + float(W @ (lnW + lnphiw - d - 1.0))
        return tm, w, Zw

    # ------------------------------------------------------------- two-phase flash
    def _flash2(self, z, T, P, K, maxss=12):
        """Two-phase flash: successive substitution with GDEM acceleration (Crowe &
        Nishio 1975, every 5th step), then Newton (retried every 10 SS steps if it fails)."""
        lnK = np.log(K)
        beta = None
        x = y = None
        next_newton = maxss
        d_prev = None
        for it in range(300):
            K = np.exp(lnK)
            beta = _rr(z, K, beta)
            if beta <= -0.5 or beta >= 1.5 or not np.isfinite(beta):
                if it > 3:
                    return None
            bb = min(max(beta, -0.5), 1.5)
            den = 1.0 + bb * (K - 1.0)
            x = z / den
            y = K * x
            x = x / x.sum()
            y = y / y.sum()
            lnphiL, ZL = self._lnphi(x, T, P)
            lnphiV, ZV = self._lnphi(y, T, P)
            lnK_n = lnphiL - lnphiV
            d = lnK_n - lnK
            err = np.abs(d).max()
            lnK = lnK_n
            if np.abs(lnK).max() < 1e-4:
                return None
            if err < 1e-10:
                break
            if it >= next_newton and 0.0 < beta < 1.0 and err < 0.1:
                res = self._flash2_newton(z, T, P, beta, x, y)
                if res is not None:
                    return res
                next_newton = it + 10
            if d_prev is not None and it % 5 == 4:
                den_ = float(d_prev @ d)
                lam = float(d @ d) / den_ if den_ != 0.0 else 0.0
                if 0.0 < lam < 0.98:
                    lnK = lnK + d * min(lam / (1.0 - lam), 20.0)
            d_prev = d
        beta = _rr(z, np.exp(lnK), beta)
        if not (0.0 < beta < 1.0):
            return None
        return self._pack2(z, T, P, beta, x, y)

    def _flash2_newton(self, z, T, P, beta, x, y):
        v = beta * y
        l = z - v
        if (l <= 0).any():
            return None
        err_prev = None
        for it in range(25):
            nV = v.sum()
            nL = l.sum()
            y = v / nV
            x = l / nL
            # keep each phase on its own cubic root during Newton (avoids root jumping
            # near the three-root region); the SS stage uses the min-Gibbs root
            lnphiV, JV, ZV = self._lnphi_jac(y, T, P, "V")
            lnphiL, JL, ZL = self._lnphi_jac(x, T, P, "L")
            g = np.log(y) + lnphiV - np.log(x) - lnphiL
            err = np.abs(g).max()
            if err < 1e-10:
                break
            if err_prev is not None and err > 10.0 * err_prev:
                return None
            err_prev = err
            H = (np.diag(1.0 / y) - 1.0 + JV) / nV + (np.diag(1.0 / x) - 1.0 + JL) / nL
            try:
                dv = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return None
            vn = v + dv
            vn = np.where(vn <= 0.0, 0.1 * v, vn)
            vn = np.where(vn >= z, v + 0.9 * (z - v), vn)
            v = vn
            l = z - v
        else:
            return None
        beta = v.sum()
        if not (0.0 < beta < 1.0) or np.abs(np.log(y / x)).max() < 1e-4:
            return None
        return self._pack2(z, T, P, beta, x, y)

    def _pack2(self, z, T, P, beta, x, y):
        pV = self._phase(y, T, P, name="vapour", beta=beta)
        pL = self._phase(x, T, P, name="liquid", beta=1.0 - beta)
        if pV.rho > pL.rho:
            pV, pL = pL, pV
            pV.name, pL.name = "vapour", "liquid"
        if self.iw is not None and pL.x[self.iw] > 0.5:
            pL.name = "aqueous"
        return FlashResult(T, P, z, [pV, pL], kind="std2", K=pV.x / pL.x, names=self.names)

    def _result1(self, z, T, P, Z=None, kind="std1"):
        zf = np.maximum(z, ZFLOOR) if (z <= 0).any() else z
        ph = self._phase(zf, T, P, Z=Z)
        return FlashResult(T, P, z, [ph], kind=kind, names=self.names)

    def _flash_std(self, z, T, P, init=None, same=False):
        zf = np.maximum(z, ZFLOOR)
        if same and init is not None and init.kind == "std1":
            root = "V" if init.phases[0].name == "vapour" else "L"
            _, Zs = self._lnphi(zf, T, P, root)
            return self._result1(z, T, P, Zs)
        if init is not None and init.kind == "std2" and init.K is not None:
            res = self._flash2(zf, T, P, init.K, maxss=1)
            if res is not None:
                return res
        lnphiz, Zz = self._lnphi(zf, T, P)
        extra = [p.x for p in init.phases] if init is not None else ()
        stable, w, tm, Zw = self.stability(zf, T, P, lnphi_z=lnphiz, extra_trials=extra)
        if stable:
            return self._result1(z, T, P, Zz)
        K = w / zf if Zw > Zz else zf / w
        res = self._flash2(zf, T, P, K)
        if res is None:   # try the other trial type once (Wilson)
            res = self._flash2(zf, T, P, self._wilson(T, P))
        if res is None:
            r = self._result1(z, T, P, Zz)
            r.info["warning"] = "stability test unstable but flash converged to trivial"
            return r
        return res

    # ------------------------------------------------------------- free water
    def _water_iapws(self, T, P):
        """Pure water at (T,P) from IAPWS-95 (CoolProp): returns (lnphi, Phase-kwargs)."""
        if self._wcache[0] == T and self._wcache[1] == P:
            return self._wcache[2]
        AS, CP_ = self._ASw, self._CP
        TM = 273.16
        if T < TM:
            # Below the triple point free water would freeze (ice / hydrates are NOT
            # modelled): treat it as supercooled liquid with properties extrapolated
            # from 273.16 K at constant cp and density; ln f from the Gibbs-Helmholtz
            # relation with constant residual enthalpy.
            lnphim, kw = self._water_iapws(TM, P)
            hres = kw["h"] - self._ig(TM)[1][self.iw]
            lnf = lnphim + math.log(P) + hres / R * (1.0 / T - 1.0 / TM)
            kw = dict(kw)
            kw["h"] = kw["h"] + kw["cp"] * (T - TM)
            kw["s"] = kw["s"] + kw["cp"] * math.log(T / TM)
            kw["u"] = kw["h"] - P * kw["v"]
            kw["Z"] = P * kw["v"] / (R * T)
            out = (lnf - math.log(P), kw)
            self._wcache = (T, P, out)
            return out
        try:
            AS.update(CP_.PT_INPUTS, P, T)
        except ValueError:
            # (T, P) exactly on the saturation line: evaluate just on the liquid side
            T = T * (1.0 - 2e-6)
            AS.update(CP_.PT_INPUTS, P, T)
        Z = AS.compressibility_factor()
        lnphi = AS.alphar() + Z - 1.0 - math.log(Z)
        cp0_i, h0_i, s0_i = self._ig(T)
        iw = self.iw
        v = 1.0 / AS.rhomolar()
        h = h0_i[iw] + AS.hmolar_residual()
        s = s0_i[iw] - R * math.log(P / P_REF) + AS.smolar_residual() + R * math.log(Z)
        cp0w = AS.cp0molar()
        cp = AS.cpmolar() - cp0w + cp0_i[iw]
        cv = AS.cvmolar() - cp0w + cp0_i[iw]
        kw = dict(Z=Z, v=v, M=self.Mw[iw], h=h, u=h - P * v, s=s, cp=cp, cv=cv,
                  w=AS.speed_sound(), dPdT_v=float("nan"), dPdv_T=AS.first_partial_deriv(
                      CP_.iP, CP_.iDmolar, CP_.iT) * (-1.0 / v ** 2))
        out = (lnphi, kw)
        self._wcache = (T, P, out)
        return out

    def _aq_phase(self, T, P, beta, Zw0=None):
        if self.water_model == "iapws":
            kw = self._water_iapws(T, P)[1]
            return Phase(name="aqueous", beta=beta, x=self._ew.copy(), T=T, P=P, **kw)
        return self._phase(self._ew.copy(), T, P, Z=Zw0, name="aqueous", beta=beta)

    def _lnfw0(self, T, P):
        if self.water_model == "iapws":
            return self._water_iapws(T, P)[0] + math.log(P), None
        lnphi, Zw = self._lnphi(self._ew, T, P)
        return lnphi[self.iw] + math.log(P), Zw

    def _fw_single(self, z, T, P, lnfw0):
        iw = self.iw
        S = z[self._nw].sum()
        lnP = math.log(P)
        yw = min(math.exp(lnfw0 - lnP), 0.9)
        y = z.copy()
        for _ in range(100):
            y = z.copy()
            y[self._nw] *= (1.0 - yw) / S
            y[iw] = yw
            lnphi, Zy = self._lnphi(y, T, P)
            ywn = math.exp(lnfw0 - lnP - lnphi[iw])
            if ywn >= 1.0:
                return None
            if abs(ywn - yw) <= 1e-12 * yw:
                yw = ywn
                break
            yw = ywn
        y = z.copy()
        y[self._nw] *= (1.0 - yw) / S
        y[iw] = yw
        lnphi, Zy = self._lnphi(y, T, P)
        bHC = S / (1.0 - yw)
        bW = 1.0 - bHC
        if bW <= 1e-14:
            return None
        return y, bHC, bW, lnphi, Zy

    @staticmethod
    def _fw_rr(zn, Kn, xw, yw, bV, bL):
        for _ in range(60):
            den = bL + bV * Kn
            if (den <= 0).any():
                return False, bV, bL
            t = zn / den
            E1 = t.sum() + xw - 1.0
            E2 = (Kn * t).sum() + yw - 1.0
            t2 = t / den
            a11 = -t2.sum()
            a12 = -(t2 * Kn).sum()
            a22 = -(t2 * Kn * Kn).sum()
            det = a11 * a22 - a12 * a12
            if det == 0.0:
                return False, bV, bL
            dL = (-E1 * a22 + E2 * a12) / det
            dV = (-E2 * a11 + E1 * a12) / det
            s = 1.0
            for _k in range(40):
                if ((bL + s * dL) + (bV + s * dV) * Kn > 0).all():
                    break
                s *= 0.5
            bL += s * dL
            bV += s * dV
            if abs(E1) + abs(E2) < 1e-14:
                return True, bV, bL
        return abs(E1) + abs(E2) < 1e-9, bV, bL

    def _fw_two(self, z, T, P, lnfw0, K, betas=None):
        iw, nw = self.iw, self._nw
        zn = z[nw]
        Kn = np.maximum(K[nw], 1e-300)
        lnP = math.log(P)
        yw = min(math.exp(lnfw0 - lnP), 0.5)
        xw = 1e-3 * yw
        S = zn.sum()
        bV, bL = betas if betas is not None else (0.5 * S, 0.5 * S)
        x = y = None
        next_newton = 3
        for it in range(500):
            ok, bV, bL = self._fw_rr(zn, Kn, xw, yw, bV, bL)
            if not ok:
                return None
            den = bL + bV * Kn
            x = np.empty_like(z)
            y = np.empty_like(z)
            x[nw] = zn / den
            y[nw] = Kn * zn / den
            x[iw] = xw
            y[iw] = yw
            x /= x.sum()
            y /= y.sum()
            lnphiL, _ = self._lnphi(x, T, P)
            lnphiV, _ = self._lnphi(y, T, P)
            Kn_n = np.exp(lnphiL[nw] - lnphiV[nw])
            xw_n = math.exp(lnfw0 - lnP - lnphiL[iw])
            yw_n = math.exp(lnfw0 - lnP - lnphiV[iw])
            err = max(np.abs(np.log(Kn_n / Kn)).max(), abs(math.log(xw_n / xw)),
                      abs(math.log(yw_n / yw)))
            Kn, xw, yw = Kn_n, min(xw_n, 0.999), min(yw_n, 0.999)
            if np.abs(np.log(Kn)).max() < 1e-4:
                return None
            if err < 1e-10:
                break
            if it > 30 and (bV < -0.2 or bL < -0.2):
                break
            if it >= next_newton and bV > 0 and bL > 0 and err < 0.1:
                res = self._fw_newton(z, T, P, lnfw0, bV * y, bL * x)
                if res is not None:
                    return res
                next_newton = it + 15
        bW = 1.0 - bV - bL
        return dict(bV=bV, bL=bL, bW=bW, x=x, y=y)

    def _fw_newton(self, z, T, P, lnfw0, v, l):
        """Newton for vapour + HC liquid + pure water.  Unknowns: vapour moles v_i (all
        components) and water moles in the HC liquid l_w; l_i = z_i - v_i (i != w).
        Equations: ln f_i^V = ln f_i^L (i != w), ln f_w^V = ln f_w^L = ln f_w(pure)."""
        iw, nw = self.iw, self._nw
        n = self.nc
        lnP = math.log(P)
        v = v.copy()
        lw = l[iw]
        idx = np.flatnonzero(nw)
        err_prev = None
        for it in range(25):
            l = z - v
            l[iw] = lw
            nV, nL = v.sum(), l.sum()
            y, x = v / nV, l / nL
            lnphiV, JV, _ = self._lnphi_jac(y, T, P, "V")
            lnphiL, JL, _ = self._lnphi_jac(x, T, P, "L")
            lfV = np.log(y) + lnphiV
            lfL = np.log(x) + lnphiL
            g = np.empty(n + 1)
            g[:n - 1] = (lfV - lfL)[idx]
            g[n - 1] = lfV[iw] + lnP - lnfw0
            g[n] = lfL[iw] + lnP - lnfw0
            err = np.abs(g).max()
            if err < 1e-10:
                break
            if err_prev is not None and err > 10.0 * err_prev:
                return None
            err_prev = err
            FV = (np.diag(1.0 / y) - 1.0 + JV) / nV
            FL = (np.diag(1.0 / x) - 1.0 + JL) / nL
            Jm = np.zeros((n + 1, n + 1))
            FLc = FL.copy()
            FLc[:, iw] = 0.0                     # l_w does not depend on v_w
            Jm[:n - 1, :n] = (FV + FLc)[idx]
            Jm[:n - 1, n] = -FL[idx, iw]
            Jm[n - 1, :n] = FV[iw]
            Jm[n, :n] = -FLc[iw]
            Jm[n, n] = FL[iw, iw]
            try:
                du = np.linalg.solve(Jm, -g)
            except np.linalg.LinAlgError:
                return None
            vn = v + du[:n]
            lwn = lw + du[n]
            vn = np.where(vn <= 0.0, 0.1 * v, vn)
            upper = z.copy()
            vn[nw] = np.where(vn[nw] >= upper[nw], v[nw] + 0.9 * (upper[nw] - v[nw]), vn[nw])
            lwn = lwn if lwn > 0.0 else 0.1 * lw
            v, lw = vn, lwn
        else:
            return None
        l = z - v
        l[iw] = lw
        bV, bL = v.sum(), l.sum()
        return dict(bV=bV, bL=bL, bW=1.0 - bV - bL, x=l / bL, y=v / bV)

    def _flash_fw(self, z, T, P, init=None, same=False):
        iw = self.iw
        zf = np.maximum(z, ZFLOOR)
        lnfw0, Zw0 = self._lnfw0(T, P)

        def pack(hc_phases, bW, kind, K=None):
            phases = list(hc_phases)
            if bW > 0:
                pw = self._aq_phase(T, P, bW, Zw0)
                phases.append(pw)
            return FlashResult(T, P, z, phases, kind=kind, K=K, names=self.names)

        def three(res):
            pV = self._phase(res["y"], T, P, name="vapour", beta=res["bV"])
            pL = self._phase(res["x"], T, P, name="liquid", beta=res["bL"])
            if pV.rho > pL.rho:
                pV, pL = pL, pV
                pV.name, pL.name = "vapour", "liquid"
            return pack([pV, pL], res["bW"], "fw2", K=pV.x / pL.x)

        def ok3(res):
            return res is not None and res["bV"] > 0 and res["bL"] > 0 and res["bW"] > 0

        if init is not None and init.kind == "fw2" and init.K is not None:
            v0 = init.beta_V * init.vapour.x
            l0 = init.beta_L * init.liquid.x
            res = None
            if (v0[self._nw] < zf[self._nw]).all():
                res = self._fw_newton(zf, T, P, lnfw0, v0, l0)
            if not ok3(res):
                res = self._fw_two(zf, T, P, lnfw0, init.K, (init.beta_V, init.beta_L))
            if ok3(res):
                return three(res)
        single = self._fw_single(zf, T, P, lnfw0)
        if single is None:
            return self._flash_std(z, T, P, init if (init is not None and init.kind == "std2") else None)
        y, bHC, bW, lnphiy, Zy = single
        if same and init is not None and init.kind == "fw1":
            ph = self._phase(y, T, P, Z=Zy, beta=bHC)
            if ph.name == "aqueous":
                ph.name = "liquid"
            return pack([ph], bW, "fw1")
        extra = [p.x for p in init.phases if p.name != "aqueous"] if init is not None else ()
        stable, w, tm, Zw = self.stability(y, T, P, lnphi_z=lnphiy, extra_trials=extra,
                                           skip_aqueous=True)
        if stable:
            ph = self._phase(y, T, P, Z=Zy, beta=bHC)
            if ph.name == "aqueous":
                ph.name = "liquid"
            return pack([ph], bW, "fw1")
        K = w / y if Zw > Zy else y / w
        res = self._fw_two(zf, T, P, lnfw0, K)
        if ok3(res):
            return three(res)
        if res is not None and res["bW"] <= 0:
            std = self._flash_std(z, T, P)
            fw_ok = all(math.log(max(p.x[iw], 1e-300)) + self._lnphi(p.x, T, P, Z=p.Z)[0][iw]
                        + math.log(P) <= lnfw0 + 1e-8 for p in std.phases)
            if fw_ok:
                return std
        ph = self._phase(y, T, P, Z=Zy, beta=bHC)
        r = pack([ph], bW, "fw1")
        r.info["warning"] = "HC phase unstable but 3-phase free-water flash failed"
        return r

    # ------------------------------------------------------------- public flashes
    def flash_PT(self, P, T, z=None, init=None, same_phases=False):
        """Isothermal flash.  init: previous FlashResult for warm start.
        same_phases=True (internal, for derivatives): assume init's phase set, no
        stability test."""
        z = self._z(z)
        if (self.iw is not None and self.free_water and 1e-12 < z[self.iw] < 1.0 - 1e-12):
            return self._flash_fw(z, T, P, init, same_phases)
        if self.iw is not None and self.free_water and z[self.iw] >= 1.0 - 1e-12:
            # pure water: use the same aqueous (IAPWS) model as the free-water phase, so a
            # water pool is thermodynamically identical whether or not traces of other
            # components are present (PR water differs by ~1.7 kJ/mol in enthalpy)
            aq = self._aq_phase(T, P, 1.0)
            return FlashResult(T, P, z, [aq], kind="water1", names=self.names)
        return self._flash_std(z, T, P, init, same_phases)

    def _uv_single(self, z, u, veos, T):
        b = z @ self._b
        if veos <= b * (1.0 + 1e-9):
            return T, -1.0, None, False
        for _ in range(60):
            aij, daij, d2aij = self._eos_T(T)
            a = z @ aij @ z
            da = z @ daij @ z
            d2a = z @ d2aij @ z
            L = math.log((veos + D1 * b) / (veos + D2 * b))
            k2 = 1.0 / (2.0 * SQ2 * b)
            cp0_i, h0_i, _ = self._ig(T)
            uc = z @ h0_i - R * T + (T * da - a) * k2 * L
            cv = z @ cp0_i - R + T * d2a * k2 * L
            dT = -(uc - u) / cv
            dT = max(-100.0, min(100.0, dT))
            Tn = max(T + dT, 20.0)
            if abs(Tn - T) < 1e-10 * T:
                T = Tn
                break
            T = Tn
        aij = self._eos_T(T)[0]
        a = z @ aij @ z
        vd1, vd2 = veos + D1 * b, veos + D2 * b
        P = R * T / (veos - b) - a / (vd1 * vd2)
        dPdv = -R * T / (veos - b) ** 2 + 2.0 * a * (veos + b) / (vd1 * vd2) ** 2
        ok = P > 0 and dPdv < 0
        return T, P, (P * veos / (R * T) if ok else None), ok

    def flash_UV(self, U, V, n, T_guess=300.0, P_guess=None, init=None, tol=1e-9):
        """Given total internal energy U [J], volume V [m3], moles n [mol] (array or dict)
        -> FlashResult (r.N = total moles).  init: previous result (warm start)."""
        if isinstance(n, dict):
            nn = np.zeros(self.nc)
            for k_, v_ in n.items():
                nn[self.names.index(ALIASES.get(k_.upper(), k_.upper()))] = v_
            n = nn
        n = np.asarray(n, float)
        N = n.sum()
        z = n / N
        u = U / N
        v = V / N
        veos = v + z @ self._c
        multi_init = init is not None and len(init.phases) > 1
        r = None
        T1 = init.T if init is not None else T_guess
        if not multi_init:
            r = self._uv_try_single(z, u, veos, T1, init)
            if r is not None and r is not False:
                r.N = N
                r.info["method"] = "single"
                return r
        T0 = init.T if init is not None else T_guess
        P0 = init.P if init is not None else (P_guess or (self._uvP1 if self._uvP1 and
                                                          self._uvP1 > 0 else 1e6))
        active = np.flatnonzero(z > 1e-12)
        if len(active) == 1:
            r = self._uv_pure(int(active[0]), z, u, v, T0)
            if r is None:
                r = self._uv_try_single(z, u, veos, T0, None)
                r = r if r else None
            if r is not None:
                r.N = N
                r.info.setdefault("method", "pure-saturation")
                return r
        r = self._uv_newton(z, u, v, T0, P0, init, tol)
        method = "newton"
        if r is None:
            r = self._uv_nested(z, u, v, T0, P0, init)
            method = "nested"
        if multi_init and len(r.phases) == 1:
            # phase disappeared: confirm by the single-phase route
            rs = self._uv_try_single(z, u, veos, r.T, None)
            if rs:
                r = rs
                method = "single"
        r.N = N
        r.info["method"] = method
        return r

    _uvT1 = None
    _uvP1 = None

    def _uv_try_single(self, z, u, veos, T, init):
        T1, P1, Z1, ok = self._uv_single(z, u, veos, T)
        self._uvT1, self._uvP1 = T1, (P1 if ok else None)
        if not ok:
            return False
        zf = np.maximum(z, ZFLOOR)
        lnphiz, _ = self._lnphi(zf, T1, P1, Z=Z1)
        if self.iw is not None and self.free_water and z[self.iw] > 1e-12:
            lnfw0, _ = self._lnfw0(T1, P1)
            if math.log(zf[self.iw]) + lnphiz[self.iw] + math.log(P1) > lnfw0 + 1e-10:
                return False
        extra = [p.x for p in init.phases] if init is not None else ()
        fw = self.iw is not None and self.free_water and z[self.iw] > 1e-12
        stable, w, tm, Zw = self.stability(zf, T1, P1, lnphi_z=lnphiz, extra_trials=extra,
                                           skip_aqueous=fw)
        if not stable:
            return False
        return self._result1(z, T1, P1, Z1)

    def _uv_pure(self, i, z, u, v, T0):
        """Single component inside the dome: T from u = u_L + beta (u_V - u_L), with
        beta from the lever rule on v, at P = Psat(T)."""
        x = np.zeros(self.nc)
        x[i] = 1.0
        Tmax = self.Tc[i] * 0.99999
        st = {}

        def f(T):
            Ps = self.psat(T, i)
            L = self._phase(x, T, Ps, "L", name="liquid")
            V = self._phase(x, T, Ps, "V", name="vapour")
            if self.iw == i:
                L.name = "aqueous"
            bV = (v - L.v) / (V.v - L.v)
            st[T] = (L, V, bV, Ps)
            return (L.u + bV * (V.u - L.u) - u) / (R * T0), T
        try:
            T, fT, _, _, _ = _solve1d(f, min(T0, Tmax * 0.999), -2.0, 20.0, Tmax, incr=True,
                                      xtol=1e-11, ftol=1e-12)
        except (RuntimeError, ValueError):
            return None
        L, V, bV, Ps = st[T]
        if not (0.0 <= bV <= 1.0):
            return None
        V.beta, L.beta = bV, 1.0 - bV
        return FlashResult(T, Ps, z, [p for p in (V, L) if p.beta > 0], kind="pure2",
                           names=self.names)

    def _uv_newton(self, z, u, v, T, P, init, tol):
        def ev(T_, lnP_, ini, same=False):
            r_ = self.flash_PT(math.exp(lnP_), T_, z, init=ini, same_phases=same)
            return r_, np.array([(r_.u - u) / (R * T_), math.log(r_.v / v)])
        try:
            x = np.array([T, math.log(P)])
            r, F = ev(x[0], x[1], init)
            J = None
            fails = 0
            for it in range(40):
                nF = np.abs(F).max()
                if nF < tol:
                    return r
                if J is None:
                    dT = 1e-5 * x[0]
                    dl = 1e-6
                    rT, FT = ev(x[0] + dT, x[1], r, True)
                    rP, FP = ev(x[0], x[1] + dl, r, True)
                    J = np.column_stack(((FT - F) / dT, (FP - F) / dl))
                dx = np.linalg.solve(J, -F)
                sc = max(abs(dx[0]) / 30.0, abs(dx[1]) / 0.5, 1.0)
                dx /= sc
                s = 1.0
                for k in range(8):
                    xn = x + s * dx
                    rn, Fn = ev(xn[0], xn[1], r)
                    if np.abs(Fn).max() < nF:
                        break
                    s *= 0.5
                else:
                    fails += 1
                    if fails > 3:
                        return None
                    J = None
                    continue
                sdx = s * dx
                J = J + np.outer(Fn - F - J @ sdx, sdx) / (sdx @ sdx)
                x, r, F = xn, rn, Fn
            return r if np.abs(F).max() < 1e3 * tol else None
        except (np.linalg.LinAlgError, ValueError, OverflowError, ZeroDivisionError, RuntimeError):
            return None

    def _uv_nested(self, z, u, v, T0, P0, init):
        st = {"r": init, "P": P0}

        def inner(T):
            def f(lnP):
                r = self.flash_PT(math.exp(lnP), T, z, init=st["r"])
                st["r"] = r
                return math.log(r.v / v), r
            x, fx, r, A, B = _solve1d(f, math.log(st["P"]), 0.3, math.log(1e-3), math.log(5e9),
                                      incr=False, xtol=1e-13, ftol=1e-11)
            if abs(fx) > 1e-9:
                r = self._blend_to(A, B, lambda q: q.v, v)
            st["P"] = r.P
            return r

        def outer(T):
            r = inner(T)
            return (r.u - u) / (R * T0), r
        T, fT, r, A, B = _solve1d(outer, T0, 2.0, 20.0, 3000.0, incr=True, xtol=1e-10, ftol=1e-10)
        if abs(fT) > 1e-8:
            r = self._blend_to(A, B, lambda q: q.u, u)
        return r

    def _blend_to(self, A, B, q, target):
        ra, rb = A[2], B[2]
        qa, qb = q(ra), q(rb)
        w = 0.0 if qb == qa else min(max((target - qa) / (qb - qa), 0.0), 1.0)
        return self._blend(ra, rb, w)

    def _blend(self, ra, rb, w):
        phases = []
        for p in ra.phases:
            phases.append(p.copy(p.beta * (1.0 - w)))
        for p in rb.phases:
            q = p.copy(p.beta * w)
            for e in phases:
                if e.name == q.name and np.abs(e.x - q.x).max() < 1e-6:
                    e.beta += q.beta
                    break
            else:
                phases.append(q)
        phases = [p for p in phases if p.beta > 0]
        r = FlashResult((1 - w) * ra.T + w * rb.T, (1 - w) * ra.P + w * rb.P, ra.z, phases,
                        kind="blend", names=self.names)
        r.info["blend"] = w
        return r

    def _water_sat_phase(self, P, q):
        """Saturated pure water (q=0 liquid, q=1 vapour) at P, IAPWS via CoolProp,
        in this model's enthalpy/entropy reference (ideal-gas part from _ig)."""
        AS, CP_ = self._ASw, self._CP
        AS.update(CP_.PQ_INPUTS, P, float(q))
        T = AS.T()
        Z = P / (AS.rhomolar() * R * T)
        cp0_i, h0_i, s0_i = self._ig(T)
        iw = self.iw
        v = 1.0 / AS.rhomolar()
        h = h0_i[iw] + AS.hmolar_residual()
        s_ = s0_i[iw] - R * math.log(P / P_REF) + AS.smolar_residual() + R * math.log(Z)
        cp0w = AS.cp0molar()
        name = "vapour" if q >= 0.5 else "aqueous"
        return Phase(name=name, beta=1.0, x=self._ew.copy(), T=T, P=P, Z=Z, v=v, M=self.Mw[iw],
                     h=h, u=h - P * v, s=s_, cp=AS.cpmolar() - cp0w + cp0_i[iw],
                     cv=AS.cvmolar() - cp0w + cp0_i[iw], w=AS.speed_sound(),
                     dPdT_v=float("nan"), dPdv_T=float("nan"))

    def water_PH(self, P, H, T_guess=None):
        """Isenthalpic flash of PURE water with IAPWS (H in J/mol): subcooled liquid,
        saturated two-phase (aqueous + steam 'vapour' phase) or superheated steam."""
        L = self._water_sat_phase(P, 0.0)
        Vp = self._water_sat_phase(P, 1.0)
        z = self._ew.copy()
        if L.h < H < Vp.h:
            bv = (H - L.h) / (Vp.h - L.h)
            L.beta, Vp.beta = 1.0 - bv, bv
            return FlashResult(L.T, P, z, [Vp, L], kind="water2", names=self.names)
        # single phase: Newton on T on the correct side of T_sat
        Ts = L.T
        liquid = H <= L.h
        T = min(T_guess or Ts - 1.0, Ts * (1 - 1e-5)) if liquid else max(T_guess or Ts + 1.0, Ts * (1 + 1e-5))
        for _ in range(60):
            kw = self._water_iapws(T, P)[1]
            dT = (H - kw["h"]) / kw["cp"]
            dT = max(min(dT, 50.0), -50.0)
            Tn = T + dT
            Tn = min(Tn, Ts * (1 - 1e-6)) if liquid else max(Tn, Ts * (1 + 1e-6))
            if abs(Tn - T) < 1e-9 * T:
                T = Tn
                break
            T = Tn
        kw = self._water_iapws(T, P)[1]
        ph = Phase(name="aqueous" if liquid else "vapour", beta=1.0, x=z, T=T, P=P, **kw)
        return FlashResult(T, P, z, [ph], kind="water1", names=self.names)

    def flash_PH(self, P, H, z=None, T_guess=300.0, init=None):
        """Isenthalpic flash: H [J/mol of feed] at P."""
        zz = self._z(z)
        if self.iw is not None and self.free_water and zz[self.iw] >= 1.0 - 1e-12:
            return self.water_PH(P, H, T_guess=init.T if init is not None else None)
        return self._flash_PX(P, H, z, T_guess, init, "h")

    def flash_PS(self, P, S, z=None, T_guess=300.0, init=None):
        """Isentropic flash: S [J/mol/K of feed] at P."""
        return self._flash_PX(P, S, z, T_guess, init, "s")

    def _flash_PX(self, P, X, z, T_guess, init, which):
        z = self._z(z)
        st = {"r": init}
        T0 = init.T if init is not None else T_guess

        def f(T):
            r = self.flash_PT(P, T, z, init=st["r"])
            st["r"] = r
            val = (r.h - X) / (R * 300.0) if which == "h" else (r.s - X) / R
            return val, r
        r0 = self.flash_PT(P, T0, z, init=init)
        cp = sum(p.beta * p.cp for p in r0.phases)
        d0 = ((r0.h - X) / cp) if which == "h" else (T0 * (r0.s - X) / cp)
        dx0 = -max(-150.0, min(150.0, d0))
        if abs(dx0) < 1e-3:
            dx0 = 1e-3 if dx0 >= 0 else -1e-3
        st["r"] = r0
        T, fT, r, A, B = _solve1d(f, T0, dx0, 20.0, 3000.0, incr=True, xtol=1e-10, ftol=1e-11)
        if abs(fT) > 1e-8:
            r = self._blend_to(A, B, (lambda q: q.h) if which == "h" else (lambda q: q.s), X)
        return r

    # ------------------------------------------------------------- saturation
    def bubble_P(self, T, x=None, P_guess=None, maxit=500):
        """Bubble pressure at T; returns (P, y_incipient)."""
        x = np.maximum(self._z(x), ZFLOOR)
        e = np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))
        P = P_guess or float(x @ (self.Pc * e))
        K = self.Pc * e / P
        for it in range(maxit):
            lnphiL, _ = self._lnphi(x, T, P, "L")
            y = K * x
            y /= y.sum()
            lnphiV, _ = self._lnphi(y, T, P, "V")
            K = np.exp(lnphiL - lnphiV)
            S = float(K @ x)
            if np.abs(np.log(K)).max() < 1e-5:
                raise ValueError("bubble point: trivial solution (above cricondenbar?)")
            P *= S
            if abs(S - 1.0) < 1e-12:
                break
        y = K * x
        return P, y / y.sum()

    def dew_P(self, T, y=None, P_guess=None, maxit=500):
        """Dew pressure at T; returns (P, x_incipient)."""
        y = np.maximum(self._z(y), ZFLOOR)
        e = np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))
        P = P_guess or float(1.0 / (y @ (1.0 / (self.Pc * e))))
        K = self.Pc * e / P
        for it in range(maxit):
            lnphiV, _ = self._lnphi(y, T, P, "V")
            x = y / K
            x /= x.sum()
            lnphiL, _ = self._lnphi(x, T, P, "L")
            K = np.exp(lnphiL - lnphiV)
            S = float((y / K).sum())
            if np.abs(np.log(K)).max() < 1e-5:
                raise ValueError("dew point: trivial solution")
            P /= S
            if abs(S - 1.0) < 1e-12:
                break
        x = y / K
        return P, x / x.sum()

    def _sat_T(self, P, z, bubble, T_guess, maxit=200):
        """Bubble (bubble=True) or dew temperature at P: Newton on T for
        f(T) = ln sum(K z) (bubble) or ln sum(z/K) (dew), with the incipient-phase
        composition updated by successive substitution.  Returns (T, incipient comp)."""
        z = np.maximum(z, ZFLOOR)
        T = T_guess
        K = self._wilson(T, P)
        w = K * z if bubble else z / K
        w /= w.sum()

        def fk(T_, w_):
            if bubble:
                lnL, _ = self._lnphi(z, T_, P, "L")
                lnV, _ = self._lnphi(w_, T_, P, "V")
                K_ = np.exp(lnL - lnV)
                return math.log(float(K_ @ z)), K_
            lnL, _ = self._lnphi(w_, T_, P, "L")
            lnV, _ = self._lnphi(z, T_, P, "V")
            K_ = np.exp(lnL - lnV)
            return math.log(float((z / K_).sum())), K_
        for it in range(maxit):
            f, K = fk(T, w)
            if np.abs(np.log(K)).max() < 1e-5:
                raise ValueError("saturation T: trivial solution (no bubble/dew point at this P?)")
            wn = K * z if bubble else z / K
            wn /= wn.sum()
            dT = 1e-4 * T
            f1, _ = fk(T, wn)
            f2, _ = fk(T + dT, wn)
            df = (f2 - f1) / dT
            step = -f1 / df if df != 0 else 5.0
            step = max(-20.0, min(20.0, step))
            w = wn
            T += step
            if abs(f1) < 1e-11 and abs(step) < 1e-8 * T:
                break
        return T, w

    def bubble_T(self, P, x=None, T_guess=None):
        """Bubble temperature at P; returns (T, y_incipient)."""
        x = self._z(x)
        return self._sat_T(P, x, True, T_guess or self._wilson_T(P, x, True))

    def dew_T(self, P, y=None, T_guess=None):
        """Dew temperature at P (branch nearest T_guess / Wilson); returns (T, x_incipient)."""
        y = self._z(y)
        return self._sat_T(P, y, False, T_guess or self._wilson_T(P, y, False))

    def _wilson_T(self, P, z, bubble):
        from scipy.optimize import brentq
        def f(T):
            K = self._wilson(T, P)
            return math.log(K @ z) if bubble else -math.log((z / K).sum())
        try:
            return brentq(f, 30.0, 1500.0)
        except ValueError:
            return 300.0

    def psat(self, T, i=None):
        """Pure-component PR vapour pressure (component index i, default the only one)."""
        if i is None:
            i = int(np.argmax(self.z))
        x = np.zeros(self.nc)
        x[i] = 1.0
        P = self.Pc[i] * math.exp(5.373 * (1 + self.omega[i]) * (1 - self.Tc[i] / T))
        for _ in range(200):
            lnL, ZL = self._lnphi(x, T, P, "L")
            lnV, ZV = self._lnphi(x, T, P, "V")
            if abs(ZV - ZL) < 1e-8:
                raise ValueError("psat: no two roots (T near/above Tc?)")
            d = (lnL[i] - lnV[i]) / (ZV - ZL)
            P *= math.exp(max(-1.0, min(1.0, d)))
            if abs(d) < 1e-13:
                break
        return P

    def heat_of_vaporization(self, T, x=None):
        """Enthalpy of vaporisation at T from the bubble point of liquid x:
        returns dict(P, dh_molar [J/mol], dh_mass [J/kg] = h_V/M_V - h_L/M_L)."""
        x = self._z(x)
        if (x > 1e-12).sum() == 1:
            i = int(np.argmax(x))
            P = self.psat(T, i)
            L = self._phase(x, T, P, "L")
            Vp = self._phase(x, T, P, "V")
        else:
            P, y = self.bubble_P(T, x)
            L = self._phase(np.maximum(x, ZFLOOR), T, P, "L")
            Vp = self._phase(y, T, P, "V")
        return dict(P=P, dh_molar=Vp.h - L.h, dh_mass=Vp.h / Vp.M - L.h / L.M)

    # ------------------------------------------------------------- transport
    def _chung_mix(self, x):
        s, e, w, M = self._csig, self._ceps, self.omega, self._cM
        X = np.outer(x, x)
        sij = np.sqrt(np.outer(s, s))
        eij = np.sqrt(np.outer(e, e))
        wij = 0.5 * (w[:, None] + w[None, :])
        Mij = 2.0 * np.outer(M, M) / (M[:, None] + M[None, :])
        kij = np.sqrt(np.outer(self._kap, self._kap))
        s3ij = sij ** 3
        s3 = float((X * s3ij).sum())
        sm = s3 ** (1.0 / 3.0)
        em = float((X * eij * s3ij).sum()) / s3
        wm = float((X * wij * s3ij).sum()) / s3
        Mm = (float((X * eij * sij ** 2 * np.sqrt(Mij)).sum()) / (em * sm * sm)) ** 2
        mu4 = s3 * float((X * np.outer(self._dip ** 2, self._dip ** 2) / s3ij).sum())
        km = float((X * kij).sum())
        Vcm = (sm / 0.809) ** 3
        Tcm = 1.2593 * em
        mur = 131.3 * mu4 ** 0.25 / math.sqrt(Vcm * Tcm)
        return Tcm, Vcm, wm, Mm, mur, km

    _CH_VIS = np.array([
        [6.324, 50.412, -51.680, 1189.0], [1.210e-3, -1.154e-3, -6.257e-3, 0.03728],
        [5.283, 254.209, -168.48, 3898.0], [6.623, 38.096, -8.464, 31.42],
        [19.745, 7.630, -14.354, 31.53], [-1.900, -12.537, 4.985, -18.15],
        [24.275, 3.450, -11.291, 69.35], [0.7972, 1.117, 0.01235, -4.117],
        [-0.2382, 0.06770, -0.8163, 4.025], [0.06863, 0.3479, 0.5926, -0.727]])
    _CH_TC = np.array([
        [2.4166, 0.74824, -0.91858, 121.72], [-0.50924, -1.5094, -49.991, 69.983],
        [6.6107, 5.6207, 64.760, 27.039], [14.543, -8.9139, -5.6379, 74.344],
        [0.79274, 0.82019, -0.69369, 6.3173], [-5.8634, 12.801, 9.5893, 65.529],
        [91.089, 128.11, -54.217, 523.81]])

    def viscosity(self, x, T, rho_mol, method="chung"):
        """Dynamic viscosity [Pa s] at T and molar density [mol/m3].
        The Chung mixture rules (polarity / association terms) can give non-physical
        (negative) values for water-containing hydrocarbon gases; LBC is used then."""
        x = self._z(x)
        if method.lower() == "lbc":
            return self._visc_lbc(x, T, rho_mol)
        mu = self._visc_chung(x, T, rho_mol)
        if not (math.isfinite(mu) and mu > 0.0):
            return self._visc_lbc(x, T, rho_mol)
        return mu

    def _visc_chung(self, x, T, rho_mol):
        Tcm, Vcm, wm, Mm, mur, km = self._chung_mix(x)
        Ts = 1.2593 * T / Tcm
        Om = 1.16145 * Ts ** -0.14874 + 0.52487 * math.exp(-0.77320 * Ts) \
            + 2.16178 * math.exp(-2.43787 * Ts)
        Fc = 1.0 - 0.2756 * wm + 0.059035 * mur ** 4 + km
        E = self._CH_VIS @ np.array([1.0, wm, mur ** 4, km])
        y = rho_mol * 1e-6 * Vcm / 6.0
        G1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3
        t1 = -math.expm1(-E[3] * y) / y if y > 1e-12 else E[3]
        G2 = (E[0] * t1 + E[1] * G1 * math.exp(E[4] * y) + E[2] * G1) / (E[0] * E[3] + E[1] + E[2])
        ess = E[6] * y * y * G2 * math.exp(E[7] + E[8] / Ts + E[9] / Ts ** 2)
        es = math.sqrt(Ts) / Om * Fc * (1.0 / G2 + E[5] * y) + ess
        return es * 36.344 * math.sqrt(Mm * Tcm) / Vcm ** (2.0 / 3.0) * 1e-7

    def _visc_lbc(self, x, T, rho_mol):
        Tc, Pa, M = self.Tc, self.Pc / 101325.0, self._cM
        xi = Tc ** (1 / 6) / (np.sqrt(M) * Pa ** (2 / 3))
        Tr = T / Tc
        mu = np.where(Tr <= 1.5, 34e-5 * Tr ** 0.94, 17.78e-5 * np.abs(4.58 * Tr - 1.67) ** 0.625) / xi
        sm = np.sqrt(M)
        mu0 = float((x * mu * sm).sum() / (x * sm).sum())
        xim = float(x @ Tc) ** (1 / 6) / (math.sqrt(float(x @ M)) * float(x @ Pa) ** (2 / 3))
        rr = rho_mol * float(x @ self.Vc)
        pol = 0.1023 + 0.023364 * rr + 0.058533 * rr ** 2 - 0.040758 * rr ** 3 + 0.0093324 * rr ** 4
        return (mu0 + (pol ** 4 - 1e-4) / xim) * 1e-3   # cP -> Pa s

    def thermal_conductivity(self, x, T, rho_mol):
        """Thermal conductivity [W/m/K], Chung et al. (1988)."""
        x = self._z(x)
        Tcm, Vcm, wm, Mm, mur, km = self._chung_mix(x)
        Ts = 1.2593 * T / Tcm
        Om = 1.16145 * Ts ** -0.14874 + 0.52487 * math.exp(-0.77320 * Ts) \
            + 2.16178 * math.exp(-2.43787 * Ts)
        Fc = 1.0 - 0.2756 * wm + 0.059035 * mur ** 4 + km
        eta0 = 40.785 * Fc * math.sqrt(Mm * T) / (Vcm ** (2.0 / 3.0) * Om) * 1e-7
        cv0 = float(x @ self._ig(T)[0]) - R
        al = cv0 / R - 1.5
        be = 0.7862 - 0.7109 * wm + 1.3168 * wm * wm
        Tr = T / Tcm
        Zc = 2.0 + 10.5 * Tr * Tr
        psi = 1.0 + al * (0.215 + 0.28288 * al - 1.061 * be + 0.26665 * Zc) \
            / (0.6366 + be * Zc + 1.061 * al * be)
        Bc = self._CH_TC @ np.array([1.0, wm, mur ** 4, km])
        y = rho_mol * 1e-6 * Vcm / 6.0
        G1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3
        t1 = -math.expm1(-Bc[3] * y) / y if y > 1e-12 else Bc[3]
        G2 = (Bc[0] * t1 + Bc[1] * G1 * math.exp(Bc[4] * y) + Bc[2] * G1) / (Bc[0] * Bc[3] + Bc[1] + Bc[2])
        Mp = Mm / 1000.0
        q = 3.586e-3 * math.sqrt(Tcm / Mp) / Vcm ** (2.0 / 3.0)
        return 31.2 * eta0 * psi / Mp * (1.0 / G2 + Bc[5] * y) + q * Bc[6] * y * y * math.sqrt(Tr) * G2

    def surface_tension(self, xL, rhoL_mol, yV=None, rhoV_mol=0.0):
        """Macleod-Sugden surface tension [N/m] (rho in mol/m3)."""
        xL = self._z(xL)
        yV = np.zeros(self.nc) if yV is None else self._z(yV)
        s = float(self._par @ (xL * rhoL_mol * 1e-6 - yV * rhoV_mol * 1e-6))
        return max(s, 0.0) ** 4 * 1e-3

    def transport_rho(self, p):
        """Molar density [mol/m3] used for transport / surface tension of a phase: the
        Peneloux-translated density (whether or not volume_shift is on), because the
        Chung and parachor methods need realistic liquid densities (plain PR liquid
        densities are typically 5-10 % high for light and low for heavy components)."""
        return 1.0 / (p.v + p.x @ self._c - p.x @ self._cpen)

    @staticmethod
    def water_surface_tension(T):
        """IAPWS (1994) release on the surface tension of ordinary water [N/m]."""
        tau = max(1.0 - T / 647.096, 0.0)
        return 0.2358 * tau ** 1.256 * (1.0 - 0.625 * tau)

    def _water_transport(self, T, P):
        """Pure-water viscosity / conductivity from the IAPWS formulations (IAPWS 2008
        viscosity, IAPWS 2011 conductivity) as implemented in CoolProp; None if unavailable."""
        try:
            import CoolProp.CoolProp as CP
            return CP.PropsSI("V", "T", T, "P", P, "Water"), CP.PropsSI("L", "T", T, "P", P, "Water")
        except Exception:
            return None

    def transport(self, r, method="chung", water="iapws"):
        """Add viscosity p.mu [Pa s] and conductivity p.k [W/m/K] to all phases of a
        FlashResult and the surface tension r.sigma [N/m] (vapour-HC liquid by
        Macleod-Sugden; vapour-aqueous = IAPWS pure-water value when no HC liquid).
        Hydrocarbon phases: Chung et al. (or LBC viscosity with method='lbc').
        Aqueous (free-water, pure) phase: IAPWS via CoolProp when water='iapws'
        (the Chung method is poor for water, see REPORT), else Chung."""
        for p in r.phases:
            wt = None
            if p.name == "aqueous" and water == "iapws" and self.iw is not None                     and p.x[self.iw] > 0.999:
                wt = self._water_transport(p.T, p.P)
            if wt is not None:
                p.mu, p.k = wt
            else:
                rho = self.transport_rho(p)
                p.mu = self.viscosity(p.x, p.T, rho, method)
                p.k = self.thermal_conductivity(p.x, p.T, rho)
        V, L = r.vapour, r.liquid
        if L is not None:
            r.sigma = self.surface_tension(L.x, self.transport_rho(L), V.x if V else None,
                                           self.transport_rho(V) if V else 0.0)
        elif r.aqueous is not None:
            r.sigma = self.water_surface_tension(r.T)
        return r
