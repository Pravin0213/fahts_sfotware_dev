"""
fluid.py - real-gas properties for a VessFire-style composition, via CoolProp.

Components use VessFire's documented names (manual, Segment.brl #Fluid) and are
mapped to CoolProp's multi-parameter Helmholtz equations of state (HEOS; pure
fluids use the reference EOS, mixtures use the GERG-2008-type mixing rules).

Single-phase (gas / supercritical) only. The working states have the gas phase
imposed; check_single_phase() runs an unforced flash and raises TwoPhaseError if
liquid is present (called by vessel.simulate at start and at every output step).

Flashes are done with robust Newton iterations on (density, temperature) inputs,
which CoolProp evaluates directly and quickly, instead of CoolProp's own
(density, energy) and (pressure, entropy) mixture flashes, which can hang.
Entropy is never used: CoolProp 8's cubic (PR/SRK) backends return entropies that
violate T(ds/dT)_P = cp (checked: ratio 1.25 for methane), while u, h, P, cp, cv
and beta are consistent.
"""

from __future__ import annotations

import math

import CoolProp.CoolProp as CP

VF_TO_COOLPROP = {
    "H2": "Hydrogen", "N2": "Nitrogen", "CO2": "CarbonDioxide", "H2S": "HydrogenSulfide",
    "H2O": "Water", "C1": "Methane", "C2": "Ethane", "C3": "Propane",
    "IC4": "IsoButane", "C4": "n-Butane", "NC4": "n-Butane", "IC5": "Isopentane",
    "C5": "n-Pentane", "NC5": "n-Pentane", "C6": "n-Hexane", "AR": "Argon", "O2": "Oxygen",
    "HE": "Helium",
}


class TwoPhaseError(RuntimeError):
    pass


class Fluid:
    """Composition in mole fractions, keys are VessFire component names."""

    def __init__(self, composition: dict[str, float], eos: str = "HEOS"):
        """eos: 'HEOS' (reference multi-parameter EOS, default), 'PR' (Peng-Robinson)
        or 'SRK' (Soave-Redlich-Kwong). Transport properties always use HEOS."""
        tot = sum(composition.values())
        self.comp = {k.upper(): v / tot for k, v in composition.items() if v > 0}
        missing = [k for k in self.comp if k not in VF_TO_COOLPROP]
        if missing:
            raise ValueError(f"no CoolProp mapping for components {missing}")
        names = "&".join(VF_TO_COOLPROP[k] for k in self.comp)
        self.eos = eos.upper()
        self.AS = CP.AbstractState(self.eos, names)     # (rho, T) evaluations
        self.AG = CP.AbstractState(self.eos, names)     # (P, T) in the gas phase
        self.AT = CP.AbstractState("HEOS", names)       # transport properties
        if len(self.comp) > 1:
            x = list(self.comp.values())
            for a in (self.AS, self.AG, self.AT):
                a.set_mole_fractions(x)
        # single-phase model: fix the phase so CoolProp evaluates the EOS directly
        # (unspecified-phase mixture (rho,T) updates run a slow, fragile flash)
        for a in (self.AS, self.AG, self.AT):
            a.specify_phase(CP.iphase_gas)
        # separate UNFORCED state, used only to check that the fluid really is single phase
        self.AC = CP.AbstractState(self.eos, names)
        if len(self.comp) > 1:
            self.AC.set_mole_fractions(list(self.comp.values()))
        self.phase_check_warning = None
        self.M = self.AS.molar_mass()                    # kg/mol
        self.label = " ".join(f"{k} {v:g}" for k, v in self.comp.items())
        if self.eos != "HEOS":
            self.label += f" [{self.eos}]"

    # ---------------------------------------------------------------- basics
    def at_rho_T(self, rho: float, T: float) -> CP.AbstractState:
        self.AS.update(CP.DmassT_INPUTS, rho, T)
        return self.AS

    def at_P_T(self, P: float, T: float) -> CP.AbstractState:
        self.AG.update(CP.PT_INPUTS, P, T)
        return self.AG

    def rho_PT(self, P, T):
        return self.at_P_T(P, T).rhomass()

    # ---------------------------------------------------------------- flashes
    def T_from_rho_u(self, rho: float, u: float, T_guess: float) -> float:
        """Temperature from density and specific internal energy (Newton on T)."""
        T = T_guess
        for _ in range(50):
            s = self.at_rho_T(rho, T)
            f = s.umass() - u
            cv = s.cvmass()
            dT = -f / cv
            dT = max(min(dT, 200.0), -0.5 * T)
            T += dT
            if abs(dT) < 1e-7 * T:
                break
        else:
            raise RuntimeError(f"T(rho,u) did not converge: rho={rho}, u={u}")
        return T

    def T_from_P_s(self, P: float, s: float, T_guess: float) -> float:
        """Temperature from pressure and specific entropy, gas phase (Newton on T)."""
        T = T_guess
        for _ in range(60):
            st = self.at_P_T(P, T)
            f = st.smass() - s
            dT = -f * T / st.cpmass()                    # ds/dT|P = cp/T
            dT = max(min(dT, 200.0), -0.5 * T)
            T += dT
            if abs(dT) < 1e-7 * T:
                break
        else:
            raise RuntimeError(f"T(P,s) did not converge: P={P}, s={s}")
        return T

    def check_single_phase(self, P: float, T: float):
        """Raise TwoPhaseError if an unforced (P, T) flash finds two phases. The working
        states have the gas phase imposed for speed, so without this check a two-phase
        fluid would silently be treated as all gas. A dense single-phase mixture is
        labelled 'liquid' by CoolProp (density-based label, still one phase) - allowed.
        If the chosen EOS cannot flash (e.g. CoolProp PR with no density guess), the
        check is repeated with the reference EOS."""
        last_err = None
        for AC in (self.AC, self._AC_heos()):
            if AC is None:
                continue
            try:
                AC.update(CP.PT_INPUTS, P, T)
            except Exception as e:
                last_err = e
                continue
            if AC.phase() == CP.iphase_twophase:
                raise TwoPhaseError(
                    f"{self.label}: two-phase (vapour fraction {AC.Q():.3f}) at "
                    f"P={P/1e5:.1f} bara, T={T-273.15:.1f} C - this model is single-phase only")
            return
        self.phase_check_warning = f"phase check not possible at P={P/1e5:.1f} bara, "                                    f"T={T-273.15:.1f} C: {str(last_err)[:80]}"

    def _AC_heos(self):
        if self.eos == "HEOS":
            return None
        if not hasattr(self, "_ach"):
            names = "&".join(VF_TO_COOLPROP[k] for k in self.comp)
            self._ach = CP.AbstractState("HEOS", names)
            if len(self.comp) > 1:
                self._ach.set_mole_fractions(list(self.comp.values()))
        return self._ach

    def state_rho_u(self, rho: float, u: float, T_guess: float) -> dict:
        T = self.T_from_rho_u(rho, u, T_guess)
        s = self.at_rho_T(rho, T)
        ph = s.phase()
        if ph in (CP.iphase_twophase,):
            raise TwoPhaseError(f"two-phase state at rho={rho:.3f}, T={T:.2f}")
        return dict(T=T, P=s.p(), h=s.hmass(), u=s.umass(), s=s.smass(),
                    cp=s.cpmass(), cv=s.cvmass(), Z=s.compressibility_factor())

    # ---------------------------------------------------------------- transport
    def film_props(self, T: float, P: float) -> dict:
        s = self.at_P_T(P, T)
        out = dict(rho=s.rhomass(), cp=s.cpmass(), beta=s.isobaric_expansion_coefficient())
        self.AT.update(CP.PT_INPUTS, P, T)
        out.update(mu=self.AT.viscosity(), k=self.AT.conductivity())
        return out

    # ---------------------------------------------------------------- nozzle
    def isentropic_mass_flux(self, P0: float, T0: float, P_back: float,
                             n: int = 40) -> tuple[float, float]:
        """Ideal (Cd = 1) mass flux [kg/m2/s] through an orifice from stagnation (P0, T0)
        to back pressure P_back: homogeneous isentropic expansion of the real gas.

        The isentrope is integrated WITHOUT entropy (CoolProp's cubic backends return
        inconsistent entropies), using the exact identities
            (dT/dP)_s = T * beta / (rho * cp),     dh = dP / rho      (along s = const)
        with RK4 in pressure. G(P) = rho * sqrt(2 (h0 - h)); the choked flux is the
        maximum of G, refined by a parabola through the three best points.
        Returns (G, throat pressure)."""
        if P_back >= P0:
            return 0.0, P0
        P_lo = max(P_back, 0.2 * P0)

        def f(P, T):
            st = self.at_P_T(P, T)
            rho = st.rhomass()
            return T * st.isobaric_expansion_coefficient() / (rho * st.cpmass()), 1.0 / rho, rho

        dP = (P_lo - P0) / n                              # negative
        P, T, dh = P0, T0, 0.0                            # dh = h0 - h  (>= 0)
        Ps, Gs = [P0], [0.0]
        for _ in range(n):
            k1T, k1h, _ = f(P, T)
            k2T, k2h, _ = f(P + dP / 2, T + dP / 2 * k1T)
            k3T, k3h, _ = f(P + dP / 2, T + dP / 2 * k2T)
            k4T, k4h, rho4 = f(P + dP, T + dP * k3T)
            T += dP / 6 * (k1T + 2 * k2T + 2 * k3T + k4T)
            dh -= dP / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
            P += dP
            rho = self.at_P_T(P, T).rhomass()
            Ps.append(P)
            Gs.append(rho * math.sqrt(max(2.0 * dh, 0.0)))
        i = max(range(len(Gs)), key=Gs.__getitem__)
        if i == len(Gs) - 1:                              # max at the lower end
            if P_lo == P_back:
                return Gs[i], P_back                      # sub-critical flow
            return Gs[i], Ps[i]
        # parabola through (i-1, i, i+1)
        x0, x1, x2 = Ps[i - 1], Ps[i], Ps[i + 1]
        y0, y1, y2 = Gs[i - 1], Gs[i], Gs[i + 1]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / den
        C = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / den
        if A < 0:
            Pt = -B / (2 * A)
            return A * Pt**2 + B * Pt + C, Pt
        return y1, x1

    def api520_mass_flux(self, P0: float, T0: float, P_back: float) -> float:
        """Ideal (Kd = 1) gas mass flux [kg/m2/s], API 520 Part I equations with the
        real-gas compressibility Z and isentropic exponent k = cp/cv at upstream."""
        st = self.at_P_T(P0, T0)
        Z = st.compressibility_factor()
        k = st.cpmass() / st.cvmass()
        R = CP.PropsSI("gas_constant", "Hydrogen") / self.M
        r_crit = (2 / (k + 1)) ** (k / (k - 1))
        r = P_back / P0
        if r <= r_crit:                                   # choked
            C = math.sqrt(k * (2 / (k + 1)) ** ((k + 1) / (k - 1)))
            return C * P0 / math.sqrt(Z * R * T0)
        # sub-critical
        F2 = math.sqrt((k / (k - 1)) * r ** (2 / k) * (1 - r ** ((k - 1) / k)) / (1 - r))
        return F2 * math.sqrt(2 * (P0 - P_back) * P0 / (Z * R * T0))
