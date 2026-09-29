"""Exchange with the surroundings: radiation plus natural/forced convection to air."""

from __future__ import annotations

import CoolProp.CoolProp as CP

from fahts.common.constants import G_ACC, P_ATM, SIGMA


class AmbientBC:
    """Natural/forced convection plus radiation exchange with surroundings at T_env."""

    def __init__(self, T_env: float, eps: float, h: float):
        self.T_env, self.eps, self.h = T_env, eps, h
        self.T_flame = None

    def __call__(self, T_s, t):
        qr = self.eps * SIGMA * (self.T_env**4 - T_s**4)
        qc = self.h * (self.T_env - T_s)
        return qr + qc, -4 * self.eps * SIGMA * T_s**3 - self.h, qr, qc


_AIR = CP.AbstractState("HEOS", "Air")


def h_ambient(T_s, T_env, D_out, spec):
    """Outside convection to still/moving air. spec > 0: fixed h; spec < 0: |spec|
    is a wind speed [m/s] - forced (Churchill-Bernstein) and natural (Churchill-Chu)
    convection combined as (h_f^3 + h_n^3)^(1/3)."""
    if spec > 0:
        return spec
    Tf = 0.5 * (T_s + T_env)
    _AIR.update(CP.PT_INPUTS, P_ATM, Tf)
    k, mu, rho, cp = _AIR.conductivity(), _AIR.viscosity(), _AIR.rhomass(), _AIR.cpmass()
    Pr = cp * mu / k
    Ra = G_ACC / Tf * abs(T_s - T_env) * D_out**3 * rho**2 * cp / (mu * k)
    Nu_n = (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    v = abs(spec) if abs(spec) < 50 else 0.0
    Re = rho * v * D_out / mu
    Nu_f = (
        0.3
        + 0.62
        * Re**0.5
        * Pr ** (1 / 3)
        / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25
        * (1 + (Re / 282000) ** (5 / 8)) ** 0.8
        if Re > 0
        else 0.0
    )
    return (Nu_n**3 + Nu_f**3) ** (1 / 3) * k / D_out


class AmbientAuto(AmbientBC):
    """AmbientBC whose convection coefficient follows the surface temperature."""

    def __init__(self, T_env, eps, D_out, spec):
        super().__init__(T_env, eps, 5.0)
        self.D_out, self.spec = D_out, spec

    def __call__(self, T_s, t):
        self.h = h_ambient(T_s, self.T_env, self.D_out, self.spec)
        return super().__call__(T_s, t)
