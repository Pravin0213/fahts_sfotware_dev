"""Fire exposure in the Scandpower / IP guideline form.

The specified heat flux is converted to an equivalent fire gas temperature T_f, which then
drives radiation and convection:

    q_net(T_s) = eps_s*sigma*(eps_f*T_f^4 - T_s^4) + h_f*(T_f - T_s)

"balance": T_f is solved so that q_net at a reference surface temperature equals the
specified heat load (the load is defined "based on initial conditions of the exposed
object", so the reference is the initial shell temperature; with eps_f = 1 this is the
EN 1991-1-2 fire-gas-temperature boundary).
"blackbody": the specified flux is incident black-body radiation, sigma*T_f^4 = q_spec,
and the absorbed flux follows from the balance above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.optimize import brentq

from fahts.common.constants import SIGMA


@dataclass
class GuidelineFire:
    """Net absorbed flux for a surface at T_s exposed to a fire whose incident
    heat load q_spec was defined at surface temperature T_ref."""

    eps_flame: float  # flame emissivity
    eps_surf: float  # vessel surface emissivity (absorptivity = emissivity)
    h_flame: float  # flame-to-surface convection coefficient [W/m2K]
    T_ref: float  # surface temperature at which q_spec applies [K]
    t_air_mode: str = "balance"
    # "balance"   : T_f solved so that q_net(T_ref) = q_spec
    # "blackbody" : q_spec is incident black-body radiation, sigma*T_f^4 = q_spec
    _cache: dict = field(default_factory=dict, repr=False)

    def q_rad(self, T_s, T_f):
        return self.eps_surf * SIGMA * (self.eps_flame * T_f**4 - T_s**4)

    def q_conv(self, T_s, T_f):
        return self.h_flame * (T_f - T_s)

    def q_net(self, T_s, T_f):
        return self.q_rad(T_s, T_f) + self.q_conv(T_s, T_f)

    def dq_dTs(self, T_s):
        return -self.h_flame - 4.0 * self.eps_surf * SIGMA * T_s**3

    def flame_temperature(self, q_spec):
        """Fire gas temperature for a specified heat load q_spec [W/m2]."""
        if q_spec <= 0.0:
            return None
        key = round(q_spec, 6)
        if key not in self._cache:
            if self.t_air_mode == "blackbody":
                self._cache[key] = (q_spec / SIGMA) ** 0.25
            else:
                f = lambda Tf: self.q_net(self.T_ref, Tf) - q_spec  # noqa: E731
                self._cache[key] = brentq(f, self.T_ref, 5000.0)
        return self._cache[key]
