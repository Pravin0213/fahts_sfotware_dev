"""Pressure safety valve opening characteristic (fraction open vs pressure).
"""

from __future__ import annotations


class PSVOpening:
    """Pressure safety valve with the opening characteristics of the VessFire manual
    (section 5.3.1 figure). Type index follows the manual's listing order
    "trapezoidal, triangular or square":
      0 trapezoidal : rising - closed until Pset, then linear 0 -> 100 % at Pfull;
                      falling - stays at its opening until Preseat, then closes.
      1 triangular  : rising - closed until Pset, then 100 %;
                      falling - linear from 100 % at Pset to 0 % at Preseat.
      2 square      : rising - closed until Pset, then 100 %;
                      falling - stays 100 % until Preseat, then closes."""

    def __init__(self, spec: dict):
        self.s = spec
        self.open_frac = 0.0

    def frac(self, P):
        s, f = self.s, self.open_frac
        Ps, Pr, Pf = s["p_set"], s["p_reseat"], s["p_full"]
        if f <= 0.0:  # closed: opens only at Pset
            if P >= Ps:
                f = min(max((P - Ps) / max(Pf - Ps, 1.0), 1e-6), 1.0) if s["type"] == 0 else 1.0
        elif s["type"] == 0:  # trapezoidal
            f = 0.0 if P < Pr else max(f, min((P - Ps) / max(Pf - Ps, 1.0), 1.0))
        elif s["type"] == 1:  # triangular
            f = 1.0 if P >= Ps else max((P - Pr) / max(Ps - Pr, 1.0), 0.0)
        else:  # square
            f = 0.0 if P < Pr else 1.0
        self.open_frac = f
        return f
