"""Pseudo-component (petroleum fraction) characterisation from SG/API and Tb.

Riazi & Daubert (1987); Lee & Kesler (1975) / Kesler & Lee (1976) acentric factor and
ideal-gas cp; Firoozabadi et al. (1988) parachor. See ``fahts.thermo.pr_mixture``.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import R


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
