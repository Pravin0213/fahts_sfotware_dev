"""Grey radiation exchange inside the vapour space (dry wall, liquid surface, gas).
"""

from __future__ import annotations


def rad_network(T1, T3, Tg, A1, A3, e1, e3, eg):
    """Grey radiation exchange in the vapour space (radiosity network; e.g. Modest,
    Radiative Heat Transfer, ch. 4 and 19; Incropera ch. 13).
    Surfaces: 1 = dry wall inner surface (area A1), 3 = liquid free surface (A3, flat,
    so F31 = 1 and F13 = A3/A1, F11 = 1 - F13); grey gas of emissivity eg (transmissivity
    1 - eg over the enclosure's mean beam length).
    Returns (Q1, Q3, Qg): net radiation LEAVING surface 1, LEAVING surface 3, and
    ABSORBED by the gas (Q1 + Q3 = Qg), in W."""
    S = 5.670374419e-8
    E1, E3, Eg = S * T1**4, S * T3**4, S * Tg**4
    tau = 1.0 - eg
    if A1 <= 0:
        return 0.0, 0.0, 0.0
    G1 = e1 * A1 / (1.0 - e1) if e1 < 1 else 1e12 * A1
    G1g = A1 * eg                                   # surface 1 <-> gas (F1,all = 1)
    if A3 <= 0:
        if eg <= 0:
            return 0.0, 0.0, 0.0
        J1 = (G1 * E1 + G1g * Eg) / (G1 + G1g)
        Q1 = G1 * (E1 - J1)
        return Q1, 0.0, Q1
    G3 = e3 * A3 / (1.0 - e3) if e3 < 1 else 1e12 * A3
    G13 = A3 * tau                                  # A1 F13 tau = A3 F31 tau
    G3g = A3 * eg
    # node equations: G1(E1-J1) + G13(J3-J1) + G1g(Eg-J1) = 0 ; same for J3
    a11, a12, b1 = G1 + G13 + G1g, -G13, G1 * E1 + G1g * Eg
    a21, a22, b2 = -G13, G3 + G13 + G3g, G3 * E3 + G3g * Eg
    det = a11 * a22 - a12 * a21
    J1 = (b1 * a22 - a12 * b2) / det
    J3 = (a11 * b2 - a21 * b1) / det
    Q1, Q3 = G1 * (E1 - J1), G3 * (E3 - J3)
    return Q1, Q3, Q1 + Q3
