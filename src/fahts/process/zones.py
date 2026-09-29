"""Gas / liquid zones of the non-equilibrium vessel model and fast zone flashes.

Each zone is internally in equilibrium; ``fast_PH_single`` / ``fast_PH_multi`` are
isenthalpic solves that assume the zone's previous phase set (used inside the pressure
iteration; a stability check afterwards decides whether a full flash is needed).
"""

from __future__ import annotations

import numpy as np

from fahts.thermo import FlashResult


def fast_PH_single(m, x, P, h, T0, root):
    """Single-phase isenthalpic state (Newton on T with the given EOS root 'V'/'L').
    Used inside the pressure iteration; phase stability is checked once afterwards."""
    T, ok = T0, False
    try:
        for _ in range(12):
            p = m.phase_props(x, T, P, root)
            if not (np.isfinite(p.cp) and p.cp > 0):
                return None
            dT = (h - p.h) / p.cp
            dT = max(min(dT, 20.0), -20.0)
            T = min(max(T + dT, 0.7 * T0), 1.5 * T0)
            if abs(dT) < 1e-7 * T:
                ok = True
                break
        if not ok:
            return None
        p = m.phase_props(x, T, P, root)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    if abs(p.h - h) > 1e-4 * max(abs(h), 1.0) + 0.5:
        return None
    p.beta = 1.0
    return FlashResult(T, P, np.asarray(x, float), [p], kind="fast1", names=m.names)


def fast_PH_multi(m, x, P, h, prev):
    """Isenthalpic flash assuming the phase set of `prev` (no stability search):
    Newton on T with flash_PT(same_phases=True). Returns None if the phase set is no
    longer valid (a phase fraction left (0, 1)) so the caller can do a full flash."""
    T = prev.T
    r = m.flash_PT(P, T, x, init=prev, same_phases=True)
    for _ in range(12):
        dT_fd = 0.01
        r2 = m.flash_PT(P, T + dT_fd, x, init=r, same_phases=True)
        cp_eff = (r2.h - r.h) / dT_fd
        if not np.isfinite(cp_eff) or cp_eff <= 0:
            return None
        dT = (h - r.h) / cp_eff
        dT = max(min(dT, 20.0), -20.0)
        T += dT
        r = m.flash_PT(P, T, x, init=r, same_phases=True)
        if abs(dT) < 1e-6:
            break
    if abs(r.h - h) > 1e-3 * max(abs(h), 1.0) + 1.0:
        return None
    if any(not (0.0 < p.beta < 1.0) for p in r.phases) or len(r.phases) != len(prev.phases):
        return None
    return r


class Zone:
    """Moles n [mol], enthalpy H [J] and the last flash result r at pressure P."""

    def __init__(self, name, n, H, r):
        self.name, self.n, self.H, self.r = name, np.asarray(n, float), float(H), r

    @property
    def N(self):
        return float(self.n.sum())

    @property
    def empty(self):
        return self.N < 1e-3

    @property
    def V(self):
        return self.N * self.r.v if not self.empty else 0.0

    @property
    def mass(self):
        return self.N * self.r.M if not self.empty else 0.0

    @property
    def T(self):
        return self.r.T


def split_phases(r, N, keep):
    """Split a zone flash result into (kept phases, moved phases): each as
    (moles vector, enthalpy J, volume m3). keep = tuple of phase names to keep."""
    kn, kH, kV = np.zeros(len(r.z)), 0.0, 0.0
    mn, mH, mV = np.zeros(len(r.z)), 0.0, 0.0
    for p in r.phases:
        nn = N * p.beta * np.asarray(p.x)
        if p.name in keep:
            kn += nn
            kH += N * p.beta * p.h
            kV += N * p.beta * p.v
        else:
            mn += nn
            mH += N * p.beta * p.h
            mV += N * p.beta * p.v
    return (kn, kH, kV), (mn, mH, mV)
