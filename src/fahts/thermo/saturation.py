"""Saturation: bubble/dew pressure and temperature, vapour pressure, heat of vaporisation.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import ZFLOOR


class SaturationMixin:
    """Saturation: bubble/dew pressure and temperature, vapour pressure, heat of vaporisation."""

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
