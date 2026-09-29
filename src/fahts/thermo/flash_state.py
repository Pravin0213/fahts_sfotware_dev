"""State-function flashes: UV (vessel zones), PH and PS, including water-only PH and
lever-rule blending across single-component phase changes.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import ALIASES, D1, D2, P_REF, R, SQ2, ZFLOOR
from fahts.thermo.numerics import solve_monotone_1d
from fahts.thermo.results import FlashResult, Phase


class StateFlashMixin:
    """State-function flashes: UV (vessel zones), PH and PS, including water-only PH and"""

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
        P0 = (
            init.P
            if init is not None
            else (P_guess or (self._uvP1 if self._uvP1 and self._uvP1 > 0 else 1e6))
        )
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
        stable, w, tm, Zw = self.stability(
            zf, T1, P1, lnphi_z=lnphiz, extra_trials=extra, skip_aqueous=fw
        )
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
            T, fT, _, _, _ = solve_monotone_1d(
                f, min(T0, Tmax * 0.999), -2.0, 20.0, Tmax, incr=True, xtol=1e-11, ftol=1e-12
            )
        except (RuntimeError, ValueError):
            return None
        L, V, bV, Ps = st[T]
        if not (0.0 <= bV <= 1.0):
            return None
        V.beta, L.beta = bV, 1.0 - bV
        return FlashResult(
            T, Ps, z, [p for p in (V, L) if p.beta > 0], kind="pure2", names=self.names
        )

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

            x, fx, r, A, B = solve_monotone_1d(
                f,
                math.log(st["P"]),
                0.3,
                math.log(1e-3),
                math.log(5e9),
                incr=False,
                xtol=1e-13,
                ftol=1e-11,
            )
            if abs(fx) > 1e-9:
                r = self._blend_to(A, B, lambda q: q.v, v)
            st["P"] = r.P
            return r

        def outer(T):
            r = inner(T)
            return (r.u - u) / (R * T0), r

        T, fT, r, A, B = solve_monotone_1d(
            outer, T0, 2.0, 20.0, 3000.0, incr=True, xtol=1e-10, ftol=1e-10
        )
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
        r = FlashResult(
            (1 - w) * ra.T + w * rb.T,
            (1 - w) * ra.P + w * rb.P,
            ra.z,
            phases,
            kind="blend",
            names=self.names,
        )
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
        return Phase(
            name=name,
            beta=1.0,
            x=self._ew.copy(),
            T=T,
            P=P,
            Z=Z,
            v=v,
            M=self.Mw[iw],
            h=h,
            u=h - P * v,
            s=s_,
            cp=AS.cpmolar() - cp0w + cp0_i[iw],
            cv=AS.cvmolar() - cp0w + cp0_i[iw],
            w=AS.speed_sound(),
            dPdT_v=float("nan"),
            dPdv_T=float("nan"),
        )

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
        T = (
            min(T_guess or Ts - 1.0, Ts * (1 - 1e-5))
            if liquid
            else max(T_guess or Ts + 1.0, Ts * (1 + 1e-5))
        )
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
        T, fT, r, A, B = solve_monotone_1d(
            f, T0, dx0, 20.0, 3000.0, incr=True, xtol=1e-10, ftol=1e-11
        )
        if abs(fT) > 1e-8:
            r = self._blend_to(A, B, (lambda q: q.h) if which == "h" else (lambda q: q.s), X)
        return r
