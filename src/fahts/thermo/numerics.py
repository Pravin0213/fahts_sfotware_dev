"""Numerical kernels of the PR model: cubic roots, Rachford-Rice, 1-D root finder.
"""

from __future__ import annotations

import math


def cubic_roots(A: float, B: float) -> list:
    """Real roots Z > B of the PR cubic, sorted ascending (Cardano/trig + Newton polish)."""
    c2 = B - 1.0
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B * B * B)
    q = (c2 * c2 - 3.0 * c1) / 9.0
    r = (2.0 * c2**3 - 9.0 * c2 * c1 + 27.0 * c0) / 54.0
    q3 = q * q * q
    if r * r < q3:
        th = math.acos(max(-1.0, min(1.0, r / math.sqrt(q3))))
        s = -2.0 * math.sqrt(q)
        roots = [
            s * math.cos(th / 3.0) - c2 / 3.0,
            s * math.cos((th + 2 * math.pi) / 3.0) - c2 / 3.0,
            s * math.cos((th - 2 * math.pi) / 3.0) - c2 / 3.0,
        ]
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


def rachford_rice(z, K, beta0=None):
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


def solve_monotone_1d(f, x0, dx0, xmin, xmax, incr, xtol, ftol, maxit=80):
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
