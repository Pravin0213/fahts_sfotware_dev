"""Tridiagonal linear solver."""

from __future__ import annotations

import numpy as np


def solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas algorithm. ``lower``/``upper`` have length n-1; inputs are not modified."""
    n = len(diag)
    d = diag.copy()
    r = rhs.copy()
    for i in range(1, n):
        m = lower[i-1] / d[i-1]
        d[i] -= m * upper[i-1]
        r[i] -= m * r[i-1]
    x = np.empty(n)
    x[-1] = r[-1] / d[-1]
    for i in range(n-2, -1, -1):
        x[i] = (r[i] - upper[i] * x[i+1]) / d[i]
    return x
