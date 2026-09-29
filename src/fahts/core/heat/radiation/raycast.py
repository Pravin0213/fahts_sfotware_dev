"""
Ray casting against a triangle soup — the geometric kernel for radiation shielding and
surface-to-surface view factors.

Acceleration: uniform 3-D grid; every triangle is binned into all cells overlapped by its
bounding box, rays walk the grid cell by cell (Amanatides & Woo 1987) and are tested with
the Möller–Trumbore algorithm (double-sided, so thin plates block from both sides).
The hot loops are compiled with numba (parallel over rays) when available, with a pure
numpy/python fallback that is only suitable for small scenes.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

try:                                                # pragma: no cover - env dependent
    import numba as _nb

    _njit = _nb.njit(cache=True)
    _njit_par = _nb.njit(parallel=True, cache=True)
    _prange = _nb.prange
    HAVE_NUMBA = True
except ImportError:                                 # pragma: no cover
    def _njit(f):
        return f
    _njit_par = _njit
    _prange = range
    HAVE_NUMBA = False


# ── numba kernels ─────────────────────────────────────────────────────────────

@_njit
def _moller(o, d, v0, v1, v2, eps):
    e1x, e1y, e1z = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    e2x, e2y, e2z = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    px = d[1] * e2z - d[2] * e2y
    py = d[2] * e2x - d[0] * e2z
    pz = d[0] * e2y - d[1] * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-300:
        return -1.0
    inv = 1.0 / det
    tx, ty, tz = o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < -eps or u > 1.0 + eps:
        return -1.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < -eps or u + v > 1.0 + eps:
        return -1.0
    return (e2x * qx + e2y * qy + e2z * qz) * inv


@_njit
def _trace_one(o, d, tmin, tmax, any_hit, tris, lo, h, res, cstart, cidx):
    """Nearest (or any) triangle hit with t in (tmin, tmax); returns (tri, t)."""
    # clip the ray to the grid box
    t0, t1 = tmin, tmax
    for a in range(3):
        hi_a = lo[a] + res[a] * h[a]
        if abs(d[a]) < 1e-300:
            if o[a] < lo[a] or o[a] > hi_a:
                return -1, tmax
        else:
            ta = (lo[a] - o[a]) / d[a]
            tb = (hi_a - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
    if t0 > t1:
        return -1, tmax
    # starting cell
    cell = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    tnext = np.empty(3)
    tdel = np.empty(3)
    for a in range(3):
        p = o[a] + t0 * d[a]
        c = int(math.floor((p - lo[a]) / h[a]))
        c = min(max(c, 0), res[a] - 1)
        cell[a] = c
        if d[a] > 0.0:
            step[a] = 1
            tnext[a] = (lo[a] + (c + 1) * h[a] - o[a]) / d[a]
            tdel[a] = h[a] / d[a]
        elif d[a] < 0.0:
            step[a] = -1
            tnext[a] = (lo[a] + c * h[a] - o[a]) / d[a]
            tdel[a] = -h[a] / d[a]
        else:
            step[a] = 0
            tnext[a] = 1e300
            tdel[a] = 1e300
    best_t = tmax
    best = -1
    while True:
        ci = cell[0] + res[0] * (cell[1] + res[1] * cell[2])
        for k in range(cstart[ci], cstart[ci + 1]):
            tr = cidx[k]
            t = _moller(o, d, tris[tr, 0], tris[tr, 1], tris[tr, 2], 1e-9)
            if t > tmin and t < best_t:
                best_t = t
                best = tr
                if any_hit:
                    return best, best_t
        # exit t of this cell
        a = 0
        if tnext[1] < tnext[a]:
            a = 1
        if tnext[2] < tnext[a]:
            a = 2
        t_exit = tnext[a]
        if best >= 0 and best_t <= t_exit:
            return best, best_t
        if t_exit > t1:
            return best, best_t
        cell[a] += step[a]
        if cell[a] < 0 or cell[a] >= res[a]:
            return best, best_t
        tnext[a] += tdel[a]


@_njit_par
def _trace_many(O, D, tmin, tmax, any_hit, tris, lo, h, res, cstart, cidx):
    n = O.shape[0]
    hit = np.empty(n, np.int64)
    tt = np.empty(n)
    for i in _prange(n):
        hit[i], tt[i] = _trace_one(O[i], D[i], tmin, tmax[i], any_hit, tris, lo, h, res,
                                   cstart, cidx)
    return hit, tt


@_njit
def _bin_count(bmin, bmax, lo, h, res):
    n = bmin.shape[0]
    cnt = np.zeros(res[0] * res[1] * res[2] + 1, np.int64)
    for t in range(n):
        i0 = min(max(int(math.floor((bmin[t, 0] - lo[0]) / h[0])), 0), res[0] - 1)
        i1 = min(max(int(math.floor((bmax[t, 0] - lo[0]) / h[0])), 0), res[0] - 1)
        j0 = min(max(int(math.floor((bmin[t, 1] - lo[1]) / h[1])), 0), res[1] - 1)
        j1 = min(max(int(math.floor((bmax[t, 1] - lo[1]) / h[1])), 0), res[1] - 1)
        k0 = min(max(int(math.floor((bmin[t, 2] - lo[2]) / h[2])), 0), res[2] - 1)
        k1 = min(max(int(math.floor((bmax[t, 2] - lo[2]) / h[2])), 0), res[2] - 1)
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    cnt[i + res[0] * (j + res[1] * k) + 1] += 1
    return cnt


@_njit
def _bin_fill(bmin, bmax, lo, h, res, cstart):
    n = bmin.shape[0]
    fill = cstart[:-1].copy()
    cidx = np.empty(cstart[-1], np.int64)
    for t in range(n):
        i0 = min(max(int(math.floor((bmin[t, 0] - lo[0]) / h[0])), 0), res[0] - 1)
        i1 = min(max(int(math.floor((bmax[t, 0] - lo[0]) / h[0])), 0), res[0] - 1)
        j0 = min(max(int(math.floor((bmin[t, 1] - lo[1]) / h[1])), 0), res[1] - 1)
        j1 = min(max(int(math.floor((bmax[t, 1] - lo[1]) / h[1])), 0), res[1] - 1)
        k0 = min(max(int(math.floor((bmin[t, 2] - lo[2]) / h[2])), 0), res[2] - 1)
        k1 = min(max(int(math.floor((bmax[t, 2] - lo[2]) / h[2])), 0), res[2] - 1)
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    c = i + res[0] * (j + res[1] * k)
                    cidx[fill[c]] = t
                    fill[c] += 1
    return cidx


# ── public API ────────────────────────────────────────────────────────────────

class TriangleScene:
    """
    Triangle soup + uniform grid for fast ray queries.

    Args:
        tris:   (n, 3, 3) triangle vertex coordinates [m].
        owner:  optional (n,) int label per triangle (e.g. face / patch id).
    """

    def __init__(self, tris: np.ndarray, owner: np.ndarray | None = None,
                 cells_per_tri: float = 1.0, max_res: int = 256) -> None:
        tris = np.ascontiguousarray(tris, dtype=float)
        self.tris = tris
        self.owner = (np.arange(len(tris)) if owner is None
                      else np.asarray(owner, dtype=np.int64))
        n = max(len(tris), 1)
        bmin = tris.min(axis=1) if len(tris) else np.zeros((0, 3))
        bmax = tris.max(axis=1) if len(tris) else np.zeros((0, 3))
        lo = (bmin.min(axis=0) if len(tris) else np.zeros(3))
        hi = (bmax.max(axis=0) if len(tris) else np.ones(3))
        ext = np.maximum(hi - lo, 1e-9)
        pad = 1e-6 * float(ext.max()) + 1e-9
        lo = lo - pad
        ext = ext + 2 * pad
        vol = float(np.prod(ext))
        cell = (vol / (cells_per_tri * n)) ** (1.0 / 3.0)
        res = np.clip(np.ceil(ext / cell).astype(np.int64), 1, max_res)
        self.lo = lo
        self.res = res
        self.h = ext / res
        cnt = _bin_count(bmin, bmax, lo, self.h, res)
        self.cstart = np.cumsum(cnt)
        self.cidx = _bin_fill(bmin, bmax, lo, self.h, res, self.cstart)
        self.scale = float(ext.max())

    def trace(self, origins: np.ndarray, dirs: np.ndarray, tmax: np.ndarray | float,
              any_hit: bool = False, tmin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        First (or any) hit along rays o + t·d, t ∈ (tmin, tmax).  ``dirs`` need not be unit.
        Returns (triangle index or −1, t).
        """
        O = np.ascontiguousarray(origins, dtype=float).reshape(-1, 3)
        D = np.ascontiguousarray(dirs, dtype=float).reshape(-1, 3)
        tm = np.broadcast_to(np.asarray(tmax, dtype=float), (len(O),)).copy()
        if len(self.tris) == 0 or len(O) == 0:
            return np.full(len(O), -1, np.int64), tm
        return _trace_many(O, D, float(tmin), tm, bool(any_hit), self.tris, self.lo,
                           self.h, self.res, self.cstart, self.cidx)

    def blocked(self, origins: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """True where the open segment origin → target hits any triangle."""
        O = np.asarray(origins, dtype=float).reshape(-1, 3)
        T = np.asarray(targets, dtype=float).reshape(-1, 3)
        hit, _ = self.trace(O, T - O, 1.0 - 1e-9, any_hit=True, tmin=1e-9)
        return hit >= 0


def brute_force_first_hit(tris: np.ndarray, o: np.ndarray, d: np.ndarray,
                          tmax: float = np.inf, tmin: float = 0.0) -> tuple[int, float]:
    """Reference (O(n)) nearest hit — for tests."""
    best, best_t = -1, tmax
    for k, tr in enumerate(tris):
        t = _moller(np.asarray(o, float), np.asarray(d, float), tr[0], tr[1], tr[2], 1e-9)
        if tmin < t < best_t:
            best, best_t = k, t
    return best, best_t
