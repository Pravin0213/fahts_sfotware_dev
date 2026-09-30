"""Structured Hex8 mesh of a horizontal cylindrical vessel shell (the steel wall).

Coordinates: axis along x (0..L), z up, y horizontal. Angle theta is measured from the top
(0 = top, 180 = bottom) towards +y, the convention of the heat load's peak zone.

Nodes are indexed (j along x, i around, k through the wall from the inner surface):
    node = (j * n_theta + i) * (n_r + 1) + k          (i periodic: no seam nodes)
Hex8 (VTK order, positive Jacobian): natural xi along x, eta along theta, zeta outward.

Boundary data used by the solver (lumped per node, faces split 1/4 to each corner):
- outer surface nodes: area under the background flux and under the peak (jet) flux
- inner surface nodes: area in contact with liquid / gas, recomputed from the liquid level
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _angle_distance(a, b):
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def _merge_close(values, tol: float, period: float | None = None) -> np.ndarray:
    """Sorted grid lines with near-duplicates merged: an inserted zone edge that coincides
    with an existing line to round-off (0.4 vs 0.4000000000000001) must not create a
    zero-length sliver element (singular capacity, ill-conditioned system)."""
    v = np.asarray(values, float)
    v = np.sort(v % period if period else v)
    v = v[np.concatenate(([True], np.diff(v) > tol))]
    if period and len(v) > 1 and (v[0] + period - v[-1]) <= tol:
        v = v[:-1]
    return v


def wetted_half_angle_deg(level: float, D: float) -> float:
    """Half-angle of the wetted arc of a horizontal cylinder, from the bottom [deg]."""
    R = D / 2.0
    if level <= 0.0:
        return 0.0
    if level >= D:
        return 180.0
    return math.degrees(math.acos(min(max((R - level) / R, -1.0), 1.0)))


@dataclass
class PeakZoneGeometry:
    xi_start: float
    xi_end: float
    circ_deg: float
    attack_deg: float

    def contains(self, theta_deg, xi):
        return (
            (np.asarray(xi) >= self.xi_start)
            & (np.asarray(xi) <= self.xi_end)
            & (_angle_distance(theta_deg, self.attack_deg) <= self.circ_deg / 2 + 1e-9)
        )


@dataclass
class VesselShellMesh:
    """Hex8 mesh of the shell between radius R (inner) and R + t (outer)."""

    D: float
    t: float
    L: float
    n_theta: int = 72
    n_length: int = 40
    n_radial: int = 6
    peak: PeakZoneGeometry | None = None
    theta: np.ndarray = field(init=False)  # node angles [deg], 0..360 exclusive
    x: np.ndarray = field(init=False)  # node axial positions [m]
    r: np.ndarray = field(init=False)  # node radii [m], inner -> outer

    def __post_init__(self):
        R = self.D / 2.0
        th = np.linspace(0.0, 360.0, self.n_theta, endpoint=False)
        xs = np.linspace(0.0, 1.0, self.n_length + 1)
        if self.peak is not None:  # grid lines on the zone edges: exact zone
            p = self.peak
            edges = np.mod([p.attack_deg - p.circ_deg / 2, p.attack_deg + p.circ_deg / 2], 360.0)
            th = _merge_close(np.concatenate((th, edges)), 1e-6, period=360.0)
            xs = _merge_close(np.concatenate((xs, np.clip([p.xi_start, p.xi_end], 0, 1))), 1e-9)
        self.theta, self.x = th, xs * self.L
        self.r = R + np.linspace(0.0, self.t, self.n_radial + 1)
        nt, nx, nr = len(th), len(xs), self.n_radial + 1
        self.nt, self.nx, self.nr = nt, nx, nr

        J, I, K = np.meshgrid(np.arange(nx), np.arange(nt), np.arange(nr), indexing="ij")
        thr = np.radians(th)[I]
        rr = self.r[K]
        self.nodes = np.column_stack(
            [self.x[J].ravel(), (rr * np.sin(thr)).ravel(), (rr * np.cos(thr)).ravel()]
        )
        self.n_nodes = len(self.nodes)

        # hexahedra
        j, i, k = np.meshgrid(np.arange(nx - 1), np.arange(nt), np.arange(nr - 1), indexing="ij")
        j, i, k = j.ravel(), i.ravel(), k.ravel()
        i1 = (i + 1) % nt
        nid = self.node_id
        self.hexes = np.column_stack(
            [
                nid(j, i, k),
                nid(j + 1, i, k),
                nid(j + 1, i1, k),
                nid(j, i1, k),
                nid(j, i, k + 1),
                nid(j + 1, i, k + 1),
                nid(j + 1, i1, k + 1),
                nid(j, i1, k + 1),
            ]
        )
        self.n_hexes = len(self.hexes)

        # surface faces (i, j) on the inner (k = 0) and outer (k = n_r) surfaces
        fj, fi = np.meshgrid(np.arange(nx - 1), np.arange(nt), indexing="ij")
        fj, fi = fj.ravel(), fi.ravel()
        fi1 = (fi + 1) % nt
        dth = np.radians((th[fi1] - th[fi]) % 360.0)
        dth[dth == 0.0] = 2 * np.pi  # single-division ring (not used)
        dx = self.x[fj + 1] - self.x[fj]
        theta_c = np.degrees(np.radians(th[fi]) + dth / 2) % 360.0
        xi_c = (self.x[fj] + dx / 2) / self.L
        self.face_theta, self.face_xi = theta_c, xi_c
        self._face_corners = [(fj, fi), (fj + 1, fi), (fj + 1, fi1), (fj, fi1)]

        def areas(radius):  # flat quads: chord x length
            return 2 * radius * np.sin(dth / 2) * dx

        self.outer_face_area = areas(self.r[-1])
        self.inner_face_area = areas(self.r[0])
        self.face_in_peak = (
            self.peak.contains(theta_c, xi_c) if self.peak is not None else np.zeros(len(fi), bool)
        )
        self.outer_nodes = nid(
            *np.meshgrid(np.arange(nx), np.arange(nt), indexing="ij"), nr - 1
        ).ravel()
        self.inner_nodes = nid(*np.meshgrid(np.arange(nx), np.arange(nt), indexing="ij"), 0).ravel()
        # outer node areas split by flux type (surface index = j * nt + i)
        self.outer_area_background = self._to_surface_nodes(
            self.outer_face_area * ~self.face_in_peak
        )
        self.outer_area_peak = self._to_surface_nodes(self.outer_face_area * self.face_in_peak)
        self.inner_area = self._to_surface_nodes(self.inner_face_area)
        self.set_level(0.0)

    # ------------------------------------------------------------------ indexing
    def node_id(self, j, i, k):
        return (np.asarray(j) * self.nt + np.asarray(i)) * self.nr + np.asarray(k)

    def _to_surface_nodes(self, face_values: np.ndarray) -> np.ndarray:
        """Distribute face values to their 4 corner surface nodes (1/4 each)."""
        out = np.zeros(self.nx * self.nt)
        for fj, fi in self._face_corners:
            np.add.at(out, fj * self.nt + fi, face_values / 4.0)
        return out

    # ------------------------------------------------------------------ liquid level
    def set_level(self, level: float) -> None:
        """Wetted inner faces for a liquid level [m] (by face centre)."""
        self.level = level
        half = wetted_half_angle_deg(level, self.D)
        self.face_wet = _angle_distance(self.face_theta, 180.0) < half
        self.inner_area_wet = self._to_surface_nodes(self.inner_face_area * self.face_wet)
        self.inner_area_dry = self.inner_area - self.inner_area_wet

    @property
    def wet_fraction(self) -> float:
        return float(self.inner_face_area[self.face_wet].sum() / self.inner_face_area.sum())

    @property
    def peak_fraction(self) -> float:
        return float(self.outer_face_area[self.face_in_peak].sum() / self.outer_face_area.sum())

    # ------------------------------------------------------------------ columns
    def surface_theta_xi(self) -> tuple[np.ndarray, np.ndarray]:
        """Angle [deg] and x/L of every surface-node column (index j * nt + i)."""
        J, I = np.meshgrid(np.arange(self.nx), np.arange(self.nt), indexing="ij")
        return self.theta[I].ravel(), (self.x[J] / self.L).ravel()

    def columns(self, T: np.ndarray) -> np.ndarray:
        """Node temperatures as (n_columns, n_r + 1), inner -> outer."""
        return np.asarray(T).reshape(self.nx * self.nt, self.nr)
