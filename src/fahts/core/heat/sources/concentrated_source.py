"""
Concentrated (point) heat source — FAHTS Theory Manual §3.5.4.

The source radiates energy in all directions from a single point.  The flux
received by a steel surface element is:

    q_i = E(t) * cos(theta_i) / (4 * pi * r_i^2)

where
    E(t)     : total emitted power [W] (may be time-dependent via a callable)
    r_i      : distance from source to element/quad centroid [m]
    theta_i  : angle between the source-to-element ray and the element's
               outward surface normal; cos(theta_i) > 0 means the face
               points toward the source

The source is omnidirectional (no preferred emission direction).  A negative
cos(theta_i) (face pointing away) contributes zero flux.

Optionally the power may be time-dependent; pass a callable ``power`` argument
that takes time in seconds and returns total emitted power in [W].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fahts.core.heat.sources.base_source import HeatSource


@dataclass
class ConcentratedSource(HeatSource):
    """
    Point heat source with inverse-square and cosine directional falloff (§3.5.4).

    Parameters
    ----------
    name      : identifier string
    center    : (3,) array — global coordinates of source position [m]
    power     : total emitted power E [W] — constant float or callable(t [s]) → float
    active    : whether this source participates in the analysis (default True)

    Physics
    -------
    The flux at a surface quad centroid is:

        q_i = E(t) * cos(theta_i) / (4 * pi * r_i^2)

    where r_i is the distance from ``center`` to the quad centroid and
    theta_i is the angle between the outward surface normal of the quad and the
    direction *from the source toward the quad* (i.e. the ray the radiation
    travels along).  cos(theta_i) > 0 when the face points toward the source;
    faces pointing away (cos <= 0) receive zero flux.

    The source temperature is not meaningful (flux is fully prescribed); the
    ``temperature`` method returns ambient 20 °C.  Set ``epsilon_m = 0`` in
    the solver when using this source so that steel re-radiation is handled
    by the separate ``eps_rerad`` path.
    """

    name: str
    center: np.ndarray                           # shape (3,) [m]
    power: float | Callable[[float], float]      # E [W] or callable(t) → E(t)
    active: bool = field(default=True)

    # ── HeatSource ABC ────────────────────────────────────────────────────────

    def temperature(self, t: float) -> float:
        """Not meaningful for ConcentratedSource; returns ambient 20 °C."""
        return 20.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a degenerate point bounding box at the source center.

        ConcentratedSource has no geometric extent; the bounding box is a
        single point.  Exposure range is unlimited (falloff to zero at infinity).
        """
        return self.center.copy(), self.center.copy()

    # ── Power evaluation ──────────────────────────────────────────────────────

    def emitted_power(self, t: float) -> float:
        """Return total emitted power E [W] at time *t* [s]."""
        if callable(self.power):
            return float(self.power(t))
        return float(self.power)

    # ── Per-quad flux (§3.5.4) ────────────────────────────────────────────────

    def flux_at(
        self,
        point: np.ndarray,
        normal: np.ndarray | None = None,
        t: float = 0.0,
    ) -> float:
        """
        Return the heat flux [W/m²] received at *point* at time *t*.

        Parameters
        ----------
        point  : (3,) global position of the quad centroid [m]
        normal : (3,) outward unit normal of the receiving surface.
                 When provided, the cosine directional term is applied.
                 When ``None``, the undirectional inverse-square flux
                 E / (4π r²) is returned (useful for distance-only checks).
        t      : simulation time [s] (used when ``power`` is callable)

        Returns
        -------
        float : flux [W/m²], clamped to zero for back-facing surfaces.
        """
        r_vec = np.asarray(point, dtype=float) - np.asarray(self.center, dtype=float)
        r = float(np.linalg.norm(r_vec))
        if r < 1e-9:
            return 0.0

        E = self.emitted_power(t)
        q_omni = E / (4.0 * math.pi * r * r)

        if normal is None:
            return q_omni

        # cos(theta_i): angle between outward normal and the ray from source to point.
        # The radiation ray travels in direction r_vec/r (source → point).
        # The face receives radiation when dot(normal, -ray) > 0, i.e. face points
        # toward the source.  This equals -dot(normal, r_hat).
        r_hat = r_vec / r
        cos_theta = float(np.dot(np.asarray(normal, dtype=float), -r_hat))
        return q_omni * max(0.0, cos_theta)

    def per_quad_flux(
        self,
        quad_centroids: np.ndarray,
        quad_normals: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute per-quad flux [W/m²] for all quads simultaneously.

        Parameters
        ----------
        quad_centroids : (n_quads, 3) global centroid positions [m]
        quad_normals   : (n_quads, 3) outward unit normals (need not be normalised)
        t              : simulation time [s]

        Returns
        -------
        (n_quads,) flux array [W/m²]; back-facing quads receive 0.
        """
        centers = np.asarray(quad_centroids, dtype=float)  # (n, 3)
        norms   = np.asarray(quad_normals,   dtype=float)  # (n, 3)

        r_vecs = centers - self.center                      # (n, 3)
        r      = np.linalg.norm(r_vecs, axis=1)            # (n,)

        E         = self.emitted_power(t)
        q_omni    = np.where(r > 1e-9, E / (4.0 * math.pi * r * r), 0.0)

        # Normalise normals and r_vecs for cos computation
        r_safe    = np.where(r > 1e-9, r, 1.0)[:, None]
        r_hat     = r_vecs / r_safe                         # (n, 3)

        # Normalise surface normals
        n_len     = np.linalg.norm(norms, axis=1, keepdims=True)
        n_safe    = np.where(n_len > 1e-9, n_len, 1.0)
        n_hat     = norms / n_safe                          # (n, 3)

        # cos(theta_i) = dot(n_hat, -r_hat) = -dot(n_hat, r_hat)
        cos_theta = -np.einsum("ij,ij->i", n_hat, r_hat)   # (n,)
        cos_theta = np.maximum(cos_theta, 0.0)

        return q_omni * cos_theta
