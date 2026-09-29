"""
Line heat source — FAHTS Theory Manual §3.5.5.

The line source is simulated by n discrete concentrated sources (§3.5.4) placed
uniformly along the line from *start* to *end*.  Each interior sub-source has
energy emittance ΔE = (E/L) · ΔL and the two end sub-sources emit 50% of that
value, matching the FAHTS manual prescription.

Flux at a receiving surface element is the sum of contributions from all n
sub-sources:

    Q = Σ_{i=1}^{n}  ΔE_i · cos(θ_i) / (4π · r_i²)

where r_i is the distance from sub-source i to the quad centroid and θ_i is the
angle between the outward surface normal and the direction from sub-source i to
the quad centroid.  cos(θ_i) is clamped to 0 for back-facing quads.

Total power E(t) [W] may be time-dependent (pass a callable).  The linear power
density is uniform along the line: e = E / L [W/m].

The source is fully directional — no convective or Stefan-Boltzmann term.  Set
epsilon_m = 0 in the solver when using this source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fahts.core.heat.sources.base_source import HeatSource


@dataclass
class LineSource(HeatSource):
    """
    Finite line heat source with inverse-square directional falloff (§3.5.5).

    The line is divided into n_segments discrete point sub-sources.  End
    sub-sources contribute 50% of the interior sub-source energy, as specified
    in the FAHTS Theory Manual.

    Parameters
    ----------
    name       : identifier string
    start      : (3,) global coordinate of source end 1 [m]
    end        : (3,) global coordinate of source end 2 [m]
    power      : total emitted power E [W] — constant float or callable(t [s]) → float
    active     : whether this source participates in the analysis (default True)

    Physics
    -------
    Linear power density: e = E(t) / L  [W/m]
    Sub-source spacing:   ΔL = L / (n − 1)  (n = n_segments discrete points)
    Interior sub-source:  ΔE = e · ΔL = E · ΔL / L
    End sub-sources:      ΔE_end = 0.5 · ΔE

    Flux at quad centroid from sub-source i:
        q_i = ΔE_i · cos(θ_i) / (4π · r_i²)

    cos(θ_i) is clamped to 0 for back-facing quads (face pointing away from the
    sub-source).  Set epsilon_m = 0 in the solver; re-radiation is handled by
    the separate eps_rerad path.

    The source temperature is not meaningful; temperature() returns ambient 20°C.
    """

    name: str
    start: np.ndarray                          # shape (3,) [m]
    end: np.ndarray                            # shape (3,) [m]
    power: float | Callable[[float], float]    # E [W] or callable(t) → E(t)
    active: bool = field(default=True)

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float)
        self.end   = np.asarray(self.end,   dtype=float)
        length = float(np.linalg.norm(self.end - self.start))
        if length < 1e-12:
            raise ValueError(
                f"LineSource '{self.name}': start and end are coincident "
                f"(distance={length:.3e} m)."
            )

    # ── HeatSource ABC ────────────────────────────────────────────────────────

    def temperature(self, t: float) -> float:
        """Not meaningful for LineSource; returns ambient 20 °C."""
        return 20.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box enclosing the line segment [m]."""
        lo = np.minimum(self.start, self.end)
        hi = np.maximum(self.start, self.end)
        return lo, hi

    # ── Power evaluation ──────────────────────────────────────────────────────

    def emitted_power(self, t: float) -> float:
        """Return total emitted power E [W] at time *t* [s]."""
        if callable(self.power):
            return float(self.power(t))
        return float(self.power)

    # ── Sub-source geometry ───────────────────────────────────────────────────

    def _sub_sources(
        self,
        t: float,
        n_segments: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (positions, energies) for the n_segments discrete sub-sources.

        Parameters
        ----------
        t          : simulation time [s]
        n_segments : number of discrete points along the line (≥ 2)

        Returns
        -------
        positions : (n_segments, 3) global coordinates [m]
        energies  : (n_segments,) energy emittance per sub-source [W]
                    (time-averaged instantaneous power, same units as §3.5.4 E)
        """
        n = max(2, n_segments)
        E = self.emitted_power(t)
        L = float(np.linalg.norm(self.end - self.start))

        # n uniformly-spaced positions along the segment
        alphas    = np.linspace(0.0, 1.0, n)                     # (n,)
        positions = self.start + alphas[:, None] * (self.end - self.start)  # (n, 3)

        # Linear power density (uniform): e = E / L [W/m]
        dL = L / (n - 1)                         # spacing between sub-sources
        dE = (E / L) * dL                        # interior sub-source energy

        energies = np.full(n, dE)
        energies[0]  *= 0.5                      # end source: 50% reduction
        energies[-1] *= 0.5                      # end source: 50% reduction

        return positions, energies

    # ── Per-quad flux (§3.5.5) ────────────────────────────────────────────────

    def flux_at(
        self,
        point: np.ndarray,
        normal: np.ndarray | None = None,
        t: float = 0.0,
        n_segments: int = 10,
    ) -> float:
        """
        Return total heat flux [W/m²] at *point* from all sub-sources at time *t*.

        Parameters
        ----------
        point      : (3,) global position of the quad centroid [m]
        normal     : (3,) outward unit normal of the receiving surface.
                     When provided, the cosine directional term is applied.
                     When None, the undirectional sum over all sub-sources is
                     returned (useful for distance-only checks).
        t          : simulation time [s]
        n_segments : number of discrete sub-sources (default 10)

        Returns
        -------
        float : flux [W/m²], clamped to zero for back-facing surfaces.
        """
        pt = np.asarray(point, dtype=float)
        positions, energies = self._sub_sources(t, n_segments)

        q_total = 0.0
        for pos, dE in zip(positions, energies):
            r_vec = pt - pos
            r     = float(np.linalg.norm(r_vec))
            if r < 1e-9:
                continue
            q_omni = dE / (4.0 * math.pi * r * r)

            if normal is None:
                q_total += q_omni
            else:
                # cos(θ): dot of outward normal with direction from sub-source to quad.
                # Radiation travels along r_vec/r; face receives it when n · (−r_hat) > 0.
                r_hat     = r_vec / r
                n_arr     = np.asarray(normal, dtype=float)
                n_len     = float(np.linalg.norm(n_arr))
                n_hat     = n_arr / n_len if n_len > 1e-9 else n_arr
                cos_theta = float(np.dot(n_hat, -r_hat))
                if cos_theta > 0.0:
                    q_total += q_omni * cos_theta

        return q_total

    def per_quad_flux(
        self,
        quad_centroids: np.ndarray,
        quad_normals: np.ndarray,
        t: float = 0.0,
        n_segments: int = 10,
    ) -> np.ndarray:
        """
        Compute per-quad flux [W/m²] for all quads simultaneously.

        Parameters
        ----------
        quad_centroids : (n_quads, 3) global centroid positions [m]
        quad_normals   : (n_quads, 3) outward unit normals (need not be unit-length)
        t              : simulation time [s]
        n_segments     : number of discrete sub-sources (default 10)

        Returns
        -------
        (n_quads,) flux array [W/m²]; back-facing quads receive 0.
        """
        centers = np.asarray(quad_centroids, dtype=float)   # (nq, 3)
        norms   = np.asarray(quad_normals,   dtype=float)   # (nq, 3)

        # Normalise surface normals once
        n_len  = np.linalg.norm(norms, axis=1, keepdims=True)   # (nq, 1)
        n_safe = np.where(n_len > 1e-9, n_len, 1.0)
        n_hat  = norms / n_safe                                  # (nq, 3)

        positions, energies = self._sub_sources(t, n_segments)

        q_total = np.zeros(len(centers))
        for pos, dE in zip(positions, energies):
            r_vecs = centers - pos                               # (nq, 3)
            r      = np.linalg.norm(r_vecs, axis=1)             # (nq,)
            valid  = r > 1e-9
            r_safe = np.where(valid, r, 1.0)

            q_omni = np.where(valid, dE / (4.0 * math.pi * r_safe * r_safe), 0.0)

            r_hat     = r_vecs / r_safe[:, None]                 # (nq, 3)
            # cos(θ) = dot(n_hat, −r_hat) = −dot(n_hat, r_hat)
            cos_theta = -np.einsum("ij,ij->i", n_hat, r_hat)    # (nq,)
            cos_theta = np.maximum(cos_theta, 0.0)

            q_total += q_omni * cos_theta

        return q_total
