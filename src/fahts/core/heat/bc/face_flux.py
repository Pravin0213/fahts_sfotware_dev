"""
Time-varying per-face directional flux from point / line sources.

A ConcentratedSource or LineSource flux is linear in its emitted power E(t) while
the geometry (distance, cos θ) is fixed, so each source contributes

    q_s(face, t) = E_s(t) · g_s(face)

where g_s is the flux pattern at unit power (1 W), computed once.  A time-invariant
part (RadiationBall) is carried separately.  Calling the object returns the total
(n_faces,) flux [W/m²] at time t, which the solvers evaluate every assembly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class TimeVaryingFaceFlux:
    """
    Per-face flux schedule q(t) = static + Σ_s E_s(t) · pattern_s   [W/m²].

    Attributes:
        n_faces: number of faces the schedule covers.
        static:  optional (n_faces,) time-invariant flux (e.g. RadiationBall).
        terms:   list of (E_s(t) [W], pattern_s (n_faces,) [W/m² per W]).
    """
    n_faces: int
    static: np.ndarray | None = None
    terms: list[tuple[Callable[[float], float], np.ndarray]] = field(default_factory=list)
    #: source object of each entry in ``terms`` (used for shielding)
    sources: list = field(default_factory=list)

    def add_source(
        self, source, pattern_fn: Callable[[object], np.ndarray]
    ) -> np.ndarray:
        """
        Add one point/line source; returns its unit-power pattern.

        ``pattern_fn(unit_source)`` must return the (n_faces,) flux of a copy of
        *source* emitting exactly 1 W (the geometry-only factor).
        """
        unit = dataclasses.replace(source, power=1.0)
        pattern = np.asarray(pattern_fn(unit), dtype=float)
        if pattern.shape != (self.n_faces,):
            raise ValueError(f"pattern shape {pattern.shape} != ({self.n_faces},)")
        if np.any(pattern > 0.0):
            self.terms.append((source.emitted_power, pattern))
            self.sources.append(source)
        return pattern

    @property
    def lit(self) -> bool:
        """True when any face can ever receive flux (static or any source pattern)."""
        if self.static is not None and np.any(self.static > 0.0):
            return True
        return bool(self.terms)

    def __call__(self, t: float) -> np.ndarray:
        q = np.zeros(self.n_faces) if self.static is None else self.static.copy()
        for power, pattern in self.terms:
            q += float(power(t)) * pattern
        return q
