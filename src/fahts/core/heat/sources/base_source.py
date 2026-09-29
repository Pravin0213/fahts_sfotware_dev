"""Abstract base class for all heat sources."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class HeatSource(ABC):
    """Base class for FAHTS heat sources (fire zones, point radiators, jet fires)."""

    name: str
    active: bool

    @abstractmethod
    def temperature(self, t: float) -> float:
        """Return fire/source gas temperature [°C] at time *t* [s]."""

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box in global coordinates [m]."""
