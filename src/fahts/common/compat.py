"""Compatibility helpers across supported library versions."""

import numpy as np

# NumPy 2 renamed trapz -> trapezoid (same function); support both.
trapezoid = getattr(np, "trapezoid", None) or np.trapz
