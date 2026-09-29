"""Access to the frozen vfpy reference (legacy/vfpy) for equivalence tests.

Only tests may import legacy code, and only through this helper.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

LEGACY_VFPY = Path(__file__).resolve().parents[1] / "legacy" / "vfpy"


def legacy(module: str):
    """Import a module of the frozen vfpy snapshot by its original flat name."""
    if not hasattr(np, "trapezoid"):          # vfpy targets NumPy 2; same function
        np.trapezoid = np.trapz
    if str(LEGACY_VFPY) not in sys.path:
        sys.path.insert(0, str(LEGACY_VFPY))
    return importlib.import_module(module)
