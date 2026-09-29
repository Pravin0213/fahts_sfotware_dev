"""Pipe friction: Darcy friction factor.
"""

from __future__ import annotations

import math


def colebrook(Re, rel_rough):
    """Darcy friction factor, Colebrook (1939); laminar 64/Re below Re = 2300."""
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    f = (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2   # Haaland (1983)
    for _ in range(6):
        f = (-2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
    return f
