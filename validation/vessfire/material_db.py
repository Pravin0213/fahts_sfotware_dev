"""Read steel property tables from a local VessFire database (vessfire.db).

Used only for calibration and golden regression, so both models get identical property
input. The database is VessFire's and is never committed; keep a local copy at
``data/reference/vessfire/vessfire.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from fahts.materials import SteelTable

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "reference" / "vessfire" / "vessfire.db"


def load_steel_table(name: str, db: Path = DEFAULT_DB) -> SteelTable:
    """Material table ``name`` from the ``metals`` table, opened read-only."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select temperature, Cp, Conduct, Dens, F_Yield, F_UTS from metals "
        "where name = ? order by temperature", (name,)).fetchall()
    con.close()
    if not rows:
        raise ValueError(f"material {name!r} not found in {db}")
    a = np.array([[r[0], r[1], r[2], r[4], r[5]] for r in rows], dtype=float)
    rho = next(r[3] for r in rows if r[3] is not None)
    return SteelTable(name, a[:, 0], a[:, 1], a[:, 2], float(rho), a[:, 3], a[:, 4])
