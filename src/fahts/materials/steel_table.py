"""Tabulated temperature-dependent steel properties for wall conduction and rupture.

Sources: the built-in EN 1993-1-2 carbon steel (``fahts.materials.en1993_carbon_steel``) or a
user table (CSV: ``SteelTable.from_csv``; embedded in case files via ``to_dict``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SteelTable:
    """Temperature-dependent steel properties (T in K), linear interpolation."""

    name: str
    T: np.ndarray
    cp: np.ndarray  # J/kg/K
    k: np.ndarray  # W/m/K
    rho: float  # kg/m3 (constant)
    f_yield: np.ndarray  # yield retention factor
    f_uts: np.ndarray  # UTS retention factor

    def __post_init__(self):
        for name in ("T", "cp", "k", "f_yield", "f_uts"):
            a = np.asarray(getattr(self, name), float)
            if a.shape != np.shape(self.T):
                raise ValueError(f"{self.name}: {name} has shape {a.shape}, T {np.shape(self.T)}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{self.name}: {name} contains non-finite values")
        if np.any(np.diff(self.T) <= 0):
            raise ValueError(f"{self.name}: temperatures must be strictly increasing")

    # ------------------------------------------------------------------ serialisation
    CSV_COLUMNS = ("T_C", "cp_J_kgK", "k_W_mK", "f_yield", "f_uts")

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            rho=float(self.rho),
            T_C=(np.asarray(self.T) - 273.15).tolist(),
            cp=list(map(float, self.cp)),
            k=list(map(float, self.k)),
            f_yield=list(map(float, self.f_yield)),
            f_uts=list(map(float, self.f_uts)),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SteelTable":
        return cls(
            name=d["name"],
            T=np.asarray(d["T_C"], float) + 273.15,
            cp=np.asarray(d["cp"], float),
            k=np.asarray(d["k"], float),
            rho=float(d["rho"]),
            f_yield=np.asarray(d["f_yield"], float),
            f_uts=np.asarray(d["f_uts"], float),
        )

    def to_csv(self, path: Path | str) -> None:
        """Write ``# name``, ``# rho_kg_m3`` header lines, then the property columns."""
        with open(path, "w", newline="") as f:
            f.write(f"# name: {self.name}\n# rho_kg_m3: {self.rho}\n")
            w = csv.writer(f)
            w.writerow(self.CSV_COLUMNS)
            for row in zip(np.asarray(self.T) - 273.15, self.cp, self.k, self.f_yield, self.f_uts):
                w.writerow([f"{v:.6g}" for v in row])

    @classmethod
    def from_csv(
        cls, path: Path | str, name: str | None = None, rho: float | None = None
    ) -> "SteelTable":
        """Read a table written by ``to_csv`` (or any CSV with those columns; ``name`` and
        ``rho`` from the ``#`` header lines or the arguments)."""
        meta, rows = {}, []
        with open(path, newline="") as f:
            lines = [ln for ln in f if ln.strip()]
        for ln in lines:
            if ln.startswith("#") and ":" in ln:
                k_, v_ = ln[1:].split(":", 1)
                meta[k_.strip()] = v_.strip()
        reader = csv.DictReader(ln for ln in lines if not ln.startswith("#"))
        missing = set(cls.CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for r in reader:
            rows.append([float(r[c]) for c in cls.CSV_COLUMNS])
        a = np.asarray(rows, float)
        rho = rho if rho is not None else float(meta.get("rho_kg_m3", "nan"))
        if not np.isfinite(rho):
            raise ValueError(f"{path}: density not given (header '# rho_kg_m3: ...' or rho=)")
        return cls(
            name=name or meta.get("name", Path(path).stem),
            T=a[:, 0] + 273.15,
            cp=a[:, 1],
            k=a[:, 2],
            rho=rho,
            f_yield=a[:, 3],
            f_uts=a[:, 4],
        )

    # ------------------------------------------------------------------ properties
    def cp_at(self, T):
        return np.interp(T, self.T, self.cp)

    def k_at(self, T):
        return np.interp(T, self.T, self.k)

    def f_yield_at(self, T):
        return np.interp(T, self.T, self.f_yield)

    def f_uts_at(self, T):
        return np.interp(T, self.T, self.f_uts)
