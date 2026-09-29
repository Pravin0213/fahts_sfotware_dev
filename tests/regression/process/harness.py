"""Golden-output harness for the process (vessel) model.

Freezes the behaviour of the original vfpy code (snapshot in ``legacy/vfpy/``) so that the
port into ``src/fahts/`` can be checked module by module against it.

- ``CASES``     the golden case set (input decks in ``cases/<id>/``)
- ``PROFILES``  named option sets (VessFire-matching default and physics-preferred)
- ``RUNS``      the (case, profile, duration) combinations that have golden files
- ``run(golden_run, impl)`` runs one with one implementation and returns
  (time series, rupture table)

Implementations: ``"legacy"`` = frozen vfpy snapshot. The ported code is added here as a
second implementation once it exists; the regression test runs every implementation
against the same golden files.

The steel property tables come from the VessFire database, which may not be committed.
A local copy is expected at ``data/reference/vessfire/vessfire.db`` (gitignored); without
it the golden tests are skipped.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE_DIR = HERE / "cases"
GOLDEN_DIR = HERE / "golden"
LEGACY_VFPY = ROOT / "legacy" / "vfpy"
MATERIAL_DB = ROOT / "data" / "reference" / "vessfire" / "vessfire.db"
P_ATM = 101325.0


@dataclass(frozen=True)
class GoldenRun:
    case_id: str
    profile: str         # key of PROFILES
    t_end: float         # simulated seconds

    @property
    def name(self) -> str:
        return f"{self.case_id}__{self.profile}__{self.t_end:.0f}s"


# One case per physics branch of the process model (input decks in cases/<id>/,
# copied from Test/validation_set/cases/<module>/<id>_*/inputs).
CASES: dict[str, str] = {
    "M03-0003": "H2, no fire, ambient exchange (wind h_field)",
    "M04-0003": "H2 cold blowdown (BDV + line)",
    "M05-0003": "LPG two-phase, PSV relief (cold, pressurised)",
    "M06-0003": "H2 closed vessel in 150 kW/m2 fire (Tresca failure ~870 s)",
    "M07-0004": "CH4 fire + BDV",
    "M08-0001": "CH4 fire + PSV (type 0)",
    "M09-0001": "CH4 fire + BDV + PSV",
    "M10-0001": "CH4 250 kW/m2 fire, stress type Y (rupture)",
    "M11-0002": "C1 + pseudo-component liquid, fire + BDV",
    "M12-0004": "gas + free water, fire + BDV",
    "M15-0002": "retrograde rich gas, fire + BDV",
}

# Option sets. "vf" = current default (VessFire-matching, MODEL_CHOICES.md N5);
# "physics" = physics-preferred alternatives, to cover the other code branches.
PROFILES: dict[str, dict] = {
    "vf": {},
    "physics": dict(wet_above_crit="single-phase", wet_boiling="full",
                    water_mode="physical", psv_liquid="gas"),
}

_NO_FIRE = ("M03-0003", "M04-0003", "M05-0003")
_LIQUID = ("M05-0003", "M11-0002", "M12-0004", "M15-0002")

# Short runs of every case (fast), physics profile on the liquid cases, and three
# full-hour runs for rupture timing and late-stage behaviour (liquid-full, dense relief).
RUNS: list[GoldenRun] = (
    [GoldenRun(c, "vf", 600.0 if c in _NO_FIRE else 900.0) for c in CASES]
    + [GoldenRun(c, "physics", 600.0 if c in _NO_FIRE else 900.0) for c in _LIQUID]
    + [GoldenRun(c, "vf", 3600.0) for c in ("M06-0003", "M10-0001", "M11-0002")]
)

OUT_EVERY = 10.0          # s between output rows


def have_material_db() -> bool:
    return MATERIAL_DB.exists()


# ------------------------------------------------------------------ input deck reader
# Verbatim copy of vfpy's vessfire_io.read_case (that module cannot be imported: it pulls
# in Windows-only tooling at import time).

def _lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(errors="replace").splitlines()]


def read_case(case_dir: Path) -> dict:
    case_dir = Path(case_dir)
    admin = {}
    for ln in _lines(case_dir / "Admin.brl"):
        p = ln.split()
        if len(p) >= 2 and p[0].startswith("#"):
            admin[p[0][1:].lower()] = p[1]

    seg = {"fluid": {}, "bdv": None, "psv": None, "back_pressure": P_ATM}
    in_fluid = False
    for ln in _lines(case_dir / "Segment.brl"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "#Fluid":
            in_fluid = True
            continue
        if in_fluid and not p[0].startswith("#"):
            if len(p) >= 4:                                   # pseudo: frac, rel. density/API, Tb
                seg.setdefault("pseudo", {})[p[0].upper()] = dict(rd=float(p[2]), Tb=float(p[3]))
            seg["fluid"][p[0].upper()] = float(p[1])
            continue
        in_fluid = False
        key = p[0]
        if key == "#Vessel":
            seg.update(tag=p[1], strength_mpa=float(p[2]), material=p[3],
                       D=float(p[4]), t=float(p[5]), L=float(p[6]))
        elif key == "#Vessel_conditions":
            seg.update(P0=float(p[1]) * 1e3, T0=float(p[2]), hc_level=float(p[3]),
                       water_level=float(p[4]), T_shell=float(p[p.index("%Shell") + 1]))
        elif key == "#Vessel_Outside_Conditions":
            seg.update(T_env=float(p[1]), h_out=float(p[2]), eps_surf=float(p[3]))
        elif key == "#Vessel_Orientation":
            seg["orientation"] = p[1].upper()
        elif key == "#StressType":
            seg["stress_type"] = p[1].upper()
        elif key == "#StressFactor":
            seg["stress_factor"] = float(p[1])
        elif key == "#External_Longitudinal_Stress":
            seg["ext_long_mpa"] = float(p[1])
        elif key == "#Blowdown_valve":
            seg["bdv"] = dict(d=float(p[1]), cd=float(p[2]), delay=float(p[3]))
        elif key == "#Blowdown_line":
            seg["bdv_line"] = dict(d=float(p[1]), t=float(p[2]), L=float(p[3]))
        elif key == "#BDV_Valve_location":
            seg["bdv_loc"] = (float(p[1]), float(p[2]))
        elif key == "#Process_safety_valve":
            seg["psv"] = dict(d=float(p[1]), cd=float(p[2]), p_set=float(p[3]) * 1e3,
                              p_full=float(p[4]) * 1e3, p_reseat=float(p[5]) * 1e3,
                              type=int(p[6]))
        elif key == "#Back_pressure":
            seg["back_pressure"] = float(p[1]) * 1e3

    hl = {}
    txt = _lines(case_dir / "heatload.scn")
    num = lambda s: float(s.split(":")[-1])  # noqa: E731
    for ln in txt:
        if ln.startswith("Longitudinal direction, start"):
            hl["xi_start"] = num(ln)
        elif ln.startswith("end [0 - 1]"):
            hl["xi_end"] = num(ln)
        elif ln.startswith("Circumferential length"):
            hl["circ_deg"] = num(ln)
        elif ln.startswith("Angle of attack"):
            hl["attack_deg"] = num(ln)
    rows = []
    for ln in txt:
        p = ln.split()
        if len(p) == 3:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    hl["series"] = np.array(rows) if rows else np.zeros((1, 3))
    return dict(admin=admin, seg=seg, hl=hl, name=case_dir.name)


# ------------------------------------------------------------------ implementations

def _legacy_modules():
    """Import the frozen vfpy snapshot, pointing its material reader at the local DB."""
    if not hasattr(np, "trapezoid"):          # vfpy was written for NumPy 2; same function
        np.trapezoid = np.trapz
    if str(LEGACY_VFPY) not in sys.path:
        sys.path.insert(0, str(LEGACY_VFPY))
    import heat_transfer
    import stress
    import vessel2

    if not getattr(heat_transfer.Material, "_golden_db_patched", False):
        orig = heat_transfer.Material.from_vessfire_db.__func__

        def from_db(cls, name, db=MATERIAL_DB):
            return orig(cls, name, db=db)

        heat_transfer.Material.from_vessfire_db = classmethod(from_db)
        heat_transfer.Material._golden_db_patched = True
    return vessel2, stress


def _run_legacy(case: dict, t_end: float, opts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    vessel2, stress = _legacy_modules()
    opt = vessel2.Options2(t_end=t_end, out_every=OUT_EVERY, **opts)
    ts, _meta = vessel2.simulate2(case, opt)
    _ss, failures = stress.evaluate(ts, case)
    return ts, failures


IMPLEMENTATIONS = {"legacy": _run_legacy}


def run(gr: GoldenRun, impl: str = "legacy") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one golden run; returns (time series, rupture/failure table)."""
    case = read_case(CASE_DIR / gr.case_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IMPLEMENTATIONS[impl](case, gr.t_end, PROFILES[gr.profile])


def golden_paths(gr: GoldenRun) -> tuple[Path, Path]:
    return GOLDEN_DIR / f"{gr.name}.csv", GOLDEN_DIR / f"{gr.name}__rupture.csv"
