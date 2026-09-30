"""Golden-output harness for the process (vessel) model.

The goldens freeze the behaviour of the product code (``fahts.coupling``), so any change in
results is visible and deliberate. History: until 2026-09-30 they were generated from the
original vfpy code (``legacy/vfpy/``) to verify the port bit for bit; they were regenerated
from ``fahts`` when known issue #1 (wall grid) was fixed - see the manifest and
``docs/process_model_known_issues.md``.

- ``CASES``     the golden case set (input decks in ``cases/<id>/``)
- ``PROFILES``  named option sets (VessFire-matching default and physics-preferred)
- ``RUNS``      the (case, profile, duration) combinations that have golden files
- ``run(golden_run, impl)`` runs one with one implementation and returns
  (time series, rupture table)

Implementations: ``"fahts"`` = the product code (checked against the goldens);
``"legacy"`` = frozen vfpy snapshot (diagnostics and before/after comparisons only).

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

from validation.vessfire.input_deck import read_case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE_DIR = HERE / "cases"
GOLDEN_DIR = HERE / "golden"
LEGACY_VFPY = ROOT / "legacy" / "vfpy"
MATERIAL_DB = ROOT / "data" / "reference" / "vessfire" / "vessfire.db"


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


def _run_fahts(case: dict, t_end: float, opts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The product code only: fahts.coupling + fahts.rupture (no legacy import)."""
    from fahts.coupling import VesselFireOptions, simulate
    from fahts.rupture import evaluate
    from validation.vessfire.material_db import load_steel_table

    mat = load_steel_table(case["seg"]["material"], MATERIAL_DB)
    opt = VesselFireOptions(t_end=t_end, out_every=OUT_EVERY, **opts)
    ts, meta = simulate(case, opt, mat)
    _ss, failures = evaluate(ts, case, mat, x_nodes=meta["x_nodes"])
    return ts, failures


IMPLEMENTATIONS = {"fahts": _run_fahts, "legacy": _run_legacy}
GOLDEN_IMPLEMENTATIONS = ("fahts",)       # implementations that must reproduce the goldens


def run(gr: GoldenRun, impl: str = "fahts") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one golden run; returns (time series, rupture/failure table)."""
    case = read_case(CASE_DIR / gr.case_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IMPLEMENTATIONS[impl](case, gr.t_end, PROFILES[gr.profile])


def golden_paths(gr: GoldenRun) -> tuple[Path, Path]:
    return GOLDEN_DIR / f"{gr.name}.csv", GOLDEN_DIR / f"{gr.name}__rupture.csv"
