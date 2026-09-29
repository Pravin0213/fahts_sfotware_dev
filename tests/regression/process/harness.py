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

import multiprocessing as mp
import sys
import types
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


# ------------------------------------------------------------------ hybrid (port in progress)
# While vfpy is being ported, the legacy driver (vessel2.simulate2) runs on top of the
# ported modules: each ported module is installed under its legacy flat name before the
# legacy code is imported. Add an entry here as each module is ported; when the driver
# itself is ported, "fahts" calls src/fahts directly and the hybrid goes away.

def _shim(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _ported_modules() -> dict[str, types.ModuleType]:
    import fahts.fire as fire
    import fahts.thermo as thermo
    from fahts.common.constants import SIGMA
    from fahts.materials import SteelTable
    from fahts.thermo import component_data
    from fahts.wall.column_1d import REFERENCE_NODES_105MM, WallColumn
    from fahts.wall.column_1d.tridiagonal import solve_tridiagonal
    from validation.vessfire.material_db import load_steel_table

    class Material(SteelTable):          # legacy API: Material.from_vessfire_db(name)
        @classmethod
        def from_vessfire_db(cls, name, db=MATERIAL_DB):
            return load_steel_table(name, db)

    import fahts.relief.blowdown as blowdown
    import fahts.relief.ideal_gas as ideal_gas
    from fahts.relief.psv import PSVOpening
    from fahts.rupture import membrane_stresses

    def PSV(_fluid, spec, _model):       # legacy signature; only the opening is used
        return PSVOpening(spec)

    return {
        "valves": ideal_gas,
        "valves2": _shim("valves2", mdot=blowdown.mdot),
        "vessel": _shim("vessel", AmbientAuto=fire.AmbientAuto, PSV=PSV,
                        stresses=membrane_stresses, H_CORRELATIONS=_legacy_h_correlations()),
        "heat_transfer": _shim("heat_transfer", SIGMA=SIGMA, Material=Material,
                               GuidelineFire=fire.GuidelineFire, FireBC=fire.FireBC,
                               PrescribedFluxBC=fire.PrescribedFluxBC,
                               AmbientBC=fire.AmbientBC, WallColumn=WallColumn,
                               VESSFIRE_LOG_NODES_105MM=REFERENCE_NODES_105MM,
                               thomas=solve_tridiagonal),
        "thermo_pr": _shim("thermo_pr", PRMixture=thermo.PRMixture,
                           FlashResult=thermo.FlashResult, Phase=thermo.Phase,
                           characterise_pseudo=thermo.characterise_pseudo),
        "thermo_data": component_data,
    }


def _legacy_h_correlations() -> dict:
    """Not ported yet (moves to process/inner_ht): taken from the legacy file by source."""
    src = (LEGACY_VFPY / "vessel.py").read_text()
    start = src.index("H_CORRELATIONS = {")
    end = src.index("}\n", start) + 2
    ns: dict = {}
    exec(src[start:end], ns)  # noqa: S102 - trusted, frozen reference file
    return ns["H_CORRELATIONS"]


def _hybrid_worker(case: dict, t_end: float, opts: dict):
    """Runs in a fresh process so legacy and ported modules never share sys.modules."""
    warnings.simplefilter("ignore")
    sys.modules.update(_ported_modules())
    vessel2, _ = _legacy_modules()
    for obj in (vessel2.PRMixture, vessel2.WallColumn, vessel2.GuidelineFire,
                vessel2.AmbientAuto, vessel2.valves2.mdot, vessel2.mdot_orifice_line,
                vessel2.stresses):
        assert obj.__module__.startswith("fahts."), f"hybrid is not using the port: {obj}"
    from fahts.rupture import evaluate
    from fahts.wall.column_1d import REFERENCE_NODES_105MM
    from validation.vessfire.material_db import load_steel_table

    opt = vessel2.Options2(t_end=t_end, out_every=OUT_EVERY, **opts)
    ts, _meta = vessel2.simulate2(case, opt)
    mat = load_steel_table(case["seg"]["material"], MATERIAL_DB)
    _ss, failures = evaluate(ts, case, mat, x_nodes=REFERENCE_NODES_105MM)
    return ts, failures


def _run_hybrid(case: dict, t_end: float, opts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(_hybrid_worker, (case, t_end, opts))


def _run_fahts(case: dict, t_end: float, opts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The product code only: fahts.coupling + fahts.rupture (no legacy import)."""
    from fahts.coupling import VesselFireOptions, simulate
    from fahts.rupture import evaluate
    from fahts.wall.column_1d import REFERENCE_NODES_105MM
    from validation.vessfire.material_db import load_steel_table

    mat = load_steel_table(case["seg"]["material"], MATERIAL_DB)
    opt = VesselFireOptions(t_end=t_end, out_every=OUT_EVERY, **opts)
    ts, _meta = simulate(case, opt, mat)
    _ss, failures = evaluate(ts, case, mat, x_nodes=REFERENCE_NODES_105MM)
    return ts, failures


IMPLEMENTATIONS = {"legacy": _run_legacy, "hybrid": _run_hybrid, "fahts": _run_fahts}


def run(gr: GoldenRun, impl: str = "legacy") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one golden run; returns (time series, rupture/failure table)."""
    case = read_case(CASE_DIR / gr.case_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IMPLEMENTATIONS[impl](case, gr.t_end, PROFILES[gr.profile])


def golden_paths(gr: GoldenRun) -> tuple[Path, Path]:
    return GOLDEN_DIR / f"{gr.name}.csv", GOLDEN_DIR / f"{gr.name}__rupture.csv"
