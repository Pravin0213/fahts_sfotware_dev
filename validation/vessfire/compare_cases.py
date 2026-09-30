"""Compare the fahts vessel model with VessFire reference results on validation-set cases.

    python -m validation.vessfire.compare_cases wall_grid_fix            # built-in study
    python -m validation.vessfire.compare_cases wall_grid_fix --per-module 2
    python -m validation.vessfire.compare_cases wall_grid_fix --report-only

Each study runs one or more model configurations on a case sample in parallel and writes
``validation/vessfire/reports/<study>.csv`` (one row per case x configuration) and
``<study>.md`` (summary). The reference data set lives in ``data/reference/vessfire/``
(gitignored; VessFire output, never committed).

Metrics (as in the vfpy calibration studies, ``Test/vfpy/compare.py``):
  P_rms_pct      RMS of (P_model - P_VF) / max(|P_VF|, 1 barg) over the run [%]
  Tgas_rms_K     gas temperature RMS difference [K]
  Tdry_rms_K     dry-wall through-wall mean vs VessFire hottest-location mean (TS_max) [K]
  Twet_rms_K     wetted-wall mean vs VessFire coldest-location mean (TS_min), liquid cases [K]
  Tliq_rms_K     liquid temperature vs T_oil while liquid is present [K]
  rupt_Tr_s      Tresca rupture time (membrane check), model and VessFire [s]
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from validation.vessfire.input_deck import read_case
from validation.vessfire.material_db import DEFAULT_DB, load_steel_table

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SET = ROOT / "data" / "reference" / "vessfire" / "validation_set"
REPORTS = Path(__file__).resolve().parent / "reports"
P_ATM = 101325.0

# vfpy's wall grid (node positions from the inner surface for a 105 mm shell), used for
# every vessel before known issue #1 was fixed. Reproduces the "before" state.
LEGACY_NODES_105MM = (0.0, 0.00525, 0.01575, 0.02625, 0.03675, 0.04725,
                      0.05775, 0.06825, 0.07875, 0.08925, 0.1017, 0.105)


# ------------------------------------------------------------------ VessFire results
def _vre(txt: str) -> pd.DataFrame:
    """VesselRes.vre -> DataFrame. Tn columns: 12 hottest-location nodes, then 12 coldest."""
    lines = txt.splitlines()
    hdr, out, k = lines[0].split("#", 1)[-1].split(), [], 0
    for name in hdr:
        if name == "Tn":
            k += 1
            out.append(f"Tn_hot{k:02d}" if k <= 12 else f"Tn_cold{k - 12:02d}")
        else:
            out.append(name)
    rows = []
    for ln in lines[1:]:
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            continue
    n = min(len(out), min(len(r) for r in rows))
    out[0] = "Time"   # the first header token is not the first data column: column 0 is time [s]
    return pd.DataFrame([r[:n] for r in rows], columns=out[:n])


def load_vessfire(case_dir: Path) -> tuple[pd.DataFrame, dict]:
    """VessFire time series (t > 0, derived columns in C / barg) and summary.json."""
    with zipfile.ZipFile(case_dir / "results.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("VesselRes.vre"))
        vf = _vre(z.read(name).decode("latin-1"))
    vf = vf[vf.Time > 0].reset_index(drop=True)
    vf["P_barg"] = (vf.Pres_applied - P_ATM) / 1e5
    vf["T_gas_C"] = vf.T_gas - 273.15
    vf["T_hot_mean_C"] = vf.TS_max - 273.15
    vf["T_cold_mean_C"] = vf.TS_min - 273.15
    vf["M_liq_total"] = vf.Mass_oil + vf.Mass_water
    return vf, json.loads((case_dir / "summary.json").read_text())


# ------------------------------------------------------------------ metrics
def _rms(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def metrics(py: pd.DataFrame, meta: dict, vf: pd.DataFrame, summ: dict) -> dict:
    vf = vf.iloc[:-1]                     # the final output row reports a zero release rate
    t = vf.Time.values
    ip = lambda col: np.interp(t, py.Time, py[col])  # noqa: E731
    P_vf = vf.P_barg.values
    out = dict(
        P_rms_pct=100 * float(np.sqrt(np.mean(((ip("P_barg") - P_vf)
                                               / np.maximum(np.abs(P_vf), 1.0)) ** 2))),
        Tgas_rms_K=_rms(ip("T_gas_C"), vf.T_gas_C),
        Tdry_rms_K=_rms(ip("background_T_mean_C"), vf.T_hot_mean_C),
        rupt_Tr_model_s=meta["rupture"]["Tresca"],
        rupt_Tr_vf_s=summ.get("rupture_Tresca_s"),
        energy_err_pct=float(py.energy_err_pct.iloc[-1]),
    )
    if vf.M_liq_total.max() > 1.0:
        live = (vf.M_liq_total > 0.01 * vf.M_liq_total.max()).values
        out["Tliq_rms_K"] = _rms(np.interp(t[live], py.Time, py.T_liq_C),
                                 vf.T_oil.values[live] - 273.15)
        out["Twet_rms_K"] = _rms(ip("wet_T_mean_C"), vf.T_cold_mean_C)
    return out


# ------------------------------------------------------------------ running
def _run_one(args) -> dict:
    case_dir, config, opts = args
    warnings.simplefilter("ignore")
    from fahts.coupling import VesselFireOptions, simulate

    row = dict(case=case_dir.name[:8], variation=case_dir.name[9:], config=config)
    t0 = time.perf_counter()
    try:
        case = read_case(case_dir / "inputs")
        row.update(wall_m=case["seg"]["t"], D_m=case["seg"]["D"])
        mat = load_steel_table(case["seg"]["material"], DEFAULT_DB)
        py, meta = simulate(case, VesselFireOptions(**opts), mat)
        vf, summ = load_vessfire(case_dir)
        row.update(metrics(py, meta, vf, summ))
    except Exception as e:  # noqa: BLE001 - report and continue with the other cases
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    row["runtime_s"] = round(time.perf_counter() - t0, 1)
    return row


def select_cases(per_module: int, seed: int = 0, include: tuple[str, ...] = ()) -> list[Path]:
    """Completed, horizontal, uninsulated cases: ``per_module`` random per module + include."""
    idx = pd.read_csv(VALIDATION_SET / "results_index.csv", low_memory=False)
    ok = idx[(idx.status == "completed") & (idx.orientation == "H") & idx.insulation.isna()]
    ids = set(include)
    for _, grp in ok.groupby("module"):
        ids.update(grp.sample(min(per_module, len(grp)), random_state=seed).case_id)
    dirs = sorted(p for p in (VALIDATION_SET / "cases").glob("*/*")
                  if p.name[:8] in ids and (p / "results.zip").exists())
    return dirs


def run_study(name: str, configs: dict[str, dict], cases: list[Path]) -> pd.DataFrame:
    jobs = [(c, cfg, opts) for c in cases for cfg, opts in configs.items()]
    log.info("%s: %d cases x %d configs = %d runs", name, len(cases), len(configs), len(jobs))
    with mp.Pool(min(len(jobs), mp.cpu_count())) as pool:
        rows = pool.map(_run_one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / f"{name}.csv", index=False)
    return df


# ------------------------------------------------------------------ report
METRICS = ["P_rms_pct", "Tgas_rms_K", "Tdry_rms_K", "Twet_rms_K", "Tliq_rms_K",
           "rupt_err_s", "energy_err_abs_pct"]


def summarize(name: str, df: pd.DataFrame | None = None) -> str:
    """Markdown summary of a study CSV: medians per configuration, by wall thickness, and
    rupture-time agreement."""
    df = pd.read_csv(REPORTS / f"{name}.csv") if df is None else df.copy()
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].copy()
    ok["rupt_err_s"] = (ok.rupt_Tr_model_s - ok.rupt_Tr_vf_s).abs()
    ok["energy_err_abs_pct"] = ok.energy_err_pct.abs()
    ok["wall"] = pd.cut(ok.wall_m, [0, 0.03, 0.07, 0.2], labels=["< 30 mm", "30-70 mm", "> 70 mm"])
    configs = list(dict.fromkeys(df.config))
    n_cases = ok.case.nunique()

    def med(frame):
        t = frame.groupby("config")[METRICS].median().reindex(configs)
        t.insert(0, "cases", frame.groupby("config").case.nunique().reindex(configs))
        return t.round(2)

    both = ok[ok.rupt_Tr_vf_s.notna() | ok.rupt_Tr_model_s.notna()]
    rup = both.groupby("config").apply(lambda g: pd.Series({
        "VF ruptures": int(g.rupt_Tr_vf_s.notna().sum()),
        "model ruptures": int(g.rupt_Tr_model_s.notna().sum()),
        "both": int((g.rupt_Tr_vf_s.notna() & g.rupt_Tr_model_s.notna()).sum()),
        "median |dt| s (both)": g.rupt_err_s.median(),
    })).reindex(configs)
    gold = ok[ok.case.isin(GOLDEN_CASES)].pivot_table(
        index="case", columns="config", values=["P_rms_pct", "Tdry_rms_K"]).round(1)
    lines = [f"# VessFire comparison: {name}", "",
             f"{n_cases} validation-set cases (completed, horizontal, uninsulated), "
             f"{len(df) - len(ok)} failed runs of {len(df)}. Medians over cases; lower is "
             "closer to VessFire. Metric definitions: `validation/vessfire/compare_cases.py`.",
             "", "## All cases", "", med(ok).to_markdown(), ""]
    for w, grp in ok.groupby("wall", observed=True):
        lines += [f"## Wall {w}", "", med(grp).to_markdown(), ""]
    lines += ["## Tresca rupture time (membrane check)", "", rup.to_markdown(), "",
              "## Golden cases", "", gold.to_markdown(), ""]
    if len(df) > len(ok):
        lines += ["## Failed runs", "", df[df.error.notna()][["case", "config", "error"]]
                  .to_markdown(index=False), ""]
    text = "\n".join(lines)
    (REPORTS / f"{name}.md").write_text(text)
    return text


# ------------------------------------------------------------------ studies
GOLDEN_CASES = ("M03-0003", "M04-0003", "M05-0003", "M06-0003", "M07-0004", "M08-0001",
                "M09-0001", "M10-0001", "M11-0002", "M12-0004", "M15-0002")

STUDIES = {
    "wall_grid_fix": {
        "before (105 mm grid)": dict(wall_nodes=LEGACY_NODES_105MM),
        "after (real t, 10 cells)": dict(),
        "after (real t, 20 cells)": dict(wall_cells=20),
    },
}


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = argv[0]
    per_module = int(argv[argv.index("--per-module") + 1]) if "--per-module" in argv else 4
    if "--report-only" not in argv:
        cases = select_cases(per_module, include=GOLDEN_CASES)
        run_study(name, STUDIES[name], cases)
    summarize(name)


if __name__ == "__main__":
    main(sys.argv[1:])
