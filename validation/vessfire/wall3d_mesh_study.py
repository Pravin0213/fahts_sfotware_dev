"""Mesh convergence of the 3-D vessel wall (coarse / default / fine) on a jet-fire and a
uniform-fire case.  python -m validation.vessfire.wall3d_mesh_study  ->  reports/wall3d_mesh.md
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings
from pathlib import Path

import pandas as pd

from fahts.coupling import VesselFireOptions, simulate
from validation.vessfire.input_deck import read_case
from validation.vessfire.material_db import DEFAULT_DB, load_steel_table

CASES = (Path(__file__).resolve().parents[2] / "data" / "reference" / "vessfire" / "validation_set"
         / "cases" / "M06_fire_closed")
REPORT = Path(__file__).resolve().parent / "reports" / "wall3d_mesh.md"
MESHES = {"coarse (36x20x4)": (36, 20, 4), "default (72x40x6)": (72, 40, 6),
          "fine (144x80x10)": (144, 80, 10)}
RUNS = [("M06-0070_F10_jet_top", "jet on top"), ("M06-0003_F01_flux_150", "uniform 150 kW/m2")]
T_END = 1500.0


def _one(args) -> dict:
    (cid, what), (mesh, (nt, nl, nr)) = args
    warnings.simplefilter("ignore")
    case = read_case(CASES / cid / "inputs")
    mat = load_steel_table(case["seg"]["material"], DEFAULT_DB)
    t0 = time.perf_counter()
    ts, meta = simulate(case, VesselFireOptions(t_end=T_END, wall3d_n_theta=nt,
                                                wall3d_n_length=nl, wall3d_n_radial=nr), mat)
    last = ts.iloc[-1]
    return {"case": f"{cid[:8]} ({what})", "mesh": mesh, "nodes": meta["wall3d"]["mesh"].n_nodes,
            "run time s": round(time.perf_counter() - t0),
            "Tresca rupture s": meta["rupture"]["Tresca"],
            "hot spot mean max C": round(ts.T_mean_hot_C.max(), 1),
            "hot spot outer max C": round(ts.hot_T_out_C.max(), 1) if "hot_T_out_C" in ts else None,
            f"P at {T_END:.0f} s bara": round(last.P_bara, 2),
            f"T gas at {T_END:.0f} s C": round(last.T_gas_C, 1),
            "energy error %": round(ts.energy_err_pct.iloc[-1], 4)}


def main() -> None:
    jobs = [(r, m) for r in RUNS for m in MESHES.items()]
    with mp.Pool(len(jobs)) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    REPORT.write_text(f"# 3-D wall mesh convergence ({T_END:.0f} s simulated)\n\n"
                      + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
