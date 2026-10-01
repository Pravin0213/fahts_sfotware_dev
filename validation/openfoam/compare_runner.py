"""
End-to-end check: the full ``run_analysis`` pipeline (3-D default settings: batched
assembly, global Jacobi-PCG, monotone conduction, free-end exposure, RadiationBall flux
per face, re-radiation to ambient) against OpenFOAM solidFoam on the SAME mesh with the
SAME per-face flux + εσ(T_amb⁴ − T⁴).

Model: one free PIPE member Ø406×13, L = 4 m (both end caps exposed).  Cases: ball
engulfing the member (uniform flux), and ball outside it (cos θ pattern).  Shielding /
radiation exchange are switched off — OpenFOAM's solid solver has no view-factor
radiation; those are verified analytically in tests/test_radiation_geometry.py.

Run: python -m validation.openfoam.compare_runner
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

import fahts.core.heat.solver.analysis_runner as ar
from fahts.core.heat.solid_mesh import FACE_INNER, FACE_OUTER
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.results.analysis_config import AnalysisConfig
from validation.openfoam.compare_openfoam import RHO, ec3_material
from validation.openfoam.foam_bridge import FoamCase

HERE = Path(__file__).resolve().parent
FEM = """ NODE 1 0.0 0.0 0.0
 NODE 2 4.0 0.0 0.0
 BEAM 1 1 2 1 1 1
 PIPE 1 0.406 0.013
 MISOIEP 1 2.100E+11 3.000E-01 3.550E+08 7.850E+03 1.200E-05
 UNITVEC 1 0.0 0.0 1.0
"""


def run_case(name: str, ball: RadiationBall, t_end: float = 1800.0, dt: float = 10.0,
             level: int = 0):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pipe.fem"
        p.write_text(FEM)
        model = read_usfos_fem(p)
    cap: dict = {}
    orig = ar.GlobalThermalSolver.__init__

    def spy(self, eids, solvers, gdof_map, n_global_dofs, **kw):
        cap["solver"], cap["gmap"] = solvers[eids[0]], gdof_map[eids[0]]
        orig(self, eids, solvers, gdof_map, n_global_dofs, **kw)

    ar.GlobalThermalSolver.__init__ = spy
    try:
        f = 2 ** level                                  # mesh refinement factor
        cfg = AnalysisConfig(t_end=t_end, dt=dt, output_dt=300.0, shielding=False,
                             radiation_exchange=False, c_circ=12 * f, n_layers_3d=2 * f,
                             axial_aspect_3d=2.0)
        res = ar.run_analysis(model, [ball], cfg)
    finally:
        ar.GlobalThermalSolver.__init__ = orig
    sv = cap["solver"]
    mesh = sv.mesh
    T_fem = res.T_section[1]                        # member-local node order
    T_cells_fem = T_fem[:, np.asarray(mesh.hexes)].mean(axis=2)

    # OpenFOAM: same mesh, outer faces = per-face ball flux + εσ(Ta⁴ − T⁴), rest adiabatic
    outer = mesh.outer_face_indices
    q = np.asarray(sv._q_per_face, dtype=float)
    eps = float(sv._eps_rerad)
    amb = np.array([[0.0, 20.0], [t_end, 20.0]])
    patches = {"outer": outer,
               "adiabatic": np.setdiff1d(np.arange(mesh.n_faces), outer)}
    mat = ec3_material()
    fc = FoamCase(HERE / "runs" / f"runner_{name}_L{level}", mesh, patches,
                  {"outer": ("robin", 0.0, amb, eps, q), "adiabatic": "adiabatic"},
                  RHO, mat.k_table, mat.c_table, 20.0, t_end, dt, 300.0)
    fc.run()
    t_fv, T_fv = fc.results()
    idx = [int(np.argmin(np.abs(res.times - t))) for t in t_fv]
    d = T_cells_fem[idx][1:] - T_fv[1:]
    rise = float(T_fv.max() - 20.0)
    info = {
        "cells": mesh.n_hexes, "end_caps_exposed": int(np.sum(
            (mesh.face_group == FACE_OUTER)
            & (np.ptp(np.asarray(mesh.nodes)[mesh.faces][:, :, 0], axis=1) < 1e-9))),
        "inner_faces": int(np.sum(mesh.face_group == FACE_INNER)),
        "rise": rise, "max_diff": float(np.abs(d).max()),
        "mean_diff_final": float(abs(T_cells_fem[idx][-1].mean() - T_fv[-1].mean())),
        "T_fem_max": float(T_cells_fem.max()), "T_fv_max": float(T_fv.max()),
    }
    return info


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    cases = {
        "engulfed": RadiationBall(name="B", center=np.array([2.0, 0.0, 0.0]), radius=5.0,
                                  flux=100e3, active=True),
        "outside": RadiationBall(name="B", center=np.array([2.0, 0.0, 3.0]), radius=1.5,
                                 flux=350e3, active=True),
    }
    rows = ["# run_analysis (3-D pipeline) vs OpenFOAM solidFoam — single free pipe", "",
            "| case | level | cells | exposed end-cap faces | ΔT rise [K] | max \\|Δ\\| [K] | "
            "max \\|Δ\\|/rise | Δ mean final [K] | Tmax FAHTS / OF [°C] |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, ball in cases.items():
        for level in (0, 1, 2):
            r = run_case(name, ball, level=level)
            rows.append(
                f"| {name} | {level} | {r['cells']} | {r['end_caps_exposed']} | "
                f"{r['rise']:.0f} | {r['max_diff']:.3g} | {r['max_diff'] / r['rise']:.2%} | "
                f"{r['mean_diff_final']:.3g} | {r['T_fem_max']:.1f} / {r['T_fv_max']:.1f} |")
            print(rows[-1])
    (HERE / "report_runner.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
