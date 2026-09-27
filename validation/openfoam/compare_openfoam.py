"""
Code-to-code verification: FAHTS 3-D Hex8 FEM solver vs OpenFOAM solidFoam (FV).

For every case both codes get the SAME hexahedral mesh (exported to an OpenFOAM
polyMesh, one FV cell per Hex8), the SAME boundary conditions, the SAME initial
temperature and the SAME tabulated material data (constant or EN 1993-1-2 steel on a
1 °C grid, linear interpolation in both codes).

The two codes discretise differently (FEM: nodal values, Galerkin; FV: cell values,
two-point fluxes), so on one mesh they are NOT expected to agree exactly.  The test is
that (a) they agree to within a small fraction of the temperature rise, and (b) the
difference SHRINKS when the mesh is refined — i.e. both converge to the same solution.
Where an exact solution exists it is reported for both codes too.

FEM cell temperature = mean of the hex's 8 nodal values (= trilinear field at the cell
centre), compared with the FV cell value.

Run:  python -m validation.openfoam.compare_openfoam      (needs the fahts-openfoam
conda env with OpenFOAM v2412).  Writes validation/openfoam/report_openfoam.md and
figures to validation/openfoam/figures/.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fahts.core.heat.bc.inner_robin_bc import InnerRobinBC  # noqa: E402
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC  # noqa: E402
from fahts.core.heat.solid_mesh import (  # noqa: E402
    FACE_END,
    FACE_INNER,
    FACE_OUTER,
    BlockSolidMesher,
    BoxSolidMesher,
    IProfileSolidMesher,
    PipeSolidMesher,
    SolidMesh,
)
from fahts.core.heat.solver.solid_solver import SolidTransientSolver  # noqa: E402
from fahts.core.model.material import SteelMaterial  # noqa: E402
from fahts.core.model.section import BoxSection, ISection, PipeSection  # noqa: E402
from validation.openfoam.foam_bridge import FoamCase  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
FIG_DIR = HERE / "figures"
RHO = 7850.0
#: FAHTS 3-D conduction scheme under test ("monotone" = production default since
#: 2026-09-27; set FAHTS_CONDUCTION=consistent to test the Galerkin operator).
CONDUCTION = __import__("os").environ.get("FAHTS_CONDUCTION", "monotone")


# ── materials & fire curves ───────────────────────────────────────────────────

@dataclass
class TableMaterial:
    """k(T), c(T) by linear interpolation of a table — identical in both codes."""
    k_table: np.ndarray
    c_table: np.ndarray
    rho: float = RHO

    def properties(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.interp(T, self.k_table[:, 0], self.k_table[:, 1]),
                np.interp(T, self.c_table[:, 0], self.c_table[:, 1]))

    def conductivity(self, T: float) -> float:
        return float(np.interp(T, self.k_table[:, 0], self.k_table[:, 1]))

    def specific_heat(self, T: float) -> float:
        return float(np.interp(T, self.c_table[:, 0], self.c_table[:, 1]))


def const_material(k: float = 45.0, c: float = 500.0) -> TableMaterial:
    return TableMaterial(np.array([[0.0, k], [1500.0, k]]), np.array([[0.0, c], [1500.0, c]]))


def ec3_material() -> TableMaterial:
    """EN 1993-1-2 carbon steel k(T), c(T) sampled every 1 °C, 0–1200 °C."""
    steel = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=RHO, alpha_T=12e-6)
    T = np.arange(0.0, 1201.0, 1.0)
    k = np.array([steel.conductivity(float(x)) for x in T])
    c = np.array([steel.specific_heat(float(x)) for x in T])
    return TableMaterial(np.column_stack([T, k]), np.column_stack([T, c]))


def iso834(t: float) -> float:
    return 20.0 + 345.0 * math.log10(8.0 * t / 60.0 + 1.0)


def hydrocarbon(t: float) -> float:
    m = t / 60.0
    return 20.0 + 1080.0 * (1.0 - 0.325 * math.exp(-0.167 * m) - 0.675 * math.exp(-2.5 * m))


def fire_table(curve: Callable[[float], float], t_end: float, dt: float = 5.0) -> np.ndarray:
    t = np.arange(0.0, t_end + dt, dt)
    return np.column_stack([t, [curve(x) for x in t]])


# ── case definition ───────────────────────────────────────────────────────────

@dataclass
class Case:
    name: str
    title: str
    mesh_fn: Callable[[int], SolidMesh]
    patch_fn: Callable[[SolidMesh], dict[str, np.ndarray]]
    bcs: dict[str, object]
    material: TableMaterial
    t_end: float
    dt: float
    write_dt: float
    levels: tuple[int, ...] = (0, 1)
    exact: Callable[[np.ndarray, float], np.ndarray] | None = None
    notes: str = ""


def _faces_where(mesh: SolidMesh, pred: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 groups: tuple[int, ...] | None = None) -> np.ndarray:
    c, n = mesh.face_centroids(), mesh.face_normals()
    ok = pred(c, n)
    if groups is not None:
        ok &= np.isin(mesh.face_group, groups)
    return np.flatnonzero(ok)


def _rest(mesh: SolidMesh, *used: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(mesh.n_faces), np.concatenate(used))


# ── FAHTS side ────────────────────────────────────────────────────────────────

def run_fahts(case: Case, mesh: SolidMesh, patches: dict[str, np.ndarray]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a SolidTransientSolver with the case BCs; returns (times, T_cells, T_nodes)."""
    group = np.full(mesh.n_faces, FACE_END)
    robin_outer = robin_inner = None
    flux_q = np.zeros(mesh.n_faces)
    has_flux = False
    fixed_nodes: list[tuple[np.ndarray, float]] = []
    robin_faces = np.zeros(mesh.n_faces, dtype=bool)
    for name, idx in patches.items():
        spec = case.bcs[name]
        if spec == "adiabatic":
            continue
        kind = spec[0]
        if kind == "robin" and name == "inner":          # cavity / contents side
            if spec[3] != 0.0:
                raise ValueError("inner Robin BC is convection-only")
            robin_inner = spec
            group[idx] = FACE_INNER
        elif kind == "robin":                              # fire / ambient side
            if robin_outer is not None and robin_outer is not spec:
                raise ValueError("FAHTS supports one outer Robin BC per member")
            robin_outer = spec
            group[idx] = FACE_OUTER
            robin_faces[idx] = True
        elif kind == "flux":
            group[idx] = FACE_OUTER
            flux_q[idx] = spec[1]
            has_flux = True
        elif kind == "fixed":
            fixed_nodes.append((np.unique(np.asarray(mesh.faces)[idx]), spec[1]))
    m2 = SolidMesh(mesh.nodes, mesh.hexes, mesh.faces, group, mesh.section_kind)
    outer = m2.outer_face_indices

    # epsilon_steel=0 disables the solver's separate steel re-radiation term (used when
    # epsilon_m == 0), which OpenFOAM's flux / convection-only BCs do not have.
    kw: dict = {"epsilon_steel": 0.0}
    if robin_outer is not None:
        _, h, tab, eps = robin_outer
        kw.update(fire_temp=lambda t, tab=tab: float(np.interp(t, tab[:, 0], tab[:, 1])),
                  epsilon_m=eps, h_conv=h)
        if has_flux:
            kw["face_exposure"] = robin_faces[outer].astype(float)
    else:
        kw.update(fire_temp=lambda t: 20.0, epsilon_m=0.0, h_conv=0.0)
    if has_flux:
        kw["q_per_face"] = flux_q[outer]
    if robin_inner is not None:
        _, h_in, tab_in, _ = robin_inner
        kw["inner_bc"] = InnerRobinBC(
            h=h_in, T_fluid=lambda t, tab=tab_in: float(np.interp(t, tab[:, 0], tab[:, 1])))
    if fixed_nodes:
        kw["prescribed_node_bcs"] = [PrescribedNodeBC(node_indices=n.tolist(), temperature=v)
                                     for n, v in fixed_nodes]
    solver = SolidTransientSolver(m2, case.material, T0=20.0, mass_matrix="lumped",
                                  nonlinear_max_iter=8, nonlinear_tol=1e-8,
                                  conduction=CONDUCTION, **kw)
    times, T_nodes = solver.run(case.t_end, case.dt, output_dt=case.write_dt)
    return times, T_nodes[:, np.asarray(mesh.hexes)].mean(axis=2), T_nodes


# ── cases ─────────────────────────────────────────────────────────────────────

def _slab_exact(L: float, T_s: float, T_i: float, alpha: float):
    def f(xc: np.ndarray, t: float) -> np.ndarray:
        x = xc[:, 0]
        s = np.zeros_like(x)
        for n in range(200):
            lam = (2 * n + 1) * math.pi / (2 * L)
            s += 4 / ((2 * n + 1) * math.pi) * np.sin(lam * x) * math.exp(-alpha * lam ** 2 * t)
        return T_s + (T_i - T_s) * s
    return f


def build_cases() -> list[Case]:
    cases: list[Case] = []
    ec3 = ec3_material()

    # 1. slab, sudden surface temperature, constant properties (exact series solution)
    L = 0.04
    cases.append(Case(
        name="slab_dirichlet_const",
        title="40 mm slab, face at 500 °C, back adiabatic, constant k/c",
        mesh_fn=lambda lv: BlockSolidMesher(L, 0.01, 0.01, 10 * 2 ** lv, 2, 2).build(),
        patch_fn=lambda m: (lambda hot: {"hot": hot, "adiabatic": _rest(m, hot)})(
            _faces_where(m, lambda c, n: np.isclose(c[:, 0], 0.0))),
        bcs={"hot": ("fixed", 500.0), "adiabatic": "adiabatic"},
        material=const_material(), t_end=300.0, dt=0.5, write_dt=30.0, levels=(0, 1, 2),
        exact=_slab_exact(L, 500.0, 20.0, 45.0 / (RHO * 500.0)),
    ))

    # 2. slab, ISO 834 fire one side (convection + radiation), EC3 steel
    iso = fire_table(iso834, 3600.0)
    cases.append(Case(
        name="plate_iso_fire_ec3",
        title="40 mm plate, ISO 834 fire on one face (h=25, ε=0.7), EC3 steel",
        mesh_fn=lambda lv: BlockSolidMesher(L, 0.02, 0.02, 8 * 2 ** lv, 2, 2).build(),
        patch_fn=lambda m: (lambda hot: {"fire": hot, "adiabatic": _rest(m, hot)})(
            _faces_where(m, lambda c, n: np.isclose(c[:, 0], 0.0))),
        bcs={"fire": ("robin", 25.0, iso, 0.7), "adiabatic": "adiabatic"},
        material=ec3, t_end=3600.0, dt=2.0, write_dt=300.0, levels=(0, 1, 2),
    ))

    # 3. hollow BOX member, hydrocarbon fire all round, EC3
    hc = fire_table(hydrocarbon, 1800.0)
    box = BoxSection(sid=1, H=0.3, W=0.3, T_side=0.012, T_bot=0.012, T_top=0.012)
    cases.append(Case(
        name="box_hc_fire_ec3",
        title="BOX 300×300×12, 1 m, HC fire outside (h=50, ε=0.7), cavity+ends adiabatic, EC3",
        mesh_fn=lambda lv: BoxSolidMesher(box, 1.0, n_top=4 * 2 ** lv, n_side=4 * 2 ** lv,
                                          n_length=2 * 2 ** lv, n_layers=2 * 2 ** lv).build(),
        patch_fn=lambda m: {"outer": m.face_indices(FACE_OUTER),
                            "adiabatic": _rest(m, m.face_indices(FACE_OUTER))},
        bcs={"outer": ("robin", 50.0, hc, 0.7), "adiabatic": "adiabatic"},
        material=ec3, t_end=1800.0, dt=2.0, write_dt=300.0, levels=(0, 1, 2),
    ))

    # 4. pipe with cooling contents (vessel-contents hook), EC3
    pipe = PipeSection(sid=2, outer_diameter=0.3, thickness=0.015)
    cool = np.array([[0.0, 20.0], [1800.0, 20.0]])
    cases.append(Case(
        name="pipe_fire_liquid_ec3",
        title="Pipe Ø300×15, HC fire outside (h=50, ε=0.8), liquid inside (h=500, 20 °C), EC3",
        mesh_fn=lambda lv: PipeSolidMesher(pipe, 0.5, c_circ=24 * 2 ** lv, n_length=2,
                                           n_layers=3 * 2 ** lv).build(),
        patch_fn=lambda m: {"outer": m.face_indices(FACE_OUTER),
                            "inner": m.face_indices(FACE_INNER),
                            "ends": m.face_indices(FACE_END)},
        bcs={"outer": ("robin", 50.0, hc, 0.8), "inner": ("robin", 500.0, cool, 0.0),
             "ends": "adiabatic"},
        material=ec3, t_end=1800.0, dt=2.0, write_dt=300.0,
    ))

    # 5. I-beam exposed on the bottom flange only (3-D conduction into web/top flange)
    ib = ISection(sid=3, h=0.3, tw=0.0071, bf_top=0.15, tf_top=0.0107,
                  bf_bot=0.15, tf_bot=0.0107)
    z_cut = -0.15 + 0.0107 + 1e-6

    def ibeam_patches(m: SolidMesh) -> dict[str, np.ndarray]:
        fire = _faces_where(m, lambda c, n: c[:, 2] < z_cut, groups=(FACE_OUTER,))
        return {"fire": fire, "adiabatic": _rest(m, fire)}

    cases.append(Case(
        name="ibeam_bottom_fire_ec3",
        title="I 300×150, ISO 834 on bottom flange only (h=25, ε=0.7), rest adiabatic, EC3",
        mesh_fn=lambda lv: IProfileSolidMesher(ib, 0.5, n_top=4 * 2 ** lv, n_side=6 * 2 ** lv,
                                               n_bottom=4 * 2 ** lv, n_length=2,
                                               n_layers=2 * 2 ** lv).build(),
        patch_fn=ibeam_patches,
        bcs={"fire": ("robin", 25.0, fire_table(iso834, 1800.0), 0.7),
             "adiabatic": "adiabatic"},
        material=ec3, t_end=1800.0, dt=2.0, write_dt=300.0, levels=(0, 1, 2),
    ))

    # 6. plate with a local jet-fire-like hot spot (prescribed flux) + cooled back face
    amb = np.array([[0.0, 20.0], [1200.0, 20.0]])

    def hotspot_patches(m: SolidMesh) -> dict[str, np.ndarray]:
        spot = _faces_where(m, lambda c, n: np.isclose(c[:, 2], 0.02)
                            & (np.abs(c[:, 0] - 0.25) < 0.05) & (np.abs(c[:, 1] - 0.25) < 0.05))
        back = _faces_where(m, lambda c, n: np.isclose(c[:, 2], 0.0))
        return {"spot": spot, "back": back, "adiabatic": _rest(m, spot, back)}

    cases.append(Case(
        name="plate_hotspot_flux_ec3",
        title="Plate 500×500×20, 100 kW/m² on a 100×100 spot, back face h=10 to 20 °C, EC3",
        mesh_fn=lambda lv: BlockSolidMesher(0.5, 0.5, 0.02, 20 * 2 ** lv, 20 * 2 ** lv,
                                            2 * 2 ** lv).build(),
        patch_fn=hotspot_patches,
        bcs={"spot": ("flux", 1.0e5), "back": ("robin", 10.0, amb, 0.0),
             "adiabatic": "adiabatic"},
        material=ec3, t_end=1200.0, dt=2.0, write_dt=300.0,
    ))
    return cases


# ── driver ────────────────────────────────────────────────────────────────────

@dataclass
class LevelResult:
    level: int
    n_cells: int
    rise: float                   # max temperature rise [K]
    max_diff: float               # max |FEM − FV| over cells & times [K]
    rms_diff: float               # RMS over cells at the final time [K]
    mean_diff: float              # |mean(FEM) − mean(FV)| at final time [K]
    t_fem: float
    t_fv: float
    err_fem: float | None = None  # max |FEM − exact| [K]
    err_fv: float | None = None
    err_fem_nodes: float | None = None   # max |FEM − exact| at FEM nodes [K]
    curves: dict = field(default_factory=dict)


def run_case(case: Case) -> list[LevelResult]:
    out: list[LevelResult] = []
    for lv in case.levels:
        mesh = case.mesh_fn(lv)
        patches = case.patch_fn(mesh)
        t0 = time.perf_counter()
        t_fem, T_fem, T_nod = run_fahts(case, mesh, patches)
        t_fem_s = time.perf_counter() - t0
        fc = FoamCase(RUN_DIR / f"{case.name}_L{lv}", mesh, patches, case.bcs, RHO,
                      case.material.k_table, case.material.c_table, 20.0,
                      case.t_end, case.dt, case.write_dt)
        t0 = time.perf_counter()
        fc.run()
        t_fv_s = time.perf_counter() - t0
        t_fv, T_fv = fc.results()
        # align output times
        idx = [int(np.argmin(np.abs(t_fem - t))) for t in t_fv]
        if not np.allclose(t_fem[idx], t_fv, atol=1e-6):
            raise RuntimeError(f"{case.name}: output times differ {t_fem} vs {t_fv}")
        T_fem, T_nod = T_fem[idx], T_nod[idx]
        # skip t = 0: a sudden Dirichlet jump is nodal in FEM (the first cell layer's
        # 8-node mean is already partly hot) but a face value in FV — not comparable.
        t_fv, T_fem, T_fv, T_nod = t_fv[1:], T_fem[1:], T_fv[1:], T_nod[1:]
        d = T_fem - T_fv
        vol = mesh.hex_volumes()
        res = LevelResult(
            level=lv, n_cells=mesh.n_hexes, rise=float(max(T_fem.max(), T_fv.max()) - 20.0),
            max_diff=float(np.abs(d).max()), rms_diff=float(np.sqrt(np.mean(d[-1] ** 2))),
            mean_diff=float(abs((T_fem[-1] - T_fv[-1]) @ vol / vol.sum())),
            t_fem=t_fem_s, t_fv=t_fv_s,
        )
        if case.exact is not None:
            xc = np.asarray(mesh.nodes)[np.asarray(mesh.hexes)].mean(axis=1)
            ex = np.array([case.exact(xc, t) for t in t_fv])
            res.err_fem = float(np.abs(T_fem - ex).max())
            res.err_fv = float(np.abs(T_fv - ex).max())
            xn = np.asarray(mesh.nodes)
            res.err_fem_nodes = float(max(np.abs(T_nod[i] - case.exact(xn, t)).max()
                                          for i, t in enumerate(t_fv)))
        # curves: hottest / coldest cell & volume mean vs time
        hot = int(np.argmax(T_fv[-1]))
        cold = int(np.argmin(T_fv[-1]))
        res.curves = {"t": t_fv, "fem_hot": T_fem[:, hot], "fv_hot": T_fv[:, hot],
                      "fem_cold": T_fem[:, cold], "fv_cold": T_fv[:, cold],
                      "fem_mean": T_fem @ vol / vol.sum(), "fv_mean": T_fv @ vol / vol.sum()}
        log.info("%s L%d: cells=%d rise=%.1f max|Δ|=%.3f K (FEM %.1fs, FV %.1fs)",
                 case.name, lv, mesh.n_hexes, res.rise, res.max_diff, t_fem_s, t_fv_s)
        out.append(res)
    return out


def _plot(case: Case, results: list[LevelResult]) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    r = results[-1]
    c = r.curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab, col in (("hot", "hottest cell", "tab:red"), ("mean", "volume mean", "tab:blue"),
                          ("cold", "coldest cell", "tab:green")):
        ax.plot(c["t"] / 60, c[f"fv_{key}"], "o", color=col, mfc="none", label=f"OpenFOAM {lab}")
        ax.plot(c["t"] / 60, c[f"fem_{key}"], "-", color=col, label=f"FAHTS {lab}")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("T [°C]")
    ax.set_title(f"{case.name} (level {r.level}, {r.n_cells} cells)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fn = FIG_DIR / f"{case.name}.png"
    fig.tight_layout()
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn.name


def write_report(all_res: dict[str, tuple[Case, list[LevelResult]]], path: Path) -> None:
    lines = [
        "# FAHTS 3-D solver vs OpenFOAM solidFoam — code-to-code comparison",
        "",
        "Same Hex8 mesh (one FV cell per hex), same BCs, same tabulated material data "
        "(EN 1993-1-2 on a 1 °C grid or constant), same initial state (20 °C), same Δt.",
        "FEM cell value = mean of the 8 nodal temperatures; compared with the FV cell value.",
        f"FAHTS conduction scheme: **{CONDUCTION}**.  "
        "OpenFOAM v2412 `solidFoam`, Crank–Nicolson 0.9, `externalWallHeatFluxTemperature` "
        "(h + εσ(Ta⁴−T⁴), identical formula).  σ: FAHTS 5.67e-8, OpenFOAM 5.670374e-8.",
        "",
        "FV values live at cell centres, FEM values at nodes: comparing at cell centres "
        "adds an interpolation error (the 8-node mean of a curved profile) that belongs to "
        "neither solver.  For the exact-solution case the FEM error is therefore also "
        "reported at its own nodes.",
        "",
        "**Fire/convection BC in OpenFOAM:** the built-in `externalWallHeatFluxTemperature` "
        "(mode coefficient) is a *mixed* T condition, which `heSolidThermo` converts to "
        "enthalpy with refValue = h(Ta).  With a temperature-dependent Cp this makes the wall "
        "flux ≈ c̄p(T_wall→Ta)/cp(T_wall) × the true flux (+17–22 % for EN 1993 steel in a "
        "fire; exact for constant Cp — verified against a lumped-capacitance ODE).  All Robin "
        "conditions are therefore imposed as a flux q = h(Ta−T) + εσ(Ta⁴−T⁴) from the current "
        "face temperature (`codedMixed`, valueFraction 0), which OpenFOAM converts correctly.",
        "",
        "**Pass criterion:** max |FEM − FV| ≤ 2 % of the temperature rise on the finest mesh "
        "AND the difference decreases with refinement (both codes converge to the same answer).",
        "",
    ]
    summary = ["| Case | cells (finest) | ΔT rise [K] | max \\|Δ\\| per level [K] | "
               "ratio | max \\|Δ\\|/rise | Result |", "|---|---|---|---|---|---|---|"]
    detail: list[str] = []
    for name, (case, res) in all_res.items():
        diffs = [r.max_diff for r in res]
        fin = res[-1]
        rel = fin.max_diff / max(fin.rise, 1e-9)
        decreasing = all(b < a for a, b in zip(diffs, diffs[1:]))
        ok = rel <= 0.02 and decreasing
        ratio = " → ".join(f"{a / b:.1f}×" for a, b in zip(diffs, diffs[1:]))
        summary.append(f"| {name} | {fin.n_cells} | {fin.rise:.0f} | "
                       f"{' / '.join(f'{d:.3g}' for d in diffs)} | {ratio} | {rel:.2%} | "
                       f"{'PASS' if ok else 'FAIL'} |")
        fig = _plot(case, res)
        detail += [f"### {name}", "", case.title, ""]
        if case.notes:
            detail += [case.notes, ""]
        hdr = "| level | cells | max \\|Δ\\| [K] | RMS Δ final [K] | Δ vol-mean final [K] |"
        sep = "|---|---|---|---|---|"
        if case.exact is not None:
            hdr += (" FEM err @cell centres [K] | FEM err @nodes [K] |"
                    " FV err @cell centres [K] |")
            sep += "---|---|---|"
        hdr += " time FAHTS / OF [s] |"
        sep += "---|"
        detail += [hdr, sep]
        for r in res:
            row = (f"| {r.level} | {r.n_cells} | {r.max_diff:.4g} | {r.rms_diff:.4g} | "
                   f"{r.mean_diff:.4g} |")
            if case.exact is not None:
                row += f" {r.err_fem:.4g} | {r.err_fem_nodes:.4g} | {r.err_fv:.4g} |"
            row += f" {r.t_fem:.1f} / {r.t_fv:.1f} |"
            detail.append(row)
        detail += ["", f"![{name}](figures/{fig})", ""]
    path.write_text("\n".join(lines + ["## Summary", ""] + summary + ["", "## Cases", ""]
                              + detail) + "\n")


def main(only: list[str] | None = None) -> dict:
    """Run all (or *only* the named) cases; results are cached per case in runs/*.pkl so
    the report always covers every case that has been run."""
    import pickle

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    all_res: dict[str, tuple[Case, list[LevelResult]]] = {}
    for case in build_cases():
        cache = RUN_DIR / f"{case.name}.pkl"
        if not only or case.name in only:
            res = run_case(case)
            cache.write_bytes(pickle.dumps(res))
        elif cache.exists():
            res = pickle.loads(cache.read_bytes())
        else:
            continue
        all_res[case.name] = (case, res)
    write_report(all_res, HERE / "report_openfoam.md")
    return all_res


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
