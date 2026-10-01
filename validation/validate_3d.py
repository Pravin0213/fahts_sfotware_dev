"""
Validation of the 3-D Hex8 heat solver (`SolidTransientSolver`) — plan WP-D.

Every case compares the solver against an independent reference (closed-form
analytical solution, a 1-D method-of-lines ODE solve, the lumped-capacitance
ODE, or the 2-D `SurfaceTransientSolver`).  The case functions are importable
(the fast pytest versions live in ``tests/test_solid_validation.py``); running
this module executes all cases at higher resolution and writes

    validation/report_3d.md       tables: case, metric, value, tolerance, PASS/FAIL
    validation/figures/*.png      plots

Usage:
    python -m validation.validate_3d
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc

from fahts.core.heat.bc.inner_robin_bc import InnerRobinBC
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.solid_mesh import (
    BlockSolidMesher,
    BoxSolidMesher,
    PipeSolidMesher,
    SolidMesh,
)
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection, PipeSection

logger = logging.getLogger(__name__)

SIGMA = 5.67e-8
K0 = 273.15
_TOL_GEOM = 1e-9


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class ConstantMaterial:
    """Duck-typed constant-property material (k, ρ, c) for analytical cases."""
    k: float
    rho: float
    cp: float

    def conductivity(self, T: float) -> float:  # noqa: ARG002
        return self.k

    def specific_heat(self, T: float) -> float:  # noqa: ARG002
        return self.cp


def steel() -> SteelMaterial:
    """EN 1993-1-2 Annex C carbon steel (temperature-dependent k, c)."""
    return SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=355e6, rho=7850.0, alpha_T=1.2e-5)


def iso834(t: float) -> float:
    """ISO 834 standard fire curve, t in seconds → °C."""
    return 20.0 + 345.0 * math.log10(8.0 * t / 60.0 + 1.0)


def march(
    solver: SolidTransientSolver,
    T_init: np.ndarray,
    dt: float,
    n_steps: int,
    callback: Callable[[float, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Advance a *fresh* solver from a non-uniform initial field with public `step`."""
    T = np.asarray(T_init, dtype=float).copy()
    t = 0.0
    for _ in range(n_steps):
        t += dt
        T = solver.step(T, dt, t)
        if callback is not None:
            callback(t, T)
    return T


def outer_centroids(mesh: SolidMesh) -> np.ndarray:
    return mesh.face_centroids()[mesh.outer_face_indices]


def outer_node_indices(mesh: SolidMesh) -> np.ndarray:
    return np.unique(mesh.faces[mesh.outer_face_indices].ravel())


def lumped_node_volumes(mesh: SolidMesh) -> np.ndarray:
    """V/8 of every hex added to its 8 nodes (exact row-sum for parallelepipeds)."""
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, mesh.hexes.ravel(), np.repeat(mesh.hex_volumes() / 8.0, 8))
    return w


def order(errors: list[float], hs: list[float]) -> list[float]:
    """Observed orders log(e_i/e_{i+1}) / log(h_i/h_{i+1})."""
    return [math.log(errors[i] / errors[i + 1]) / math.log(hs[i] / hs[i + 1])
            for i in range(len(errors) - 1)]


_GP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) / math.sqrt(3.0)


def face_flux_integral(
    mesh: SolidMesh, T: np.ndarray, q_of_T: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> float:
    """∫ q(T, face) dA over FACE_OUTER, 2×2 Gauss on bilinear faces (own implementation)."""
    faces = mesh.faces[mesh.outer_face_indices]
    P = mesh.nodes[faces]                                  # (n_f, 4, 3)
    Tf = T[faces]                                          # (n_f, 4)
    total = 0.0
    for xi, eta in _GP:
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        t1 = np.einsum("j,fjk->fk", dxi, P)
        t2 = np.einsum("j,fjk->fk", deta, P)
        dA = np.linalg.norm(np.cross(t1, t2), axis=1)
        total += float(np.sum(q_of_T(Tf @ N, np.arange(len(faces))) * dA))
    return total


# ── Case 1: sin·sin·sin decay in the unit cube ────────────────────────────────

def case1_cube_decay(n: int, dt: float, t_end: float, mass: str = "lumped",
                     solver_cls: type | None = None) -> dict:
    """
    Unit cube, k = ρ = c = 1, T = 0 on all boundary nodes, initial
    T = sin πx sin πy sin πz → T = T(0) exp(−3π²t).  Errors are normalised by
    the analytical amplitude exp(−3π² t_end).
    """
    mesh = BlockSolidMesher(1.0, 1.0, 1.0, n, n, n).build()
    X = mesh.nodes
    T0 = np.prod(np.sin(np.pi * X), axis=1)
    bnd = np.flatnonzero(np.any((X < _TOL_GEOM) | (X > 1.0 - _TOL_GEOM), axis=1))
    cls = solver_cls if solver_cls is not None else SolidTransientSolver
    solver = cls(
        mesh, ConstantMaterial(1.0, 1.0, 1.0), fire_temp=lambda _t: 0.0,
        epsilon_m=0.0, h_conv=0.0, T0=0.0, mass_matrix=mass,
        prescribed_node_bcs=[PrescribedNodeBC(bnd.tolist(), 0.0)],
    )
    n_steps = int(round(t_end / dt))
    T = march(solver, T0, dt, n_steps)
    amp = math.exp(-3.0 * math.pi ** 2 * t_end)
    err = (T - T0 * amp) / amp
    return {"n": n, "h": 1.0 / n, "dt": dt, "mass": mass, "T": T, "T0": T0,
            "err_max": float(np.max(np.abs(err))),
            "err_rms": float(np.sqrt(np.mean(err ** 2)))}


class ReducedInitRateSolver(SolidTransientSolver):
    """
    DIAGNOSTIC ONLY (not library code): initial rate from the *free-DOF* system
        M_ff Ṫ_f = (Q − K T0)_f,   Ṫ_prescribed = 0.
    The library solves the full M Ṫ = Q − K T0 and then zeroes the prescribed
    rows (solid_solver.py:430-433), which is exact for a diagonal (lumped) M but,
    with a consistent M, lets the (non-zero) boundary-row residual pollute the
    interior Ṫ0.  The CN rate recursion never damps that error → O(Δt) error.
    """

    def _init_rate(self, T0: np.ndarray, t0: float = 0.0) -> None:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        K0, M0, Q0 = self._assemble_step(T0, t0)
        self._K_prev, self._M_prev = K0, M0
        r = Q0 - K0 @ T0
        if not sp.issparse(M0):
            Td = r / M0
        else:
            f = np.flatnonzero(self._free_mask > 0.0)
            Td = np.zeros_like(T0)
            Td[f] = spla.spsolve(M0.tocsc()[f][:, f], r[f])
        self._T_dot_prev = Td * self._free_mask


def case1_consistent_dt_sensitivity(n: int = 8, t_end: float = 0.05,
                                    steps: tuple[int, ...] = (25, 50, 100, 200, 400),
                                    solver_cls: type | None = None) -> dict:
    """Consistent-mass cube error vs Δt: should be Δt-independent (time error ≪ h² error)."""
    errs = [case1_cube_decay(n, t_end / m, t_end, "consistent", solver_cls)["err_max"]
            for m in steps]
    return {"steps": list(steps), "errs": errs, "spread": max(errs) - min(errs)}


# ── Case 2: semi-infinite solid (erfc / ierfc) ────────────────────────────────

SEMI_MAT = ConstantMaterial(k=45.0, rho=7850.0, cp=600.0)


def _slab_mesh(L: float, nx: int) -> SolidMesh:
    return BlockSolidMesher(L, 0.01, 0.01, nx, 1, 1).build()


def case2_semi_infinite_dirichlet(
    nx: int = 120, L: float = 0.6, dt: float = 2.0, t_end: float = 1200.0,
    out_dt: float = 300.0, T_i: float = 20.0, T_s: float = 1000.0,
) -> dict:
    """Sudden surface temperature on x = 0: T = T_s + (T_i − T_s) erf(x / 2√(αt))."""
    mat = SEMI_MAT
    alpha = mat.k / (mat.rho * mat.cp)
    mesh = _slab_mesh(L, nx)
    x = mesh.nodes[:, 0]
    surf = np.flatnonzero(x < _TOL_GEOM)
    solver = SolidTransientSolver(
        mesh, mat, fire_temp=lambda _t: T_i, epsilon_m=0.0, h_conv=0.0, T0=T_i,
        prescribed_node_bcs=[PrescribedNodeBC(surf.tolist(), T_s)],
    )
    times, hist = solver.run(t_end, dt, out_dt)
    res = {"x": x, "times": times, "hist": hist, "exact": [], "err": []}
    for t, T in zip(times[1:], hist[1:]):
        Tex = T_i + (T_s - T_i) * erfc(x / (2.0 * math.sqrt(alpha * t)))
        res["exact"].append(Tex)
        res["err"].append(float(np.max(np.abs(T - Tex)) / (T_s - T_i)))
    res["err_max"] = max(res["err"])
    res["penetration_ratio"] = L / (4.0 * math.sqrt(alpha * t_end))
    return res


def case2_semi_infinite_flux(
    nx: int = 120, L: float = 0.6, dt: float = 2.0, t_end: float = 1200.0,
    out_dt: float = 300.0, q: float = 5.0e4, T_i: float = 20.0,
) -> dict:
    """
    Constant flux q on x = 0 only (q_per_face, zero elsewhere, no re-radiation):
        T − T_i = (2q/k) √(αt/π) exp(−x²/4αt) − (q x / k) erfc(x / 2√(αt)).
    """
    mat = SEMI_MAT
    alpha = mat.k / (mat.rho * mat.cp)
    mesh = _slab_mesh(L, nx)
    x = mesh.nodes[:, 0]
    qf = np.where(outer_centroids(mesh)[:, 0] < _TOL_GEOM, q, 0.0)
    solver = SolidTransientSolver(
        mesh, mat, fire_temp=lambda _t: T_i, epsilon_m=0.0, h_conv=0.0, T0=T_i,
        q_per_face=qf, epsilon_steel=0.0,
    )
    times, hist = solver.run(t_end, dt, out_dt)
    res = {"x": x, "times": times, "hist": hist, "exact": [], "err": []}
    for t, T in zip(times[1:], hist[1:]):
        s = math.sqrt(alpha * t)
        Tex = T_i + (2 * q / mat.k) * s / math.sqrt(math.pi) * np.exp(-x ** 2 / (4 * s * s)) \
            - (q * x / mat.k) * erfc(x / (2 * s))
        res["exact"].append(Tex)
        dT_surf = Tex[0] - T_i
        res["err"].append(float(np.max(np.abs(T - Tex)) / dT_surf))
    res["err_max"] = max(res["err"])
    return res


# ── Case 3: steady plane wall ─────────────────────────────────────────────────

UNIT_MAT = ConstantMaterial(1.0, 1.0, 1.0)


def case3_wall_dirichlet(
    nx: int = 10, T1: float = 100.0, T2: float = 20.0, dt: float = 0.01, t_end: float = 5.0,
) -> dict:
    """Fixed temperatures on x = 0 and x = 1 (k = ρc = 1) → exact linear profile."""
    mesh = BlockSolidMesher(1.0, 0.2, 0.2, nx, 2, 2).build()
    x = mesh.nodes[:, 0]
    solver = SolidTransientSolver(
        mesh, UNIT_MAT, fire_temp=lambda _t: 0.0, epsilon_m=0.0, h_conv=0.0, T0=T2,
        prescribed_node_bcs=[
            PrescribedNodeBC(np.flatnonzero(x < _TOL_GEOM).tolist(), T1),
            PrescribedNodeBC(np.flatnonzero(x > 1 - _TOL_GEOM).tolist(), T2),
        ],
    )
    _, hist = solver.run(t_end, dt, t_end)
    Tex = T1 + (T2 - T1) * x
    return {"x": x, "T": hist[-1], "exact": Tex,
            "err_max": float(np.max(np.abs(hist[-1] - Tex)) / abs(T1 - T2))}


def case3_wall_robin(
    nx: int = 10, h1: float = 2.0, h2: float = 5.0, Tf1: float = 100.0, Tf2: float = 20.0,
    dt: float = 0.01, t_end: float = 10.0,
) -> dict:
    """
    Convection on both faces (k = 1, L = 1), sides adiabatic:
        x = 0: fire_temp = Tf1 with h_conv = h1 (face_exposure 1)
        x = 1: face_exposure = h2/h1 and q_per_face = h2 (Tf2 − Tf1)
               → −k∂T/∂n = h2 (T − Tf2)
    Analytical: q = (Tf1 − Tf2) / (1/h1 + L/k + 1/h2), T linear.
    """
    mesh = BlockSolidMesher(1.0, 0.2, 0.2, nx, 2, 2).build()
    x = mesh.nodes[:, 0]
    cx = outer_centroids(mesh)[:, 0]
    left, right = cx < _TOL_GEOM, cx > 1 - _TOL_GEOM
    exposure = np.where(left, 1.0, np.where(right, h2 / h1, 0.0))
    qf = np.where(right, h2 * (Tf2 - Tf1), 0.0)
    solver = SolidTransientSolver(
        mesh, UNIT_MAT, fire_temp=lambda _t: Tf1, epsilon_m=0.0, h_conv=h1,
        T0=0.5 * (Tf1 + Tf2), face_exposure=exposure, q_per_face=qf, epsilon_steel=0.0,
    )
    _, hist = solver.run(t_end, dt, t_end)
    q = (Tf1 - Tf2) / (1.0 / h1 + 1.0 + 1.0 / h2)
    Tex = Tf1 - q / h1 - q * x
    return {"x": x, "T": hist[-1], "exact": Tex,
            "err_max": float(np.max(np.abs(hist[-1] - Tex)) / abs(Tf1 - Tf2))}


# ── Case 4: steady radial conduction through a pipe wall ──────────────────────

PIPE_SEC = PipeSection(sid=1, outer_diameter=1.0, thickness=0.2)   # r_i = 0.3, r_o = 0.5


def _pipe_mesh(c_circ: int, n_layers: int) -> tuple[SolidMesh, np.ndarray]:
    mesh = PipeSolidMesher(PIPE_SEC, 0.1, c_circ=c_circ, n_length=1,
                           n_layers=n_layers).build()
    r = np.hypot(mesh.nodes[:, 1], mesh.nodes[:, 2])
    return mesh, r


def case4_pipe_dirichlet(
    c_circ: int, n_layers: int, T_in: float = 100.0, T_out: float = 20.0,
    dt: float = 0.002, t_end: float = 0.3,
) -> dict:
    """T_in on bore nodes, T_out on outer nodes → T = T_in + ΔT ln(r/r_i)/ln(r_o/r_i)."""
    mesh, r = _pipe_mesh(c_circ, n_layers)
    ri, ro = PIPE_SEC.inner_radius, PIPE_SEC.outer_radius
    inner = mesh.inner_node_indices
    outer = outer_node_indices(mesh)
    solver = SolidTransientSolver(
        mesh, UNIT_MAT, fire_temp=lambda _t: 0.0, epsilon_m=0.0, h_conv=0.0,
        prescribed_node_bcs=[PrescribedNodeBC(inner.tolist(), T_in),
                             PrescribedNodeBC(outer.tolist(), T_out)],
    )
    # smooth (linear-in-r) start → no high-frequency CN ringing from a BC jump
    T_init = T_in + (T_out - T_in) * (r - ri) / (ro - ri)
    T = march(solver, T_init, dt, int(round(t_end / dt)))
    Tex = T_in + (T_out - T_in) * np.log(r / ri) / math.log(ro / ri)
    return {"c_circ": c_circ, "n_layers": n_layers, "r": r, "T": T, "exact": Tex,
            "err_max": float(np.max(np.abs(T - Tex)) / abs(T_in - T_out))}


def case4_pipe_inner_robin(
    c_circ: int, n_layers: int, h_o: float = 5.0, T_f: float = 100.0,
    h_i: float = 10.0, T_fl: float = 20.0, dt: float = 0.002, t_end: float = 0.4,
) -> dict:
    """
    Outer convection (fire_temp = T_f, h_conv = h_o), inner InnerRobinBC(h_i, T_fl),
    k = 1.  Series resistance per unit length:
        R' = 1/(2π r_o h_o) + ln(r_o/r_i)/(2π k) + 1/(2π r_i h_i),  q' = (T_f − T_fl)/R'
        T(r) = T_fl + q' [1/(2π r_i h_i) + ln(r/r_i)/(2π k)]
    """
    mesh, r = _pipe_mesh(c_circ, n_layers)
    ri, ro = PIPE_SEC.inner_radius, PIPE_SEC.outer_radius
    solver = SolidTransientSolver(
        mesh, UNIT_MAT, fire_temp=lambda _t: T_f, epsilon_m=0.0, h_conv=h_o,
        inner_bc=InnerRobinBC(h=h_i, T_fluid=T_fl),
    )
    R_o = 1.0 / (2 * math.pi * ro * h_o)
    R_w = math.log(ro / ri) / (2 * math.pi)
    R_i = 1.0 / (2 * math.pi * ri * h_i)
    qp = (T_f - T_fl) / (R_o + R_w + R_i)
    Tex = T_fl + qp * (R_i + np.log(r / ri) / (2 * math.pi))
    T = march(solver, Tex.mean() * np.ones(mesh.n_nodes), dt, int(round(t_end / dt)))
    return {"c_circ": c_circ, "n_layers": n_layers, "r": r, "T": T, "exact": Tex,
            "q_per_length": qp,
            "err_max": float(np.max(np.abs(T - Tex)) / abs(T_f - T_fl))}


# ── Case 5: thin-wall (lumped) limit, BOX under ISO 834 ──────────────────────

THIN_BOX = BoxSection(sid=1, H=0.30, T_side=0.006, T_bot=0.006, T_top=0.006, W=0.20)


def case5_thin_box(
    t_end: float = 3600.0, dt: float = 10.0, n_top: int = 4, n_side: int = 4,
    n_length: int = 2, n_layers: int = 2, h: float = 25.0, eps: float = 0.7,
) -> dict:
    """
    Thin 6 mm BOX, all outer faces exposed to ISO 834 (h = 25, ε_m = 0.7), EC3 steel.
    Compares the 3-D volume-mean temperature with (a) the 2-D SurfaceTransientSolver
    and (b) the lumped ODE  ρ c(T) A_s dT/dt = P_outer [h(T_f − T) + εσ(T_f⁴ − T⁴)].
    """
    mat = steel()
    s, L = THIN_BOX, 1.0
    mesh = BoxSolidMesher(s, L, n_top, n_side, n_length, n_layers).build()
    sol3 = SolidTransientSolver(mesh, mat, fire_temp=iso834, epsilon_m=eps, h_conv=h)
    t3, h3 = sol3.run(t_end, dt, dt)
    w = mesh.node_volume_weights
    T3_mean = h3 @ w
    T3_spread = h3.max(axis=1) - h3.min(axis=1)

    smesh = BoxSurfaceMesher(s, L, n_top, n_side, n_length).build()
    sol2 = SurfaceTransientSolver(smesh, mat, fire_temp=iso834, epsilon_m=eps, h_conv=h)
    t2, h2 = sol2.run(t_end, dt, dt)
    T2_mean = h2.mean(axis=1)

    A_s = s.W * s.H - (s.W - 2 * s.T_side) * (s.H - s.T_top - s.T_bot)
    P = 2.0 * (s.W + s.H)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        T = float(y[0])
        Tf = iso834(t)
        q = h * (Tf - T) + eps * SIGMA * ((Tf + K0) ** 4 - (T + K0) ** 4)
        return [P * q / (mat.rho * mat.specific_heat(T) * A_s)]

    ode = solve_ivp(rhs, (0.0, t_end), [20.0], t_eval=t3, rtol=1e-9, atol=1e-9,
                    max_step=2.0, method="RK45")
    T_lump = ode.y[0]
    A_2d = P * s.T_side  # surface-solver steel area (outer perimeter × t)
    return {"t": t3, "T3": T3_mean, "T2": T2_mean, "T_lump": T_lump,
            "spread3": T3_spread,
            "dev_3d_lumped": float(np.max(np.abs(T3_mean - T_lump))),
            "dev_3d_2d": float(np.max(np.abs(T3_mean - T2_mean))),
            "dev_2d_lumped": float(np.max(np.abs(T2_mean - T_lump))),
            "max_spread3": float(T3_spread.max()),
            "area_ratio_2d": A_2d / A_s,
            "vol_ratio_3d": mesh.volume / (A_s * L)}


# ── Case 6: energy conservation ───────────────────────────────────────────────

def _energy_block() -> SolidMesh:
    return BlockSolidMesher(0.2, 0.1, 0.05, 8, 4, 2).build()


def case6_energy_flux(
    material: object | None = None, t_end: float = 900.0, dt: float = 5.0,
    mass: str = "lumped", seed: int = 1,
) -> dict:
    """
    Random per-face flux (0 … 100 kW/m²) plus a uniform time-varying flux
    q(t) = 20 kW/m² (1 + sin 2πt/600), no convection / radiation / re-radiation.
    Stored energy Σ V_i ρ ∫c dT is compared with ∫∫ q dA dt (trapezoid in time,
    the exact CN balance for linear problems).
    """
    mat = material if material is not None else ConstantMaterial(45.0, 7850.0, 600.0)
    mesh = _energy_block()
    rng = np.random.default_rng(seed)
    qf = rng.uniform(0.0, 1.0e5, len(mesh.outer_face_indices))
    qt = lambda t: 2.0e4 * (1.0 + math.sin(2.0 * math.pi * t / 600.0))  # noqa: E731
    area = mesh.face_areas()[mesh.outer_face_indices]
    solver = SolidTransientSolver(
        mesh, mat, fire_temp=lambda _t: 20.0, epsilon_m=0.0, h_conv=0.0, T0=20.0,
        q_per_face=qf, q_prescribed_fn=qt, epsilon_steel=0.0, mass_matrix=mass,
    )
    _, hist = solver.run(t_end, dt, t_end)
    ts = np.linspace(0.0, t_end, int(round(t_end / dt)) + 1)
    Qdot = float(qf @ area) + np.array([qt(t) for t in ts]) * area.sum()
    E_in = float(np.sum(0.5 * (Qdot[1:] + Qdot[:-1])) * dt)
    E_st = _stored_energy(mesh, mat, hist[-1], 20.0)
    return {"E_in": E_in, "E_stored": E_st, "rel_err": abs(E_st - E_in) / E_in,
            "T_max": float(hist[-1].max()), "T_min": float(hist[-1].min())}


def _stored_energy(mesh: SolidMesh, mat: object, T: np.ndarray, T_ref: float) -> float:
    """Σ V_i ρ ∫_{T_ref}^{T_i} c(T) dT with fine trapezoidal enthalpy integration."""
    V = lumped_node_volumes(mesh)
    Tg = np.linspace(T_ref - 1.0, max(float(T.max()), T_ref) + 1.0, 40001)
    cg = np.array([mat.specific_heat(float(x)) for x in Tg])
    H = np.concatenate([[0.0], np.cumsum(0.5 * (cg[1:] + cg[:-1]) * np.diff(Tg))])
    Hn = np.interp(T, Tg, H) - np.interp(T_ref, Tg, H)
    return float(mat.rho * np.sum(V * Hn))


def case6_energy_fire(
    material: object | None = None, t_end: float = 1800.0, dt: float = 5.0,
    h: float = 25.0, eps: float = 0.7,
) -> dict:
    """
    ISO 834 convection + radiation on one half of the outer faces (face_exposure 0/1).
    Net boundary power evaluated independently each step from the solved T
    (own 2×2 Gauss face integration), trapezoid in time, vs stored energy.
    """
    mat = material if material is not None else ConstantMaterial(45.0, 7850.0, 600.0)
    mesh = _energy_block()
    c = outer_centroids(mesh)
    exposure = (c[:, 0] < 0.1).astype(float)

    def power(t: float, T: np.ndarray) -> float:
        Tf = iso834(t)

        def q(Tg: np.ndarray, fidx: np.ndarray) -> np.ndarray:
            return exposure[fidx] * (h * (Tf - Tg)
                                     + eps * SIGMA * ((Tf + K0) ** 4 - (Tg + K0) ** 4))
        return face_flux_integral(mesh, T, q)

    solver = SolidTransientSolver(mesh, mat, fire_temp=iso834, epsilon_m=eps, h_conv=h,
                                  face_exposure=exposure, nonlinear_tol=1e-10,
                                  nonlinear_max_iter=20)
    P = [power(0.0, np.full(mesh.n_nodes, 20.0))]
    T_last: list[np.ndarray] = []

    def cb(t: float, T: np.ndarray) -> None:
        P.append(power(t, T))
        T_last[:] = [T.copy()]

    solver.run(t_end, dt, t_end, callback=cb)
    P_arr = np.array(P)
    E_in = float(np.sum(0.5 * (P_arr[1:] + P_arr[:-1])) * dt)
    E_st = _stored_energy(mesh, mat, T_last[0], 20.0)
    return {"E_in": E_in, "E_stored": E_st, "rel_err": abs(E_st - E_in) / E_in,
            "T_max": float(T_last[0].max())}


# ── Case 7: Crank-Nicolson time convergence ──────────────────────────────────

def case7_time_order_linear(n: int = 6, t_end: float = 0.05,
                            steps: tuple[int, ...] = (4, 8, 16, 32),
                            ref_steps: int = 1024) -> dict:
    """Cube decay on a fixed mesh; errors vs a very-small-dt run on the same mesh."""
    ref = case1_cube_decay(n, t_end / ref_steps, t_end)["T"]
    errs = []
    for m in steps:
        T = case1_cube_decay(n, t_end / m, t_end)["T"]
        errs.append(float(np.max(np.abs(T - ref))))
    dts = [t_end / m for m in steps]
    return {"dts": dts, "errs": errs, "orders": order(errs, dts)}


class CNCurrentHistorySolver(SolidTransientSolver):
    """
    DIAGNOSTIC ONLY (not library code): Crank-Nicolson with the history terms
    evaluated with the *current* iterate matrices,
        (K_i + 2M_i/Δt) ΔT = Q_i − K_i T_{i−1} + M_i Ṫ_{i−1},
    which makes M_i Ṫ_i = Q_i − K_i T_i hold exactly (trapezoidal rule on
    Ṫ = M⁻¹(Q − K T)).  The library uses K_{i−1}, M_{i−1} (SINTEF Eq. 3.2.30),
    leaving an O(Δt) residual (M_{i−1} − M_i) Ṫ_{i−1} + (K_i − K_{i−1}) T_{i−1}.
    No Dirichlet support (not needed for the diagnostic).
    """

    def step(self, T_prev: np.ndarray, dt: float, t: float) -> np.ndarray:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        if self._K_prev is None:
            self._init_rate(T_prev, t0=0.0)
        Td = self._T_dot_prev
        T_iter = T_prev + dt * Td
        for _ in range(self._nonlinear_max_iter):
            K, M, Q = self._assemble_step(T_iter, t)
            Mm = M if sp.issparse(M) else sp.diags(M)
            A = (K + Mm * (2.0 / dt)).tocsc()
            dT = spla.spsolve(A, Q - K @ T_prev + Mm @ Td)
            T_new = T_prev + dT
            if self._nonlinear_converged(T_new, T_iter):
                break
            T_iter = T_new
        self._T_dot_prev = 2.0 / dt * dT - Td
        self._K_prev = K
        return T_new


def _plate_solver(nx: int, thick: float = 0.04, h: float = 25.0, eps: float = 0.7,
                  material: object | None = None, solver_cls: type = SolidTransientSolver,
                  **kw: object) -> tuple[SolidTransientSolver, SolidMesh]:
    mesh = BlockSolidMesher(thick, 0.01, 0.01, nx, 1, 1).build()
    exposure = (outer_centroids(mesh)[:, 0] < _TOL_GEOM).astype(float)
    solver = solver_cls(mesh, material if material is not None else steel(),
                        fire_temp=iso834, epsilon_m=eps, h_conv=h, face_exposure=exposure,
                        **kw)
    return solver, mesh


def case7_time_order_nonlinear(nx: int = 8, t_end: float = 1200.0,
                               dts: tuple[float, ...] = (60.0, 30.0, 15.0, 7.5),
                               dt_ref: float = 0.46875, material: object | None = None,
                               eps: float = 0.7,
                               solver_cls: type = SolidTransientSolver) -> dict:
    """40 mm plate, one face ISO 834 (convection [+ radiation], k(T), c(T))."""
    def final(dt: float) -> np.ndarray:
        s, _ = _plate_solver(nx, eps=eps, material=material, solver_cls=solver_cls,
                             nonlinear_tol=1e-12, nonlinear_max_iter=50)
        return s.run(t_end, dt, t_end)[1][-1]
    ref = final(dt_ref)
    errs = [float(np.max(np.abs(final(d) - ref))) for d in dts]
    return {"dts": list(dts), "errs": errs, "orders": order(errs, list(dts))}


@dataclass
class LinearPropMaterial:
    """Smooth test material: k = 50 − 0.03 T (if k_var), c = 450 + 0.5 T (if c_var)."""
    k_var: bool
    c_var: bool
    rho: float = 7850.0

    def conductivity(self, T: float) -> float:
        return 50.0 - 0.03 * T if self.k_var else 45.0

    def specific_heat(self, T: float) -> float:
        return 450.0 + 0.5 * T if self.c_var else 600.0


def case7_property_breakdown() -> dict[str, list[float]]:
    """Which nonlinearity degrades the CN order?  (smooth materials, plate case)"""
    cases = {
        "const k,c + radiation": (LinearPropMaterial(False, False), 0.7),
        "k(T) only, convection": (LinearPropMaterial(True, False), 0.0),
        "c(T) only, convection": (LinearPropMaterial(False, True), 0.0),
        "EC3 k(T),c(T) + radiation": (None, 0.7),
    }
    return {name: case7_time_order_nonlinear(material=m, eps=e)["orders"]
            for name, (m, e) in cases.items()}


# ── Case 8': thick plate through-thickness gradient vs 1-D MOL reference ─────

def plate_1d_reference(
    thick: float = 0.04, n: int = 401, t_end: float = 3600.0, t_eval: np.ndarray | None = None,
    h: float = 25.0, eps: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vertex-centred finite-volume method of lines, EC3 k(T), c(T) (scalar SteelMaterial
    functions tabulated at 0.05 °C), stiff BDF integration.  x = 0 exposed to ISO 834,
    x = thick adiabatic.  Returns (t, x, T[t, x]).
    """
    mat = steel()
    Tg = np.arange(-50.0, 1500.0, 0.05)
    kg = np.array([mat.conductivity(float(v)) for v in Tg])
    cg = np.array([mat.specific_heat(float(v)) for v in Tg])
    x = np.linspace(0.0, thick, n)
    dx = x[1] - x[0]
    vol = np.full(n, dx)
    vol[0] = vol[-1] = 0.5 * dx

    def rhs(t: float, T: np.ndarray) -> np.ndarray:
        kf = np.interp(0.5 * (T[1:] + T[:-1]), Tg, kg)
        flux = -kf * np.diff(T) / dx                     # (n−1,) +x direction
        Tf = iso834(t)
        q0 = h * (Tf - T[0]) + eps * SIGMA * ((Tf + K0) ** 4 - (T[0] + K0) ** 4)
        net = np.zeros(n)
        net[0] = q0 - flux[0]
        net[1:-1] = flux[:-1] - flux[1:]
        net[-1] = flux[-1]
        return net / (mat.rho * np.interp(T, Tg, cg) * vol)

    if t_eval is None:
        t_eval = np.linspace(0.0, t_end, 61)
    sol = solve_ivp(rhs, (0.0, t_end), np.full(n, 20.0), method="BDF", t_eval=t_eval,
                    rtol=1e-8, atol=1e-6, max_step=5.0)
    return sol.t, x, sol.y.T


def case8_thick_plate(nx: int = 16, dt: float = 5.0, t_end: float = 3600.0,
                      n_ref: int = 401) -> dict:
    """40 mm plate, ISO 834 on x = 0, adiabatic elsewhere, EC3 steel: 3-D vs 1-D MOL."""
    solver, mesh = _plate_solver(nx)
    t3, h3 = solver.run(t_end, dt, 60.0)
    x = mesh.nodes[:, 0]
    front = np.flatnonzero(x < _TOL_GEOM)
    back = np.flatnonzero(x > 0.04 - _TOL_GEOM)
    Tf3, Tb3 = h3[:, front].mean(axis=1), h3[:, back].mean(axis=1)
    tr, xr, Tr = plate_1d_reference(n=n_ref, t_end=t_end, t_eval=t3)
    Tfr, Tbr = Tr[:, 0], Tr[:, -1]
    return {"t": t3, "front3": Tf3, "back3": Tb3, "front_ref": Tfr, "back_ref": Tbr,
            "dT3": Tf3 - Tb3, "dT_ref": Tfr - Tbr,
            "err_front": float(np.max(np.abs(Tf3 - Tfr))),
            "err_back": float(np.max(np.abs(Tb3 - Tbr))),
            "err_dT": float(np.max(np.abs((Tf3 - Tb3) - (Tfr - Tbr)))),
            "dT_max_ref": float(np.max(Tfr - Tbr)),
            "x3": x, "T3_final": h3[-1], "x_ref": xr, "T_ref_final": Tr[-1]}


# ── Report driver ─────────────────────────────────────────────────────────────

@dataclass
class Row:
    case: str
    metric: str
    value: float
    tol: float | None
    kind: str = "<="    # "<=", ">=" or "info"

    @property
    def status(self) -> str:
        if self.tol is None or self.kind == "info":
            return "info"
        ok = self.value <= self.tol if self.kind == "<=" else self.value >= self.tol
        return "PASS" if ok else "FAIL"


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, *a: object, **kw: object) -> None:
        self.rows.append(Row(*a, **kw))  # type: ignore[arg-type]


def _fmt(v: float) -> str:
    return f"{v:.3e}" if (abs(v) < 1e-2 or abs(v) >= 1e4) and v != 0 else f"{v:.4g}"


def main() -> None:  # noqa: C901 — linear script
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(__file__).resolve().parent
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    rep = Report()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def tick(name: str, t0: float) -> None:
        timings[name] = time.perf_counter() - t0
        logger.info("%s: %.1f s", name, timings[name])

    # 1 ── cube decay, spatial order
    t0 = time.perf_counter()
    ns = [4, 8, 12, 16]
    c1: dict[str, list[dict]] = {}
    for mass in ("lumped", "consistent"):
        c1[mass] = [case1_cube_decay(n, 0.05 / 100, 0.05, mass) for n in ns]
        e = [r["err_max"] for r in c1[mass]]
        rms = [r["err_rms"] for r in c1[mass]]
        o = order(e, [1 / n for n in ns])
        orms = order(rms, [1 / n for n in ns])
        for n, ei in zip(ns, e):
            rep.add("1 cube decay", f"{mass}: max rel err n={n}", ei, None, "info")
        rep.add("1 cube decay", f"{mass}: max rel err n={ns[-1]}", e[-1],
                0.03 if mass == "lumped" else 0.01)
        kind = ">=" if mass == "lumped" else "info"
        rep.add("1 cube decay", f"{mass}: observed order Linf (finest pair)", o[-1], 1.8, kind)
        rep.add("1 cube decay", f"{mass}: observed order RMS (finest pair)", orms[-1], 1.8,
                kind)
        rep.notes.append(f"Case 1 {mass} orders (Linf): "
                         + ", ".join(f"{v:.3f}" for v in o)
                         + "; (RMS): " + ", ".join(f"{v:.3f}" for v in orms))
    c1["consistent, reduced Ṫ0 (diag.)"] = [
        case1_cube_decay(n, 0.05 / 100, 0.05, "consistent", ReducedInitRateSolver) for n in ns]
    ef = [r["err_max"] for r in c1["consistent, reduced Ṫ0 (diag.)"]]
    of = order(ef, [1 / n for n in ns])
    rep.add("1 cube decay", "consistent, diagnostic reduced-Ṫ0: order Linf (finest pair)",
            of[-1], 1.8, ">=")
    rep.notes.append("Case 1 consistent + diagnostic reduced-system Ṫ0 orders (Linf): "
                     + ", ".join(f"{v:.3f}" for v in of))
    s_lib = case1_consistent_dt_sensitivity()
    s_fix = case1_consistent_dt_sensitivity(solver_cls=ReducedInitRateSolver)
    rep.add("1 cube decay", "consistent n=8: err spread over Δt=T/25…T/400 (library)",
            s_lib["spread"], 1e-3)
    rep.add("1 cube decay", "consistent n=8: same with reduced-system Ṫ0 (diag.)",
            s_fix["spread"], 1e-3)
    rep.notes.append("Case 1 consistent n=8 error vs steps " + str(s_lib["steps"]) + ": library "
                     + ", ".join(f"{v:.5f}" for v in s_lib["errs"]) + "; reduced Ṫ0 "
                     + ", ".join(f"{v:.5f}" for v in s_fix["errs"])
                     + " (reduced free-DOF Ṫ0 in the library since 2026-09-27; was O(Δt) before).")
    fig, ax = plt.subplots(figsize=(5, 4))
    for mass in c1:
        ax.loglog([1 / n for n in ns], [r["err_max"] for r in c1[mass]], "o-", label=mass)
    hh = np.array([1 / n for n in ns])
    ax.loglog(hh, c1["lumped"][0]["err_max"] * (hh / hh[0]) ** 2, "k--", label="slope 2")
    ax.set_xlabel("h"), ax.set_ylabel("max |T−T_ex| / amplitude")
    ax.set_title("Case 1: sin·sin·sin decay, t = 0.05"), ax.legend(), ax.grid(True, which="both")
    fig.tight_layout(), fig.savefig(figs / "case1_convergence.png", dpi=120), plt.close(fig)
    tick("case1", t0)

    # 2 ── semi-infinite
    t0 = time.perf_counter()
    d = case2_semi_infinite_dirichlet(nx=240, dt=1.0)
    f = case2_semi_infinite_flux(nx=240, dt=1.0)
    d_coarse = case2_semi_infinite_dirichlet(nx=120, dt=2.0)
    f_coarse = case2_semi_infinite_flux(nx=120, dt=2.0)
    rep.add("2 semi-inf Dirichlet", "max |err|/(Ts−Ti), t≥300 s, h=2.5 mm", d["err_max"], 0.01)
    rep.add("2 semi-inf Dirichlet", "same, h=5 mm", d_coarse["err_max"], None, "info")
    rep.add("2 semi-inf flux", "max |err|/ΔT_surf, t≥300 s, h=2.5 mm", f["err_max"], 0.005)
    rep.add("2 semi-inf flux", "same, h=5 mm", f_coarse["err_max"], None, "info")
    rep.add("2 semi-inf", "domain depth / 4√(αt_end)", d["penetration_ratio"], 1.0, ">=")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, r, title in ((axs[0], d, "sudden surface T (erfc)"),
                         (axs[1], f, "constant flux (ierfc)")):
        o = np.argsort(r["x"])
        for t, T, Tex in zip(r["times"][1:], r["hist"][1:], r["exact"]):
            ax.plot(r["x"][o] * 1e3, T[o], "-", label=f"3-D t={t:.0f}s")
            ax.plot(r["x"][o] * 1e3, Tex[o], "k:", lw=1)
        ax.set_xlim(0, 300), ax.set_xlabel("depth x [mm]"), ax.set_ylabel("T [°C]")
        ax.set_title(f"Case 2: {title}  (dotted = analytical)"), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(figs / "case2_semi_infinite.png", dpi=120), plt.close(fig)
    tick("case2", t0)

    # 3 ── steady wall
    t0 = time.perf_counter()
    w1 = case3_wall_dirichlet()
    w2 = case3_wall_robin()
    rep.add("3 steady wall Dirichlet", "max |err| / ΔT", w1["err_max"], 1e-9)
    rep.add("3 steady wall Robin", "max |err| / ΔT_f", w2["err_max"], 1e-9)
    tick("case3", t0)

    # 4 ── pipe
    t0 = time.perf_counter()
    levels = [(16, 2), (32, 4), (64, 8), (128, 16)]
    p_d = [case4_pipe_dirichlet(c, nl) for c, nl in levels]
    p_r = [case4_pipe_inner_robin(c, nl) for c, nl in levels]
    hs = [1 / nl for _, nl in levels]
    od, orr = order([r["err_max"] for r in p_d], hs), order([r["err_max"] for r in p_r], hs)
    for (c, nl), a, b in zip(levels, p_d, p_r):
        rep.add("4 pipe Dirichlet ln(r)", f"max rel err c={c}, layers={nl}", a["err_max"],
                None, "info")
        rep.add("4 pipe InnerRobin series-R", f"max rel err c={c}, layers={nl}",
                b["err_max"], None, "info")
    rep.add("4 pipe Dirichlet ln(r)", "max rel err finest", p_d[-1]["err_max"], 1e-3)
    rep.add("4 pipe Dirichlet ln(r)", "observed order (finest pair)", od[-1], 1.8, ">=")
    rep.add("4 pipe InnerRobin series-R", "max rel err finest", p_r[-1]["err_max"], 1e-3)
    rep.add("4 pipe InnerRobin series-R", "observed order (finest pair)", orr[-1], 1.8, ">=")
    rep.notes.append("Case 4 orders (refining c_circ and n_layers together): Dirichlet "
                     + ", ".join(f"{v:.3f}" for v in od) + "; InnerRobin "
                     + ", ".join(f"{v:.3f}" for v in orr))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, title in ((axs[0], p_d, "Dirichlet ln(r)"), (axs[1], p_r, "InnerRobin")):
        r = res[1]
        o = np.argsort(r["r"])
        ax.plot(r["r"][o], r["T"][o], "o", ms=3, label=f"3-D c={r['c_circ']}, "
                                                        f"layers={r['n_layers']}")
        ax.plot(r["r"][o], r["exact"][o], "k-", label="analytical")
        ax.set_xlabel("r [m]"), ax.set_ylabel("T"), ax.set_title(f"Case 4: {title}")
        ax.legend()
    fig.tight_layout(), fig.savefig(figs / "case4_pipe.png", dpi=120), plt.close(fig)
    tick("case4", t0)

    # 5 ── thin BOX lumped limit
    t0 = time.perf_counter()
    b = case5_thin_box(dt=5.0)
    rep.add("5 thin BOX ISO834", "max |T̄3D − T_lumped| [°C]", b["dev_3d_lumped"], 5.0)
    rep.add("5 thin BOX ISO834", "max |T̄3D − T̄2D| [°C]", b["dev_3d_2d"], 20.0)
    rep.add("5 thin BOX ISO834", "max |T̄2D − T_lumped| [°C]", b["dev_2d_lumped"], None, "info")
    rep.add("5 thin BOX ISO834", "max 3-D spread in section [°C]", b["max_spread3"], None,
            "info")
    rep.notes.append("Case 5: the 3-D section spread (~15 °C at t≈8 min, mesh-converged "
                     "≈16.5 °C) sits at the four outer corners (double exposed perimeter "
                     "per steel volume); away from the corners the spread is ≈6 °C "
                     "(through-thickness q·t/2k plus hoop).  The 2-D solver models the "
                     "wall at the outer perimeter with steel area P·t, "
                     f"{100 * (b['area_ratio_2d'] - 1):.1f} % too large → slower heating.")
    rep.add("5 thin BOX ISO834", "2-D steel area / exact", b["area_ratio_2d"], None, "info")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b["t"] / 60, b["T3"], label="3-D solid (volume mean)")
    ax.plot(b["t"] / 60, b["T2"], "--", label="2-D surface solver")
    ax.plot(b["t"] / 60, b["T_lump"], ":", label="lumped ODE")
    ax.plot(b["t"] / 60, [iso834(t) for t in b["t"]], color="0.6", lw=1, label="ISO 834")
    ax.set_xlabel("t [min]"), ax.set_ylabel("T [°C]"), ax.legend()
    ax.set_title("Case 5: thin BOX 300×200×6, EC3 steel")
    fig.tight_layout(), fig.savefig(figs / "case5_thin_box.png", dpi=120), plt.close(fig)
    tick("case5", t0)

    # 6 ── energy
    t0 = time.perf_counter()
    e_lin = case6_energy_flux()
    e_lin_c = case6_energy_flux(mass="consistent")
    e_ec3 = case6_energy_flux(steel(), dt=5.0)
    e_ec3_f = case6_energy_flux(steel(), dt=1.0)
    e_fire = case6_energy_fire()
    e_fire_ec3 = case6_energy_fire(steel())
    rep.add("6 energy", "const props, flux, lumped: rel err", e_lin["rel_err"], 1e-10)
    rep.add("6 energy", "const props, flux, consistent: rel err", e_lin_c["rel_err"], 1e-10)
    rep.add("6 energy", "EC3 props, flux, dt=5: rel err", e_ec3["rel_err"], 0.01)
    rep.add("6 energy", "EC3 props, flux, dt=1: rel err", e_ec3_f["rel_err"], 0.01)
    rep.add("6 energy", "const props, ISO fire h+rad: rel err", e_fire["rel_err"], 1e-6)
    rep.add("6 energy", "EC3 props, ISO fire h+rad: rel err", e_fire_ec3["rel_err"], 0.01)
    rep.notes.append(f"Case 6 EC3 flux: T range after 900 s = {e_ec3['T_min']:.0f}"
                     f"–{e_ec3['T_max']:.0f} °C (crosses the 735 °C c(T) peak).  "
                     "With constant properties the balance is exact to round-off; the EC3 "
                     "residual (~0.1 %) has an O(Δt) part from the lagged M_(i−1) history "
                     "term (see Case 7) and a Δt-independent part because c is evaluated at "
                     "the hex-mean T while the reference enthalpy uses nodal T.")
    tick("case6", t0)

    # 7 ── time order
    t0 = time.perf_counter()
    tl = case7_time_order_linear(n=8)
    tn = case7_time_order_nonlinear()
    tn_fix = case7_time_order_nonlinear(solver_cls=CNCurrentHistorySolver)
    brk = case7_property_breakdown()
    rep.add("7 CN time order", "linear cube: order (finest pair)", tl["orders"][-1], 1.8, ">=")
    rep.add("7 CN time order", "EC3 plate ISO fire: order (finest pair)",
            tn["orders"][-1], 1.8, ">=")
    rep.add("7 CN time order", "EC3 plate, diagnostic CN with B=Q−K_i T_(i−1)+M_i Ṫ_(i−1)",
            tn_fix["orders"][-1], 1.8, ">=")
    for name, o in brk.items():
        rep.add("7 CN time order", f"order (finest pair), {name}", o[-1], None, "info")
    rep.notes.append("Case 7 diagnostic (validation-only subclass, now identical to the library CN form): "
                     "orders " + ", ".join(f"{v:.3f}" for v in tn_fix["orders"]))
    rep.notes.append("Case 7 orders: linear " + ", ".join(f"{v:.3f}" for v in tl["orders"])
                     + " (dt = " + ", ".join(f"{v:.4g}" for v in tl["dts"]) + "); nonlinear "
                     + ", ".join(f"{v:.3f}" for v in tn["orders"])
                     + " (dt = " + ", ".join(f"{v:.4g}" for v in tn["dts"]) + " s)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(np.array(tl["dts"]) / tl["dts"][0], tl["errs"], "o-", label="linear cube")
    ax.loglog(np.array(tn["dts"]) / tn["dts"][0], tn["errs"], "s-", label="EC3 plate ISO")
    ax.loglog(np.array(tn_fix["dts"]) / tn_fix["dts"][0], tn_fix["errs"], "^-",
              label="EC3 plate, current-M history (diag.)")
    r = np.array([1, 1 / 8])
    ax.loglog(r, tn["errs"][0] * r ** 2, "k--", label="slope 2")
    ax.set_xlabel("dt / dt_max"), ax.set_ylabel("max |T − T_ref|"), ax.legend()
    ax.grid(True, which="both"), ax.set_title("Case 7: CN time convergence")
    fig.tight_layout(), fig.savefig(figs / "case7_time_order.png", dpi=120), plt.close(fig)
    tick("case7", t0)

    # 8' ── thick plate
    t0 = time.perf_counter()
    p = case8_thick_plate(nx=32, dt=2.0)
    p16 = case8_thick_plate(nx=16, dt=5.0)
    rep.add("8' 40 mm plate vs 1-D MOL", "max |ΔT_front| [°C] (nx=32)", p["err_front"], 3.0)
    rep.add("8' 40 mm plate vs 1-D MOL", "max |ΔT_back| [°C] (nx=32)", p["err_back"], 3.0)
    rep.add("8' 40 mm plate vs 1-D MOL", "max |err in ΔT across| [°C] (nx=32)", p["err_dT"],
            3.0)
    rep.add("8' 40 mm plate vs 1-D MOL", "max |err in ΔT across| [°C] (nx=16)",
            p16["err_dT"], None, "info")
    rep.add("8' 40 mm plate vs 1-D MOL", "peak ΔT across thickness [°C] (ref)",
            p["dT_max_ref"], None, "info")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    tm = p["t"] / 60
    axs[0].plot(tm, p["front3"], label="3-D front"), axs[0].plot(tm, p["back3"], label="3-D back")
    axs[0].plot(tm, p["front_ref"], "k:", label="1-D MOL"), axs[0].plot(tm, p["back_ref"], "k:")
    axs[0].set_xlabel("t [min]"), axs[0].set_ylabel("T [°C]"), axs[0].legend()
    axs[0].set_title("Case 8': 40 mm plate, ISO 834 one side")
    axs[1].plot(tm, p["dT3"], label="3-D"), axs[1].plot(tm, p["dT_ref"], "k:", label="1-D MOL")
    axs[1].set_xlabel("t [min]"), axs[1].set_ylabel("T_front − T_back [°C]"), axs[1].legend()
    fig.tight_layout(), fig.savefig(figs / "case8_thick_plate.png", dpi=120), plt.close(fig)
    dt_table = [(int(t / 60), a, bb) for t, a, bb in zip(p["t"], p["dT3"], p["dT_ref"])
                if int(round(t)) % 600 == 0 and t > 0]
    tick("case8", t0)

    total = time.perf_counter() - t_start
    _write_report(out / "report_3d.md", rep, timings, total, dt_table)
    n_fail = sum(r.status == "FAIL" for r in rep.rows)
    logger.info("Done in %.1f s — %d FAIL", total, n_fail)


def _write_report(path: Path, rep: Report, timings: dict[str, float], total: float,
                  dt_table: list[tuple[int, float, float]]) -> None:
    L = ["# 3-D Hex8 solver validation report (WP-D)", "",
         "Generated by `python -m validation.validate_3d`.  Solver: "
         "`fahts/core/heat/solver/solid_solver.py` (`SolidTransientSolver`, CN θ=½).", "",
         "| Case | Metric | Value | Tolerance | Result |", "|---|---|---|---|---|"]
    for r in rep.rows:
        tol = "—" if r.tol is None or r.kind == "info" else f"{r.kind} {_fmt(r.tol)}"
        L.append(f"| {r.case} | {r.metric} | {_fmt(r.value)} | {tol} | {r.status} |")
    L += ["", "## Observed convergence orders", ""] + [f"- {n}" for n in rep.notes]
    L += ["", "## Case 8' — through-thickness temperature difference (40 mm plate)", "",
          "| t [min] | ΔT 3-D [°C] | ΔT 1-D ref [°C] |", "|---|---|---|"]
    L += [f"| {m} | {a:.1f} | {b:.1f} |" for m, a, b in dt_table]
    L += ["", "## Runtime", ""] + [f"- {k}: {v:.1f} s" for k, v in timings.items()]
    L += [f"- total: {total:.1f} s", "", "## Figures", ""]
    for p in sorted((path.parent / "figures").glob("*.png")):
        L.append(f"![{p.stem}](figures/{p.name})")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
