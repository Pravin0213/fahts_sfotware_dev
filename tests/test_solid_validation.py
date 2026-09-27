"""
WP-D validation of the 3-D Hex8 solver (`SolidTransientSolver`) against
analytical / independent references — fast versions of `validation/validate_3d.py`.

Tolerances are set from the discretisation error expected at these (coarse)
resolutions, with ~2× head-room over the observed value; see
`validation/report_3d.md` for the high-resolution run.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation.validate_3d import (
    LinearPropMaterial,
    case1_cube_decay,
    case2_semi_infinite_dirichlet,
    case2_semi_infinite_flux,
    case3_wall_dirichlet,
    case3_wall_robin,
    case4_pipe_dirichlet,
    case4_pipe_inner_robin,
    case5_thin_box,
    case6_energy_fire,
    case6_energy_flux,
    case7_time_order_linear,
    case7_time_order_nonlinear,
    case8_thick_plate,
    order,
    steel,
)


# ── 1. sin·sin·sin decay: 2nd-order spatial convergence ──────────────────────

@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_cube_decay_spatial_order(mass: str) -> None:
    # dt small enough that the CN time error (~1e-5 rel.) is negligible vs O(h²)
    e = [case1_cube_decay(n, 0.05 / 50, 0.05, mass)["err_max"] for n in (4, 8)]
    assert order(e, [0.25, 0.125])[0] > 1.8          # Q1 elements: O(h²)
    # observed n=8: lumped 0.097, consistent 0.020 (relative to amplitude)
    assert e[1] < (0.15 if mass == "lumped" else 0.04)


# ── 2. semi-infinite solid ───────────────────────────────────────────────────

def test_semi_infinite_sudden_surface_temperature() -> None:
    r = case2_semi_infinite_dirichlet(nx=60, dt=4.0)       # h = 10 mm
    assert r["penetration_ratio"] > 1.0                    # truly semi-infinite
    assert r["err_max"] < 0.01                             # 1 % of the imposed jump


def test_semi_infinite_constant_flux() -> None:
    r = case2_semi_infinite_flux(nx=60, dt=4.0)
    assert r["err_max"] < 0.01                             # 1 % of surface rise


# ── 3. steady plane wall: linear profile reproduced to round-off ─────────────

def test_steady_wall_dirichlet_exact() -> None:
    assert case3_wall_dirichlet()["err_max"] < 1e-9


def test_steady_wall_robin_both_sides_exact() -> None:
    assert case3_wall_robin()["err_max"] < 1e-9


# ── 4. radial conduction through a pipe wall ─────────────────────────────────

def test_pipe_dirichlet_log_profile_converges() -> None:
    e = [case4_pipe_dirichlet(c, nl)["err_max"] for c, nl in ((16, 2), (32, 4))]
    assert e[1] < 5e-4                                     # observed 1.7e-4
    assert order(e, [0.5, 0.25])[0] > 1.8


def test_pipe_inner_robin_series_resistance() -> None:
    e = [case4_pipe_inner_robin(c, nl)["err_max"] for c, nl in ((16, 2), (32, 4))]
    assert e[1] < 2e-3                                     # observed 8.3e-4
    assert order(e, [0.5, 0.25])[0] > 1.8


# ── 5. thin-wall lumped limit (EC3 steel, ISO 834) ───────────────────────────

def test_thin_box_matches_lumped_and_surface_solver() -> None:
    r = case5_thin_box(t_end=3600.0, dt=10.0)
    assert abs(r["vol_ratio_3d"] - 1.0) < 1e-12            # exact steel volume
    assert r["dev_3d_lumped"] < 5.0                        # °C; observed 2.4
    # 2-D solver carries 2.5 % extra steel (P·t) → slightly slower; observed 5 °C
    assert r["dev_3d_2d"] < 10.0


# ── 6. energy conservation ───────────────────────────────────────────────────

@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_energy_conservation_constant_properties(mass: str) -> None:
    # Linear problem: CN trapezoid balance must hold to round-off
    assert case6_energy_flux(mass=mass)["rel_err"] < 1e-10


def test_energy_conservation_fire_bc() -> None:
    # Radiation + convection (nonlinear BC), constant props; Picard tol 1e-10
    assert case6_energy_fire(t_end=900.0)["rel_err"] < 1e-6


def test_energy_conservation_ec3_properties() -> None:
    # c(T) peak at 735 °C is crossed; hex-mean property evaluation → ~0.1 %
    assert case6_energy_flux(steel())["rel_err"] < 0.01


# ── 7. CN time convergence ───────────────────────────────────────────────────

def test_cn_time_order_linear() -> None:
    r = case7_time_order_linear(n=4, steps=(4, 8, 16), ref_steps=256)
    assert min(r["orders"]) > 1.9


def test_cn_time_order_nonlinear_bc_constant_props() -> None:
    r = case7_time_order_nonlinear(material=LinearPropMaterial(False, False))
    assert r["orders"][-1] > 1.8                           # observed 1.96


def test_cn_time_order_temperature_dependent_capacity() -> None:
    r = case7_time_order_nonlinear()
    assert r["orders"][-1] > 1.8


# ── 8'. thick plate: through-thickness gradient vs 1-D MOL reference ─────────

def test_thick_plate_through_thickness_gradient() -> None:
    r = case8_thick_plate(nx=8, dt=10.0, n_ref=201)
    assert r["dT_max_ref"] > 20.0                          # a real gradient exists
    assert np.all(r["dT3"][1:] > 0.0)
    assert r["err_front"] < 3.0 and r["err_back"] < 3.0    # °C
    assert r["err_dT"] < 2.0                               # °C


def test_consistent_mass_dirichlet_error_is_dt_independent() -> None:
    from validation.validate_3d import case1_consistent_dt_sensitivity
    r = case1_consistent_dt_sensitivity(n=4, steps=(25, 400))
    assert r["spread"] < 1e-3        # was 0.016 before the reduced-system Ṫ0 fix


def test_consistent_mass_dirichlet_reduced_init_diagnostic() -> None:
    from validation.validate_3d import ReducedInitRateSolver, case1_consistent_dt_sensitivity
    r = case1_consistent_dt_sensitivity(n=4, steps=(25, 400), solver_cls=ReducedInitRateSolver)
    assert r["spread"] < 1e-3
