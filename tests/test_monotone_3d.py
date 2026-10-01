"""
Monotone 3-D discretisation (fix for unphysical hot/cold spots next to a RadiationBall,
found on model_t1.fem 2026-09-27: −69 °C / +1577 °C with a 350 kW/m² ball, equilibrium
1450 °C).  Causes: consistent trilinear K on thin elongated hexes (positive couplings),
consistent boundary face mass, interpolated joint ties, re-radiation to 0 K, and too few
axial elements on long members.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.solid_mesh import BlockSolidMesher, PipeSolidMesher
from fahts.core.heat.solver import solid_integration as si
from fahts.core.heat.solver.fem_3d import (
    hex8_conductivity_base,
    hex8_conductivity_twopoint_base,
)
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection, PipeSection
from fahts.core.results.analysis_config import AnalysisConfig

T_EQ_350 = (350e3 / (0.7 * 5.67e-8)) ** 0.25 - 273.15      # 1450.1 °C


def _mat() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _offdiag_max(K: np.ndarray) -> float:
    return float((K - np.einsum("...ii->...i", K)[..., None] * np.eye(K.shape[-1])).max())


def test_twopoint_is_m_matrix_and_linear_exact():
    for dims in [(2.0, 0.5, 0.01), (0.1, 0.1, 0.1), (0.3, 0.02, 1.5)]:
        m = BlockSolidMesher(*dims, 1, 1, 1).build()
        X = m.nodes[m.hexes]
        K = hex8_conductivity_twopoint_base(X)[0]
        Kc = hex8_conductivity_base(X)[0]
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        assert np.abs(K.sum(axis=1)).max() < 1e-12
        assert _offdiag_max(K) <= 0.0
        x = X[0]
        T = 3 * x[:, 0] - 2 * x[:, 1] + 0.5 * x[:, 2]
        np.testing.assert_allclose(K @ T, Kc @ T, atol=1e-10 * np.abs(Kc @ T).max())
    # thin elongated brick: the consistent K has positive couplings, the two-point one not
    m = BlockSolidMesher(2.0, 0.5, 0.01, 1, 1, 1).build()
    assert _offdiag_max(hex8_conductivity_base(m.nodes[m.hexes])[0]) > 0.0


def test_twopoint_rotation_invariant_and_curved_pipe_linear_exact():
    """Rotated bricks give the same K; on a pipe mesh a linear field's flux matches
    the consistent K (orthogonal annular sectors) — catches metric-transpose errors."""
    m = BlockSolidMesher(2.0, 0.3, 0.02, 1, 1, 1).build()
    X = m.nodes[m.hexes]
    a, b = 0.7, -0.4
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Xr = X @ (Rz @ Rx).T + np.array([5.0, -3.0, 2.0])
    np.testing.assert_allclose(hex8_conductivity_twopoint_base(Xr),
                               hex8_conductivity_twopoint_base(X), atol=1e-10)
    sec = PipeSection(sid=1, outer_diameter=0.406, thickness=0.013)
    pm = PipeSolidMesher(sec, 1.0, c_circ=64, n_length=4, n_layers=2).build()
    Xp = pm.nodes[pm.hexes]
    K2 = hex8_conductivity_twopoint_base(Xp)
    Kc = hex8_conductivity_base(Xp)
    T = 2.0 * Xp[..., 0] + 0.0 * Xp[..., 1]           # linear along the axis
    r2 = np.einsum("nij,nj->ni", K2, T)
    rc = np.einsum("nij,nj->ni", Kc, T)
    # curved annular-sector hexes are not parallelepipeds: the centre-metric two-point
    # stencil carries an O(h²) curvature error (0.56 % at c_circ=64); a transposed metric
    # (the bug this guards against) gave O(1) errors on rotated/curved elements.
    np.testing.assert_allclose(r2, rc, rtol=0, atol=2e-2 * np.abs(rc).max())


@pytest.mark.parametrize("conduction, n_len, bounded", [
    ("monotone", 48, True),
    ("consistent", 4, False),        # the original artefact
])
def test_half_lit_long_pipe_stays_physical(conduction, n_len, bounded):
    """10 m Ø406×13 pipe, one half lit by 350 kW/m² (step at cosθ = 0)."""
    sec = PipeSection(sid=1, outer_diameter=0.406, thickness=0.013)
    mesh = PipeSolidMesher(sec, 10.0, c_circ=12, n_length=n_len, n_layers=2).build()
    n = mesh.face_normals()[mesh.outer_face_indices]
    q = np.where(n[:, 2] > 0.0, 350e3, 0.0)
    s = SolidTransientSolver(mesh, _mat(), fire_temp=lambda t: 20.0, epsilon_m=0.0,
                             h_conv=0.0, q_per_face=q, conduction=conduction)
    _, T = s.run(900.0, 30.0, output_dt=900.0)
    ok = T.min() >= 20.0 - 1e-6 and T.max() <= T_EQ_350 + 1.0
    assert ok == bounded, (T.min(), T.max())


def test_reradiation_is_to_ambient_not_zero_kelvin():
    """Unlit steel in prescribed-flux mode must not cool below the 20 °C ambient."""
    m = BlockSolidMesher(0.2, 0.2, 0.013, 2, 2, 2).build()
    q = np.zeros(len(m.outer_face_indices))
    for cond in ("consistent", "monotone"):
        s = SolidTransientSolver(m, _mat(), fire_temp=lambda t: 20.0, epsilon_m=0.0,
                                 h_conv=0.0, q_per_face=q, conduction=cond)
        _, T = s.run(3600.0, 60.0, output_dt=3600.0)
        np.testing.assert_allclose(T, 20.0, atol=1e-9)


def test_joint_ties_have_no_positive_couplings():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_open_issue_fixes import BOX, SMALL, _coupled, _model

    model = _model({10: BOX, 11: SMALL}, [(1, 1, 2, 10), (2, 2, 3, 10), (3, 2, 4, 11)],
                   {1: (-1, 0, 0), 2: (0, 0, 0), 3: (1, 0, 0), 4: (0, 0, 1.5)})
    *_, K = _coupled(model, AnalysisConfig(t_end=60, dt=30, output_dt=30), hot=1)
    off = K - np.diag(K.diagonal())
    assert off.max() <= 0.0
    assert np.abs(np.asarray(K.sum(axis=1))).max() < 1e-9 * abs(K).max()


def test_axial_divisions_follow_aspect_limit():
    cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)          # aspect 2, c_circ_3d 12
    pipe = PipeSection(sid=1, outer_diameter=0.406, thickness=0.013)
    h = np.pi * 0.406 / cfg.c_circ_3d
    n = si.axial_divisions(pipe, 10.0, cfg, 4)
    assert 10.0 / n <= 2.0 * h + 1e-12 and 10.0 / (n - 1) > 2.0 * h
    assert si.axial_divisions(pipe, 0.2, cfg, 4) == 4               # minimum kept
    cfg0 = AnalysisConfig(t_end=60, dt=30, output_dt=30, axial_aspect_3d=0.0)
    assert si.axial_divisions(pipe, 10.0, cfg0, 4) == 4             # disabled
    box = BoxSection(sid=2, H=0.3, W=0.3, T_side=0.01, T_bot=0.01, T_top=0.01)
    h_box = max(0.3 / cfg.n_top, 0.3 / cfg.n_side)                  # larger in-plane size
    assert si.axial_divisions(box, 6.0, cfg, 4) == int(np.ceil(6.0 / (2 * h_box)))
