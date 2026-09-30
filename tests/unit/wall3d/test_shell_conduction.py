"""3-D vessel shell conduction: mesh geometry, energy conservation, radial agreement with an
independent fine 1-D radial finite-volume solution, lateral spreading from a hot patch."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.solver.fem_3d import hex8_jacobians
from fahts.materials import SteelTable
from fahts.wall.fem_3d import PeakZoneGeometry, ShellConduction3D, VesselShellMesh


def const_steel(k=45.0, cp=500.0):
    T = np.array([200.0, 2500.0])
    return SteelTable("const", T, np.full(2, cp), np.full(2, k), 7850.0, np.ones(2), np.ones(2))


def test_mesh_geometry():
    peak = PeakZoneGeometry(0.4, 0.6, 90.0, 0.0)
    m = VesselShellMesh(D=2.0, t=0.06, L=6.0, n_theta=72, n_length=30, n_radial=4, peak=peak)
    detJ = np.linalg.det(hex8_jacobians(m.nodes[m.hexes]))
    assert detJ.min() > 0.0                                             # all hexes valid
    # zone edges that coincide with grid lines to round-off must not leave sliver elements
    assert np.diff(m.x).min() > 0.5 * 6.0 / 30 and np.diff(m.theta).min() > 1.0
    R = 1.0
    s = ShellConduction3D(m, const_steel(), 293.15)
    assert s.volume == pytest.approx(np.pi * ((R + 0.06) ** 2 - R ** 2) * 6.0, rel=2e-3)
    assert m.outer_face_area.sum() == pytest.approx(2 * np.pi * 1.06 * 6.0, rel=2e-3)
    assert m.peak_fraction == pytest.approx(0.2 * 90 / 360, rel=1e-6)   # zone edges exact
    m.set_level(1.0)
    assert m.wet_fraction == pytest.approx(0.5, abs=0.01)
    assert (m.inner_area_wet + m.inner_area_dry).sum() == pytest.approx(m.inner_face_area.sum())


def test_energy_conservation_uniform_flux():
    m = VesselShellMesh(D=2.0, t=0.05, L=3.0, n_theta=36, n_length=10, n_radial=4)
    mat = const_steel()
    s = ShellConduction3D(m, mat, 300.0)
    A_o = m.outer_area_background + m.outer_area_peak
    q = 50e3
    E0, Q = s.energy(), 0.0
    for _ in range(100):
        qo, _qi = s.step(1.0, lambda T: (q * A_o, np.zeros_like(T)),
                         lambda T: (np.zeros_like(T), np.zeros_like(T)))
        Q += qo.sum() * 1.0
    assert s.energy() - E0 == pytest.approx(Q, rel=1e-9)
    assert Q == pytest.approx(q * A_o.sum() * 100.0)
    assert s.T[m.outer_nodes].mean() > s.T[m.inner_nodes].mean()        # gradient inwards


def radial_fv(R_in, t, n, k, rho_cp, T0, q_out, h_in, T_fl, dt, n_steps):
    """Reference: implicit finite-volume conduction in a cylinder wall (constant properties),
    n cells, flux q_out on the outer surface, convection h_in to T_fl inside. Returns the
    cell-centre radii and temperatures."""
    rf = R_in + np.linspace(0.0, t, n + 1)                 # faces
    rc = 0.5 * (rf[1:] + rf[:-1])
    V = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)                 # per radian and metre
    G = k * rf[1:-1] / np.diff(rc)                         # interior face conductances
    A = np.diag(rho_cp * V / dt)
    for i, g in enumerate(G):
        A[i, i] += g
        A[i + 1, i + 1] += g
        A[i, i + 1] -= g
        A[i + 1, i] -= g
    g_in = 1.0 / (1.0 / (h_in * rf[0]) + (rc[0] - rf[0]) / (k * rf[0]))
    A[0, 0] += g_in
    T = np.full(n, T0)
    for _ in range(n_steps):
        b = rho_cp * V / dt * T
        b[0] += g_in * T_fl
        b[-1] += q_out * rf[-1]
        T = np.linalg.solve(A, b)
    return rc, T


def test_radial_profile_matches_1d_reference():
    """Uniform fire outside, convection inside: the 3-D field is axisymmetric and equals a
    fine 1-D radial finite-volume solution."""
    D, t, h_in, T_fl, q = 2.0, 0.06, 500.0, 300.0, 100e3
    mat = const_steel()
    m = VesselShellMesh(D=D, t=t, L=1.0, n_theta=24, n_length=2, n_radial=24)
    s = ShellConduction3D(m, mat, 300.0)
    A_o, A_i = m.outer_area_background, m.inner_area

    for _ in range(300):
        s.step(1.0, lambda T: (q * A_o, np.zeros_like(T)),
               lambda T: (h_in * A_i * (T - T_fl), h_in * A_i))
    rc, T_ref = radial_fv(D / 2, t, 200, 45.0, 7850.0 * 500.0, 300.0, q, h_in, T_fl, 1.0, 300)
    prof = m.columns(s.T)
    assert np.ptp(prof, axis=0).max() < 1e-6                            # axisymmetric
    T1 = np.interp(m.r, rc, T_ref)                                      # (flat beyond ends)
    assert np.abs(prof[0] - T1).max() < 1.0                             # K, after 300 s
    assert prof[0][-1] - prof[0][0] > 50.0                              # real gradient


def test_heating_never_undershoots():
    """M-matrix operator + lumped capacity: heating only can never cool any node (a sliver
    element from a duplicated grid line broke this and the solver's conditioning)."""
    peak = PeakZoneGeometry(0.4, 0.6, 60.0, 0.0)
    m = VesselShellMesh(D=2.0, t=0.03, L=2.0, n_theta=72, n_length=40, n_radial=3, peak=peak)
    s = ShellConduction3D(m, const_steel(), 300.0)
    for _ in range(20):
        s.step(1.0, lambda T: (200e3 * m.outer_area_peak, np.zeros_like(T)),
               lambda T: (np.zeros_like(T), np.zeros_like(T)))
    assert s.T.min() >= 300.0 - 1e-6
    assert s.cg_iterations / 20 < 10                                   # line preconditioner


def test_hot_patch_spreads_sideways():
    """Flux only in the peak zone: heat conducts into the surrounding wall (the 1-D region
    model cannot do this)."""
    peak = PeakZoneGeometry(0.4, 0.6, 60.0, 0.0)
    m = VesselShellMesh(D=2.0, t=0.03, L=2.0, n_theta=72, n_length=40, n_radial=3, peak=peak)
    s = ShellConduction3D(m, const_steel(), 300.0)
    for _ in range(300):
        s.step(1.0, lambda T: (200e3 * m.outer_area_peak, np.zeros_like(T)),
               lambda T: (np.zeros_like(T), np.zeros_like(T)))
    th, xi = m.surface_theta_xi()
    T_out = s.T[m.outer_nodes]
    near = (np.abs(((th + 180) % 360) - 180) < 45) & (np.abs(xi - 0.5) < 0.2) & \
        ~peak.contains(th, xi)
    far = (np.abs(((th - 180 + 180) % 360) - 180) < 30)                 # bottom
    assert T_out[near].max() > 330.0                                    # heated by conduction
    assert T_out[far].max() < 300.01                                    # not reached yet
    assert T_out.max() > 500.0                                          # the patch itself
