"""3-D vessel shell conduction: mesh geometry, energy conservation, radial agreement with the
1-D wall column, lateral spreading from a hot patch."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.solver.fem_3d import hex8_jacobians
from fahts.materials import SteelTable
from fahts.wall.column_1d import WallColumn, radial_nodes
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


def test_radial_profile_matches_1d_column():
    """Uniform fire outside, convection inside: the 3-D field is axisymmetric and equals the
    1-D radial column (same physics, fine grids)."""
    D, t, h_in, T_fl, q = 2.0, 0.06, 500.0, 300.0, 100e3
    mat = const_steel()
    m = VesselShellMesh(D=D, t=t, L=1.0, n_theta=24, n_length=2, n_radial=24)
    s = ShellConduction3D(m, mat, 300.0)
    A_o, A_i = m.outer_area_background, m.inner_area

    class Flux:
        T_flame = None

        def __call__(self, T_s, time):
            return q, 0.0, q, 0.0

    col = WallColumn(mat, D / 2, radial_nodes(t, 48), Flux(), 300.0)
    for n in range(300):
        s.step(1.0, lambda T: (q * A_o, np.zeros_like(T)),
               lambda T: (h_in * A_i * (T - T_fl), h_in * A_i))
        col.step(1.0, n + 0.5, T_fl, h_in)
    prof = m.columns(s.T)
    assert np.ptp(prof, axis=0).max() < 1e-6                            # axisymmetric
    r3 = m.r - D / 2
    T1 = np.interp(r3, col.R - D / 2, col.T)
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
