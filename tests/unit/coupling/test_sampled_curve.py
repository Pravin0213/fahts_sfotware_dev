"""Heat-transfer curves for the 3-D wall are sampled where the nodes that use them are."""

from __future__ import annotations

import numpy as np

from fahts.coupling.wall3d_coupling import sampled_curve
from fahts.process.inner_ht.nucleate_only import nucleate_only_flux

SAT = dict(T_sat=320.0, rho_l=480.0, rho_v=40.0, h_l=2.0e5, h_v=5.5e5, h_fg=3.5e5, sigma=0.006,
           mu_l=1.0e-4, mu_v=1.0e-5, k_l=0.09, k_v=0.02, cp_l=2800.0, cp_v=2000.0, P=15e5,
           P_c=42.5e5, M=44.1)
LIQ = dict(rho=480.0, cp=2800.0, mu=1.0e-4, k=0.09, beta=3e-3)


def test_boiling_curve_accurate_on_wetted_nodes_next_to_a_hot_dry_wall():
    """Jet on top: dry nodes at 500-1000 K must not dilute the samples of the steep boiling
    curve on the wetted nodes a few K above saturation."""
    fn = lambda Tw: nucleate_only_flux(Tw, 318.0, SAT, LIQ, 1.0)  # noqa: E731
    wet = np.linspace(322.0, 335.0, 200)
    T = np.concatenate((wet, np.linspace(500.0, 1000.0, 400)))
    use = np.arange(len(T)) < 200
    q, dq = sampled_curve(fn, T, 16, use=use)
    exact = np.array([fn(t) for t in wet])
    assert np.max(np.abs(q[:200] - exact) / exact) < 0.02
    slope = np.array([(fn(t + 0.01) - fn(t - 0.01)) / 0.02 for t in wet])
    assert np.median(np.abs(dq[:200] - slope) / slope) < 0.05
