"""Time-dependent fire boundary conditions for a wall surface."""

from __future__ import annotations

import numpy as np

from fahts.fire.guideline_fire import GuidelineFire


class FireBC:
    """Fire with a specified heat-load time series, converted to a fire gas
    temperature (GuidelineFire). Where the load is zero, falls back to `after`
    (normally AmbientBC)."""

    def __init__(self, fire: GuidelineFire, t_series, q_series, after=None):
        self.fire, self.t, self.q, self.after = (
            fire,
            np.asarray(t_series),
            np.asarray(q_series),
            after,
        )
        self.T_flame = None

    def __call__(self, T_s, t):
        q_spec = float(np.interp(t, self.t, self.q))
        Tf = self.fire.flame_temperature(q_spec)
        self.T_flame = Tf
        if Tf is None:
            return self.after(T_s, t) if self.after else (0.0, 0.0, 0.0, 0.0)
        qr, qc = self.fire.q_rad(T_s, Tf), self.fire.q_conv(T_s, Tf)
        return qr + qc, self.fire.dq_dTs(T_s), qr, qc


class PrescribedFluxBC:
    """Net absorbed flux given as a time series (diagnostic: e.g. another code's output)."""

    def __init__(self, t_series, q_series, rad_series=None):
        self.t, self.q = np.asarray(t_series), np.asarray(q_series)
        self.qr = None if rad_series is None else np.asarray(rad_series)
        self.T_flame = None

    def __call__(self, T_s, t):
        q = float(np.interp(t, self.t, self.q))
        qr = float(np.interp(t, self.t, self.qr)) if self.qr is not None else q
        return q, 0.0, qr, q - qr
