"""3-D wall for the vessel model: the shell as a Hex8 solid (``wall.fem_3d``) instead of 1-D
radial columns per region. Heat conducts through the thickness, around the circumference and
along the length, so a jet-fire hot spot spreads into the surrounding steel.

Per time step (``Wall3DCoupling.step``, replaces the model's dry / wet wall stages):
  outer surface  fire flux per node (background or peak zone, black-body + convection,
                 vectorised) or ambient exchange after the fire (sampled curve)
  inner surface  the vessel model's heat-transfer physics as curves of wall temperature,
                 sampled on a few temperatures and interpolated per node:
                   gas side   natural convection (film properties at the mean dry-wall film
                              temperature) + internal radiation network (linearised)
                   liquid     boiling (nucleate-only / full curve) or single-phase convection
                              to a dense pool, with the model's saturation logic
  -> one implicit conduction step; the heat that actually crossed each surface goes to the
     gas and liquid zones (linearised to the new wall temperatures: energy conserved).
Wall regions for output: the background / wetted / peak (dry, wetted) averages through the
thickness and ``hot``, the through-thickness profile at the hottest point of the wall, which
drives the rupture check.
"""

from __future__ import annotations

import numpy as np

from fahts.process.fluid_properties import film_gas, phase_dict
from fahts.process.inner_ht.boiling_curve import boiling_flux_and_derivative
from fahts.process.inner_ht.internal_radiation import rad_network
from fahts.process.inner_ht.natural_convection import H_CORRELATIONS
from fahts.process.inner_ht.nucleate_only import nucleate_only_flux
from fahts.wall.fem_3d import PeakZoneGeometry, ShellConduction3D, VesselShellMesh
from fahts.wall.fem_3d.shell_mesh import _angle_distance, wetted_half_angle_deg

G_ACC = 9.81  # as the 1-D path (docs/process_model_known_issues.md #5)
REGIONS = ("background", "wet", "peak", "peak_wet", "hot")


def sampled_curve(fn, T_nodes: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a scalar function f(T) on ``n`` temperatures spanning the node temperatures,
    then interpolate value and slope at every node (piecewise linear, slope from the grid)."""
    lo, hi = float(np.min(T_nodes)), float(np.max(T_nodes))
    if hi - lo < 1e-6:
        f0 = fn(lo)
        return np.full(T_nodes.shape, f0), np.full(T_nodes.shape, (fn(lo + 0.05) - f0) / 0.05)
    grid = np.linspace(lo - 0.02 * (hi - lo) - 0.05, hi + 0.02 * (hi - lo) + 0.05, n)
    vals = np.array([fn(float(T)) for T in grid])
    slope = np.gradient(vals, grid)
    return np.interp(T_nodes, grid, vals), np.interp(T_nodes, grid, slope)


class Wall3DCoupling:
    """Owns the 3-D shell of one ``VesselFireModel`` (see module docstring)."""

    def __init__(self, model) -> None:
        self.model = m = model
        opt, s = m.opt, m.s
        peak = None
        if m.peak is not None:
            p = m.peak
            peak = PeakZoneGeometry(p.xi_start, p.xi_end, p.circ_deg, p.attack_deg)
        self.mesh = VesselShellMesh(
            D=m.D,
            t=m.t_w,
            L=m.L,
            n_theta=opt.wall3d_n_theta,
            n_length=opt.wall3d_n_length,
            n_radial=opt.wall3d_n_radial,
            peak=peak,
        )
        self.solver = ShellConduction3D(self.mesh, m.mat, s["T_shell"])
        self.bc_background = m._outer_bc(False)
        self.bc_peak = m._outer_bc(True) if m.peak is not None else self.bc_background
        self.n_curve = opt.wall3d_curve_points
        th, xi = self.mesh.surface_theta_xi()
        self._col_theta = th
        self._col_peak = peak.contains(th, xi) if peak is not None else np.zeros(len(th), bool)
        # column weights for region averages: outer surface area of each node column
        self._col_area = self.mesh.outer_area_background + self.mesh.outer_area_peak
        self.x_nodes = self.mesh.r - self.mesh.r[0]
        self.q_out_last = np.zeros(len(self._col_area))  # W per outer node, last step
        self.times: list[float] = []
        self.history: list[np.ndarray] = []
        self.set_level(m._level())

    # ------------------------------------------------------------------ geometry state
    def set_level(self, level: float) -> None:
        self.mesh.set_level(level)
        half = wetted_half_angle_deg(level, self.mesh.D)
        self._col_wet = _angle_distance(self._col_theta, 180.0) < half

    def fractions(self) -> dict[str, float]:
        """Shell area fractions of the background / peak wet regions (as 1-D ``frac``)."""
        m = self.mesh
        A = m.inner_face_area
        tot = A.sum()
        return dict(
            wet=float(A[m.face_wet & ~m.face_in_peak].sum() / tot),
            peak_wet=float(A[m.face_wet & m.face_in_peak].sum() / tot),
            dry=float(A[~m.face_wet & ~m.face_in_peak].sum() / tot),
            peak_dry=float(A[~m.face_wet & m.face_in_peak].sum() / tot),
        )

    # ------------------------------------------------------------------ outer boundary
    def _outer_rates(self, bc, area: np.ndarray, T: np.ndarray, t: float):
        """Heat rate INTO the wall [W] and its derivative for one flux type."""
        if not np.any(area):
            return np.zeros_like(T), np.zeros_like(T)
        fire = getattr(bc, "fire", None)
        q_spec = float(np.interp(t, bc.t, bc.q)) if fire is not None else 0.0
        Tf = fire.flame_temperature(q_spec) if fire is not None else None
        if Tf is not None:
            q = fire.q_rad(T, Tf) + fire.q_conv(T, Tf)
            dq = fire.dq_dTs(T)
        else:  # ambient: scalar air properties
            ambient = bc.after if fire is not None else bc
            q, dq = sampled_curve(lambda Ts: ambient(Ts, t)[0], T, self.n_curve)
        return area * q, area * dq

    # ------------------------------------------------------------------ one step
    def step(self, c) -> None:
        """Replaces the model's dry- and wet-wall stages for one time step."""
        md = self.model
        opt, mm, G, Lz, P, info, sat = md.opt, md.m, md.G, md.Lz, md.P, md.info, md.sat
        mesh, sol, dt = self.mesh, self.solver, md.dt
        T_in = sol.T[mesh.inner_nodes]
        A_dry, A_wet = mesh.inner_area_dry, mesh.inner_area_wet
        A1 = float(A_dry.sum())
        T_fl = G.T if c.src_is_gas else Lz.T

        # ---- gas side: convection (+ internal radiation) as curves of the wall temperature
        T_dry_mean = float(np.average(T_in, weights=A_dry)) if A1 > 0 else float(G.T)
        pg = film_gas(mm, c.yG, 0.5 * (T_dry_mean + G.T), P)
        sat["pg"] = pg
        Pr = pg["cp"] * pg["mu"] / pg["k"]
        ra_coef = (
            G_ACC * abs(pg["beta"]) * md.D**3 * pg["rho"] ** 2 * pg["cp"] / (pg["mu"] * pg["k"])
        )
        corr = H_CORRELATIONS[opt.h_corr]

        def h_gas(Tw):
            dTg = abs(Tw - G.T)
            return corr(ra_coef * dTg, Pr) * pg["k"] / md.D if dTg > 1e-3 else 0.0

        h_n, dh_n = sampled_curve(h_gas, T_in, self.n_curve)
        q_cv = h_n * (T_in - T_fl)  # W/m2 out of the wall
        dq_cv = h_n + dh_n * (T_in - T_fl)
        h_r = T_ref_r = 0.0
        Q1 = Q3 = 0.0
        rad_on = opt.rad_internal and A1 > 0 and c.src_is_gas
        if rad_on:
            A3 = md.geom.interface_area(md.geom.level(Lz.V)) if not Lz.empty else 0.0
            T3 = Lz.T if not Lz.empty else G.T
            args = (T3, G.T, A1, A3, opt.eps_wall_in, opt.eps_liq, opt.eps_gas)
            Q1, Q3, _ = rad_network(T_dry_mean, *args)
            Q1b = rad_network(T_dry_mean + 0.05, *args)[0]
            h_r = max((Q1b - Q1) / 0.05 / A1, 1e-6)
            T_ref_r = T_dry_mean - (Q1 / A1) / h_r
        q_rd = h_r * (T_in - T_ref_r) if rad_on else np.zeros_like(T_in)

        # ---- liquid side: the model's wet-wall logic as a curve of the wall temperature
        sd = c.sd
        if (
            c.liq_present
            and sd is None
            and opt.wet_above_crit == "boiling"
            and sat.get("last") is not None
            and not (mm.iw is not None and (Lz.n / Lz.N)[mm.iw] > 0.99)
        ):
            sd = dict(sat["last"], T_sat=Lz.T, P=P)
        q_wt = dq_wt = np.zeros_like(T_in)
        regime, h_wet = "", 0.0
        if c.liq_present and A_wet.sum() > 0:
            T_wet_mean = float(np.average(T_in, weights=A_wet))
            fn, regime = self._wet_curve(sd, T_wet_mean)
            q_wt, dq_wt = sampled_curve(fn, T_in, self.n_curve)
            h_wet = float(
                np.average(
                    np.divide(
                        q_wt, T_in - Lz.T, out=np.zeros_like(q_wt), where=np.abs(T_in - Lz.T) > 1e-6
                    ),
                    weights=A_wet,
                )
            )

        def inner(T):
            Q = A_dry * (q_cv + q_rd) + A_wet * q_wt
            dQ = A_dry * (dq_cv + h_r) + A_wet * dq_wt
            return Q, dQ

        t_mid = c.t_mid

        def outer(T):
            qb, dqb = self._outer_rates(self.bc_background, mesh.outer_area_background, T, t_mid)
            qp, dqp = self._outer_rates(self.bc_peak, mesh.outer_area_peak, T, t_mid)
            return qb + qp, dqb + dqp

        Q_in, _Q_out = sol.step(dt, outer, inner)
        dT = sol.T[mesh.inner_nodes] - T_in
        Q_cv = float(np.sum(A_dry * (q_cv + dq_cv * dT)))
        Q_rad = float(np.sum(A_dry * (q_rd + h_r * dT)))
        Q_wl = float(np.sum(A_wet * (q_wt + dq_wt * dT)))
        if rad_on:
            scale = Q_rad / Q1 if abs(Q1) > 1.0 else 1.0
            Q_rad_l = -Q3 * scale  # share received by the liquid surface
            Q_rad_g = Q_rad - Q_rad_l
            info["h_rad"] = h_r
        else:
            Q_rad_l = Q_rad_g = 0.0
        md.Q_fire += float(Q_in.sum()) * dt
        self.q_out_last = Q_in
        h_dry = float(np.average(h_n, weights=A_dry)) if A1 > 0 else 0.0
        info["h_dry"], info["Q_rad_l"], info["Q_rad_g"] = h_dry, Q_rad_l, Q_rad_g
        if c.liq_present and A_wet.sum() > 0:
            info["h_wet"], info["regime"] = h_wet, regime
        c.h_dry, c.Q_wg, c.Q_rad_l, c.Q_wl, c.sd = h_dry, Q_cv + Q_rad_g, Q_rad_l, Q_wl, sd

    def _wet_curve(self, sd, T_wet_mean):
        """q(T_wall) into the liquid [W/m2] and the regime name (the 1-D model's logic)."""
        md = self.model
        opt, mm, Lz, P = md.opt, md.m, md.Lz, md.P
        Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
        Lnc = md.geom.liquid_char_length(md.geom.level(Lz.V))
        if sd is not None:
            pl = phase_dict(mm, Lph)
            if opt.wet_boiling == "nucleate_only":
                return (lambda Tw: nucleate_only_flux(Tw, Lz.T, sd, pl, Lnc)), "nucleate_only"

            def full(Tw):
                try:
                    return boiling_flux_and_derivative(Tw, Lz.T, sd, pl, Lnc)[0]
                except Exception:  # noqa: BLE001 - as the 1-D path: no flux if the curve fails
                    return 0.0

            return full, "boiling"
        # single-phase natural convection to a dense (compressed / supercritical) pool:
        # film properties at the mean wetted-wall film temperature
        try:
            pl = (
                phase_dict(mm, mm.phase_props(Lph.x, 0.5 * (T_wet_mean + Lz.T), P, "L"))
                if Lph.name != "aqueous"
                else phase_dict(mm, Lph)
            )
        except Exception:  # noqa: BLE001
            pl = phase_dict(mm, Lph)
        corr = H_CORRELATIONS[opt.h_corr_liq or opt.h_corr]
        Pr_l = pl["cp"] * pl["mu"] / pl["k"]
        use_drho = opt.liq_grashof == "drho" and Lph.name != "aqueous"

        def q_single(Tw):
            dTl = abs(Tw - Lz.T)
            if dTl <= 1e-3:
                return 0.0
            if use_drho:
                try:
                    rho_w = mm.phase_props(Lph.x, Tw, P, "L").rho
                except Exception:  # noqa: BLE001
                    rho_w = Lph.rho
                Ra = (
                    G_ACC
                    * abs(Lph.rho - rho_w)
                    * Lnc**3
                    * pl["rho"]
                    * pl["cp"]
                    / (pl["mu"] * pl["k"])
                )
            else:
                Ra = (
                    G_ACC
                    * abs(pl["beta"])
                    * dTl
                    * Lnc**3
                    * pl["rho"] ** 2
                    * pl["cp"]
                    / (pl["mu"] * pl["k"])
                )
            return corr(Ra, Pr_l) * pl["k"] / Lnc * (Tw - Lz.T)

        return q_single, "single-phase"

    # ------------------------------------------------------------------ output
    def energy(self) -> float:
        return self.solver.energy()

    def region_profiles(self) -> dict[str, tuple[np.ndarray, float, float]]:
        """{region: (node T through the wall [K] inner->outer, area share, outer flux W/m2)}.
        Regions without area are omitted; ``hot`` is the column with the highest
        through-wall mean temperature."""
        cols = self.mesh.columns(self.solver.T)  # (n_col, n_r+1)
        w_r = self.mesh.r / self.mesh.r.sum()  # radius weighting
        col_mean = cols @ w_r
        q_col = self.q_out_last / np.where(self._col_area > 0, self._col_area, 1.0)
        masks = {
            "background": ~self._col_peak & ~self._col_wet,
            "wet": ~self._col_peak & self._col_wet,
            "peak": self._col_peak & ~self._col_wet,
            "peak_wet": self._col_peak & self._col_wet,
        }
        out = {}
        tot = self._col_area.sum()
        for key, mk in masks.items():
            a = self._col_area * mk
            if a.sum() > 0:
                out[key] = (a @ cols / a.sum(), float(a.sum() / tot), float(a @ q_col / a.sum()))
        i = int(np.argmax(col_mean))
        out["hot"] = (cols[i].copy(), 0.0, float(q_col[i]))
        return out

    def through_wall_mean(self, profile: np.ndarray) -> float:
        r = self.mesh.r
        return float(np.sum(profile * r) / r.sum())

    def store(self, time: float) -> None:
        self.times.append(float(time))
        self.history.append(self.solver.T.astype(np.float32))
