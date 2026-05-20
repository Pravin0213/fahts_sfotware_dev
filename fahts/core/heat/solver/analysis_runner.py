"""
Analysis orchestration — FAHTS heat transfer solver.

Outer loop: time steps (matching USFOS terminal output style).
Inner loop: all exposed elements advanced in parallel at each time step.

This produces per-time-step global temperature statistics printed in real-time,
identical to what the USFOS FAHTS binary prints to the terminal.

Supports all structural element and section types using FAHTS axial × hoop
surface-shell approach (SINTEF FAHTS §3.2.2):
  - BOX      → BoxSurfaceMesher      + SurfaceTransientSolver
  - IHPROFIL → IProfileSurfaceMesher + SurfaceTransientSolver
  - PIPE     → PipeSurfaceMesher     + SurfaceTransientSolver
  - Shell    → PlateSurfaceMesher    + SurfaceTransientSolver  (QUADSHEL only)
               ShellMesher           + Shell1DSolver           (TRISHELL fallback)

Supports heat sources:
  - FireZone    — rectangular zone with fire curve (ISO 834, HC, user-defined)
  - RadiationBall — two-zone spherical USERFLUX source (prescribed flux, no convection)
"""
from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import numpy as np

from fahts.core.heat.bc.view_factor import exposed_element_ids, geometric_view_factor
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.iprofil_surface_mesher import IProfileSurfaceMesher
from fahts.core.heat.section_mesh.pipe_surface_mesher import PipeSurfaceMesher
from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
from fahts.core.heat.section_mesh.shell_mesh import ShellMesher
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.shell_1d_solver import Shell1DSolver
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig
from fahts.core.results.temperature_field import TemperatureField

log = logging.getLogger(__name__)

# Minimum falloff flux [W/m²] below which an element is not considered exposed
# by RadiationBall inverse-square falloff (§3.5.4).  Prevents solving thousands
# of elements that receive negligible radiation at extreme distances.
_MIN_FALLOFF_FLUX: float = 1.0


class AnalysisCancelledError(Exception):
    """Raised by run_analysis when the cancel_check callback returns True."""


def run_analysis(
    model: FEMModel,
    fire_zones: list,
    config: AnalysisConfig,
    progress_cb: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> TemperatureField:
    """
    Run the transient heat-transfer analysis for all exposed elements.

    Uses a time-step-outer / element-inner loop so that per-step global
    temperature statistics can be reported in real-time (matching USFOS output).
    Elements do not exchange heat with each other — each is driven solely by
    the fire boundary condition.

    Parameters
    ----------
    model:       Loaded FEMModel.
    fire_zones:  Active fire sources (FireZone and/or RadiationBall).
    config:      AnalysisConfig from RunAnalysisDialog.
    progress_cb: Optional callback(step, total_steps, -1) called each time step.
    cancel_check: Optional callable that returns True when cancelled.
    log_cb:      Optional callable(str) for USFOS-style console output.

    Returns
    -------
    TemperatureField with merged results for all analysed elements.

    Raises
    ------
    AnalysisCancelledError
        If cancel_check() returns True during the run.
    ValueError
        If no supported elements are exposed to any fire source.
    """
    config.validate()

    active_sources = [s for s in fire_zones if s.active]
    active_zones   = [s for s in active_sources if isinstance(s, FireZone)]
    active_balls   = [s for s in active_sources if isinstance(s, RadiationBall)]

    # ── Determine exposed elements ────────────────────────────────────────────
    if config.element_ids:
        beam_eids  = [e for e in config.element_ids if e in model.elements]
        shell_eids = [e for e in config.element_ids if e in model.shell_elements]
    else:
        beam_eids  = sorted(_exposed_beam_ids(model, active_zones, active_balls))
        shell_eids = sorted(_exposed_shell_ids(model, active_zones, active_balls))

    # ── Build beam solvers ────────────────────────────────────────────────────
    solvers:       dict[int, object]         = {}
    meshes:        dict[int, object]         = {}
    T_states:      dict[int, np.ndarray]     = {}
    sec_type_name: dict[int, str]            = {}
    is_surface:    dict[int, bool]           = {}   # True → from_surface_solver_run

    valid_beam_eids: list[int] = []
    for eid in beam_eids:
        elem = model.elements[eid]
        sec  = model.sections.get(elem.geom_id)
        mat  = model.materials.get(elem.mat_id)
        if mat is not None and mat.rho == 0.0:
            log.debug("Element %d: rho=0 (rigid material %d) — skipped.", eid, mat.mid)
            continue
        if not isinstance(sec, (BoxSection, ISection, PipeSection)):
            log.warning(
                "Element %d: section type %s not supported — skipped.",
                eid, type(sec).__name__ if sec else "None",
            )
            continue
        midpoint = elem.midpoint(model.nodes)
        fire_temp, epsilon_m, h_conv, q_fn, covering_zones, ref_zone = _bc_for_element(
            midpoint, active_zones, active_balls, config
        )

        # §3.5.4 falloff: element is beyond r2 of every active ball
        falloff_ball: RadiationBall | None = None
        if fire_temp is None:
            falloff_ball = _bc_falloff_ball(midpoint, active_balls)
            if falloff_ball is None:
                log.warning("Element %d: no active source covers midpoint — skipped.", eid)
                continue
            fire_temp     = lambda t: 20.0   # noqa: E731
            epsilon_m     = 0.0
            h_conv        = 0.0
            q_fn          = lambda t: 0.0    # noqa: E731
            covering_zones = []
            ref_zone      = None

        mesh = _build_beam_surface_mesh(sec, elem.length, config)

        # §3.3.4 geometric view factors (only for FireZone sources)
        vf = eps_s = eps_f = None
        if covering_zones:
            vf    = _element_view_factors(mesh, elem, model.nodes, covering_zones)
            eps_s = 0.7
            eps_f = ref_zone.epsilon_fire

        # §3.5.4 per-face directional flux for falloff elements
        q_per_quad: np.ndarray | None = None
        if falloff_ball is not None:
            q_per_quad = _rad_ball_per_quad_flux(falloff_ball, mesh, elem, model.nodes)
            if np.all(q_per_quad == 0.0):
                log.debug("Element %d: falloff flux zero for all faces — skipped.", eid)
                continue

        solvers[eid]       = SurfaceTransientSolver(
            mesh=mesh, material=mat, fire_temp=fire_temp,
            epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
            epsilon_steel=eps_s, epsilon_fire=eps_f, view_factors=vf,
            q_per_quad=q_per_quad,
        )
        meshes[eid]        = mesh
        T_states[eid]      = np.full(mesh.n_nodes, 20.0)
        sec_type_name[eid] = type(sec).__name__
        is_surface[eid]    = True
        valid_beam_eids.append(eid)

    # ── Build shell solvers ───────────────────────────────────────────────────
    valid_shell_eids: list[int] = []
    for eid in shell_eids:
        shell = model.shell_elements[eid]
        sec   = model.sections.get(shell.geom_id)
        if not isinstance(sec, PlateSection):
            log.warning("Shell %d: section not PlateSection — skipped.", eid)
            continue
        mat      = model.materials[shell.mat_id]
        midpoint = np.mean([model.nodes[nid].xyz for nid in shell.nodes], axis=0)
        fire_temp, epsilon_m, h_conv, q_fn, _czones, _rzone = _bc_for_element(
            midpoint, active_zones, active_balls, config
        )

        # §3.5.4 falloff for shells beyond r2
        shell_falloff_ball: RadiationBall | None = None
        if fire_temp is None:
            shell_falloff_ball = _bc_falloff_ball(midpoint, active_balls)
            if shell_falloff_ball is None:
                log.warning("Shell %d: no active source covers midpoint — skipped.", eid)
                continue
            fire_temp  = lambda t: 20.0   # noqa: E731
            epsilon_m  = 0.0
            h_conv     = 0.0
            q_fn       = lambda t: 0.0    # noqa: E731

        if len(shell.nodes) == 4:
            corners = np.array([model.nodes[nid].xyz for nid in shell.nodes])
            mesh    = PlateSurfaceMesher(
                section=sec, corners=corners,
                mesh_12=config.mesh_12, mesh_14=config.mesh_14,
            ).build()
            # Per-quad directional falloff flux for QUADSHEL
            q_per_quad_sh: np.ndarray | None = None
            if shell_falloff_ball is not None:
                dist     = float(np.linalg.norm(midpoint - shell_falloff_ball.center))
                r_hat    = (midpoint - shell_falloff_ball.center) / dist
                # Shell normal from corner geometry
                c        = corners
                sn       = np.cross(c[1] - c[0], c[3] - c[0])
                sn_norm  = np.linalg.norm(sn)
                sn       = sn / sn_norm if sn_norm > 1e-9 else sn
                cos_th   = float(np.dot(sn, -r_hat))
                base_q   = shell_falloff_ball.flux2 * (shell_falloff_ball.r2 / dist) ** 2
                # Uniform value across all quads (shell is flat)
                q_val    = base_q * max(0.0, cos_th)
                if q_val == 0.0:
                    log.debug("Shell %d: falloff flux zero (facing away) — skipped.", eid)
                    continue
                q_per_quad_sh = np.full(mesh.n_quads, q_val)
            solver  = SurfaceTransientSolver(
                mesh=mesh, material=mat, fire_temp=fire_temp,
                epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
                q_per_quad=q_per_quad_sh,
            )
            is_surf = True
            stype   = "QUADSHEL"
        else:
            if shell_falloff_ball is not None:
                log.debug("Shell %d: TRISHELL falloff not supported — skipped.", eid)
                continue
            mesh    = ShellMesher(sec, n_layers=config.n_layers).build()
            solver  = Shell1DSolver(
                mesh=mesh, material=mat, fire_temp=fire_temp,
                epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
            )
            is_surf = False
            stype   = "TRISHELL"
        solvers[eid]       = solver
        meshes[eid]        = mesh
        T_states[eid]      = np.full(mesh.n_nodes, 20.0)
        sec_type_name[eid] = stype
        is_surface[eid]    = is_surf
        valid_shell_eids.append(eid)

    all_eids = valid_beam_eids + valid_shell_eids
    total    = len(all_eids)
    if total == 0:
        raise ValueError(
            "No supported elements are exposed to fire. "
            "Supported cross-sections: BOX, IHPROFIL, PIPE, QUADSHEL/TRISHELL."
        )

    # ── USFOS-style header + step-table column headers ────────────────────────
    n_steps   = max(1, round(config.t_end / config.dt))
    out_every = max(1, round(config.output_dt / config.dt))

    if log_cb is not None:
        _log_header(log_cb, model, config, len(valid_beam_eids), len(valid_shell_eids),
                    total, active_zones, active_balls)

    # ── Output storage ────────────────────────────────────────────────────────
    out_times: list[float]                   = [0.0]
    out_T:     dict[int, list[np.ndarray]]   = {eid: [T_states[eid].copy()] for eid in all_eids}

    # ── Time-step outer loop ──────────────────────────────────────────────────
    t        = 0.0
    step_log = 0

    # Use threads to advance independent elements in parallel.
    # NumPy/SciPy release the GIL during BLAS/LAPACK calls so threads give
    # real CPU overlap even in CPython.  A single worker avoids thread
    # overhead for trivially small models.
    n_workers = min(total, os.cpu_count() or 4) if total > 1 else 1

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for step_i in range(n_steps):
            dt_step = min(config.dt, config.t_end - t)
            if dt_step <= 0.0:
                break
            t += dt_step

            if cancel_check is not None and cancel_check():
                raise AnalysisCancelledError(f"Analysis cancelled at t={t:.2f} s.")

            # Advance all elements in parallel — each solver is independent
            futures = {
                executor.submit(solvers[eid].step, T_states[eid], dt_step, t): eid
                for eid in all_eids
            }
            for fut in as_completed(futures):
                T_states[futures[fut]] = fut.result()

            # Emit one USFOS-style step row at every internal step
            step_log += 1
            if log_cb is not None:
                T_max_now = max(float(np.max(T_states[eid])) for eid in all_eids)
                T_min_now = min(float(np.min(T_states[eid])) for eid in all_eids)
                _log_step_row(log_cb, step_log, t / 60.0, T_max_now, T_min_now)

            # Store output at intervals
            if (step_i + 1) % out_every == 0 or step_i == n_steps - 1:
                out_times.append(t)
                for eid in all_eids:
                    out_T[eid].append(T_states[eid].copy())

            if progress_cb is not None:
                progress_cb(step_i + 1, n_steps, -1)

    if progress_cb is not None:
        progress_cb(n_steps, n_steps, -1)

    # ── Build TemperatureField per element ────────────────────────────────────
    times_arr = np.array(out_times)
    fields: list[TemperatureField] = []

    if log_cb is not None:
        _log_element_section_header(log_cb)

    for eid in valid_beam_eids:
        T_hist = np.array(out_T[eid])
        tf     = TemperatureField.from_surface_solver_run(
            eid=eid, times=times_arr, T_history=T_hist, mesh=meshes[eid]
        )
        fields.append(tf)
        t_peak = tf.peak_centroid_temperature(eid)
        log.info("Beam %d (%s) solved: T_peak=%.1f °C", eid, sec_type_name[eid], t_peak)
        if log_cb is not None:
            _log_element_row(log_cb, eid, sec_type_name[eid], t_peak)

    for eid in valid_shell_eids:
        T_hist = np.array(out_T[eid])
        mesh   = meshes[eid]
        if is_surface[eid]:
            tf = TemperatureField.from_surface_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh
            )
        else:
            tf = TemperatureField.from_shell_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh
            )
        fields.append(tf)
        t_peak = tf.peak_centroid_temperature(eid)
        log.info("Shell %d solved: T_peak=%.1f °C", eid, t_peak)
        if log_cb is not None:
            _log_element_row(log_cb, eid, sec_type_name[eid], t_peak)

    if not fields:
        raise ValueError("Analysis produced no results — all elements were skipped.")

    merged = TemperatureField.merge(fields)
    log.info(
        "Analysis complete: %d elements solved, %d time steps.",
        merged.n_elements, merged.n_steps,
    )
    if log_cb is not None:
        _log_summary(log_cb, merged)

    return merged


# ── Exposure helpers ──────────────────────────────────────────────────────────

def _exposed_beam_ids(model, active_zones, active_balls) -> set[int]:
    result: set[int] = set()
    if active_zones:
        result |= exposed_element_ids(model.elements, active_zones, model.nodes)
    for ball in active_balls:
        result |= ball.exposed_element_ids(model.elements, model.nodes).keys()
        # §3.5.4 falloff: elements beyond r2 with flux > _MIN_FALLOFF_FLUX
        result |= ball.falloff_element_ids(
            model.elements, model.nodes, min_flux=_MIN_FALLOFF_FLUX
        )
    return result


def _exposed_shell_ids(model, active_zones, active_balls) -> set[int]:
    if not model.shell_elements:
        return set()
    result: set[int] = set()
    for shell in model.shell_elements.values():
        mid = np.mean([model.nodes[nid].xyz for nid in shell.nodes], axis=0)
        for zone in active_zones:
            if zone.contains_midpoint(mid):
                result.add(shell.eid)
                break
        else:
            for ball in active_balls:
                dist = float(np.linalg.norm(mid - ball.center))
                if ball.flux_at(dist) > 0.0:
                    result.add(shell.eid)
                    break
                # §3.5.4 falloff beyond r2
                if dist > ball.r2:
                    base_flux = ball.flux2 * (ball.r2 / dist) ** 2
                    if base_flux > _MIN_FALLOFF_FLUX:
                        result.add(shell.eid)
                        break
    return result


def _bc_for_element(
    midpoint: np.ndarray,
    active_zones: list[FireZone],
    active_balls: list[RadiationBall],
    config: AnalysisConfig,
) -> tuple:
    """
    Determine fire BC parameters for an element at `midpoint`.

    Returns (fire_temp_fn, epsilon_m, h_conv, q_prescribed_fn,
             covering_zones, ref_zone).
    Returns (None, ...) when no source covers the midpoint.

    RadiationBall takes precedence over FireZone when both cover an element;
    in that case covering_zones is empty.
    """
    covering_balls: list[tuple[RadiationBall, float]] = []
    for b in active_balls:
        dist = float(np.linalg.norm(midpoint - b.center))
        flux = b.flux_at(dist)
        if flux > 0.0:
            covering_balls.append((b, flux))

    if covering_balls:
        _ref_ball, ref_flux = max(covering_balls, key=lambda x: x[1])
        fire_temp = lambda t: 20.0              # noqa: E731
        q_fn      = lambda t, _f=ref_flux: _f  # noqa: E731
        return fire_temp, 0.0, 0.0, q_fn, [], None

    covering_zones = [z for z in active_zones if z.contains_midpoint(midpoint)]
    if not covering_zones:
        return None, 0.0, 0.0, lambda _t: 0.0, [], None

    if len(covering_zones) == 1:
        fire_temp = covering_zones[0].temperature
    else:
        def fire_temp(t, _zones=covering_zones):
            return max(z.temperature(t) for z in _zones)

    ref_zone  = max(covering_zones, key=lambda z: z.temperature(config.t_end))
    epsilon_m = ref_zone.epsilon_fire * 0.7
    h_conv    = ref_zone.h_conv
    return fire_temp, epsilon_m, h_conv, lambda _t: 0.0, covering_zones, ref_zone


def _bc_falloff_ball(
    midpoint: np.ndarray,
    active_balls: list[RadiationBall],
) -> RadiationBall | None:
    """
    Return the RadiationBall providing the strongest §3.5.4 falloff flux
    to *midpoint* (beyond r2), or None if no ball exceeds _MIN_FALLOFF_FLUX.
    """
    best_ball: RadiationBall | None = None
    best_flux = _MIN_FALLOFF_FLUX  # must strictly exceed threshold
    for ball in active_balls:
        dist = float(np.linalg.norm(midpoint - ball.center))
        if dist <= ball.r2:
            continue  # already handled by direct coverage path
        base_flux = ball.flux2 * (ball.r2 / dist) ** 2
        if base_flux > best_flux:
            best_flux = base_flux
            best_ball = ball
    return best_ball


def _rad_ball_per_quad_flux(
    ball: RadiationBall,
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
) -> np.ndarray:
    """
    Compute per-quad prescribed flux [W/m²] for a beam element outside r2.

    FAHTS §3.5.4 concentrated source in far-field:
        q_face = flux2 · (r2/r)² · cos(θ_face)

    where θ_face is the angle between the face outward normal and the direction
    from the ball to the element (i.e. the face must point toward the ball to
    receive positive flux).  Faces pointing away receive 0.

    Parameters
    ----------
    ball        : RadiationBall source
    mesh        : BeamSurfaceMesh for this element (beam-local coords)
    elem        : BeamElement (provides direction, local_z, n1 for frame)
    model_nodes : {nid: Node} global node positions

    Returns
    -------
    (n_quads,) array of per-quad flux values [W/m²]
    """
    R      = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    midpoint = elem.midpoint(model_nodes)
    r_vec    = midpoint - ball.center
    r        = float(np.linalg.norm(r_vec))
    if r < 1e-9:
        return np.zeros(mesh.n_quads)

    # Unit vector FROM ball TOWARD element; radiation travels in this direction.
    r_hat     = r_vec / r
    base_flux = ball.flux2 * (ball.r2 / r) ** 2

    q_per_quad = np.zeros(mesh.n_quads)
    for q in range(mesh.n_quads):
        normal_local  = _quad_outward_normal_local(mesh, q)
        normal_global = R @ normal_local
        norm = np.linalg.norm(normal_global)
        if norm > 1e-9:
            normal_global /= norm
        # Face receives radiation when its outward normal points toward the ball,
        # i.e. dot(normal, direction_to_ball) > 0, where direction_to_ball = -r_hat.
        cos_theta = float(np.dot(normal_global, -r_hat))
        if cos_theta > 0.0:
            q_per_quad[q] = base_flux * cos_theta

    return q_per_quad


def _beam_local_to_global(elem) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (origin_global, R) for transforming beam-local → global coords.

    origin_global : global position of beam start node (ignoring eccentricity,
                    which is small relative to fire zone dimensions)
    R             : (3,3) rotation matrix; columns = [local_x, local_y, local_z]
    """
    local_x = np.asarray(elem.direction, dtype=float)
    local_z = np.asarray(elem.local_z, dtype=float)
    local_y = np.cross(local_x, local_z)
    norm_y = np.linalg.norm(local_y)
    local_y = local_y / norm_y if norm_y > 1e-9 else np.array([0.0, 1.0, 0.0])
    R = np.column_stack([local_x, local_y, local_z])
    return R


def _quad_outward_normal_local(mesh: BeamSurfaceMesh, q: int) -> np.ndarray:
    """
    Outward unit normal of quad q in beam-local coordinates.

    BOX/I sections: determined from local_cols (face-plane orientation + sign).
    PIPE sections : radial direction from the beam axis in the y-z plane.
    """
    node_q = mesh.nodes[mesh.quads[q]]          # (4, 3) beam-local
    if mesh.local_coords_2d is not None:
        # PIPE: outward = radial from beam axis
        c = node_q.mean(axis=0)
        radial = np.array([0.0, c[1], c[2]])
        norm = np.linalg.norm(radial)
        return radial / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    lc = mesh.local_cols[q]
    if lc[1] == 1:          # (0,1) → x-y plane, constant-z face (top/bottom)
        z_val = float(node_q[0, 2])
        return np.array([0.0, 0.0, np.sign(z_val)])
    else:                   # (0,2) → x-z plane, constant-y face (left/right)
        y_val = float(node_q[0, 1])
        return np.array([0.0, np.sign(y_val), 0.0])


def _element_view_factors(
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
    covering_zones: list[FireZone],
    n_sub: int = 4,
) -> np.ndarray:
    """
    Compute FAHTS §3.3.4 geometric view factor for every quad in *mesh*.

    Both the steel surface and each fire zone face are treated as sub-patches of
    their respective surfaces. The steel quad is the sub-patch of surface 1; the
    fire zone face sub-patches (from FireZone.face_patches) are surface 2.

    F[q] = Σ_j  cosθ_i · cosθ_j / (π·r²) · A_j      (clamped to [0, 1])

    Parameters
    ----------
    mesh          : BeamSurfaceMesh in beam-local coordinates
    elem          : BeamElement (provides direction, local_z, n1)
    model_nodes   : {nid: Node} dict for global node positions
    covering_zones: active FireZone objects covering this element
    n_sub         : fire zone face subdivision count (default 4)

    Returns
    -------
    (n_quads,) view factor array
    """
    R = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    # Collect sub-patches from all covering fire zones
    all_patches: list[tuple[np.ndarray, np.ndarray, float]] = []
    for zone in covering_zones:
        all_patches.extend(zone.face_patches(n_sub))

    F = np.zeros(mesh.n_quads)
    for q in range(mesh.n_quads):
        centroid_local = mesh.nodes[mesh.quads[q]].mean(axis=0)
        centroid_global = origin + R @ centroid_local

        normal_local = _quad_outward_normal_local(mesh, q)
        normal_global = R @ normal_local
        norm = np.linalg.norm(normal_global)
        if norm > 1e-9:
            normal_global /= norm

        F[q] = geometric_view_factor(centroid_global, normal_global, all_patches)

    return F


def _build_beam_surface_mesh(sec, length: float, config: AnalysisConfig) -> BeamSurfaceMesh:
    """Build the FAHTS surface mesh for a beam element based on section type."""
    if isinstance(sec, BoxSection):
        return BoxSurfaceMesher(
            section=sec, length=length,
            n_top=config.n_top, n_side=config.n_side, n_length=config.n_length,
        ).build()
    if isinstance(sec, ISection):
        return IProfileSurfaceMesher(
            section=sec, length=length,
            n_top=config.n_top_i, n_side=config.n_side_i,
            n_bottom=config.n_bottom_i, n_length=config.n_length_i,
        ).build()
    if isinstance(sec, PipeSection):
        return PipeSurfaceMesher(
            section=sec, length=length,
            c_circ=config.c_circ, n_length=config.n_length_p,
        ).build()
    raise TypeError(f"Unsupported section type: {type(sec).__name__}")


def _compute_M_extra(sec, mesh, elem_length: float) -> np.ndarray | None:
    """
    Heat accumulation element mass for hollow BOX/PIPE cross-section meshes.

    Retained for tests and backward compatibility; not called in the main analysis path.
    """
    inner_ids = mesh.inner_node_indices
    if not inner_ids:
        return None

    if isinstance(sec, BoxSection):
        A_inner = sec.inner_height * sec.inner_width
    elif isinstance(sec, PipeSection):
        A_inner = math.pi * sec.inner_radius ** 2
    else:
        return None

    if A_inner <= 0.0:
        return None

    m_acc   = A_inner * elem_length * 1200.0 / len(inner_ids)
    M_extra = np.zeros(mesh.n_nodes)
    for idx in inner_ids:
        M_extra[idx] = m_acc
    return M_extra


# ── USFOS-style console log helpers ──────────────────────────────────────────

def _log_header(
    log_cb: Callable[[str], None],
    model: FEMModel,
    config: AnalysisConfig,
    n_beams: int,
    n_shells: int,
    total: int,
    active_zones: list,
    active_balls: list,
) -> None:
    n_sources = len(active_zones) + len(active_balls)
    n_steps   = max(1, round(config.t_end / config.dt))
    src_name  = model.source_file.name if model.source_file else "unknown"
    stamp     = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    lines: list[str] = [
        "",
        "                              - F A H T S -                  ",
        "",
        "                          Fire And Heat Transfer             ",
        "",
        "                      Simulations of Frame Structures        ",
        "",
        "",
        f"                    - Analysis initiated at -",
        f"                    Date : {stamp}",
        "",
        "",
        "     ======  A N A L Y S I S   P A R A M E T E R S  ======",
        "",
        f"               Model file                     : {src_name}",
        f"               Number of beam elements        = {n_beams:>6d}",
        f"               Number of shell elements       = {n_shells:>6d}",
        f"               Total elements to solve        = {total:>6d}",
        f"               Number of active fire sources  = {n_sources:>6d}",
        "",
        f"               Total simulation time          : {config.t_end:>10.2f}  s"
        f"  ( {config.t_end / 60.0:.2f} min )",
        f"               Time step  (dt)                : {config.dt:>10.4f}  s",
        f"               Output interval                : {config.output_dt:>10.2f}  s",
        f"               Number of time steps           : {n_steps:>6d}",
        f"               Initial temperature            : {20.0:>10.2f}  deg C",
        "",
        "",
        "         Step      Time        Max         Min ",
        "          no.      (min)     (degr C)    (degr C)",
        "",
    ]
    for line in lines:
        log_cb(line)


def _log_step_row(
    log_cb: Callable[[str], None],
    step_no: int,
    t_min: float,
    T_max: float,
    T_min: float,
) -> None:
    log_cb(f"         {step_no:>4d}       {t_min:>7.4f}     {T_max:>8.1f}    {T_min:>8.1f}")


def _log_element_section_header(log_cb: Callable[[str], None]) -> None:
    lines: list[str] = [
        "",
        "",
        "     ======  E L E M E N T   R E S U L T S  ======",
        "",
        "      Elem no    Section       T_peak",
        "      " + "-" * 38,
    ]
    for line in lines:
        log_cb(line)


def _log_element_row(
    log_cb: Callable[[str], None],
    eid: int,
    sec_type: str,
    t_peak: float,
) -> None:
    log_cb(f"       {eid:>6d}    {sec_type:<12s}  {t_peak:>8.1f} deg C")


def _log_summary(log_cb: Callable[[str], None], merged: TemperatureField) -> None:
    critical   = merged.critical_elements(T_crit=600.0)
    t_peak_all = (
        max(merged.peak_centroid_temperature(e) for e in merged.element_ids)
        if merged.element_ids else 0.0
    )

    lines: list[str] = [
        "",
        "",
        "     ======  R E S U L T S   S U M M A R Y  ======",
        "",
        f"               Elements solved                = {merged.n_elements:>6d}",
        f"               Critical (>= 600 deg C)        = {len(critical):>6d}",
        f"               Peak temperature               : {t_peak_all:>8.1f}  deg C",
        "",
        "     " + "=" * 62,
        "              A N A L Y S I S   C O M P L E T E",
        "     " + "=" * 62,
        "",
    ]
    for line in lines:
        log_cb(line)
