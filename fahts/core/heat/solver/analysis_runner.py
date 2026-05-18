"""
Task 3.8 (extended) — Analysis orchestration logic (pure Python, no Qt).

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
from typing import Callable

import numpy as np

from fahts.core.heat.bc.view_factor import exposed_element_ids
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


class AnalysisCancelledError(Exception):
    """Raised by run_analysis when the cancel_check callback returns True."""


def run_analysis(
    model: FEMModel,
    fire_zones: list,
    config: AnalysisConfig,
    progress_cb: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TemperatureField:
    """
    Run the transient heat-transfer analysis for all exposed elements.

    Handles BOX, IHPROFIL, PIPE beam sections and QUADSHEL/TRISHELL shell elements.
    Accepts both FireZone and RadiationBall heat sources.

    Parameters
    ----------
    model:
        Loaded FEMModel containing elements, sections, materials, nodes.
    fire_zones:
        Active fire sources — list of FireZone and/or RadiationBall objects.
    config:
        AnalysisConfig produced by RunAnalysisDialog.
    progress_cb:
        Optional callback(current, total, eid) called before each element.
    cancel_check:
        Optional callable that returns True when cancelled.
        Raises AnalysisCancelledError when True.

    Returns
    -------
    TemperatureField
        Merged results for all analysed elements.

    Raises
    ------
    AnalysisCancelledError
        If cancel_check() returns True during the run.
    ValueError
        If no supported elements are exposed to any fire source.
    """
    config.validate()

    active_sources = [s for s in fire_zones if s.active]
    active_zones = [s for s in active_sources if isinstance(s, FireZone)]
    active_balls = [s for s in active_sources if isinstance(s, RadiationBall)]

    # ── Determine exposed beam elements ───────────────────────────────────────
    if config.element_ids:
        beam_eids = [e for e in config.element_ids if e in model.elements]
        shell_eids = [e for e in config.element_ids if e in model.shell_elements]
    else:
        beam_eids = sorted(_exposed_beam_ids(model, active_zones, active_balls))
        shell_eids = sorted(_exposed_shell_ids(model, active_zones, active_balls))

    # ── Sort beam elements by section type ────────────────────────────────────
    supported_beam_eids: list[int] = []
    for eid in beam_eids:
        sec = model.sections.get(model.elements[eid].geom_id)
        if isinstance(sec, (BoxSection, ISection, PipeSection)):
            supported_beam_eids.append(eid)
        else:
            log.warning(
                "Element %d: section type %s not supported — skipped.",
                eid,
                type(sec).__name__ if sec else "None",
            )

    total = len(supported_beam_eids) + len(shell_eids)
    if total == 0:
        raise ValueError(
            "No supported elements are exposed to fire. "
            "Supported cross-sections: BOX, IHPROFIL, PIPE, QUADSHEL/TRISHELL."
        )

    fields: list[TemperatureField] = []
    done = 0

    # ── Solve beam elements ───────────────────────────────────────────────────
    for eid in supported_beam_eids:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelledError(
                f"Analysis cancelled after {done} of {total} elements."
            )
        if progress_cb is not None:
            progress_cb(done, total, eid)

        elem = model.elements[eid]
        sec = model.sections[elem.geom_id]
        mat = model.materials[elem.mat_id]
        midpoint = elem.midpoint(model.nodes)

        fire_temp, epsilon_m, h_conv, q_fn = _bc_for_element(
            midpoint, active_zones, active_balls, config
        )
        if fire_temp is None:
            log.warning("Element %d: no active source covers midpoint — skipped.", eid)
            done += 1
            continue

        mesh = _build_beam_surface_mesh(sec, elem.length, config)
        solver = SurfaceTransientSolver(
            mesh=mesh,
            material=mat,
            fire_temp=fire_temp,
            epsilon_m=epsilon_m,
            h_conv=h_conv,
            T0=20.0,
            q_prescribed_fn=q_fn,
        )
        times, T_history = solver.run(
            t_end=config.t_end, dt=config.dt, output_dt=config.output_dt
        )
        tf = TemperatureField.from_surface_solver_run(
            eid=eid, times=times, T_history=T_history, mesh=mesh
        )
        fields.append(tf)
        log.info(
            "Beam %d (%s) solved: T_peak=%.1f °C",
            eid, type(sec).__name__, tf.peak_centroid_temperature(eid),
        )
        done += 1

    # ── Solve shell elements ──────────────────────────────────────────────────
    for eid in shell_eids:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelledError(
                f"Analysis cancelled after {done} of {total} elements."
            )
        if progress_cb is not None:
            progress_cb(done, total, eid)

        shell = model.shell_elements[eid]
        sec = model.sections.get(shell.geom_id)
        if not isinstance(sec, PlateSection):
            log.warning("Shell %d: section not PlateSection — skipped.", eid)
            done += 1
            continue

        mat = model.materials[shell.mat_id]
        midpoint = np.mean(
            [model.nodes[nid].xyz for nid in shell.nodes], axis=0
        )

        fire_temp, epsilon_m, h_conv, q_fn = _bc_for_element(
            midpoint, active_zones, active_balls, config
        )
        if fire_temp is None:
            log.warning("Shell %d: no active source covers midpoint — skipped.", eid)
            done += 1
            continue

        # QUADSHEL (4 nodes): surface mesh on plate face
        # TRISHELL (3 nodes): fall back to 1-D through-thickness solver
        if len(shell.nodes) == 4:
            corners = np.array([model.nodes[nid].xyz for nid in shell.nodes])
            surf_mesh = PlateSurfaceMesher(
                section=sec,
                corners=corners,
                mesh_12=config.mesh_12,
                mesh_14=config.mesh_14,
            ).build()
            solver = SurfaceTransientSolver(
                mesh=surf_mesh,
                material=mat,
                fire_temp=fire_temp,
                epsilon_m=epsilon_m,
                h_conv=h_conv,
                T0=20.0,
                q_prescribed_fn=q_fn,
            )
            times, T_history = solver.run(
                t_end=config.t_end, dt=config.dt, output_dt=config.output_dt
            )
            tf = TemperatureField.from_surface_solver_run(
                eid=eid, times=times, T_history=T_history, mesh=surf_mesh
            )
        else:
            # TRISHELL fallback: 1-D through-thickness
            shell_mesh = ShellMesher(sec, n_layers=config.n_layers).build()
            solver_1d = Shell1DSolver(
                mesh=shell_mesh,
                material=mat,
                fire_temp=fire_temp,
                epsilon_m=epsilon_m,
                h_conv=h_conv,
                T0=20.0,
                q_prescribed_fn=q_fn,
            )
            times, T_history = solver_1d.run(
                t_end=config.t_end, dt=config.dt, output_dt=config.output_dt
            )
            tf = TemperatureField.from_shell_solver_run(
                eid=eid, times=times, T_history=T_history, mesh=shell_mesh
            )
        fields.append(tf)
        log.info(
            "Shell %d solved: T_peak=%.1f °C", eid, tf.peak_centroid_temperature(eid)
        )
        done += 1

    if not fields:
        raise ValueError("Analysis produced no results — all elements were skipped.")

    if progress_cb is not None:
        progress_cb(total, total, -1)

    merged = TemperatureField.merge(fields)
    log.info(
        "Analysis complete: %d elements solved, %d time steps.",
        merged.n_elements, merged.n_steps,
    )
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exposed_beam_ids(model, active_zones, active_balls) -> set[int]:
    result: set[int] = set()
    if active_zones:
        result |= exposed_element_ids(model.elements, active_zones, model.nodes)
    for ball in active_balls:
        result |= ball.exposed_element_ids(model.elements, model.nodes).keys()
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
    return result


def _bc_for_element(
    midpoint: np.ndarray,
    active_zones: list[FireZone],
    active_balls: list[RadiationBall],
    config: AnalysisConfig,
) -> tuple:
    """
    Determine fire BC parameters for an element at `midpoint`.

    Returns (fire_temp_fn, epsilon_m, h_conv, q_prescribed_fn).
    Returns (None, ...) if no source covers the midpoint.

    RadiationBall takes precedence over FireZone when both cover an element
    (matching USFOS USERFLUX behaviour).
    """
    # Check RadiationBalls first (USERFLUX takes precedence)
    covering_balls: list[tuple[RadiationBall, float]] = []
    for b in active_balls:
        dist = float(np.linalg.norm(midpoint - b.center))
        flux = b.flux_at(dist)
        if flux > 0.0:
            covering_balls.append((b, flux))

    if covering_balls:
        # Use the ball with the highest flux (conservative)
        _ref_ball, ref_flux = max(covering_balls, key=lambda x: x[1])
        # fire_temp is not used (epsilon_m=0, h_conv=0) but required by solver interface
        fire_temp = lambda t: 20.0              # noqa: E731
        q_fn = lambda t, _f=ref_flux: _f       # noqa: E731
        return fire_temp, 0.0, 0.0, q_fn

    # Check FireZones
    covering_zones = [z for z in active_zones if z.contains_midpoint(midpoint)]
    if not covering_zones:
        return None, 0.0, 0.0, lambda _t: 0.0

    if len(covering_zones) == 1:
        fire_temp = covering_zones[0].temperature
    else:
        def fire_temp(t, _zones=covering_zones):
            return max(z.temperature(t) for z in _zones)

    ref_zone = max(covering_zones, key=lambda z: z.temperature(config.t_end))
    epsilon_m = ref_zone.epsilon_fire * 0.7
    h_conv = ref_zone.h_conv
    return fire_temp, epsilon_m, h_conv, lambda _t: 0.0


def _compute_M_extra(sec, mesh, elem_length: float) -> np.ndarray | None:
    """
    Heat accumulation element mass for hollow BOX/PIPE cross-section meshes.

    Used with the legacy 2-D cross-section solver (TransientSolver).  Retained
    for tests and backward compatibility; not called in the main analysis path
    which now uses the FAHTS surface-shell approach.

    Distributes trapped-air thermal mass ρ·c ≈ 1200 J/(m³·K) over inner
    surface nodes.  Returns None when there is no enclosed cavity.
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

    m_acc = A_inner * elem_length * 1200.0 / len(inner_ids)
    M_extra = np.zeros(mesh.n_nodes)
    for idx in inner_ids:
        M_extra[idx] = m_acc
    return M_extra


def _build_beam_surface_mesh(sec, length: float, config) -> BeamSurfaceMesh:
    """Build the FAHTS surface mesh for a beam element based on section type."""
    if isinstance(sec, BoxSection):
        return BoxSurfaceMesher(
            section=sec,
            length=length,
            n_top=config.n_top,
            n_side=config.n_side,
            n_length=config.n_length,
        ).build()
    if isinstance(sec, ISection):
        return IProfileSurfaceMesher(
            section=sec,
            length=length,
            n_top=config.n_top_i,
            n_side=config.n_side_i,
            n_bottom=config.n_bottom_i,
            n_length=config.n_length_i,
        ).build()
    if isinstance(sec, PipeSection):
        return PipeSurfaceMesher(
            section=sec,
            length=length,
            c_circ=config.c_circ,
            n_length=config.n_length_p,
        ).build()
    raise TypeError(f"Unsupported section type: {type(sec).__name__}")
