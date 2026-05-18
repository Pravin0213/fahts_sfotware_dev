"""
Phase 2.5 — Exposure logic: which beam faces are inside which fire zones.

Phase 2 uses a binary in/out check on the beam midpoint.
Phase 5 will replace this with analytical view-factor geometry and ray casting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fahts.core.heat.bc.net_flux import net_heat_flux
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.model.element import BeamElement


@dataclass
class ElementHeatBC:
    """Per-element heat-flux boundary conditions at a single time step."""

    eid: int
    face_fluxes: dict[str, float]  # {'top': q, 'bot': q, 'left': q, 'right': q} [W/m²]
    t: float                        # time [s]

    @property
    def total_flux(self) -> float:
        """Sum of all face fluxes [W/m²]."""
        return sum(self.face_fluxes.values())

    @property
    def is_exposed(self) -> bool:
        return any(q > 0.0 for q in self.face_fluxes.values())


# Face labels used throughout (BOX convention: top/bot in local-z, left/right in local-y)
FACE_NAMES: tuple[str, ...] = ("top", "bot", "left", "right")


def exposure_flags(
    beam: BeamElement,
    fire_zones: list[FireZone],
    nodes: dict,
    tolerance: float = 0.0,
) -> dict[str, float]:
    """
    Return per-face exposure fractions for one beam element.

    Phase 2: binary — 1.0 for all faces if the midpoint is inside any active
    zone, 0.0 otherwise.  Phase 5 will compute partial fractions via view factors.

    Parameters
    ----------
    beam       : the beam element to test
    fire_zones : list of FireZone objects to test against
    nodes      : FEMModel.nodes dict  {nid: Node}
    tolerance  : extra margin [m] added to each zone's bounding box

    Returns
    -------
    dict[str, float]
        Keys: 'top', 'bot', 'left', 'right'.  Values in [0, 1].
    """
    midpoint = beam.midpoint(nodes)
    for zone in fire_zones:
        if zone.active and zone.contains_midpoint(midpoint, tolerance=tolerance):
            return {face: 1.0 for face in FACE_NAMES}
    return {face: 0.0 for face in FACE_NAMES}


def exposed_element_ids(
    elements: dict[int, BeamElement],
    fire_zones: list[FireZone],
    nodes: dict,
    tolerance: float = 0.0,
) -> set[int]:
    """
    Return the set of element IDs whose midpoints fall inside any active fire zone.

    Parameters
    ----------
    elements   : FEMModel.elements dict
    fire_zones : list of FireZone objects
    nodes      : FEMModel.nodes dict
    tolerance  : extra margin [m] added to each zone bounding box
    """
    active_zones = [z for z in fire_zones if z.active]
    if not active_zones:
        return set()

    result: set[int] = set()
    for beam in elements.values():
        mid = beam.midpoint(nodes)
        for zone in active_zones:
            if zone.contains_midpoint(mid, tolerance=tolerance):
                result.add(beam.eid)
                break
    return result


def compute_element_bc(
    beam: BeamElement,
    fire_zones: list[FireZone],
    nodes: dict,
    t: float,
    T_steel: float = 20.0,
    epsilon_steel: float = 0.7,
    tolerance: float = 0.0,
) -> ElementHeatBC | None:
    """
    Compute net heat-flux BCs for one beam element at time *t*.

    Returns None if the beam is not exposed to any fire zone.

    Parameters
    ----------
    beam          : beam element
    fire_zones    : list of FireZone objects
    nodes         : FEMModel.nodes dict
    t             : current time [s]
    T_steel       : current steel surface temperature [°C] (uniform in Phase 2)
    epsilon_steel : steel surface emissivity (EN 1993-1-2 §3.1: 0.7 default)
    tolerance     : zone bounding-box margin [m]
    """
    flags = exposure_flags(beam, fire_zones, nodes, tolerance=tolerance)
    if not any(v > 0.0 for v in flags.values()):
        return None

    midpoint = beam.midpoint(nodes)
    covering_zones = [
        z for z in fire_zones
        if z.active and z.contains_midpoint(midpoint, tolerance=tolerance)
    ]

    # Use the hottest covering zone at time t
    hottest = max(covering_zones, key=lambda z: z.temperature(t))
    T_fire = hottest.temperature(t)
    epsilon_m = hottest.epsilon_fire * epsilon_steel
    h_conv = hottest.h_conv

    face_fluxes: dict[str, float] = {}
    for face in FACE_NAMES:
        fraction = flags[face]
        if fraction > 0.0:
            q = net_heat_flux(T_fire, T_steel, epsilon_m, h_conv)
            face_fluxes[face] = q * fraction
        else:
            face_fluxes[face] = 0.0

    return ElementHeatBC(eid=beam.eid, face_fluxes=face_fluxes, t=t)


def compute_all_bcs(
    elements: dict[int, BeamElement],
    fire_zones: list[FireZone],
    nodes: dict,
    t: float,
    T_steel: float = 20.0,
    epsilon_steel: float = 0.7,
    tolerance: float = 0.0,
) -> list[ElementHeatBC]:
    """
    Compute heat-flux BCs for all exposed elements at time *t*.

    Returns a list containing only the elements that are actually exposed
    (i.e. have at least one non-zero face flux).
    """
    result: list[ElementHeatBC] = []
    for beam in elements.values():
        bc = compute_element_bc(
            beam, fire_zones, nodes, t,
            T_steel=T_steel,
            epsilon_steel=epsilon_steel,
            tolerance=tolerance,
        )
        if bc is not None:
            result.append(bc)
    return result
