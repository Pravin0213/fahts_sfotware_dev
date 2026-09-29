"""Outer-surface exposure of equipment walls: fire loads and ambient exchange.

Every boundary condition here is a callable ``bc(T_s, t) -> (q_net, dq_net/dT_s, q_rad,
q_conv)`` in W/m2 for surface temperature ``T_s`` [K] at time ``t`` [s], so any wall solver
(1-D column or 3-D FEM) can use any exposure.
"""

from fahts.fire.ambient import AmbientAuto, AmbientBC, h_ambient
from fahts.fire.boundary import FireBC, PrescribedFluxBC
from fahts.fire.guideline_fire import GuidelineFire

__all__ = ["AmbientAuto", "AmbientBC", "FireBC", "GuidelineFire", "PrescribedFluxBC", "h_ambient"]
