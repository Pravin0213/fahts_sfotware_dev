"""
Prescribed (forced) nodal boundary temperature — FAHTS §3.5.2.

The user specifies the actual temperature to be applied to certain thermal DOFs
(surface mesh node indices).  These temperatures are held fixed at every time
step by Dirichlet elimination in the FEM system:

    K[i, :] = 0,  K[i, i] = 1,  rhs[i] = T_prescribed(t)

This is the standard row-zeroing / diagonal-pinning approach for Dirichlet BCs
in FEM.  It is numerically exact (no penalty parameter), backward-compatible with
the CN state update, and does not require changes to the capacitance matrix.

The FAHTS theory manual (§3.5.2) describes the BC in terms of structural node(s):
all heat-transfer-model nodes belonging to elements connected to the structural
node receive the prescribed temperature for the entire simulation.  In this
implementation the caller maps structural nodes to surface-mesh node indices
before construction, keeping this class free of structural model knowledge.

Usage example
-------------
>>> from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
>>> bc = PrescribedNodeBC(node_indices=[0, 4], temperature=500.0)
>>> bc_time = PrescribedNodeBC(node_indices=[1, 5], temperature=lambda t: 20.0 + 10.0*t)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PrescribedNodeBC:
    """
    Prescribed Dirichlet temperature constraint for a subset of surface mesh nodes.

    Implements §3.5.2 "Nodal Boundary Temperature": the listed node DOFs are
    pinned to the given temperature value at every time step via row-zeroing /
    diagonal-pinning elimination in the assembled FEM system.

    Attributes
    ----------
    node_indices:
        Indices into the surface mesh node list (0-based, same numbering as
        ``BeamSurfaceMesh.nodes``).  Must be non-empty; duplicates are silently
        ignored by the solver.
    temperature:
        Prescribed temperature [°C] as either:
        - a ``float`` — constant value held for the entire simulation, or
        - a ``Callable[[float], float]`` — ``T(t_seconds) → °C`` for time-varying
          prescriptions.  The callable is evaluated once per time step at the
          step-end time ``t``.
    """

    node_indices: list[int]
    temperature: float | Callable[[float], float]

    def __post_init__(self) -> None:
        if not self.node_indices:
            raise ValueError("PrescribedNodeBC.node_indices must not be empty.")
        if not callable(self.temperature) and not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError(
                "PrescribedNodeBC.temperature must be a float or a callable "
                f"(t → float), got {type(self.temperature)!r}."
            )

    # ── Helper ────────────────────────────────────────────────────────────────

    def eval(self, t: float) -> float:
        """
        Evaluate the prescribed temperature at time *t* [s].

        Returns:
            Temperature [°C].
        """
        if callable(self.temperature):
            return float(self.temperature(t))
        return float(self.temperature)
