"""
TemperatureField — results container for transient heat transfer analysis (Task 3.6).

Stores cross-section nodal temperatures and area-weighted centroid temperatures for
all analysed beam elements over the full simulation time history.

Data contract (stable, referenced by renderer and post-processor):
    times:       (n_steps,)             output times [s]
    element_ids: list[int]              element IDs in column order
    T_centroid:  (n_steps, n_elems)     area-weighted centroid temperature [°C]
    T_section:   {eid: (n_steps, n_nodes)}  full nodal temperatures [°C]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from fahts.core.heat.section_mesh.section_mesh import SectionMesh


@dataclass
class TemperatureField:
    """
    Time-history of temperatures for one or more beam elements.

    Attributes:
        times:       (n_steps,) output times [s], starting at 0.
        element_ids: ordered list of element IDs (length n_elems).
        T_centroid:  (n_steps, n_elems) area-weighted centroid temperature [°C].
        T_section:   {eid: (n_steps, n_section_nodes)} full cross-section nodal
                     temperatures [°C]; only populated for elements where the
                     2-D section solver was run.
    """
    times: np.ndarray
    element_ids: list[int]
    T_centroid: np.ndarray
    T_section: dict[int, np.ndarray] = field(default_factory=dict)

    # Reverse lookup built in __post_init__; not part of the public contract
    _eid_to_idx: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._eid_to_idx = {eid: i for i, eid in enumerate(self.element_ids)}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_steps(self) -> int:
        """Number of output time steps."""
        return len(self.times)

    @property
    def n_elements(self) -> int:
        """Number of beam elements in this result set."""
        return len(self.element_ids)

    # ── Per-element queries ───────────────────────────────────────────────────

    def centroid_temperature(self, eid: int, t_idx: int = -1) -> float:
        """
        Area-weighted centroid temperature for element `eid` at time index `t_idx`.

        Args:
            eid:   Element ID.
            t_idx: Time step index (default -1 = last step).

        Returns:
            Temperature [°C].
        """
        return float(self.T_centroid[t_idx, self._eid_index(eid)])

    def section_temperatures(self, eid: int, t_idx: int = -1) -> np.ndarray:
        """
        Full cross-section nodal temperatures for element `eid` at `t_idx`.

        Args:
            eid:   Element ID (must be in T_section).
            t_idx: Time step index (default -1 = last step).

        Returns:
            (n_section_nodes,) nodal temperature array [°C].
        """
        if eid not in self.T_section:
            raise KeyError(
                f"Element {eid} has no section temperature data. "
                "Only elements solved with the 2-D section solver are stored."
            )
        return self.T_section[eid][t_idx]

    def peak_centroid_temperature(self, eid: int) -> float:
        """Maximum centroid temperature over the full simulation [°C]."""
        return float(self.T_centroid[:, self._eid_index(eid)].max())

    def time_to_critical(
        self,
        eid: int,
        T_crit: float = 600.0,
    ) -> float | None:
        """
        Time [s] when the centroid temperature first reaches or exceeds `T_crit` [°C].

        Uses linear interpolation between output steps.
        Returns None if `T_crit` is never reached within the simulation.
        """
        i = self._eid_index(eid)
        T = self.T_centroid[:, i]
        crossings = np.where(T >= T_crit)[0]
        if len(crossings) == 0:
            return None
        idx = int(crossings[0])
        if idx == 0:
            return float(self.times[0])
        t0, t1 = float(self.times[idx - 1]), float(self.times[idx])
        T0, T1 = float(T[idx - 1]), float(T[idx])
        if T1 <= T0:
            return t0
        return t0 + (T_crit - T0) / (T1 - T0) * (t1 - t0)

    # ── Bulk queries ──────────────────────────────────────────────────────────

    @property
    def peak_temperatures(self) -> dict[int, float]:
        """Peak centroid temperature for every element: {eid: T_peak [°C]}."""
        return {eid: self.peak_centroid_temperature(eid) for eid in self.element_ids}

    def critical_elements(self, T_crit: float = 600.0) -> list[int]:
        """
        Element IDs whose peak centroid temperature reached or exceeded `T_crit` [°C].
        Ordered as they appear in `element_ids`.
        """
        return [
            eid for eid in self.element_ids
            if self.peak_centroid_temperature(eid) >= T_crit
        ]

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_solver_run(
        cls,
        eid: int,
        times: np.ndarray,
        T_history: np.ndarray,
        mesh: SectionMesh,
    ) -> TemperatureField:
        """
        Create a single-element TemperatureField from TransientSolver.run() output.

        The centroid temperature is computed as the area-weighted mean of all nodal
        temperatures (consistent with the lumped mass formulation).

        Args:
            eid:       Beam element ID.
            times:     (n_steps,) output times [s] from solver.run().
            T_history: (n_steps, n_nodes) nodal temperatures [°C] from solver.run().
            mesh:      SectionMesh used for the analysis (needed for area weights).

        Returns:
            TemperatureField with a single element.
        """
        from fahts.core.heat.solver.fem_2d_section import assemble_C_lumped

        # Nodal tributary areas (rho=1, cp=1 → C_lump = area per node)
        node_areas = assemble_C_lumped(mesh, rho=1.0, cp=1.0)
        total_area = node_areas.sum()

        T_hist = np.asarray(T_history, dtype=float)
        T_cen = (T_hist @ node_areas) / total_area          # (n_steps,)

        return cls(
            times=np.asarray(times, dtype=float),
            element_ids=[eid],
            T_centroid=T_cen[:, np.newaxis],                # (n_steps, 1)
            T_section={eid: T_hist},
        )

    @classmethod
    def from_surface_solver_run(
        cls,
        eid: int,
        times: np.ndarray,
        T_history: np.ndarray,
        mesh: object,
    ) -> "TemperatureField":
        """
        Create a single-element TemperatureField from SurfaceTransientSolver output.

        The centroid temperature is the area-weighted mean of all surface-node
        temperatures (using BeamSurfaceMesh.node_area_weights).

        Args:
            eid:       Beam element ID.
            times:     (n_steps,) output times [s].
            T_history: (n_steps, n_nodes) surface nodal temperatures [°C].
            mesh:      BeamSurfaceMesh used for the analysis.

        Returns:
            TemperatureField with a single element.
        """
        weights = mesh.node_area_weights          # (n_nodes,) normalised

        T_hist = np.asarray(T_history, dtype=float)
        T_cen  = T_hist @ weights                 # (n_steps,)

        return cls(
            times=np.asarray(times, dtype=float),
            element_ids=[eid],
            T_centroid=T_cen[:, np.newaxis],      # (n_steps, 1)
            T_section={eid: T_hist},
        )

    @classmethod
    def from_shell_solver_run(
        cls,
        eid: int,
        times: np.ndarray,
        T_history: np.ndarray,
        mesh: object,
    ) -> TemperatureField:
        """
        Create a single-element TemperatureField from Shell1DSolver.run() output.

        The centroid temperature is the thickness-weighted mean nodal temperature.

        Args:
            eid:       Shell element ID.
            times:     (n_steps,) output times [s].
            T_history: (n_steps, n_nodes) nodal temperatures [°C].
            mesh:      ShellMesh1D used for the analysis.

        Returns:
            TemperatureField with a single element.
        """
        import numpy as _np
        nodes = _np.asarray(mesh.nodes, dtype=float)
        n = len(nodes)
        node_len = _np.zeros(n)
        for a, b in mesh.segments:
            a, b = int(a), int(b)
            L = abs(nodes[b] - nodes[a])
            node_len[a] += L * 0.5
            node_len[b] += L * 0.5
        total = node_len.sum() if node_len.sum() > 0 else 1.0

        T_hist = _np.asarray(T_history, dtype=float)
        T_cen = (T_hist @ node_len) / total         # (n_steps,)

        return cls(
            times=_np.asarray(times, dtype=float),
            element_ids=[eid],
            T_centroid=T_cen[:, _np.newaxis],        # (n_steps, 1)
            T_section={eid: T_hist},
        )

    @classmethod
    def merge(cls, fields: list[TemperatureField]) -> TemperatureField:
        """
        Combine single- or multi-element TemperatureFields into one container.

        All fields must share the same `times` array (same simulation run).

        Args:
            fields: Non-empty list of TemperatureField objects.

        Returns:
            Merged TemperatureField with all elements.
        """
        if not fields:
            raise ValueError("Cannot merge an empty list of TemperatureFields.")
        times = fields[0].times
        element_ids: list[int] = []
        T_cols: list[np.ndarray] = []
        T_section: dict[int, np.ndarray] = {}

        for f in fields:
            element_ids.extend(f.element_ids)
            T_cols.append(f.T_centroid)
            T_section.update(f.T_section)

        return cls(
            times=times,
            element_ids=element_ids,
            T_centroid=np.hstack(T_cols),
            T_section=T_section,
        )

    # ── Gradient computation ──────────────────────────────────────────────────

    def section_gradient(
        self,
        eid: int,
        t_idx: int,
        mesh: SectionMesh,
    ) -> tuple[float, float]:
        """
        Linearised temperature gradients (βy, βz) at time step t_idx.

        Uses area-weighted first moments of Quad4 element centroid temperatures
        over the cross-section (SINTEF FAHTS §3.4.2):

            βz = Σ(T_k · y_k · A_k) / Iz,  Iz = Σ(y_k² · A_k)
            βy = Σ(T_k · z_k · A_k) / Iy,  Iy = Σ(z_k² · A_k)

        Returns (βy [°C/m], βz [°C/m]).  Returns (0.0, 0.0) if the element
        has no section data (e.g. shell elements solved with Shell1DSolver).
        """
        if eid not in self.T_section:
            return 0.0, 0.0

        T_nodes = self.T_section[eid][t_idx]          # (n_nodes,)
        nodes = mesh.nodes                             # (n_nodes, 2): columns [y, z]

        sum_Ty_A = 0.0
        sum_Tz_A = 0.0
        Iz = 0.0   # Σ(y_k² · A_k)
        Iy = 0.0   # Σ(z_k² · A_k)

        for quad in mesh.quads:
            y = nodes[quad, 0]
            z = nodes[quad, 1]
            # Element area — same cross-product formula as SectionMesh.steel_area
            dy1 = y[2] - y[0]; dz1 = z[2] - z[0]
            dy2 = y[3] - y[1]; dz2 = z[3] - z[1]
            A_k = 0.5 * abs(dy1 * dz2 - dy2 * dz1)

            y_k = float(y.mean())
            z_k = float(z.mean())
            T_k = float(T_nodes[quad].mean())

            sum_Ty_A += T_k * y_k * A_k
            sum_Tz_A += T_k * z_k * A_k
            Iz += y_k * y_k * A_k
            Iy += z_k * z_k * A_k

        beta_z = sum_Ty_A / Iz if Iz > 1e-20 else 0.0
        beta_y = sum_Tz_A / Iy if Iy > 1e-20 else 0.0
        return beta_y, beta_z

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eid_index(self, eid: int) -> int:
        try:
            return self._eid_to_idx[eid]
        except KeyError:
            raise KeyError(
                f"Element ID {eid} not found. Available IDs: {self.element_ids}"
            )
