"""
AnalysisConfig — parameters that drive a single heat-transfer analysis run.

Produced by RunAnalysisDialog (Task 3.7) and consumed by the solver thread
(Task 3.8).  Kept free of Qt dependencies so it can be constructed and tested
without a display.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fahts.core.model.material import MATERIAL_STANDARD

if TYPE_CHECKING:
    from fahts.core.heat.bc.insulation import InsulationLayer
    from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
    from fahts.core.heat.sources.concentrated_source import ConcentratedSource
    from fahts.core.heat.sources.line_source import LineSource


@dataclass
class AnalysisConfig:
    """
    Parameters for one transient heat-transfer analysis run.

    Attributes:
        t_end:      Total fire duration [s].
        dt:         Solver time step [s].  Must satisfy 0 < dt <= t_end.
        output_dt:  Result storage interval [s].  Must satisfy dt <= output_dt <= t_end.
                    The solver stores a snapshot every output_dt seconds (plus t=0).

    BOX surface mesh (FAHTS axial × hoop approach):
        n_top:     Elements across top / bottom face (width / hoop).  Default 2.
        n_side:    Elements across left / right face (height / hoop).  Default 3.
        n_length:  Elements along beam axis.  Default 4.

    IHPROFIL surface mesh (FAHTS axial × hoop approach):
        n_top_i:    Elements across top flange width.  Default 4.
        n_side_i:   Elements along web height (each web face).  Default 2.
        n_bottom_i: Elements across bottom flange width.  Default 2.
        n_length_i: Elements along beam axis.  Default 2.

    PIPE surface mesh (FAHTS axial × hoop approach):
        c_circ:    Elements around the circumference (hoop).  Default 8.
        n_length_p: Elements along beam axis.  Default 4.

    Shell / plate surface mesh (FAHTS 2-D face approach):
        mesh_12:   Elements along n1 → n2 edge direction.  Default 4.
        mesh_14:   Elements along n1 → n4 edge direction.  Default 2.

    Legacy (retained for backward compatibility, no longer used by the surface solver):
        n_layers:   Number of FE layers through wall thickness.  Default 1.
        elem_size:  Target element size [m].  None = auto.

    3-D solid solver (default path since 2026-09):
        solver_dim:  "3d" (default) — Hex8 solid mesh per member (`SolidTransientSolver`),
                     real wall thickness with ``n_layers_3d`` elements through it;
                     "2d" — legacy axial × hoop surface-shell solver
                     (`SurfaceTransientSolver`), wall thickness as a scalar.
        linear_solver: "cg" (default — Jacobi-preconditioned conjugate gradient,
                     warm-started, rtol 1e-10; falls back to spsolve if it does not
                     converge; 2–4× faster than direct on the project models) or
                     "direct" (sparse LU via spsolve).  Applies to the global system.
        n_layers_3d: Hex8 layers through each wall / plate thickness in 3-D.  Default 2.
        axial_aspect_3d: 3-D only — axial element length is limited to this multiple of
                     the member's cross-section element size (default 2; 0 disables).
                     Long members with few axial elements give very elongated thin-wall
                     hexes, which violate the discrete maximum principle: a sharp lit /
                     shadowed flux edge then produces unphysical hot/cold spots (found
                     on model_t1.fem: −69 °C next to a 350 kW/m² fire ball).
                     (Independent of the legacy ``n_layers`` used by TRISHELL.)
        PIPE hoop count in 3-D: ``c_circ_3d = max(c_circ, 12)`` — a coarser polygon
                     misrepresents the curved wall volume/area of a solid annulus.

    mass_matrix: "lumped" (default) or "consistent".
    analysis_mode: Selects between strict theory-manual conformance and faster
                engineering approximations.  Valid values:

                ``"engineering"`` (default — preserves all existing behaviour):
                    - Lumped mass matrix (faster, already the default).
                    - Centroid-point view-factor form used at the default n_steel_sub=2
                      level; no additional overrides.
                    - Midpoint exposure check only (existing behaviour).

                ``"theory"`` (strict SINTEF FAHTS manual conformance):
                    - Consistent mass matrix overrides ``mass_matrix`` locally inside
                      ``run_analysis``; the ``AnalysisConfig.mass_matrix`` field is NOT
                      mutated.
                    - Full double-area view-factor integration with n_steel_sub=2 (already
                      the default; confirmed and left as-is).
                    - Logs an INFO message when theory mode is active.
                    - Nonlinear Picard iteration enabled (already on by default inside
                      ``GlobalThermalSolver``).

                Both modes use the same Crank-Nicolson θ=1/2 time integration and the
                same physical BC formulas — only the numerical approximation quality
                differs.
    material_standard: Identifies the standard from which thermal material
                properties (k(T), cp(T), ρ) are taken.  Default is the module-level
                constant ``MATERIAL_STANDARD`` from ``fahts.core.model.material``
                ("EN1993-1-2:2005 Annex C").  Override only when using a non-EC3
                material model (e.g. the USFOS thermpar tables).
    element_ids: Beam element IDs to analyse.  Empty list = all exposed elements.
    insulation: Optional InsulationLayer (§3.3.3) applied to all analysed elements.
                When set, the insulation replaces the direct fire–steel convective/
                radiative BC with a series-resistance boundary.  Default None.
    prescribed_node_bcs: List of PrescribedNodeBC (§3.5.2).  Each entry pins
                selected surface-mesh node indices to a temperature (constant or
                time-varying) by Dirichlet elimination.  Default empty list.
    """

    t_end: float
    dt: float
    output_dt: float
    # BOX surface mesh
    n_top:    int = 2
    n_side:   int = 3
    n_length: int = 4
    # IHPROFIL surface mesh
    n_top_i:    int = 4
    n_side_i:   int = 2
    n_bottom_i: int = 2
    n_length_i: int = 2
    # PIPE surface mesh
    c_circ:    int = 8
    n_length_p: int = 4
    # Shell / plate surface mesh
    mesh_12: int = 4
    mesh_14: int = 2
    # Legacy / other sections
    n_layers: int = 1
    elem_size: float | None = None
    # 3-D solid solver selection (see class docstring)
    solver_dim: str = "3d"
    linear_solver: str = "cg"
    n_layers_3d: int = 2
    # 3-D: axial element length ≤ axial_aspect_3d × cross-section element size (0 = off);
    # n_length / n_length_i / n_length_p then act as MINIMUM axial counts.
    axial_aspect_3d: float = 2.0
    # 3-D conduction operator: "monotone" (two-point edge stencil, M-matrix) or
    # "consistent" (Galerkin trilinear) — see SolidTransientSolver
    conduction_3d: str = "monotone"
    # 3-D radiation geometry: shielding of RadiationBall / point / line sources by any
    # member surface, and surface-to-surface radiation exchange between members
    shielding: bool = True
    radiation_exchange: bool = True
    rad_patch_size: float = 0.5          # exchange patch size [m]
    rad_rays_per_patch: int = 256        # Monte Carlo rays per patch (view factors)
    mass_matrix: str = "lumped"
    analysis_mode: str = "engineering"
    material_standard: str = field(default_factory=lambda: MATERIAL_STANDARD)
    # ── Steel thermal material + emissivity (GUI Material dialog) ──────────────
    # property_model selects the temperature-dependent property source:
    #   "en1993" — EN 1993-1-2 Annex C formulas (k/cp from formulas; ρ, ε editable)
    #   "usfos"  — USFOS thermpar × tempdepy tables (k_ref/c_ref × fixed factor curve)
    # epsilon_steel is the grey-body STEEL surface emissivity ε_steel (PDF §3.2.4),
    # which multiplies the whole net radiant flux; the fire/gas emissivity ε_gas is
    # carried per FireZone (1.0 for ISO/HC).
    # Defaults preserve historical production behaviour (EN 1993-1-2, ε_steel=0.7);
    # the GUI Material dialog pre-loads the fahts.fem values (property_model="usfos",
    # density=7850, c_ref=510, k_ref=50, epsilon_steel=0.85).
    property_model: str = "en1993"
    density: float = 7850.0
    c_ref: float = 510.0
    k_ref: float = 50.0
    epsilon_steel: float = 0.7
    # Volumetric heat capacity of enclosed gas/air inside hollow sections [J/m³·K].
    # Used for BOX/PIPE enclosed-air thermal mass.  Default matches standard air.
    enclosed_gas_rho_c: float = 1200.0
    # Temperature-dependent factor tables for USFOS property model.
    # Each entry is (T_celsius, factor); empty list → use module defaults in material.py.
    # cp(T) = c_ref × interp(cp_factor_table, T)
    # k(T)  = k_ref × interp(k_factor_table, T)
    cp_factor_table: list[tuple[float, float]] = field(default_factory=list)
    k_factor_table:  list[tuple[float, float]] = field(default_factory=list)
    # USFOS benchmark mode (opt-in, default OFF).  When True the solver switches
    # steel thermal properties to the USFOS thermpar/tempdepy multiplier tables
    # (SteelMaterial.usfos_mode=True) and sets the RadiationBall re-radiation
    # emissivity to 0.85 (matching ``emiss`` in validation/usfos/reference/fahts.fem).
    # Used ONLY for direct comparison against a USFOS reference run; production
    # analyses use EN 1993-1-2 Annex C (the architectural default).
    usfos_benchmark_mode: bool = False
    element_ids: list[int] = field(default_factory=list)
    # Optional insulation layer applied to all analysed elements (§3.3.3)
    insulation: InsulationLayer | None = field(default=None)
    # Optional prescribed nodal boundary temperatures §3.5.2.
    # Each entry pins a set of surface-mesh node indices to a given temperature
    # at every time step via Dirichlet elimination.  When non-empty, the BCs are
    # forwarded to SurfaceTransientSolver unchanged.
    prescribed_node_bcs: list[PrescribedNodeBC] = field(default_factory=list)
    # Optional concentrated point sources §3.5.4.  Each source radiates energy
    # in all directions; each surface quad receives q_i = E·cos(θ)/(4π·r²).
    # Forwarded to run_analysis() as additional entries in fire_zones.
    concentrated_sources: list[ConcentratedSource] = field(default_factory=list)
    # Optional line sources §3.5.5.  Each source distributes energy along a
    # finite line segment as n discrete concentrated sub-sources (50% at ends).
    # Forwarded to run_analysis() alongside concentrated_sources.
    line_sources: list[LineSource] = field(default_factory=list)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Raise ``ValueError`` if the configuration is physically inconsistent.

        Called by ``RunAnalysisDialog`` before emitting, and can be called
        programmatically when constructing a config without the dialog.
        """
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.dt > self.t_end:
            raise ValueError(
                f"dt ({self.dt} s) must not exceed t_end ({self.t_end} s)"
            )
        if self.output_dt < self.dt:
            raise ValueError(
                f"output_dt ({self.output_dt} s) must be >= dt ({self.dt} s)"
            )
        if self.output_dt > self.t_end:
            raise ValueError(
                f"output_dt ({self.output_dt} s) must be <= t_end ({self.t_end} s)"
            )
        if self.n_top < 1:
            raise ValueError(f"n_top must be >= 1, got {self.n_top}")
        if self.n_side < 1:
            raise ValueError(f"n_side must be >= 1, got {self.n_side}")
        if self.n_length < 1:
            raise ValueError(f"n_length must be >= 1, got {self.n_length}")
        if self.n_top_i < 1:
            raise ValueError(f"n_top_i must be >= 1, got {self.n_top_i}")
        if self.n_side_i < 1:
            raise ValueError(f"n_side_i must be >= 1, got {self.n_side_i}")
        if self.n_bottom_i < 1:
            raise ValueError(f"n_bottom_i must be >= 1, got {self.n_bottom_i}")
        if self.n_length_i < 1:
            raise ValueError(f"n_length_i must be >= 1, got {self.n_length_i}")
        if self.c_circ < 3:
            raise ValueError(f"c_circ must be >= 3, got {self.c_circ}")
        if self.n_length_p < 1:
            raise ValueError(f"n_length_p must be >= 1, got {self.n_length_p}")
        if self.mesh_12 < 1:
            raise ValueError(f"mesh_12 must be >= 1, got {self.mesh_12}")
        if self.mesh_14 < 1:
            raise ValueError(f"mesh_14 must be >= 1, got {self.mesh_14}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.solver_dim not in {"2d", "3d"}:
            raise ValueError(f"solver_dim must be '2d' or '3d', got {self.solver_dim!r}")
        if self.linear_solver not in {"direct", "cg"}:
            raise ValueError(
                f"linear_solver must be 'direct' or 'cg', got {self.linear_solver!r}"
            )
        if not 1 <= self.n_layers_3d <= 16:
            raise ValueError(f"n_layers_3d must be in [1, 16], got {self.n_layers_3d}")
        if self.rad_patch_size <= 0 or self.rad_rays_per_patch < 1:
            raise ValueError("rad_patch_size must be > 0 and rad_rays_per_patch >= 1")
        if self.conduction_3d not in {"monotone", "consistent"}:
            raise ValueError(
                f"conduction_3d must be 'monotone' or 'consistent', got {self.conduction_3d!r}"
            )
        if self.axial_aspect_3d < 0:
            raise ValueError(f"axial_aspect_3d must be >= 0, got {self.axial_aspect_3d}")
        if self.elem_size is not None and self.elem_size <= 0:
            raise ValueError(f"elem_size must be positive, got {self.elem_size}")
        if self.mass_matrix not in {"lumped", "consistent"}:
            raise ValueError(
                "mass_matrix must be 'lumped' or 'consistent', "
                f"got {self.mass_matrix!r}"
            )
        if self.analysis_mode not in {"engineering", "theory"}:
            raise ValueError(
                "analysis_mode must be 'engineering' or 'theory', "
                f"got {self.analysis_mode!r}"
            )
        if self.property_model not in {"en1993", "usfos"}:
            raise ValueError(
                "property_model must be 'en1993' or 'usfos', "
                f"got {self.property_model!r}"
            )
        if self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")
        if self.c_ref <= 0:
            raise ValueError(f"c_ref must be positive, got {self.c_ref}")
        if self.k_ref <= 0:
            raise ValueError(f"k_ref must be positive, got {self.k_ref}")
        if not 0.0 <= self.epsilon_steel <= 1.0:
            raise ValueError(
                f"epsilon_steel must be in [0, 1], got {self.epsilon_steel}"
            )
        if self.enclosed_gas_rho_c < 0.0:
            raise ValueError(
                f"enclosed_gas_rho_c must be >= 0, got {self.enclosed_gas_rho_c}"
            )
        for label, table in (("cp_factor_table", self.cp_factor_table),
                              ("k_factor_table",  self.k_factor_table)):
            if len(table) > 1:
                Ts = [t for t, _ in table]
                if any(Ts[i] >= Ts[i + 1] for i in range(len(Ts) - 1)):
                    raise ValueError(
                        f"{label}: temperatures must be strictly increasing"
                    )

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def n_output_steps(self) -> int:
        """Approximate number of stored output steps (including t=0)."""
        import math
        return 1 + math.ceil(self.t_end / self.output_dt)

    @property
    def c_circ_3d(self) -> int:
        """PIPE hoop element count used by the 3-D solid mesher (at least 12)."""
        return max(self.c_circ, 12)

    def summary(self) -> str:
        """One-line human-readable description of the configuration."""
        dur_min = self.t_end / 60.0
        n_el = len(self.element_ids) if self.element_ids else "all exposed"
        return (
            f"t_end={dur_min:.0f} min  dt={self.dt:.0f} s  "
            f"out_dt={self.output_dt:.0f} s  "
            f"BOX(top={self.n_top},side={self.n_side},len={self.n_length})  "
            f"I(top={self.n_top_i},side={self.n_side_i},bot={self.n_bottom_i},"
            f"len={self.n_length_i})  "
            f"PIPE(circ={self.c_circ},len={self.n_length_p})  "
            f"Shell(12={self.mesh_12},14={self.mesh_14})  "
            f"mass={self.mass_matrix}  "
            f"solver={self.solver_dim}"
            + (f"(layers={self.n_layers_3d},{self.linear_solver})  "
               if self.solver_dim == "3d" else f"({self.linear_solver})  ")
            + f"elements={n_el}"
        )
