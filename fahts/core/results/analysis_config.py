"""
AnalysisConfig — parameters that drive a single heat-transfer analysis run.

Produced by RunAnalysisDialog (Task 3.7) and consumed by the solver thread
(Task 3.8).  Kept free of Qt dependencies so it can be constructed and tested
without a display.
"""
from __future__ import annotations
from dataclasses import dataclass, field


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

    element_ids: Beam element IDs to analyse.  Empty list = all exposed elements.
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
    element_ids: list[int] = field(default_factory=list)

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
        if self.elem_size is not None and self.elem_size <= 0:
            raise ValueError(f"elem_size must be positive, got {self.elem_size}")

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def n_output_steps(self) -> int:
        """Approximate number of stored output steps (including t=0)."""
        import math
        return 1 + math.ceil(self.t_end / self.output_dt)

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
            f"elements={n_el}"
        )
