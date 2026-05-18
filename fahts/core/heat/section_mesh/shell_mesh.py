"""
1-D through-thickness mesh for shell/plate elements (QUADSHEL, TRISHELL).

x = 0 is the outer (fire-exposed) face; x = thickness is the inner face.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fahts.core.model.section import PlateSection


@dataclass
class ShellMesh1D:
    """
    1-D FEM mesh through a plate/shell thickness.

    Attributes:
        nodes:      (n_nodes,) x-coordinates [m], x=0 is outer face.
        segments:   (n_elems, 2) node-index pairs [(0,1), (1,2), ...].
        outer_node: Index of the node at x=0 (fire-exposed face).
        inner_node: Index of the node at x=thickness (inner/adiabatic face).
        thickness:  Total plate thickness [m].
    """
    nodes: np.ndarray
    segments: np.ndarray
    outer_node: int
    inner_node: int
    thickness: float

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elems(self) -> int:
        return len(self.segments)


class ShellMesher:
    """
    1-D mesh generator for shell elements.

    Args:
        section:  PlateSection (thickness).
        n_layers: Number of 1-D elements through the thickness (default 1).
    """

    def __init__(self, section: PlateSection, n_layers: int = 1) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self._sec = section
        self._n = n_layers

    def build(self) -> ShellMesh1D:
        t = self._sec.thickness
        n = self._n
        xs = np.linspace(0.0, t, n + 1)
        segs = np.array([[i, i + 1] for i in range(n)], dtype=np.intp)
        return ShellMesh1D(
            nodes=xs,
            segments=segs,
            outer_node=0,
            inner_node=n,
            thickness=t,
        )
