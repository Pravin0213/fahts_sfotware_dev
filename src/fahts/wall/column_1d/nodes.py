"""Radial node layouts through the wall."""

from __future__ import annotations

import numpy as np


def radial_nodes(t: float, n_cells: int = 10) -> np.ndarray:
    """Node positions [m] through a wall of thickness ``t``, measured from the inner surface.

    Cell-centred layout with surface nodes: the wall is split into ``n_cells`` equal cells,
    with one node at each cell centre plus one on each surface (``n_cells + 2`` nodes):
    0, t/(2n), 3t/(2n), ..., t - t/(2n), t. With the vertex-centred control volumes of
    ``WallColumn`` the two surface nodes get thin (quarter-cell) volumes, which resolves the
    steep surface gradients under fire.
    """
    if t <= 0.0:
        raise ValueError(f"wall thickness must be positive, got {t}")
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")
    centres = (np.arange(n_cells) + 0.5) / n_cells
    return t * np.concatenate(([0.0], centres, [1.0]))
