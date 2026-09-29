"""
BOX cross-section 2-D FEM mesh generator (Quad4 elements).

Mesh layout (H=0.20, W=0.20, T=0.008):

  ╔═══╤═══════════════════╤═══╗  z = +H/2   (top outer)
  ║   │     top plate     │   ║
  ╠═══╪═══════════════════╪═══╣  z = +H/2 - T_top  (top inner)
  ║ L │                   │ R ║
  ║ e │    hollow (not    │ i ║
  ║ f │      meshed)      │ g ║
  ║ t │                   │ h ║
  ╠═══╪═══════════════════╪═══╣  z = -H/2 + T_bot  (bottom inner)
  ║   │    bottom plate   │   ║
  ╚═══╧═══════════════════╧═══╝  z = -H/2   (bottom outer)

  y = -W/2  -W/2+T_s             W/2-T_s  W/2

Strategy
--------
Bottom and top plates span the full width (including corner blocks).
Left and right webs span only the inner height [z_bi, z_ti].
Corner blocks are thus exclusively owned by the plates, avoiding overlap.
Nodes at plate/web junctions are shared via coordinate-based deduplication.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from fahts.core.model.section import BoxSection
from .section_mesh import SectionMesh

_DEDUP_TOL = 1e-10   # snap tolerance for node deduplication [m]


class BoxMesher:
    """
    Structured Quad4 mesh generator for a hollow BOX cross-section.

    Args:
        section:   The BoxSection to mesh.
        elem_size: Target element size along the wall [m].
                   Defaults to the thinnest wall thickness.
        n_layers:  Number of Quad4 element layers through the wall thickness (≥ 1).
    """

    def __init__(
        self,
        section: BoxSection,
        elem_size: Optional[float] = None,
        n_layers: int = 1,
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if section.inner_width <= 0 or section.inner_height <= 0:
            raise ValueError(
                f"BOX section {section.sid} has zero or negative hollow interior; "
                "cannot mesh."
            )
        self._sec = section
        self._n = n_layers
        self._es = (
            elem_size
            if elem_size is not None
            else min(section.T_side, section.T_bot, section.T_top)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> SectionMesh:
        """Build and return the complete SectionMesh."""
        s = self._sec
        n, es = self._n, self._es

        # Key boundary coordinates
        y0, y1 = -s.W / 2, s.W / 2    # outer left / right
        z0, z1 = -s.H / 2, s.H / 2    # outer bottom / top
        yl = y0 + s.T_side             # inner left surface
        yr = y1 - s.T_side             # inner right surface
        zb = z0 + s.T_bot              # inner bottom surface
        zt = z1 - s.T_top              # inner top surface

        # Node counts along each direction
        nl = n + 1                                         # through side wall
        ny_in = max(2, _n_divs(yr - yl, es) + 1)          # along inner width
        nz_in = max(2, _n_divs(zt - zb, es) + 1)          # along inner height
        nz_b = n + 1                                       # through bottom plate
        nz_t = n + 1                                       # through top plate

        # y-grid for full-width plates: left strip + inner + right strip
        y_L = np.linspace(y0, yl, nl)
        y_M = np.linspace(yl, yr, ny_in)
        y_R = np.linspace(yr, y1, nl)
        y_plate = _unique_1d([y_L, y_M, y_R])

        # z-grids
        z_bot = np.linspace(z0, zb, nz_b)
        z_web = np.linspace(zb, zt, nz_in)
        z_top = np.linspace(zt, z1, nz_t)

        # Build global node pool with coordinate deduplication
        pool: list[np.ndarray] = []
        idx: dict[tuple[float, float], int] = {}

        def gid(y: float, z: float) -> int:
            key = (_snap(y), _snap(z))
            if key not in idx:
                idx[key] = len(pool)
                pool.append(np.array([y, z], dtype=float))
            return idx[key]

        quads: list[tuple[int, int, int, int]] = []

        def mesh_patch(ys: np.ndarray, zs: np.ndarray) -> None:
            """Add all Quad4 elements for a rectangular y×z patch."""
            for j in range(len(zs) - 1):
                for i in range(len(ys) - 1):
                    quads.append((
                        gid(ys[i],     zs[j]),
                        gid(ys[i + 1], zs[j]),
                        gid(ys[i + 1], zs[j + 1]),
                        gid(ys[i],     zs[j + 1]),
                    ))

        mesh_patch(y_plate, z_bot)   # bottom plate (full width)
        mesh_patch(y_plate, z_top)   # top plate (full width)
        mesh_patch(y_L, z_web)       # left web (inner height only)
        mesh_patch(y_R, z_web)       # right web (inner height only)

        nodes_arr = np.array(pool, dtype=float)
        quads_arr = np.array(quads, dtype=np.intp)

        outer, inner = _classify_boundary_edges(
            quads_arr, nodes_arr, y0, y1, z0, z1, yl, yr, zb, zt
        )

        return SectionMesh(
            nodes=nodes_arr,
            quads=quads_arr,
            outer_edge_pairs=(
                np.array(outer, dtype=np.intp) if outer
                else np.empty((0, 2), dtype=np.intp)
            ),
            inner_edge_pairs=(
                np.array(inner, dtype=np.intp) if inner
                else np.empty((0, 2), dtype=np.intp)
            ),
        )


# ── Module helpers ────────────────────────────────────────────────────────────

def _snap(v: float) -> float:
    """Round to nearest _DEDUP_TOL for dict-key deduplication."""
    return round(v / _DEDUP_TOL) * _DEDUP_TOL


def _n_divs(length: float, elem_size: float) -> int:
    """Number of element divisions for a given length and target element size."""
    return max(1, round(length / elem_size))


def _unique_1d(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate 1-D arrays preserving order, removing floating-point duplicates."""
    seen: set[float] = set()
    out: list[float] = []
    for arr in arrays:
        for v in arr.tolist():
            k = round(v, 12)
            if k not in seen:
                seen.add(k)
                out.append(v)
    return np.array(out, dtype=float)


def _classify_boundary_edges(
    quads: np.ndarray,
    nodes: np.ndarray,
    y0: float, y1: float,
    z0: float, z1: float,
    yl: float, yr: float,
    zb: float, zt: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Identify outer (Robin BC) and inner (adiabatic) boundary edges.

    An edge is a boundary edge if it appears in exactly one quad.
    Outer edges lie on the outer rectangle; inner edges on the hollow perimeter.
    """
    edge_count: dict[tuple[int, int], int] = {}
    edge_dir: dict[tuple[int, int], tuple[int, int]] = {}

    for quad in quads:
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_dir:
                edge_dir[key] = (a, b)

    outer: list[tuple[int, int]] = []
    inner: list[tuple[int, int]] = []
    tol = max(y1 - y0, z1 - z0) * 1e-9

    for key, cnt in edge_count.items():
        if cnt != 1:
            continue
        a, b = edge_dir[key]
        ya, za = nodes[a, 0], nodes[a, 1]
        yb, zb_n = nodes[b, 0], nodes[b, 1]

        if _on_outer(ya, za, yb, zb_n, y0, y1, z0, z1, tol):
            outer.append((a, b))
        elif _on_inner(ya, za, yb, zb_n, yl, yr, zb, zt, tol):
            inner.append((a, b))


    return outer, inner


def _on_outer(
    ya: float, za: float, yb: float, zb_n: float,
    y0: float, y1: float, z0: float, z1: float, tol: float,
) -> bool:
    return (
        (abs(za - z0) < tol and abs(zb_n - z0) < tol)   # bottom outer face
        or (abs(za - z1) < tol and abs(zb_n - z1) < tol)  # top outer face
        or (abs(ya - y0) < tol and abs(yb - y0) < tol)    # left outer face
        or (abs(ya - y1) < tol and abs(yb - y1) < tol)    # right outer face
    )


def _on_inner(
    ya: float, za: float, yb: float, zb_n: float,
    yl: float, yr: float, zb: float, zt: float, tol: float,
) -> bool:
    # Bottom inner face: z = zb, y in [yl, yr]
    if (abs(za - zb) < tol and abs(zb_n - zb) < tol
            and ya > yl - tol and yb > yl - tol
            and ya < yr + tol and yb < yr + tol):
        return True
    # Top inner face: z = zt, y in [yl, yr]
    if (abs(za - zt) < tol and abs(zb_n - zt) < tol
            and ya > yl - tol and yb > yl - tol
            and ya < yr + tol and yb < yr + tol):
        return True
    # Left inner face: y = yl, z in [zb, zt]
    if (abs(ya - yl) < tol and abs(yb - yl) < tol
            and za > zb - tol and zb_n > zb - tol
            and za < zt + tol and zb_n < zt + tol):
        return True
    # Right inner face: y = yr, z in [zb, zt]
    if (abs(ya - yr) < tol and abs(yb - yr) < tol
            and za > zb - tol and zb_n > zb - tol
            and za < zt + tol and zb_n < zt + tol):
        return True
    return False
