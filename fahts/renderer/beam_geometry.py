"""
Phase 1.4 / extended — beam_geometry.py
Extrude cross-sections along beam axes; build flat shell surfaces.

Supported cross-section types
──────────────────────────────
  BoxSection  (USFOS BOX)      — hollow rectangular tube
  PipeSection (USFOS PIPE)     — circular hollow tube (N-gon approximation)
  ISection    (USFOS IHPROFIL) — I-beam / H-section

Shell elements
──────────────
  QUADSHEL / TRISHELL — flat polygon faces using node positions directly

Local frame convention (same as USFOS):
    local_x  = unit vector n1 → n2  (beam axis)
    local_z  = UNITVEC from .fem file (cross-section local-z)
    local_y  = cross(local_x, local_z), normalised

BOX corner layout in the cross-section plane (local y–z):
    0: +H/2, -W/2   (top-left)
    1: +H/2, +W/2   (top-right)
    2: -H/2, +W/2   (bottom-right)
    3: -H/2, -W/2   (bottom-left)
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pyvista as pv

from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, PipeSection, ISection

# ── Constants ─────────────────────────────────────────────────────────────────

_FALLBACK_Z  = np.array([0.0, 0.0, 1.0])  # global Z used when UNITVEC unavailable
_PIPE_SIDES  = 16                           # polygon sides for circular tube approximation

RenderMode = Literal["section", "centreline"]


# ── Local frame ───────────────────────────────────────────────────────────────

def _local_frame(
    direction: np.ndarray,
    local_z: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (local_x, local_y, local_z) orthonormal triad.

    *local_z* is the UNITVEC from the .fem file.  If it is missing or nearly
    parallel to *direction* (sin θ < 1e-6), a fallback perpendicular is used.
    """
    lx = direction / np.linalg.norm(direction)

    candidate_z = local_z if local_z is not None else _FALLBACK_Z
    candidate_z = candidate_z / np.linalg.norm(candidate_z)

    ly_raw = np.cross(lx, candidate_z)
    norm_y = np.linalg.norm(ly_raw)

    if norm_y < 1e-6:
        perp = _FALLBACK_Z if abs(lx[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ly_raw = np.cross(lx, perp)
        norm_y = np.linalg.norm(ly_raw)

    ly = ly_raw / norm_y
    lz = np.cross(ly, lx)
    lz /= np.linalg.norm(lz)

    return lx, ly, lz


# ── Cross-section offset generators ───────────────────────────────────────────

def _outer_corners_2d(section: BoxSection) -> np.ndarray:
    """Return (4, 2) outer corner offsets for a BOX section.

    Col 0 → local-y offset (width direction, ⊥ to UNITVEC).
    Col 1 → local-z offset (height direction, ‖ to UNITVEC).

    USFOS convention: H is the section height measured along local-z (UNITVEC),
    W is the width measured along local-y.
    """
    w = section.W / 2.0  # half-width  → local-y
    h = section.H / 2.0  # half-height → local-z (UNITVEC direction)
    return np.array([
        [ w, -h],   # 0
        [ w,  h],   # 1
        [-w,  h],   # 2
        [-w, -h],   # 3
    ])


def _pipe_ring_2d(n_sides: int, radius: float) -> np.ndarray:
    """Return (n_sides, 2) ring offsets for a circular cross-section."""
    angles = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]) * radius


def _ihprofil_corners_2d(section: ISection) -> np.ndarray:
    """
    Return (12, 2) corner offsets for an I-beam cross-section.

    Col 0 → local-y offset (flange/width direction, ⊥ to UNITVEC).
    Col 1 → local-z offset (height/web direction, ‖ to UNITVEC).

    USFOS convention: h is measured along local-z (UNITVEC); bf along local-y.

    Vertex layout (CCW in the ly–lz cross-section plane):
      0: outer top-left      6: outer bottom-right
      1: outer top-right     7: outer bottom-left
      2: inner top-right     8: inner bottom-left
      3: web top-right       9: web bottom-left
      4: web bottom-right   10: web top-left
      5: inner bottom-right 11: inner top-left
    """
    lz  = section.h / 2.0        # half total height  → local-z
    lz1 = section.tf_top         # top flange thickness
    lz2 = section.tf_bot         # bottom flange thickness
    lyw = section.tw / 2.0       # half web thickness  → local-y
    lyf1 = section.bf_top / 2.0  # top flange half-width
    lyf2 = section.bf_bot / 2.0  # bottom flange half-width
    return np.array([
        [-lyf1,  lz        ],   # 0  outer top-left
        [ lyf1,  lz        ],   # 1  outer top-right
        [ lyf1,  lz - lz1  ],   # 2  inner top-right
        [ lyw,   lz - lz1  ],   # 3  web top-right
        [ lyw,  -lz + lz2  ],   # 4  web bottom-right
        [ lyf2, -lz + lz2  ],   # 5  inner bottom-right
        [ lyf2, -lz        ],   # 6  outer bottom-right
        [-lyf2, -lz        ],   # 7  outer bottom-left
        [-lyf2, -lz + lz2  ],   # 8  inner bottom-left
        [-lyw,  -lz + lz2  ],   # 9  web bottom-left
        [-lyw,   lz - lz1  ],   # 10 web top-left
        [-lyf1,  lz - lz1  ],   # 11 inner top-left
    ])


def _corners_global(
    origin: np.ndarray,
    ly: np.ndarray,
    lz: np.ndarray,
    offsets_2d: np.ndarray,
) -> np.ndarray:
    """Project 2-D cross-section offsets to 3-D global coords at *origin*."""
    return origin + offsets_2d[:, 0:1] * ly + offsets_2d[:, 1:2] * lz


# ── Per-beam mesh builders ─────────────────────────────────────────────────────

def _eccentric_endpoints(
    element: BeamElement,
    nodes: dict[int, Node],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p1_eff, p2_eff, direction) with eccentricity applied."""
    p1 = nodes[element.n1].xyz.copy()
    p2 = nodes[element.n2].xyz.copy()
    if element.ecc1 is not None:
        p1 = p1 + element.ecc1
    if element.ecc2 is not None:
        p2 = p2 + element.ecc2
    vec = p2 - p1
    L = np.linalg.norm(vec)
    direction = vec / L if L > 1e-12 else element.direction
    return p1, p2, direction


def build_beam_mesh(
    element: BeamElement,
    section: BoxSection,
    nodes: dict[int, Node],
) -> pv.PolyData:
    """
    Extrude a BOX cross-section along the beam.
    Returns 6-face closed box shell PolyData with cell-data ``element_id``.
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)

    lx, ly, lz = _local_frame(direction, element.local_z)
    offsets = _outer_corners_2d(section)

    c0 = _corners_global(p1, ly, lz, offsets)
    c1 = _corners_global(p2, ly, lz, offsets)
    pts = np.vstack([c0, c1])

    faces = np.array([
        [4, 0, 1, 5, 4],   # top side
        [4, 1, 2, 6, 5],   # right side
        [4, 2, 3, 7, 6],   # bottom side
        [4, 3, 0, 4, 7],   # left side
        [4, 0, 3, 2, 1],   # end-0 (n1), normal → -lx
        [4, 4, 5, 6, 7],   # end-1 (n2), normal → +lx
    ], dtype=np.int_)

    mesh = pv.PolyData(pts, faces.ravel())
    mesh["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


def build_pipe_mesh(
    element: BeamElement,
    section: PipeSection,
    nodes: dict[int, Node],
    n_sides: int = _PIPE_SIDES,
) -> pv.PolyData:
    """
    Extrude a hollow circular PIPE cross-section along the beam.

    Point layout (4 rings × n_sides):
      O1 = 0          outer ring at p1
      O2 = n_sides    outer ring at p2
      I1 = 2*n_sides  inner ring at p1
      I2 = 3*n_sides  inner ring at p2

    Faces (4 × n_sides quads):
      outer lateral, inner lateral, end-0 annular cap, end-1 annular cap
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)

    lx, ly, lz = _local_frame(direction, element.local_z)
    outer_ring = _pipe_ring_2d(n_sides, section.outer_radius)
    inner_ring = _pipe_ring_2d(n_sides, section.inner_radius)

    O1, O2 = 0,          n_sides
    I1, I2 = 2 * n_sides, 3 * n_sides

    pts = np.vstack([
        _corners_global(p1, ly, lz, outer_ring),   # O1
        _corners_global(p2, ly, lz, outer_ring),   # O2
        _corners_global(p1, ly, lz, inner_ring),   # I1
        _corners_global(p2, ly, lz, inner_ring),   # I2
    ])

    face_list: list[int] = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        # Outer lateral face
        face_list += [4, O1+i, O1+j, O2+j, O2+i]
        # Inner lateral face (reversed winding → inward normal)
        face_list += [4, I1+i, I2+i, I2+j, I1+j]
        # End-0 annular cap (outward normal → -lx)
        face_list += [4, O1+i, O1+j, I1+j, I1+i]
        # End-1 annular cap (outward normal → +lx)
        face_list += [4, O2+i, I2+i, I2+j, O2+j]

    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.int_))
    # Explicit cell_data assignment avoids PyVista ambiguity when n_cells == n_points
    mesh.cell_data["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


def build_isection_mesh(
    element: BeamElement,
    section: ISection,
    nodes: dict[int, Node],
) -> pv.PolyData:
    """
    Extrude an I-beam cross-section along the beam.
    Returns 12 lateral quad faces + 6 end-cap quad faces (3 per end).
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)

    lx, ly, lz = _local_frame(direction, element.local_z)
    offsets = _ihprofil_corners_2d(section)   # (12, 2)

    c0 = _corners_global(p1, ly, lz, offsets)   # (12, 3) at end-0
    c1 = _corners_global(p2, ly, lz, offsets)   # (12, 3) at end-1
    pts = np.vstack([c0, c1])                   # (24, 3)

    face_list: list[int] = []

    # 12 lateral quad faces (connect consecutive corners end-0 → end-1)
    for i in range(12):
        j = (i + 1) % 12
        face_list += [4, i, j, j + 12, i + 12]

    # End-0 cap (3 quads, winding gives outward normal towards -lx):
    #   top flange [0,11,2,1], web [3,10,9,4], bottom flange [5,8,7,6]
    face_list += [4, 0, 11, 2, 1]
    face_list += [4, 3, 10, 9, 4]
    face_list += [4, 5, 8, 7, 6]

    # End-1 cap (3 quads, same corner sets offset by 12 → normal towards +lx):
    #   top flange [12,13,14,23], web [15,16,21,22], bottom flange [17,18,19,20]
    face_list += [4, 12, 13, 14, 23]
    face_list += [4, 15, 16, 21, 22]
    face_list += [4, 17, 18, 19, 20]

    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.int_))
    mesh["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


# ── Shell surface mesh builder ─────────────────────────────────────────────────

def build_shell_surface_mesh(
    shell_elements: dict[int, ShellElement],
    nodes: dict[int, Node],
    centroid: np.ndarray,
) -> pv.PolyData:
    """
    Build a flat PolyData mesh for all shell elements (QUADSHEL / TRISHELL).

    Each shell element contributes exactly one face (triangle or quad).
    Points are centroid-shifted to match build_model_mesh().

    Returns an empty PolyData if shell_elements is empty.
    """
    if not shell_elements:
        return pv.PolyData()

    # Collect all unique node IDs referenced by shells
    all_nids: set[int] = set()
    for se in shell_elements.values():
        all_nids.update(se.nodes)

    sorted_nids = sorted(all_nids)
    nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

    pts = np.array([
        [nodes[nid].x - centroid[0],
         nodes[nid].y - centroid[1],
         nodes[nid].z - centroid[2]]
        for nid in sorted_nids
    ], dtype=float)

    face_list: list[int] = []
    eid_list:  list[int] = []

    for se in shell_elements.values():
        n = len(se.nodes)
        face_list.append(n)
        face_list.extend(nid_to_idx[nid] for nid in se.nodes)
        eid_list.append(se.eid)

    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.int_))
    mesh["element_id"] = np.array(eid_list, dtype=np.int32)
    return mesh


def mesh_section_at(
    element: BeamElement,
    section: BoxSection,
    nodes: dict[int, Node],
    t: float = 0.0,
) -> pv.PolyData:
    """
    Return a 2-D cross-section ring (inner + outer) at parametric position *t*
    along the beam (0 = n1, 1 = n2).  Useful for section-view rendering.
    Only implemented for BoxSection; returns empty PolyData for other types.
    """
    if not isinstance(section, BoxSection):
        return pv.PolyData()

    p1 = nodes[element.n1].xyz
    p2 = nodes[element.n2].xyz
    origin = p1 + t * (p2 - p1)

    _, ly, lz = _local_frame(element.direction, element.local_z)

    outer = _outer_corners_2d(section)
    iw = section.inner_width  / 2.0  # inner half-width  → local-y
    ih = section.inner_height / 2.0  # inner half-height → local-z
    inner = np.array([
        [ iw, -ih],
        [ iw,  ih],
        [-iw,  ih],
        [-iw, -ih],
    ])

    pts_outer = _corners_global(origin, ly, lz, outer)
    pts_inner = _corners_global(origin, ly, lz, inner)
    pts = np.vstack([pts_outer, pts_inner])

    outer_loop = [5, 0, 1, 2, 3, 0]
    inner_loop = [5, 4, 5, 6, 7, 4]
    lines = np.array(outer_loop + inner_loop, dtype=np.int_)

    return pv.PolyData(pts, lines=lines)


# ── Model-level builders ───────────────────────────────────────────────────────

def build_model_mesh(
    model: FEMModel,
    skip_missing: bool = True,
) -> pv.PolyData:
    """
    Build a single merged PolyData for all beams and shells in *model*.

    Supported beam section types: BoxSection, PipeSection, ISection.
    Shell elements (QUADSHEL, TRISHELL) are rendered as flat polygon faces.

    Cell-data array ``element_id`` (int32) identifies which element each face
    belongs to — used for colour-by-group and temperature mapping.

    Points are centroid-shifted so PyVista renders correctly regardless of
    global vessel coordinates.
    """
    meshes: list[pv.PolyData] = []
    centroid = model.centroid()

    # ── Beam elements ─────────────────────────────────────────────────────────
    for eid, elem in model.elements.items():
        if elem.direction is None:
            if not skip_missing:
                raise ValueError(f"Element {eid} has no direction vector")
            continue

        section = model.sections.get(elem.geom_id)

        if isinstance(section, BoxSection):
            mesh = build_beam_mesh(elem, section, model.nodes)
        elif isinstance(section, PipeSection):
            mesh = build_pipe_mesh(elem, section, model.nodes)
        elif isinstance(section, ISection):
            mesh = build_isection_mesh(elem, section, model.nodes)
        else:
            if not skip_missing:
                raise ValueError(
                    f"Element {eid}: section {elem.geom_id} is unsupported "
                    f"(type {type(section).__name__ if section else 'None'})"
                )
            continue

        mesh.points -= centroid
        meshes.append(mesh)

    # ── Shell elements ────────────────────────────────────────────────────────
    if model.shell_elements:
        shell_mesh = build_shell_surface_mesh(
            model.shell_elements, model.nodes, centroid
        )
        if shell_mesh.n_cells > 0:
            meshes.append(shell_mesh)

    if not meshes:
        return pv.PolyData()

    combined = meshes[0].copy()
    for m in meshes[1:]:
        combined = combined.merge(m)
    return combined


def build_centreline_mesh(model: FEMModel) -> pv.PolyData:
    """
    Return a PolyData of line segments for all elements:
    - Beams: n1 → n2 axis line (with eccentricity applied)
    - Shells: boundary edges of each polygon

    Used for wire-frame / overview rendering (fast, low memory).
    Points are centroid-shifted, matching build_model_mesh().
    """
    centroid = model.centroid()
    pts_list: list[np.ndarray] = []
    lines: list[int] = []
    idx = 0

    # Beam centrelines — handle eccentricity per beam
    for elem in model.elements.values():
        if elem.n1 not in model.nodes or elem.n2 not in model.nodes:
            continue
        p1 = model.nodes[elem.n1].xyz.copy()
        p2 = model.nodes[elem.n2].xyz.copy()
        if elem.ecc1 is not None:
            p1 = p1 + elem.ecc1
        if elem.ecc2 is not None:
            p2 = p2 + elem.ecc2
        pts_list.append(p1 - centroid)
        pts_list.append(p2 - centroid)
        lines += [2, idx, idx + 1]
        idx += 2

    # Shell boundary edges (no eccentricity for shells)
    for se in model.shell_elements.values():
        n = len(se.nodes)
        for k in range(n):
            nid0 = se.nodes[k]
            nid1 = se.nodes[(k + 1) % n]
            if nid0 not in model.nodes or nid1 not in model.nodes:
                continue
            pts_list.append(model.nodes[nid0].xyz - centroid)
            pts_list.append(model.nodes[nid1].xyz - centroid)
            lines += [2, idx, idx + 1]
            idx += 2

    if not pts_list:
        return pv.PolyData()

    pts = np.array(pts_list, dtype=float)
    if not lines:
        return pv.PolyData(pts)
    return pv.PolyData(pts, lines=np.array(lines, dtype=np.int_))
