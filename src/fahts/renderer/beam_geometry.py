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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyvista as pv

from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, PipeSection, ISection, PlateSection

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
    Build a BOX cross-section as 4 flat mid-surface panels (USFOS-style, no end caps).
    Returns 4-face PolyData with cell-data ``element_id``.
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)

    lx, ly, lz = _local_frame(direction, element.local_z)
    offsets = _outer_corners_2d(section)

    c0 = _corners_global(p1, ly, lz, offsets)
    c1 = _corners_global(p2, ly, lz, offsets)
    pts = np.vstack([c0, c1])

    # 4 lateral panels only — no end caps (USFOS render style)
    faces = np.array([
        [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7],
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
    Build a PIPE as a single outer-surface cylindrical shell (USFOS-style, no wall thickness).

    Point layout (2 rings × n_sides):
      O1 = 0         outer ring at p1
      O2 = n_sides   outer ring at p2

    Faces: n_sides outer lateral quads only.
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)

    lx, ly, lz = _local_frame(direction, element.local_z)
    outer_ring = _pipe_ring_2d(n_sides, section.outer_radius)

    O1, O2 = 0, n_sides
    pts = np.vstack([
        _corners_global(p1, ly, lz, outer_ring),   # O1
        _corners_global(p2, ly, lz, outer_ring),   # O2
    ])

    face_list: list[int] = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        face_list += [4, O1+i, O1+j, O2+j, O2+i]

    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.int_))
    mesh.cell_data["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


def build_isection_mesh(
    element: BeamElement,
    section: ISection,
    nodes: dict[int, Node],
) -> pv.PolyData:
    """
    Build an I-beam as 3 flat mid-surface panels (USFOS-style, no wall thickness):
    top flange, web, bottom flange.  3 cells, 12 points.
    """
    p1, p2, direction = _eccentric_endpoints(element, nodes)
    lx, ly, lz = _local_frame(direction, element.local_z)

    lz_h    = section.h / 2    # half total height → local-z
    lyf_top = section.bf_top / 2
    lyf_bot = section.bf_bot / 2

    def _pt(p: np.ndarray, y: float, z: float) -> np.ndarray:
        return p + y * ly + z * lz

    # 5 panels — each flange is split at y=0 (the web centreline) so the
    # shared edge between the two halves renders as a clean polygon boundary
    # line rather than a z-fighting coincident edge.  Web spans full height.
    panels: list[list[np.ndarray]] = [
        # Top flange — left half  (y: -lyf_top → 0, z = +lz_h)
        [_pt(p1, -lyf_top, lz_h), _pt(p1, 0.0, lz_h),
         _pt(p2, 0.0,      lz_h), _pt(p2, -lyf_top, lz_h)],
        # Top flange — right half (y: 0 → +lyf_top, z = +lz_h)
        [_pt(p1, 0.0,      lz_h), _pt(p1, lyf_top, lz_h),
         _pt(p2, lyf_top,  lz_h), _pt(p2, 0.0,     lz_h)],
        # Web — y = 0, full height
        [_pt(p1, 0.0, -lz_h), _pt(p1, 0.0, lz_h),
         _pt(p2, 0.0,  lz_h), _pt(p2, 0.0, -lz_h)],
        # Bottom flange — left half  (y: -lyf_bot → 0, z = -lz_h)
        [_pt(p1, -lyf_bot, -lz_h), _pt(p1, 0.0,     -lz_h),
         _pt(p2, 0.0,      -lz_h), _pt(p2, -lyf_bot, -lz_h)],
        # Bottom flange — right half (y: 0 → +lyf_bot, z = -lz_h)
        [_pt(p1, 0.0,      -lz_h), _pt(p1, lyf_bot, -lz_h),
         _pt(p2, lyf_bot,  -lz_h), _pt(p2, 0.0,     -lz_h)],
    ]

    all_pts: list[np.ndarray] = []
    face_list: list[int] = []
    for i, panel in enumerate(panels):
        base = i * 4
        all_pts.extend(panel)
        face_list += [4, base, base + 1, base + 2, base + 3]

    mesh = pv.PolyData(np.array(all_pts, dtype=float), np.array(face_list, dtype=np.int_))
    mesh["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


# ── Shell surface mesh builder ─────────────────────────────────────────────────

def build_thick_member_mesh(
    element: BeamElement,
    section: BoxSection | PipeSection | ISection,
    nodes: dict[int, Node],
    n_pipe_sides: int = _PIPE_SIDES,
) -> pv.PolyData | None:
    """
    Build a beam member with its REAL wall thickness (outer + inner surfaces + end rings).

    Uses the same Hex8 solid meshers as the 3-D solver (one layer, coarse divisions) and
    returns their boundary faces placed in the member's local frame (incl. eccentricity),
    so the rendered solid matches the analysed geometry.  Cell-data ``element_id``.
    Returns None for unsupported sections.
    """
    from fahts.core.heat.solid_mesh import BoxSolidMesher, IProfileSolidMesher, PipeSolidMesher

    p1, p2, direction = _eccentric_endpoints(element, nodes)
    L = float(np.linalg.norm(p2 - p1))
    if L <= 1e-12:
        return None
    lx, ly, lz = _local_frame(direction, element.local_z)
    if isinstance(section, PipeSection):
        solid = PipeSolidMesher(section, L, c_circ=max(3, n_pipe_sides), n_length=1,
                                n_layers=1).build()
    elif isinstance(section, BoxSection):
        solid = BoxSolidMesher(section, L, n_top=1, n_side=1, n_length=1, n_layers=1).build()
    elif isinstance(section, ISection):
        solid = IProfileSolidMesher(section, L, n_top=2, n_side=1, n_bottom=2, n_length=1,
                                    n_layers=1).build()
    else:
        return None
    R = np.column_stack([lx, ly, lz])
    pts = p1 + np.asarray(solid.nodes) @ R.T
    faces = np.asarray(solid.faces)
    mesh = pv.PolyData(pts, np.hstack([np.full((len(faces), 1), 4), faces]).ravel())
    mesh.cell_data["element_id"] = np.full(mesh.n_cells, element.eid, dtype=np.int32)
    return mesh


def build_thick_shell_mesh(
    shell_elements: dict[int, ShellElement],
    nodes: dict[int, Node],
    sections: dict,
    centroid: np.ndarray,
) -> pv.PolyData:
    """
    Shell elements with their plate thickness: QUADSHEL as a one-layer Hex8 plate
    (both faces + edges); TRISHELL as a triangular prism (mid-surface ± t/2).
    Points are centroid-shifted like build_shell_surface_mesh().
    """
    from fahts.core.heat.solid_mesh import PlateSolidMesher

    meshes: list[pv.PolyData] = []
    for se in shell_elements.values():
        sec = sections.get(se.geom_id)
        corners = np.array([nodes[n].xyz for n in se.nodes], dtype=float)
        t = float(getattr(sec, "thickness", 0.0)) if sec is not None else 0.0
        if len(se.nodes) == 4 and isinstance(sec, PlateSection) and t > 0.0:
            solid = PlateSolidMesher(sec, corners, mesh_12=1, mesh_14=1, n_layers=1).build()
            faces = np.asarray(solid.faces)
            m = pv.PolyData(np.asarray(solid.nodes) - centroid,
                            np.hstack([np.full((len(faces), 1), 4), faces]).ravel())
        elif len(se.nodes) == 3 and t > 0.0:
            nrm = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            nrm /= max(np.linalg.norm(nrm), 1e-12)
            pts = np.vstack([corners - 0.5 * t * nrm, corners + 0.5 * t * nrm]) - centroid
            f = [3, 0, 2, 1, 3, 3, 4, 5, 4, 0, 1, 4, 3, 4, 1, 2, 5, 4, 4, 2, 0, 3, 5]
            m = pv.PolyData(pts, np.array(f))
        else:
            m = pv.PolyData(corners - centroid,
                            np.array([len(se.nodes), *range(len(se.nodes))]))
        m.cell_data["element_id"] = np.full(m.n_cells, se.eid, dtype=np.int32)
        meshes.append(m)
    if not meshes:
        return pv.PolyData()
    return pv.merge(meshes)


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
    n_pipe_sides: int = _PIPE_SIDES,
    show_thickness: bool = False,
) -> pv.PolyData:
    """
    Build a single merged PolyData for all beams and shells in *model*.

    Supported beam section types: BoxSection, PipeSection, ISection.
    Shell elements (QUADSHEL, TRISHELL) are rendered as flat polygon faces.

    Cell-data array ``element_id`` (int32) identifies which element each face
    belongs to — used for colour-by-group and temperature mapping.

    Points are centroid-shifted so PyVista renders correctly regardless of
    global vessel coordinates.

    ``show_thickness=True`` draws every member with its real wall / plate thickness
    (outer + inner surfaces and end rings, from the 3-D solid meshers) instead of the
    USFOS-style zero-thickness mid-surface panels.
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

        if show_thickness and isinstance(section, (BoxSection, PipeSection, ISection)):
            mesh = build_thick_member_mesh(elem, section, model.nodes, n_pipe_sides)
            if mesh is not None:
                mesh.points -= centroid
                meshes.append(mesh)
                continue

        if isinstance(section, BoxSection):
            mesh = build_beam_mesh(elem, section, model.nodes)
        elif isinstance(section, PipeSection):
            mesh = build_pipe_mesh(elem, section, model.nodes, n_sides=n_pipe_sides)
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
        shell_mesh = (
            build_thick_shell_mesh(model.shell_elements, model.nodes, model.sections, centroid)
            if show_thickness
            else build_shell_surface_mesh(model.shell_elements, model.nodes, centroid)
        )
        if shell_mesh.n_cells > 0:
            meshes.append(shell_mesh)

    if not meshes:
        return pv.PolyData()

    combined = meshes[0].copy()
    for m in meshes[1:]:
        combined = combined.merge(m)
    return combined


def build_analysis_mesh_overlay(
    model: FEMModel,
    solved_element_ids: list[int],
    config: Any,
    centroid: np.ndarray,
) -> pv.PolyData:
    """
    Build a quad-surface PolyData matching the FEM analysis mesh for all solved elements.

    Each element's BeamSurfaceMesh (built with the same mesher params used during
    analysis) is transformed from beam-local coords to global 3-D scene coords.
    The returned mesh should be rendered as a wireframe overlay so the user can
    see the exact mesh density that was used in the solver.

    Parameters
    ----------
    model             : loaded FEMModel.
    solved_element_ids: element IDs that were solved (TemperatureField.element_ids).
    config            : AnalysisConfig — carries n_top/n_side/n_length etc.
    centroid          : model centroid [m]; matches the shift applied by build_model_mesh.

    Returns
    -------
    pv.PolyData with one quad face per analysis mesh element.
    Empty PolyData if no supported elements are found.
    """
    from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
    from fahts.core.heat.section_mesh.iprofil_surface_mesher import IProfileSurfaceMesher
    from fahts.core.heat.section_mesh.pipe_surface_mesher import PipeSurfaceMesher
    from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
    from fahts.core.model.section import PlateSection

    all_pts: list[np.ndarray] = []
    face_list: list[int] = []
    cell_eid: list[int] = []
    pt_offset = 0

    for eid in solved_element_ids:
        elem = model.elements.get(eid)
        if elem is None or elem.direction is None:
            continue
        section = model.sections.get(elem.geom_id)

        if isinstance(section, BoxSection):
            bsm = BoxSurfaceMesher(
                section, elem.length,
                config.n_top, config.n_side, config.n_length,
            ).build()
        elif isinstance(section, ISection):
            bsm = IProfileSurfaceMesher(
                section, elem.length,
                config.n_top_i, config.n_side_i, config.n_bottom_i, config.n_length_i,
            ).build()
        elif isinstance(section, PipeSection):
            bsm = PipeSurfaceMesher(
                section, elem.length,
                config.c_circ, config.n_length_p,
            ).build()
        else:
            continue

        # Transform beam-local nodes (x=axial, y=width, z=height) to global scene coords
        p1_eff, _p2, direction = _eccentric_endpoints(elem, model.nodes)
        lx, ly, lz = _local_frame(direction, elem.local_z)

        if isinstance(section, PipeSection):
            nodes = bsm.nodes.copy()
            nodes[:, 1:3] *= 1.01  # push overlay outside solid surface to avoid z-fighting
        else:
            nodes = bsm.nodes  # (n_nodes, 3)
        global_pts = (
            p1_eff[np.newaxis, :]
            + nodes[:, 0:1] * lx[np.newaxis, :]
            + nodes[:, 1:2] * ly[np.newaxis, :]
            + nodes[:, 2:3] * lz[np.newaxis, :]
        ) - centroid[np.newaxis, :]

        all_pts.append(global_pts)
        for quad in bsm.quads:
            face_list += [
                4,
                int(quad[0]) + pt_offset,
                int(quad[1]) + pt_offset,
                int(quad[2]) + pt_offset,
                int(quad[3]) + pt_offset,
            ]
            cell_eid.append(eid)
        pt_offset += len(nodes)

    # ── Shell elements (QUADSHEL only; PlateSurfaceMesher outputs global coords) ─
    for se in model.shell_elements.values():
        if len(se.nodes) != 4:
            continue  # TRISHELL — no surface mesher
        section = model.sections.get(se.geom_id)
        if not isinstance(section, PlateSection):
            continue
        corners = np.array([model.nodes[nid].xyz for nid in se.nodes], dtype=float)
        try:
            bsm = PlateSurfaceMesher(
                section, corners, config.mesh_12, config.mesh_14,
            ).build()
        except ValueError:
            continue

        # PlateSurfaceMesher nodes are already in global 3D coords
        global_pts = bsm.nodes - centroid[np.newaxis, :]
        all_pts.append(global_pts)
        for quad in bsm.quads:
            face_list += [
                4,
                int(quad[0]) + pt_offset,
                int(quad[1]) + pt_offset,
                int(quad[2]) + pt_offset,
                int(quad[3]) + pt_offset,
            ]
            cell_eid.append(se.eid)
        pt_offset += len(bsm.nodes)

    if not all_pts:
        return pv.PolyData()

    pts = np.vstack(all_pts)
    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.intp))
    mesh.cell_data["element_id"] = np.array(cell_eid, dtype=np.int32)
    return mesh


@dataclass
class MeshInspectorData:
    """
    Pre-built FEM surface mesh with topology metadata for interactive connectivity inspection.

    Attributes
    ----------
    mesh            : global scene-coord quad PolyData (inspector picking target).
                      Cell data: ``inspector_beam_eid``, ``inspector_quad_idx``.
    cell_beam_eid   : (n_cells,) beam/shell element ID per cell.
    cell_quad_idx   : (n_cells,) local quad index within that element.
    beam_quads      : eid → (n_quads, 4) local node index connectivity.
    beam_cell_offset: eid → index of first cell for this element in *mesh*.
    beam_node_offset: eid → index of first global point for this element in *mesh*.
    beam_node_gdof  : eid → (n_nodes,) global DOF index per local node.
                      Matches the DOF numbering used by the solver's K-matrix assembly:
                      co-located nodes from different elements share the same index.
    """
    mesh: pv.PolyData
    cell_beam_eid: np.ndarray
    cell_quad_idx: np.ndarray
    beam_quads: dict[int, np.ndarray]
    beam_cell_offset: dict[int, int]
    beam_node_offset: dict[int, int]
    beam_node_gdof: dict[int, np.ndarray]


def _compute_inspector_gdof_map(
    eid_world_pts: dict[int, np.ndarray],
    eid_meshes: dict[int, Any],
    model: Any,
    tol: float = 1e-3,
    struct_tol: float = 0.025,
) -> dict[int, np.ndarray]:
    """
    Assign global DOF indices matching the solver's two-pass K-matrix assembly.

    Pass 1 — 1 mm proximity: exact end-to-end merges.
    Pass 2 — structural-node-aware (25 mm): for every FEM structural node,
    merges end-face mesh nodes from different elements within struct_tol.
    Catches perpendicular-crossing junctions where web offset places nearest
    nodes ~tw/√2 ≈ 4–14 mm apart.

    Returns {eid: (n_nodes,) int array} mapping local node → global DOF index.
    """
    from scipy.spatial import cKDTree

    coords: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for eid, pts in eid_world_pts.items():
        for i in range(len(pts)):
            coords.append(pts[i])
            labels.append((eid, i))

    n = len(labels)
    if n == 0:
        return {}

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    coords_arr = np.array(coords)

    # Pass 1: exact proximity
    tree = cKDTree(coords_arr)
    for i, j in tree.query_pairs(tol):
        _union(i, j)

    # Pass 2: structural-node-aware merge
    if model is not None:
        label_to_flat: dict[tuple[int, int], int] = {lbl: i for i, lbl in enumerate(labels)}
        for nid in model.nodes:
            end_flat: list[int] = []
            end_eids: list[int] = []
            for eid, mesh in eid_meshes.items():
                elem = model.elements.get(eid)
                if elem is None:
                    continue
                if elem.n1 != nid and elem.n2 != nid:
                    continue
                if eid not in eid_world_pts:
                    continue
                x_target = 0.0 if elem.n1 == nid else elem.length
                x_local  = mesh.nodes[:, 0]
                for local_idx in np.where(np.abs(x_local - x_target) < 1e-9)[0]:
                    key = (eid, int(local_idx))
                    if key in label_to_flat:
                        end_flat.append(label_to_flat[key])
                        end_eids.append(eid)
            if len(end_flat) < 2:
                continue
            local_tree = cKDTree(coords_arr[end_flat])
            for li, lj in local_tree.query_pairs(struct_tol):
                if end_eids[li] != end_eids[lj]:
                    _union(end_flat[li], end_flat[lj])

    root_to_dof: dict[int, int] = {}
    dof_idx = 0
    label_to_dof: dict[tuple[int, int], int] = {}
    for i, lbl in enumerate(labels):
        root = _find(i)
        if root not in root_to_dof:
            root_to_dof[root] = dof_idx
            dof_idx += 1
        label_to_dof[lbl] = root_to_dof[root]

    return {
        eid: np.array([label_to_dof[(eid, i)] for i in range(len(pts))], dtype=np.intp)
        for eid, pts in eid_world_pts.items()
    }


def build_mesh_inspector_data(
    model: FEMModel,
    centroid: np.ndarray,
    config: Any = None,
) -> MeshInspectorData:
    """
    Build ``MeshInspectorData`` for all supported elements in *model*.

    Surface meshes are built using params from *config* (an AnalysisConfig duck-type);
    default mesh densities are used when *config* is ``None``.

    The returned ``mesh`` has the same centroid shift as ``build_model_mesh``
    so it overlays correctly on the rendered structure.
    """
    from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
    from fahts.core.heat.section_mesh.iprofil_surface_mesher import IProfileSurfaceMesher
    from fahts.core.heat.section_mesh.pipe_surface_mesher import PipeSurfaceMesher
    from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
    from fahts.core.model.section import PlateSection

    n_top    = getattr(config, "n_top",      2)
    n_side   = getattr(config, "n_side",     3)
    n_length = getattr(config, "n_length",   4)
    n_top_i  = getattr(config, "n_top_i",    4)
    n_side_i = getattr(config, "n_side_i",   2)
    n_bot_i  = getattr(config, "n_bottom_i", 2)
    n_len_i  = getattr(config, "n_length_i", 2)
    c_circ   = getattr(config, "c_circ",     8)
    n_len_p  = getattr(config, "n_length_p", 4)
    mesh_12  = getattr(config, "mesh_12",    4)
    mesh_14  = getattr(config, "mesh_14",    2)

    all_pts: list[np.ndarray] = []
    face_list: list[int] = []
    cell_beam_eid: list[int] = []
    cell_quad_idx: list[int] = []
    beam_quads: dict[int, np.ndarray] = {}
    beam_cell_offset: dict[int, int] = {}
    beam_node_offset: dict[int, int] = {}
    # Per-element world-space positions and mesh objects for DOF merging.
    _eid_world_pts: dict[int, np.ndarray] = {}
    _eid_meshes: dict[int, Any] = {}
    pt_offset = 0
    cell_offset = 0

    for eid, elem in model.elements.items():
        if elem.direction is None:
            continue
        section = model.sections.get(elem.geom_id)

        if isinstance(section, BoxSection):
            bsm = BoxSurfaceMesher(section, elem.length, n_top, n_side, n_length).build()
        elif isinstance(section, ISection):
            bsm = IProfileSurfaceMesher(
                section, elem.length, n_top_i, n_side_i, n_bot_i, n_len_i,
            ).build()
        elif isinstance(section, PipeSection):
            bsm = PipeSurfaceMesher(section, elem.length, c_circ, n_len_p).build()
        else:
            continue

        p1_eff, _p2, direction = _eccentric_endpoints(elem, model.nodes)
        lx, ly, lz_ax = _local_frame(direction, elem.local_z)
        nodes_local = bsm.nodes  # (n_nodes, 3) beam-local coords
        world_pts = (
            p1_eff[np.newaxis, :]
            + nodes_local[:, 0:1] * lx[np.newaxis, :]
            + nodes_local[:, 1:2] * ly[np.newaxis, :]
            + nodes_local[:, 2:3] * lz_ax[np.newaxis, :]
        )
        global_pts = world_pts - centroid[np.newaxis, :]

        all_pts.append(global_pts)
        _eid_world_pts[eid] = world_pts
        _eid_meshes[eid]    = bsm
        beam_quads[eid] = bsm.quads.copy()
        beam_cell_offset[eid] = cell_offset
        beam_node_offset[eid] = pt_offset

        for q_idx, quad in enumerate(bsm.quads):
            face_list += [4,
                int(quad[0]) + pt_offset, int(quad[1]) + pt_offset,
                int(quad[2]) + pt_offset, int(quad[3]) + pt_offset,
            ]
            cell_beam_eid.append(eid)
            cell_quad_idx.append(q_idx)

        pt_offset += len(nodes_local)
        cell_offset += len(bsm.quads)

    # QUADSHEL shell plates
    for se in model.shell_elements.values():
        if len(se.nodes) != 4:
            continue
        section = model.sections.get(se.geom_id)
        if not isinstance(section, PlateSection):
            continue
        corners = np.array([model.nodes[nid].xyz for nid in se.nodes], dtype=float)
        try:
            bsm = PlateSurfaceMesher(section, corners, mesh_12, mesh_14).build()
        except ValueError:
            continue

        world_pts = bsm.nodes.copy()
        global_pts = world_pts - centroid[np.newaxis, :]
        all_pts.append(global_pts)
        _eid_world_pts[se.eid] = world_pts
        _eid_meshes[se.eid]    = bsm
        beam_quads[se.eid] = bsm.quads.copy()
        beam_cell_offset[se.eid] = cell_offset
        beam_node_offset[se.eid] = pt_offset

        for q_idx, quad in enumerate(bsm.quads):
            face_list += [4,
                int(quad[0]) + pt_offset, int(quad[1]) + pt_offset,
                int(quad[2]) + pt_offset, int(quad[3]) + pt_offset,
            ]
            cell_beam_eid.append(se.eid)
            cell_quad_idx.append(q_idx)

        pt_offset += len(bsm.nodes)
        cell_offset += len(bsm.quads)

    if not all_pts:
        return MeshInspectorData(
            mesh=pv.PolyData(),
            cell_beam_eid=np.array([], dtype=np.int32),
            cell_quad_idx=np.array([], dtype=np.int32),
            beam_quads={},
            beam_cell_offset={},
            beam_node_offset={},
            beam_node_gdof={},
        )

    pts = np.vstack(all_pts)
    mesh = pv.PolyData(pts, np.array(face_list, dtype=np.intp))
    eid_arr  = np.array(cell_beam_eid, dtype=np.int32)
    qidx_arr = np.array(cell_quad_idx, dtype=np.int32)
    mesh.cell_data["inspector_beam_eid"] = eid_arr
    mesh.cell_data["inspector_quad_idx"] = qidx_arr

    # ── Global DOF map: same two-pass merge as the solver ────────────────────
    beam_node_gdof = _compute_inspector_gdof_map(_eid_world_pts, _eid_meshes, model)

    return MeshInspectorData(
        mesh=mesh,
        cell_beam_eid=eid_arr,
        cell_quad_idx=qidx_arr,
        beam_quads=beam_quads,
        beam_cell_offset=beam_cell_offset,
        beam_node_offset=beam_node_offset,
        beam_node_gdof=beam_node_gdof,
    )


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
