"""Tests for the Hex8 solid meshers (fahts/core/heat/solid_mesh)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fahts.core.heat.section_mesh.box_mesher import BoxMesher
from fahts.core.heat.solid_mesh import (
    FACE_END, FACE_INNER, FACE_OUTER,
    BlockSolidMesher, BoxSolidMesher, IProfileSolidMesher, PipeSolidMesher,
    PlateSolidMesher, SolidMesh, extrude_section,
)
from fahts.core.heat.solid_mesh._hex_topology import HEX_FACES
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection

REL = 1e-10

BOX = BoxSection(sid=1, H=0.30, T_side=0.010, T_bot=0.012, T_top=0.016, W=0.20)
ISEC = ISection(sid=2, h=0.40, tw=0.012, bf_top=0.20, tf_top=0.020,
                bf_bot=0.16, tf_bot=0.016)
PIPE = PipeSection(sid=3, outer_diameter=0.3239, thickness=0.0127)
PLATE = PlateSection(sid=4, thickness=0.015)
PLATE_CORNERS = np.array([[1.0, 2.0, 3.0], [3.0, 2.5, 3.0],
                          [3.2, 3.9, 3.8], [0.9, 3.2, 3.6]])


def _planar_corners() -> np.ndarray:
    """Planar but skewed/non-parallelogram quad in a tilted plane."""
    e1 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    e2 = np.array([-1.0, 1.0, 2.0]) / math.sqrt(6)
    uv = np.array([[0.0, 0.0], [2.0, 0.1], [2.3, 1.5], [-0.2, 1.1]])
    return np.array([5.0, -1.0, 2.0]) + uv[:, :1] * e1 + uv[:, 1:] * e2


def _poly_area(p: np.ndarray) -> float:
    """Area of a planar polygon in 3-D."""
    s = np.zeros(3)
    for i in range(len(p)):
        s += np.cross(p[i], p[(i + 1) % len(p)])
    return 0.5 * float(np.linalg.norm(s))


# ── mesh factories ────────────────────────────────────────────────────────────

L = 1.7


def _cases() -> dict[str, SolidMesh]:
    return {
        "box_default": BoxSolidMesher(BOX, L).build(),
        "box_fine": BoxSolidMesher(BOX, L, n_top=5, n_side=7, n_length=3, n_layers=3).build(),
        "box_1layer": BoxSolidMesher(BOX, L, n_top=1, n_side=1, n_length=1,
                                     n_layers=1).build(),
        "ibeam_default": IProfileSolidMesher(ISEC, L).build(),
        "ibeam_fine": IProfileSolidMesher(ISEC, L, n_top=5, n_side=6, n_bottom=3,
                                          n_length=3, n_layers=3).build(),
        "ibeam_1": IProfileSolidMesher(ISEC, L, 1, 1, 1, 1, 1).build(),
        "pipe_default": PipeSolidMesher(PIPE, L).build(),
        "pipe_3layer": PipeSolidMesher(PIPE, L, c_circ=7, n_length=2, n_layers=3).build(),
        "plate_default": PlateSolidMesher(PLATE, _planar_corners()).build(),
        "plate_3layer": PlateSolidMesher(PLATE, _planar_corners(), 5, 3, 3).build(),
        "block": BlockSolidMesher(1.0, 0.5, 0.25, 4, 3, 2, origin=(1, 2, 3)).build(),
        "extruded_legacy_box": extrude_section(BoxMesher(BOX, elem_size=0.05,
                                                         n_layers=2).build(), L, 3),
    }


CASES = _cases()


@pytest.fixture(params=sorted(CASES))
def mesh(request) -> SolidMesh:
    return CASES[request.param]


# ── generic checks ────────────────────────────────────────────────────────────

class TestGenericProperties:
    def test_positive_volumes(self, mesh):
        assert np.all(mesh.hex_volumes() > 0.0)

    def test_divergence_outward(self, mesh):
        """Σ_f (n_f · c_f) A_f = 3 V for planar outward faces."""
        s = np.sum(np.einsum("fk,fk->f", mesh.face_normals(), mesh.face_centroids())
                   * mesh.face_areas())
        assert s == pytest.approx(3.0 * mesh.volume, rel=REL)

    def test_divergence_per_axis(self, mesh):
        """Σ_f n_f,x · c_f,x A_f = V (per axis) — catches compensating errors."""
        n, c, a = mesh.face_normals(), mesh.face_centroids(), mesh.face_areas()
        for ax in range(3):
            assert np.sum(n[:, ax] * c[:, ax] * a) == pytest.approx(mesh.volume, rel=1e-9)

    def test_closed_surface(self, mesh):
        """Σ n_f A_f = 0 for a closed surface."""
        v = (mesh.face_normals() * mesh.face_areas()[:, None]).sum(axis=0)
        assert np.linalg.norm(v) < 1e-10 * mesh.face_areas().sum()

    def test_watertight_consistent(self, mesh):
        """Each directed boundary edge appears once and its reverse once."""
        f = mesh.faces
        e = np.stack([f, np.roll(f, -1, axis=1)], axis=2).reshape(-1, 2)
        directed = {tuple(x) for x in e.tolist()}
        assert len(directed) == len(e)
        for a, b in directed:
            assert (b, a) in directed
        und = np.sort(e, axis=1)
        _, cnt = np.unique(und, axis=0, return_counts=True)
        assert np.all(cnt == 2)

    def test_no_duplicate_nodes(self, mesh):
        r = np.round(mesh.nodes / 1e-9).astype(np.int64)
        assert len(np.unique(r, axis=0)) == mesh.n_nodes

    def test_all_nodes_used(self, mesh):
        assert len(np.unique(mesh.hexes)) == mesh.n_nodes

    def test_boundary_faces_are_single_hex_faces(self, mesh):
        allf = mesh.hexes[:, HEX_FACES].reshape(-1, 4)
        keys, cnt = np.unique(np.sort(allf, axis=1), axis=0, return_counts=True)
        assert cnt.max() <= 2
        once = {tuple(k) for k in keys[cnt == 1].tolist()}
        bnd = {tuple(k) for k in np.sort(mesh.faces, axis=1).tolist()}
        assert once == bnd
        assert len(bnd) == mesh.n_faces

    def test_face_group_valid(self, mesh):
        assert len(mesh.face_group) == mesh.n_faces
        assert set(np.unique(mesh.face_group)) <= {FACE_OUTER, FACE_INNER, FACE_END}


# ── BOX ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_box_volume_and_areas(n_layers):
    m = BoxSolidMesher(BOX, L, n_top=3, n_side=4, n_length=5, n_layers=n_layers).build()
    assert m.section_kind == "BOX"
    assert m.volume == pytest.approx(BOX.cross_section_area * L, rel=REL)
    a = m.face_areas()
    assert a[m.outer_face_indices].sum() == pytest.approx(2 * (BOX.W + BOX.H) * L, rel=REL)
    inner = 2 * (BOX.inner_width + BOX.inner_height) * L
    assert a[m.inner_face_indices].sum() == pytest.approx(inner, rel=REL)
    assert a[m.face_indices(FACE_END)].sum() == pytest.approx(
        2 * BOX.cross_section_area, rel=REL)


def test_box_geometry_axes_and_counts():
    nt, ns, nlen, nl = 3, 4, 5, 2
    m = BoxSolidMesher(BOX, L, nt, ns, nlen, nl).build()
    # W along y, H along z (BoxSurfaceMesher convention)
    assert m.nodes[:, 1].min() == pytest.approx(-BOX.W / 2)
    assert m.nodes[:, 1].max() == pytest.approx(BOX.W / 2)
    assert m.nodes[:, 2].min() == pytest.approx(-BOX.H / 2)
    assert m.nodes[:, 2].max() == pytest.approx(BOX.H / 2)
    assert m.nodes[:, 0].min() == 0.0 and m.nodes[:, 0].max() == pytest.approx(L)
    # orthogonal ring: corner blocks (nl × nl) are shared by the adjacent walls
    hoop = 2 * (nt + ns)
    assert m.n_hexes == ((nt + 2 * nl) * (ns + 2 * nl) - nt * ns) * nlen
    assert len(m.outer_face_indices) == (hoop + 8 * nl) * nlen
    assert len(m.inner_face_indices) == hoop * nlen
    # every hex is a rectangular brick (needed by the monotone two-point stencil)
    X = m.nodes[m.hexes]
    ext = X.max(axis=1) - X.min(axis=1)
    assert np.allclose(np.abs(m.hex_volumes()), np.prod(ext, axis=1))
    # outer faces lie on the outer rectangle with axis-aligned normals
    c = m.face_centroids()[m.outer_face_indices]
    on = np.isclose(np.abs(c[:, 1]), BOX.W / 2) | np.isclose(np.abs(c[:, 2]), BOX.H / 2)
    assert np.all(on)
    # top-plate thickness correct: inner top at H/2 - T_top
    ci = m.face_centroids()[m.inner_face_indices]
    assert np.isclose(ci[:, 2].max(), BOX.H / 2 - BOX.T_top)
    assert np.isclose(ci[:, 2].min(), -BOX.H / 2 + BOX.T_bot)


def test_box_rejects_solid():
    with pytest.raises(ValueError):
        BoxSolidMesher(BoxSection(9, 0.1, 0.06, 0.06, 0.06, 0.1), 1.0)


# ── I-profile ────────────────────────────────────────────────────────────────

def _i_perimeter(s: ISection) -> float:
    return (2 * s.bf_top + 2 * s.tf_top + 2 * s.bf_bot + 2 * s.tf_bot
            + 2 * s.web_height - 2 * s.tw)


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_ibeam_volume_and_area(n_layers):
    m = IProfileSolidMesher(ISEC, L, n_top=4, n_side=3, n_bottom=2, n_length=3,
                            n_layers=n_layers).build()
    assert m.section_kind == "IPROFILE"
    assert m.volume == pytest.approx(ISEC.cross_section_area * L, rel=REL)
    a = m.face_areas()
    assert a[m.outer_face_indices].sum() == pytest.approx(_i_perimeter(ISEC) * L, rel=REL)
    assert len(m.inner_face_indices) == 0
    assert a[m.face_indices(FACE_END)].sum() == pytest.approx(
        2 * ISEC.cross_section_area, rel=REL)


def test_ibeam_placement():
    m = IProfileSolidMesher(ISEC, L).build()
    y, z = m.nodes[:, 1], m.nodes[:, 2]
    assert z.max() == pytest.approx(ISEC.h / 2) and z.min() == pytest.approx(-ISEC.h / 2)
    assert y.max() == pytest.approx(ISEC.bf_top / 2)
    web = (z > -ISEC.h / 2 + ISEC.tf_bot + 1e-9) & (z < ISEC.h / 2 - ISEC.tf_top - 1e-9)
    assert np.allclose(np.abs(y[web]).max(), ISEC.tw / 2)


def test_ibeam_connected():
    """Web and flanges share nodes (single connected component)."""
    m = IProfileSolidMesher(ISEC, L, n_top=3, n_side=2, n_bottom=5, n_layers=2).build()
    parent = np.arange(m.n_nodes)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for h in m.hexes:
        r0 = find(h[0])
        for n in h[1:]:
            parent[find(n)] = r0
    assert len({find(i) for i in range(m.n_nodes)}) == 1


# ── PIPE ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("c_circ,n_layers", [(3, 1), (8, 2), (16, 3), (33, 2)])
def test_pipe_polygon_exact(c_circ, n_layers):
    m = PipeSolidMesher(PIPE, L, c_circ=c_circ, n_length=2, n_layers=n_layers).build()
    ro, ri = PIPE.outer_diameter / 2, PIPE.outer_diameter / 2 - PIPE.thickness
    k = 0.5 * c_circ * math.sin(2 * math.pi / c_circ)
    assert m.volume == pytest.approx(k * (ro ** 2 - ri ** 2) * L, rel=REL)
    side = 2 * math.sin(math.pi / c_circ)
    a = m.face_areas()
    assert a[m.outer_face_indices].sum() == pytest.approx(c_circ * side * ro * L, rel=REL)
    assert a[m.inner_face_indices].sum() == pytest.approx(c_circ * side * ri * L, rel=REL)


def test_pipe_nodes_on_circles():
    nl = 3
    m = PipeSolidMesher(PIPE, L, c_circ=12, n_layers=nl).build()
    r = np.hypot(m.nodes[:, 1], m.nodes[:, 2])
    ro = PIPE.outer_diameter / 2
    ri = ro - PIPE.thickness
    expected = ro + (ri - ro) * np.arange(nl + 1) / nl
    assert np.all(np.min(np.abs(r[:, None] - expected[None, :]), axis=1) < 1e-14)
    out_nodes = np.unique(m.faces[m.outer_face_indices])
    in_nodes = m.inner_node_indices
    assert np.allclose(r[out_nodes], ro, atol=1e-14)
    assert np.allclose(r[in_nodes], ri, atol=1e-14)
    assert m.n_nodes == 12 * (nl + 1) * (4 + 1)          # periodic seam shared


def test_pipe_volume_converges():
    ro, ri = PIPE.outer_diameter / 2, PIPE.outer_diameter / 2 - PIPE.thickness
    exact = math.pi * (ro ** 2 - ri ** 2) * L
    errs = [abs(PipeSolidMesher(PIPE, L, c_circ=c).build().volume - exact) / exact
            for c in (8, 16, 32, 64)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
    assert errs[-1] < 2e-3
    assert errs[-2] / errs[-1] == pytest.approx(4.0, rel=0.05)   # O(1/c²)


# ── PLATE ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_plate_volume_and_area(n_layers):
    corners = _planar_corners()
    m = PlateSolidMesher(PLATE, corners, mesh_12=3, mesh_14=5, n_layers=n_layers).build()
    assert m.section_kind == "PLATE"
    A = _poly_area(corners)
    assert m.volume == pytest.approx(A * PLATE.thickness, rel=REL)
    a = m.face_areas()
    assert a[m.outer_face_indices].sum() == pytest.approx(2 * A, rel=REL)
    assert len(m.inner_face_indices) == 0
    perim = sum(np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4))
    assert a[m.face_indices(FACE_END)].sum() == pytest.approx(
        perim * PLATE.thickness, rel=REL)


def test_plate_offsets_along_normal():
    corners = _planar_corners()
    mesher = PlateSolidMesher(PLATE, corners, n_layers=2)
    m = mesher.build()
    n = mesher.normal
    d = (m.nodes - corners[0]) @ n
    assert np.allclose(np.unique(np.round(d, 12)), [-PLATE.thickness / 2, 0.0,
                                                     PLATE.thickness / 2])
    fn = m.face_normals()[m.outer_face_indices]
    assert np.allclose(np.abs(fn @ n), 1.0)


def test_plate_warped_builds():
    m = PlateSolidMesher(PLATE, PLATE_CORNERS, n_layers=2).build()
    assert np.all(m.hex_volumes() > 0)


# ── BLOCK ────────────────────────────────────────────────────────────────────

def test_block_structure():
    nx, ny, nz = 4, 3, 2
    mesher = BlockSolidMesher(1.0, 0.5, 0.25, nx, ny, nz, origin=(1.0, 2.0, 3.0))
    m = mesher.build()
    assert m.volume == pytest.approx(0.125, rel=REL)
    assert np.all(m.face_group == FACE_OUTER)
    assert m.face_areas().sum() == pytest.approx(2 * (0.5 + 0.25 + 0.125), rel=REL)
    for (i, j, k) in [(0, 0, 0), (4, 3, 2), (2, 1, 1), (3, 0, 2)]:
        nid = i + (nx + 1) * (j + (ny + 1) * k)
        assert nid == mesher.node_id(i, j, k)
        assert np.allclose(m.nodes[nid], [1 + i * 0.25, 2 + j * 0.5 / 3, 3 + k * 0.125])
    assert np.allclose(m.hex_volumes(), 0.125 / (nx * ny * nz))


# ── extrude_section with legacy 2-D mesher ───────────────────────────────────

def test_extrude_legacy_box():
    sec = BoxMesher(BOX, elem_size=0.04, n_layers=2).build()
    m = extrude_section(sec, L, 3)
    assert m.volume == pytest.approx(BOX.cross_section_area * L, rel=REL)
    a = m.face_areas()
    assert a[m.outer_face_indices].sum() == pytest.approx(2 * (BOX.W + BOX.H) * L, rel=REL)
    inner = 2 * (BOX.inner_width + BOX.inner_height) * L
    assert a[m.inner_face_indices].sum() == pytest.approx(inner, rel=REL)


def test_extrude_flips_cw_quads():
    sec = BoxMesher(BOX, elem_size=0.05).build()
    sec.quads = sec.quads[:, ::-1].copy()
    m = extrude_section(sec, L, 2)
    assert np.all(m.hex_volumes() > 0)
    assert m.volume == pytest.approx(BOX.cross_section_area * L, rel=REL)


@pytest.mark.parametrize("bad", [dict(length=0.0, n_length=2), dict(length=1.0, n_length=0)])
def test_extrude_rejects_bad_args(bad):
    with pytest.raises(ValueError):
        extrude_section(BoxMesher(BOX).build(), **bad)
