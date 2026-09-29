"""Tests for new geometry types: PipeSection, ISection, ShellElement."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.node import Node
from fahts.core.model.section import PipeSection, ISection, PlateSection
from fahts.renderer.beam_geometry import (
    build_pipe_mesh,
    build_isection_mesh,
    build_shell_surface_mesh,
    build_model_mesh,
    build_centreline_mesh,
    _pipe_ring_2d,
    _ihprofil_corners_2d,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_nodes(p1=(0, 0, 0), p2=(2, 0, 0)):
    return {
        1: Node(nid=1, x=p1[0], y=p1[1], z=p1[2]),
        2: Node(nid=2, x=p2[0], y=p2[1], z=p2[2]),
    }


def _make_pipe_elem(p1=(0, 0, 0), p2=(2, 0, 0)):
    p1a, p2a = np.array(p1, float), np.array(p2, float)
    d = p2a - p1a
    return BeamElement(
        eid=10, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
        length=float(np.linalg.norm(d)),
        direction=d / np.linalg.norm(d),
        local_z=np.array([0., 0., 1.]),
    )


def _make_isect_elem(p1=(0, 0, 0), p2=(3, 0, 0)):
    p1a, p2a = np.array(p1, float), np.array(p2, float)
    d = p2a - p1a
    return BeamElement(
        eid=20, n1=1, n2=2, mat_id=1, geom_id=2, lcoor_id=1,
        length=float(np.linalg.norm(d)),
        direction=d / np.linalg.norm(d),
        local_z=np.array([0., 0., 1.]),
    )


# ── PipeSection dataclass ─────────────────────────────────────────────────────

class TestPipeSection:
    def test_inner_diameter(self):
        p = PipeSection(sid=1, outer_diameter=0.356, thickness=0.019)
        assert abs(p.inner_diameter - (0.356 - 2 * 0.019)) < 1e-9

    def test_radii(self):
        p = PipeSection(sid=1, outer_diameter=0.200, thickness=0.010)
        assert p.outer_radius == pytest.approx(0.100)
        assert p.inner_radius == pytest.approx(0.090)

    def test_area_annulus(self):
        p = PipeSection(sid=1, outer_diameter=0.200, thickness=0.010)
        expected = np.pi / 4 * (0.200 ** 2 - 0.180 ** 2)
        assert p.cross_section_area == pytest.approx(expected)

    def test_section_factor(self):
        p = PipeSection(sid=1, outer_diameter=0.200, thickness=0.010)
        assert p.section_factor_Am_V > 0

    def test_str(self):
        p = PipeSection(sid=3, outer_diameter=0.508, thickness=0.025)
        assert "PIPE" in str(p) and "508" in str(p)


# ── ISection dataclass ────────────────────────────────────────────────────────

class TestISection:
    def test_web_height(self):
        s = ISection(sid=4, h=0.850, tw=0.015, bf_top=0.4, tf_top=0.03, bf_bot=0.4, tf_bot=0.03)
        assert s.web_height == pytest.approx(0.850 - 0.03 - 0.03)

    def test_area(self):
        s = ISection(sid=4, h=0.850, tw=0.015, bf_top=0.4, tf_top=0.03, bf_bot=0.4, tf_bot=0.03)
        expected = 0.4 * 0.03 + 0.015 * s.web_height + 0.4 * 0.03
        assert s.cross_section_area == pytest.approx(expected)

    def test_section_factor_positive(self):
        s = ISection(sid=4, h=0.850, tw=0.015, bf_top=0.4, tf_top=0.03, bf_bot=0.4, tf_bot=0.03)
        assert s.section_factor_Am_V > 0

    def test_str(self):
        s = ISection(sid=4, h=0.850, tw=0.015, bf_top=0.4, tf_top=0.03, bf_bot=0.4, tf_bot=0.03)
        assert "IHPROFIL" in str(s)


# ── PlateSection dataclass ────────────────────────────────────────────────────

class TestPlateSection:
    def test_thickness(self):
        p = PlateSection(sid=6, thickness=0.008)
        assert p.thickness == 0.008

    def test_str(self):
        p = PlateSection(sid=6, thickness=0.008)
        assert "PLTHICK" in str(p)


# ── _pipe_ring_2d ─────────────────────────────────────────────────────────────

class TestPipeRing2d:
    def test_shape(self):
        ring = _pipe_ring_2d(16, 0.1)
        assert ring.shape == (16, 2)

    def test_radius_correct(self):
        r = 0.178
        ring = _pipe_ring_2d(16, r)
        radii = np.sqrt(ring[:, 0] ** 2 + ring[:, 1] ** 2)
        assert np.allclose(radii, r)

    def test_evenly_spaced(self):
        ring = _pipe_ring_2d(8, 1.0)
        # Each point should be 45 degrees apart
        angles = np.arctan2(ring[:, 1], ring[:, 0])
        diffs = np.diff(angles) % (2 * np.pi)
        assert np.allclose(diffs, 2 * np.pi / 8, atol=1e-10)


# ── _ihprofil_corners_2d ──────────────────────────────────────────────────────

class TestIHProfilCorners2d:
    def setup_method(self):
        self.s = ISection(sid=4, h=0.850, tw=0.015,
                          bf_top=0.4, tf_top=0.03,
                          bf_bot=0.4, tf_bot=0.03)

    def test_shape(self):
        corners = _ihprofil_corners_2d(self.s)
        assert corners.shape == (12, 2)

    def test_top_corners_at_h2(self):
        # col 1 = lz (height/UNITVEC direction); corners 0 and 1 are at top (lz = +h/2)
        corners = _ihprofil_corners_2d(self.s)
        assert corners[0, 1] == pytest.approx(0.425)   # h/2 in lz column
        assert corners[1, 1] == pytest.approx(0.425)

    def test_bottom_corners_at_minus_h2(self):
        # corners 6 and 7 are at bottom (lz = -h/2)
        corners = _ihprofil_corners_2d(self.s)
        assert corners[6, 1] == pytest.approx(-0.425)
        assert corners[7, 1] == pytest.approx(-0.425)

    def test_flange_half_width(self):
        # col 0 = ly (flange/width direction); corners 0 and 1 are ±bf_top/2
        corners = _ihprofil_corners_2d(self.s)
        assert abs(corners[0, 0]) == pytest.approx(0.2)   # bf_top/2 in ly column
        assert abs(corners[1, 0]) == pytest.approx(0.2)

    def test_web_half_width(self):
        # corner 3 is the web top-right: col 0 = tw/2
        corners = _ihprofil_corners_2d(self.s)
        assert abs(corners[3, 0]) == pytest.approx(0.0075)  # tw/2 in ly column


# ── build_pipe_mesh ───────────────────────────────────────────────────────────

class TestBuildPipeMesh:
    def test_returns_polydata(self):
        sec = PipeSection(sid=1, outer_diameter=0.356, thickness=0.019)
        elem = _make_pipe_elem()
        nodes = _make_nodes()
        mesh = build_pipe_mesh(elem, sec, nodes)
        assert isinstance(mesh, pv.PolyData)

    def test_cell_count(self):
        # Outer lateral quads only (USFOS-style, no inner wall or end caps)
        n = 16
        sec = PipeSection(sid=1, outer_diameter=0.356, thickness=0.019)
        elem = _make_pipe_elem()
        mesh = build_pipe_mesh(elem, sec, _make_nodes(), n_sides=n)
        assert mesh.n_cells == n

    def test_element_id_in_cell_data(self):
        sec = PipeSection(sid=1, outer_diameter=0.356, thickness=0.019)
        elem = _make_pipe_elem()
        mesh = build_pipe_mesh(elem, sec, _make_nodes())
        assert "element_id" in mesh.cell_data
        assert all(mesh.cell_data["element_id"] == elem.eid)

    def test_mesh_has_correct_points(self):
        # 2 rings (outer × end-0/end-1) × n_sides (USFOS-style)
        sec = PipeSection(sid=1, outer_diameter=0.200, thickness=0.010)
        elem = _make_pipe_elem()
        mesh = build_pipe_mesh(elem, sec, _make_nodes(), n_sides=8)
        assert mesh.n_points == 2 * 8

    def test_different_radii(self):
        sec_large = PipeSection(sid=1, outer_diameter=0.610, thickness=0.025)
        sec_small = PipeSection(sid=2, outer_diameter=0.219, thickness=0.010)
        elem = _make_pipe_elem()
        nodes = _make_nodes()
        m1 = build_pipe_mesh(elem, sec_large, nodes)
        m2 = build_pipe_mesh(elem, sec_small, nodes)
        # Larger pipe has larger bounding box
        bb1 = m1.bounds
        bb2 = m2.bounds
        span1 = max(bb1[3] - bb1[2], bb1[5] - bb1[4])
        span2 = max(bb2[3] - bb2[2], bb2[5] - bb2[4])
        assert span1 > span2


# ── build_isection_mesh ───────────────────────────────────────────────────────

class TestBuildIsectionMesh:
    def test_returns_polydata(self):
        sec = ISection(sid=4, h=0.850, tw=0.015,
                       bf_top=0.4, tf_top=0.03,
                       bf_bot=0.4, tf_bot=0.03)
        elem = _make_isect_elem()
        mesh = build_isection_mesh(elem, sec, _make_nodes(p2=(3, 0, 0)))
        assert isinstance(mesh, pv.PolyData)

    def test_cell_count(self):
        # 5 panels: top flange (×2 halves), web, bottom flange (×2 halves)
        sec = ISection(sid=4, h=0.850, tw=0.015,
                       bf_top=0.4, tf_top=0.03,
                       bf_bot=0.4, tf_bot=0.03)
        elem = _make_isect_elem()
        mesh = build_isection_mesh(elem, sec, _make_nodes(p2=(3, 0, 0)))
        assert mesh.n_cells == 5

    def test_element_id(self):
        sec = ISection(sid=4, h=0.850, tw=0.015,
                       bf_top=0.4, tf_top=0.03,
                       bf_bot=0.4, tf_bot=0.03)
        elem = _make_isect_elem()
        mesh = build_isection_mesh(elem, sec, _make_nodes(p2=(3, 0, 0)))
        assert all(mesh.cell_data["element_id"] == elem.eid)

    def test_point_count(self):
        sec = ISection(sid=4, h=0.850, tw=0.015,
                       bf_top=0.4, tf_top=0.03,
                       bf_bot=0.4, tf_bot=0.03)
        elem = _make_isect_elem()
        mesh = build_isection_mesh(elem, sec, _make_nodes(p2=(3, 0, 0)))
        assert mesh.n_points == 20  # 5 panels × 4 corners


# ── build_shell_surface_mesh ──────────────────────────────────────────────────

class TestBuildShellSurfaceMesh:
    def test_empty_shells_returns_empty(self):
        mesh = build_shell_surface_mesh({}, {}, np.zeros(3))
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_cells == 0

    def test_one_quad_shell(self):
        nodes = {
            1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0),
            3: Node(3, 1, 1, 0), 4: Node(4, 0, 1, 0),
        }
        shells = {
            10: ShellElement(eid=10, nodes=(1, 2, 3, 4), mat_id=1, geom_id=6)
        }
        mesh = build_shell_surface_mesh(shells, nodes, np.zeros(3))
        assert mesh.n_cells == 1
        assert mesh.n_points == 4

    def test_one_tri_shell(self):
        nodes = {
            1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0), 3: Node(3, 0, 1, 0),
        }
        shells = {
            11: ShellElement(eid=11, nodes=(1, 2, 3), mat_id=1, geom_id=6)
        }
        mesh = build_shell_surface_mesh(shells, nodes, np.zeros(3))
        assert mesh.n_cells == 1
        assert mesh.n_points == 3

    def test_element_id_in_cell_data(self):
        nodes = {
            1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0),
            3: Node(3, 1, 1, 0), 4: Node(4, 0, 1, 0),
        }
        shells = {
            99: ShellElement(eid=99, nodes=(1, 2, 3, 4), mat_id=1, geom_id=6)
        }
        mesh = build_shell_surface_mesh(shells, nodes, np.zeros(3))
        assert mesh.cell_data["element_id"][0] == 99

    def test_centroid_shift_applied(self):
        nodes = {1: Node(1, 10, 10, 10), 2: Node(2, 11, 10, 10),
                 3: Node(3, 11, 11, 10), 4: Node(4, 10, 11, 10)}
        shells = {1: ShellElement(eid=1, nodes=(1, 2, 3, 4), mat_id=1, geom_id=6)}
        centroid = np.array([10.0, 10.0, 10.0])
        mesh = build_shell_surface_mesh(shells, nodes, centroid)
        # Points should be near origin
        assert np.allclose(mesh.points[:, 2], 0.0)

    def test_multiple_shells(self):
        nodes = {
            1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0),
            3: Node(3, 1, 1, 0), 4: Node(4, 0, 1, 0),
            5: Node(5, 2, 0, 0), 6: Node(6, 2, 1, 0),
        }
        shells = {
            10: ShellElement(eid=10, nodes=(1, 2, 3, 4), mat_id=1, geom_id=6),
            11: ShellElement(eid=11, nodes=(2, 5, 6, 3), mat_id=1, geom_id=6),
        }
        mesh = build_shell_surface_mesh(shells, nodes, np.zeros(3))
        assert mesh.n_cells == 2


# ── build_model_mesh with mixed types ────────────────────────────────────────

class TestBuildModelMeshMixed:
    @pytest.fixture
    def real_model_t1(self):
        p = Path(__file__).parents[1] / "examples" / "models" / "model_t1.fem"
        if not p.exists():
            pytest.skip("model_t1.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        return read_usfos_fem(p)

    def test_mesh_has_cells(self, real_model_t1):
        mesh = build_model_mesh(real_model_t1)
        assert mesh.n_cells > 0

    def test_mesh_has_element_id(self, real_model_t1):
        mesh = build_model_mesh(real_model_t1)
        assert "element_id" in mesh.cell_data

    def test_element_ids_cover_beams_and_shells(self, real_model_t1):
        mesh = build_model_mesh(real_model_t1)
        mesh_eids = set(int(e) for e in mesh.cell_data["element_id"])
        # Beam element IDs should appear
        beam_eid = next(iter(real_model_t1.elements))
        assert beam_eid in mesh_eids
        # Shell element IDs should appear
        shell_eid = next(iter(real_model_t1.shell_elements))
        assert shell_eid in mesh_eids

    def test_wire_mesh_has_cells(self, real_model_t1):
        wire = build_centreline_mesh(real_model_t1)
        assert wire.n_cells > 0

    def test_scene_manager_loads_t1(self, real_model_t1):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model_t1)
        assert scene._full_solid_mesh.n_cells > 0
        assert "element_id" in scene._full_solid_mesh.cell_data
        scene.close()

    def test_group_colouring_includes_shells(self, real_model_t1):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model_t1)
        # Shell elements in groups should appear in elem_to_group
        shell_eid = next(iter(real_model_t1.shell_elements))
        assert shell_eid in scene._elem_to_group
        scene.close()


# ── Reader smoke test ─────────────────────────────────────────────────────────

class TestReaderT1:
    @pytest.fixture
    def model(self):
        p = Path(__file__).parents[1] / "examples" / "models" / "model_t1.fem"
        if not p.exists():
            pytest.skip("model_t1.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        return read_usfos_fem(p)

    def test_has_pipe_sections(self, model):
        from fahts.core.model.section import PipeSection
        pipes = [s for s in model.sections.values() if isinstance(s, PipeSection)]
        assert len(pipes) > 0

    def test_has_isections(self, model):
        from fahts.core.model.section import ISection
        isects = [s for s in model.sections.values() if isinstance(s, ISection)]
        assert len(isects) > 0

    def test_has_plate_sections(self, model):
        from fahts.core.model.section import PlateSection
        plates = [s for s in model.sections.values() if isinstance(s, PlateSection)]
        assert len(plates) > 0

    def test_has_shell_elements(self, model):
        assert len(model.shell_elements) > 0

    def test_quad_shell_has_4_nodes(self, model):
        quads = [s for s in model.shell_elements.values() if len(s.nodes) == 4]
        assert len(quads) > 0

    def test_tri_shell_has_3_nodes(self, model):
        tris = [s for s in model.shell_elements.values() if len(s.nodes) == 3]
        assert len(tris) > 0

    def test_all_shell_nodes_exist(self, model):
        for se in model.shell_elements.values():
            for nid in se.nodes:
                assert nid in model.nodes, f"Shell {se.eid} refs missing node {nid}"

    def test_model_file_still_works(self):
        p = Path(__file__).parents[1] / "examples" / "models" / "model_file.fem"
        if not p.exists():
            pytest.skip("model_file.fem not found")
        from fahts.core.io.usfos_reader import read_usfos_fem
        model = read_usfos_fem(p)
        assert len(model.elements) > 0
        assert len(model.shell_elements) == 0
        mesh = build_model_mesh(model)
        assert mesh.n_cells > 0

    def test_eccentricity_parsed(self, model):
        """At least some beams in model_t1.fem should have non-zero eccentricity."""
        ecc_beams = [e for e in model.elements.values()
                     if e.ecc1 is not None or e.ecc2 is not None]
        assert len(ecc_beams) > 0

    def test_eccentricity_is_ndarray(self, model):
        """Eccentricity vectors must be numpy arrays."""
        for elem in model.elements.values():
            if elem.ecc1 is not None:
                assert isinstance(elem.ecc1, np.ndarray), f"ecc1 on elem {elem.eid} is not ndarray"
                assert elem.ecc1.shape == (3,)
            if elem.ecc2 is not None:
                assert isinstance(elem.ecc2, np.ndarray)
                assert elem.ecc2.shape == (3,)

    def test_zero_eccent_id_gives_none(self, model):
        """BEAM cards with ecc_id=0 must result in ecc=None, not zero vector."""
        beams_with_ecc1_none = [e for e in model.elements.values() if e.ecc1 is None]
        assert len(beams_with_ecc1_none) > 0

    def test_eccentricity_shifts_mesh(self, model):
        """Beam with eccentricity must render at different position than without."""
        ecc_elem = next(
            (e for e in model.elements.values()
             if e.ecc2 is not None and abs(np.linalg.norm(e.ecc2)) > 0.01),
            None,
        )
        if ecc_elem is None:
            pytest.skip("No element with substantial ecc2 found")
        from fahts.core.model.section import ISection, BoxSection, PipeSection
        section = model.sections.get(ecc_elem.geom_id)
        if section is None:
            pytest.skip("Section not found")

        # Build mesh with eccentricity (normal call)
        if isinstance(section, ISection):
            mesh_ecc = build_isection_mesh(ecc_elem, section, model.nodes)
        elif isinstance(section, BoxSection):
            mesh_ecc = build_beam_mesh(ecc_elem, section, model.nodes)
        else:
            pytest.skip("Pipe section — centroid check not relevant")

        # Build mesh without eccentricity by temporarily zeroing it
        ecc2_saved = ecc_elem.ecc2
        ecc_elem.ecc2 = None
        if isinstance(section, ISection):
            mesh_no_ecc = build_isection_mesh(ecc_elem, section, model.nodes)
        else:
            mesh_no_ecc = build_beam_mesh(ecc_elem, section, model.nodes)
        ecc_elem.ecc2 = ecc2_saved  # restore

        # The two meshes must differ
        assert not np.allclose(mesh_ecc.points, mesh_no_ecc.points, atol=1e-6)


# ── Section orientation sanity checks ────────────────────────────────────────

class TestSectionOrientation:
    """Verify H/W are mapped to correct local-z/local-y directions."""

    def test_box_H_in_lz_direction(self):
        """For a horizontal X-beam with UNITVEC=(0,0,1), H should extend in Z."""
        from fahts.renderer.beam_geometry import build_beam_mesh
        from fahts.core.model.section import BoxSection
        nodes = {
            1: Node(1, 0.0, 0.0, 0.0),
            2: Node(2, 1.0, 0.0, 0.0),
        }
        direction = np.array([1.0, 0.0, 0.0])
        section = BoxSection(sid=1, H=0.60, W=0.20, T_side=0.01, T_bot=0.01, T_top=0.01)
        elem = BeamElement(
            eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
            length=1.0, direction=direction,
            local_z=np.array([0.0, 0.0, 1.0]),  # UNITVEC = global Z
        )
        mesh = build_beam_mesh(elem, section, nodes)
        bb = mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
        # H=0.60 should be in Z (lz = UNITVEC = global Z)
        assert abs((bb[5] - bb[4]) - 0.60) < 1e-9, f"Z span {bb[5]-bb[4]:.3f} != H=0.60"
        # W=0.20 should be in Y (ly = global -Y for this beam)
        assert abs((bb[3] - bb[2]) - 0.20) < 1e-9, f"Y span {bb[3]-bb[2]:.3f} != W=0.20"

    def test_isection_height_in_lz_direction(self):
        """For a horizontal X-beam with UNITVEC=(0,0,1), h should extend in Z."""
        nodes = {
            1: Node(1, 0.0, 0.0, 0.0),
            2: Node(2, 2.0, 0.0, 0.0),
        }
        direction = np.array([1.0, 0.0, 0.0])
        section = ISection(sid=4, h=0.850, tw=0.015,
                           bf_top=0.4, tf_top=0.03, bf_bot=0.4, tf_bot=0.03)
        elem = BeamElement(
            eid=4, n1=1, n2=2, mat_id=1, geom_id=4, lcoor_id=1,
            length=2.0, direction=direction,
            local_z=np.array([0.0, 0.0, 1.0]),
        )
        mesh = build_isection_mesh(elem, section, nodes)
        bb = mesh.bounds
        # h=0.850 should be in Z (lz = UNITVEC = global Z)
        assert abs((bb[5] - bb[4]) - 0.850) < 1e-9, f"Z span {bb[5]-bb[4]:.3f} != h=0.850"
        # bf_top=0.4 should be in Y (ly direction)
        assert abs((bb[3] - bb[2]) - 0.4) < 1e-9, f"Y span {bb[3]-bb[2]:.3f} != bf=0.4"
