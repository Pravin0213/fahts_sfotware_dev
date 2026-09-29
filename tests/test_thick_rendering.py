"""Real-thickness rendering (View → Show Wall Thickness): geometry builders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.model.element import BeamElement
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, ISection, PipeSection
from fahts.renderer.beam_geometry import build_model_mesh, build_thick_member_mesh

ROOT = Path(__file__).resolve().parents[1]


def _beam() -> tuple[BeamElement, dict[int, Node]]:
    nodes = {1: Node(nid=1, x=0.0, y=0.0, z=0.0), 2: Node(nid=2, x=2.0, y=0.0, z=0.0)}
    elem = BeamElement(eid=7, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=0, length=2.0,
                       direction=np.array([1.0, 0, 0]), local_z=np.array([0.0, 0, 1]))
    return elem, nodes


@pytest.mark.parametrize("section, n_faces", [
    (PipeSection(sid=1, outer_diameter=3.0, thickness=0.15), 4 * 16),   # outer, inner, 2 ends
    # orthogonal ring, 1 layer: outer 4+8, inner 4, two end caps of 3×3−1 cells
    (BoxSection(sid=1, H=0.3, W=0.2, T_side=0.01, T_bot=0.012, T_top=0.012), 12 + 4 + 16),
    (ISection(sid=1, h=0.3, tw=0.0071, bf_top=0.15, tf_top=0.0107, bf_bot=0.15,
              tf_bot=0.0107), None),
])
def test_member_is_closed_solid_with_thickness(section, n_faces):
    elem, nodes = _beam()
    m = build_thick_member_mesh(elem, section, nodes, n_pipe_sides=16)
    if n_faces is not None:
        assert m.n_cells == n_faces
    assert np.all(m["element_id"] == 7)
    # watertight surface → encloses a positive volume ≈ steel volume × length
    edges = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=False)
    assert edges.n_cells == 0
    vol = m.triangulate().volume
    if isinstance(section, PipeSection):
        ro, ri = 1.5, 1.35
        n = 16   # polygonal annulus area
        exact = 0.5 * n * np.sin(2 * np.pi / n) * (ro ** 2 - ri ** 2) * 2.0
        assert vol == pytest.approx(exact, rel=1e-6)
    elif isinstance(section, BoxSection):
        assert vol == pytest.approx(section.cross_section_area * 2.0, rel=1e-6)


def test_tank_thick_vs_thin():
    model = read_usfos_fem(ROOT / "examples" / "models" / "tank_horizontal.fem")
    thin = build_model_mesh(model)
    thick = build_model_mesh(model, show_thickness=True)
    np.testing.assert_allclose(thin.bounds, thick.bounds, atol=1e-9)
    assert set(np.unique(thin["element_id"])) == set(np.unique(thick["element_id"]))
    assert thick.n_cells == 4 * thin.n_cells          # outer + inner + two end rings


def test_mixed_model_thick_keeps_all_elements():
    model = read_usfos_fem(ROOT / "examples" / "models" / "model_t1.fem")
    thin = build_model_mesh(model)
    thick = build_model_mesh(model, show_thickness=True)
    assert set(np.unique(thin["element_id"])) == set(np.unique(thick["element_id"]))
    np.testing.assert_allclose(thin.bounds, thick.bounds, atol=0.05)
