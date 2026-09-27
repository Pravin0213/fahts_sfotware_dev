"""
Free plate edges and free member ends receive heat on their thickness / end faces
(regression 2026-09-27: an 80 cm plate level with a RadiationBall was skipped entirely —
its edges were adiabatic FACE_END and its big faces edge-on to the ball; unlit elements
were also dropped although they conduct heat in 3-D).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.solid_mesh import FACE_END, FACE_OUTER, PlateSolidMesher
from fahts.core.heat.solver.analysis_runner import _model_topology, run_analysis
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.model.section import PlateSection
from fahts.core.results.analysis_config import AnalysisConfig

ROOT = Path(__file__).resolve().parents[1]
SQ = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float)


@pytest.mark.parametrize("flags", [(False,) * 4, (True,) * 4, (True, False, False, False),
                                   (False, True, False, True)])
def test_plate_exposed_edges_area(flags):
    t = 0.3
    m = PlateSolidMesher(PlateSection(sid=1, thickness=t), SQ, mesh_12=4, mesh_14=2,
                         n_layers=2, exposed_edges=flags).build()
    lengths = (2.0, 1.0, 2.0, 1.0)                 # edges 12, 23, 34, 41
    a = m.face_areas()
    outer = a[m.face_group == FACE_OUTER].sum()
    end = a[m.face_group == FACE_END].sum()
    edge_exp = sum(L * t for L, f in zip(lengths, flags) if f)
    assert outer == pytest.approx(2 * 2.0 + edge_exp)
    assert end == pytest.approx(sum(L * t for L in lengths) - edge_exp)


def _thick(tmp_path) -> Path:
    src = (ROOT / "parallel_plates_unequal.fem").read_text()
    p = tmp_path / "thick.fem"
    p.write_text(src.replace(" PLTHICK          1     0.02", " PLTHICK          1     0.8"))
    return p


def test_topology_free_edges_and_ends():
    m = read_usfos_fem(ROOT / "parallel_plates_unequal.fem")
    node_use, edge_use = _model_topology(m)
    counts = np.array(list(edge_use.values()))
    # 4×4 + 6×6 grids: boundary edges used once, interior edges twice
    assert (counts == 1).sum() == 4 * 4 + 4 * 6
    assert set(counts) <= {1, 2}
    t = read_usfos_fem(ROOT / "tank_horizontal.fem")
    nu, _ = _model_topology(t)
    assert nu[1] == 1 and nu[13] == 1 and all(nu[k] == 2 for k in range(2, 13))


def test_thick_plate_heated_through_edge(tmp_path):
    m = read_usfos_fem(_thick(tmp_path))
    ball = RadiationBall(name="B", center=np.array([4.0, 1.0, 0.0]), radius=0.8,
                         flux=350e3, active=True)
    r = run_analysis(m, [ball], AnalysisConfig(t_end=3600.0, dt=60.0, output_dt=3600.0))
    assert len(r.element_ids) == len(m.shell_elements)          # nothing skipped
    cen = {e: np.mean([m.nodes[n].xyz for n in m.shell_elements[e].nodes], axis=0)
           for e in r.element_ids}
    small = [e for e in r.element_ids if cen[e][2] < 0.5]
    T = dict(zip(r.element_ids, r.T_centroid[-1]))
    near = max(small, key=lambda e: cen[e][0])
    far = min(small, key=lambda e: cen[e][0])
    assert T[near] > T[far] + 10.0 > 30.0                        # heat enters at the edge


def test_tank_open_ends_exposed():
    """Single line of PIPE members: the two free ends expose their annular end caps."""
    m = read_usfos_fem(ROOT / "tank_horizontal.fem")
    from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone

    zone = FireZone(name="Z", center=np.array([12.0, 0.0, 2.5]), dims=np.array([30.0, 6, 6]),
                    curve=FireCurve(curve_type=FireCurveType.HYDROCARBON), h_conv=50.0,
                    epsilon_fire=1.0, active=True)
    r = run_analysis(m, [zone], AnalysisConfig(t_end=300.0, dt=60.0, output_dt=300.0,
                                               radiation_exchange=False))
    T = dict(zip(r.element_ids, r.T_centroid[-1]))
    assert T[1] > T[6] and T[12] > T[6]      # end members also heat through the end cap
