"""
WP-C integration tests — 3-D Hex8 solid path of run_analysis.

Covers: end-to-end 3-D runs on model_file.fem / model_t1.fem (BOX, I, PIPE, QUADSHEL),
RadiationBall in 3-D, 3-D vs 2-D agreement for a thin-wall BOX, safe DOF merging,
joint-link conduction, CG vs direct agreement, vectorised view factors and the
solid TemperatureField helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.bc.view_factor import geometric_view_factor_double_area
from fahts.core.heat.solid_mesh import BoxSolidMesher, SolidMesh
from fahts.core.heat.solver import solid_integration as si
from fahts.core.heat.solver.analysis_runner import (
    GlobalThermalSolver,
    _beam_local_to_global,
    _compute_global_node_positions,
    run_analysis,
)
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.model.element import BeamElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, ISection, PipeSection
from fahts.core.results.analysis_config import AnalysisConfig
from fahts.core.results.temperature_field import TemperatureField

ROOT = Path(__file__).parents[1]


# ── helpers ───────────────────────────────────────────────────────────────────

def _mat() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _frame_model(sections: dict, elems: list[tuple[int, int, int, int]],
                 coords: dict[int, tuple[float, float, float]]) -> FEMModel:
    """elems: (eid, n1, n2, geom_id); local_z = global Z unless member is vertical."""
    nodes = {nid: Node(nid=nid, x=x, y=y, z=z) for nid, (x, y, z) in coords.items()}
    elements = {}
    for eid, a, b, gid in elems:
        pa, pb = nodes[a].xyz, nodes[b].xyz
        L = float(np.linalg.norm(pb - pa))
        d = (pb - pa) / L
        lz = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        lz = lz - d * np.dot(lz, d)
        lz /= np.linalg.norm(lz)
        elements[eid] = BeamElement(eid=eid, n1=a, n2=b, mat_id=1, geom_id=gid, lcoor_id=0,
                                    length=L, direction=d, local_z=lz)
    return FEMModel(nodes=nodes, elements=elements, sections=sections,
                    materials={1: _mat()}, groups={}, unitvecs={},
                    source_file=Path("synthetic.fem"))


def _zone(center, dims, curve=FireCurveType.HYDROCARBON) -> FireZone:
    return FireZone(name="Z", center=np.asarray(center, dtype=float),
                    dims=np.asarray(dims, dtype=float), curve=FireCurve(curve_type=curve),
                    epsilon_fire=1.0, h_conv=50.0, active=True)


def _single_box_model(t: float = 0.008) -> FEMModel:
    sec = BoxSection(sid=10, H=0.20, W=0.20, T_side=t, T_bot=t, T_top=t)
    return _frame_model({10: sec}, [(101, 1, 2, 10)], {1: (0, 0, 0), 2: (1, 0, 0)})


def _zone_around(model: FEMModel, eids: list[int], margin: float = 1.0) -> FireZone:
    pts = []
    for e in eids:
        if e in model.elements:
            el = model.elements[e]
            pts += [model.nodes[el.n1].xyz, model.nodes[el.n2].xyz]
        else:
            pts += [model.nodes[n].xyz for n in model.shell_elements[e].nodes]
    pts = np.array(pts)
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    return _zone((lo + hi) / 2, hi - lo)


def _check_field(tf: TemperatureField) -> None:
    assert np.all(np.isfinite(tf.T_centroid))
    for eid in tf.element_ids:
        T = tf.T_section[eid]
        assert np.all(np.isfinite(T))
        nodes, faces = tf.nodal_geometry[eid]
        assert T.shape[1] == len(nodes)
        assert faces.ndim == 2 and faces.shape[1] == 4
        assert faces.min() >= 0 and faces.max() < len(nodes)
        Tc = tf.T_centroid[:, tf.element_ids.index(eid)]
        # heating is monotone-ish under a growing hydrocarbon fire
        assert np.all(np.diff(Tc) > -0.5)
        assert Tc[-1] > Tc[0] + 5.0


def _nearest(model: FEMModel, point: np.ndarray, cls, n: int) -> list[int]:
    cand = [e for e, el in model.elements.items()
            if isinstance(model.sections.get(el.geom_id), cls)]
    cand.sort(key=lambda e: np.linalg.norm(model.elements[e].midpoint(model.nodes) - point))
    return cand[:n]


@pytest.fixture(scope="module")
def model_file():
    from fahts.core.io.usfos_reader import read_usfos_fem
    p = ROOT / "model_file.fem"
    if not p.exists():
        pytest.skip("model_file.fem not found")
    return read_usfos_fem(p)


@pytest.fixture(scope="module")
def model_t1():
    from fahts.core.io.usfos_reader import read_usfos_fem
    p = ROOT / "model_t1.fem"
    if not p.exists():
        pytest.skip("model_t1.fem not found")
    return read_usfos_fem(p)


# ── config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        c = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        assert c.solver_dim == "3d" and c.linear_solver == "cg" and c.n_layers_3d == 2
        assert c.c_circ_3d == 12
        assert "solver=3d" in c.summary()

    @pytest.mark.parametrize("kw", [{"solver_dim": "1d"}, {"linear_solver": "gmres"},
                                    {"n_layers_3d": 0}])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            AnalysisConfig(t_end=60, dt=30, output_dt=30, **kw).validate()


# ── end-to-end on real models ─────────────────────────────────────────────────

class TestRealModels:
    def test_model_file_3d(self, model_file):
        c = model_file.centroid()
        eids = _nearest(model_file, c, BoxSection, 10)
        cfg = AnalysisConfig(t_end=300.0, dt=30.0, output_dt=60.0, element_ids=eids)
        tf = run_analysis(model_file, [_zone_around(model_file, eids)], cfg)
        assert sorted(tf.element_ids) == sorted(eids)
        _check_field(tf)

    def test_model_t1_mixed_sections_3d(self, model_t1):
        quads = [e for e, s in model_t1.shell_elements.items() if len(s.nodes) == 4]
        assert quads
        p = np.mean([model_t1.nodes[n].xyz for n in model_t1.shell_elements[quads[0]].nodes],
                    axis=0)
        eids = (_nearest(model_t1, p, PipeSection, 3) + _nearest(model_t1, p, ISection, 3)
                + _nearest(model_t1, p, BoxSection, 1) + quads[:3])
        cfg = AnalysisConfig(t_end=300.0, dt=30.0, output_dt=60.0, element_ids=eids)
        tf = run_analysis(model_t1, [_zone_around(model_t1, eids)], cfg)
        kinds = {type(model_t1.sections[model_t1.elements[e].geom_id]).__name__
                 for e in tf.element_ids if e in model_t1.elements}
        assert {"PipeSection", "ISection", "BoxSection"} <= kinds
        assert any(e in model_t1.shell_elements for e in tf.element_ids)
        _check_field(tf)

    def test_model_file_dof_map_never_collapses_member(self, model_file):
        c = model_file.centroid()
        eids = _nearest(model_file, c, BoxSection, 40)
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30, n_layers_3d=3)
        meshes = {e: si.build_beam_solid_mesh(
            model_file.sections[model_file.elements[e].geom_id],
            model_file.elements[e].length, cfg) for e in eids}
        pos = _compute_global_node_positions(eids, meshes, model_file, {e: True for e in eids})
        gmap, n = si.build_solid_dof_map(eids, meshes, pos)
        total = sum(m.n_nodes for m in meshes.values())
        assert n < total                       # members are joined somewhere
        for e in eids:
            assert len(np.unique(gmap[e])) == meshes[e].n_nodes
            assert gmap[e].max() < n


# ── 3-D physics / coupling checks ─────────────────────────────────────────────

class TestSolidPhysics:
    def test_radiation_ball_3d(self):
        model = _single_box_model()
        ball = RadiationBall(name="B", center=np.array([0.5, 0.0, 1.0]), radius=0.5,
                             flux=50_000.0)
        cfg = AnalysisConfig(t_end=300.0, dt=30.0, output_dt=60.0)
        tf = run_analysis(model, [ball], cfg)
        T = tf.T_section[101][-1]
        assert np.all(np.isfinite(T)) and tf.T_centroid[-1, 0] > 25.0
        # top wall (facing the ball) is hotter than the bottom wall
        nodes = tf.nodal_geometry[101][0]
        top, bot = nodes[:, 2] > 0.09, nodes[:, 2] < -0.09
        assert T[top].mean() > T[bot].mean() + 1.0

    def test_radiation_ball_iprofile_both_web_sides(self):
        sec = ISection(sid=10, h=0.30, tw=0.010, bf_top=0.15, tf_top=0.012,
                       bf_bot=0.15, tf_bot=0.012)
        model = _frame_model({10: sec}, [(101, 1, 2, 10)], {1: (0, 0, 0), 2: (1, 0, 0)})
        # model y axis = local_x × local_z = x × z = −y  → ball on either side must heat
        for y in (+1.0, -1.0):
            ball = RadiationBall(name="B", center=np.array([0.5, y, 0.0]), radius=0.2,
                                 flux=100_000.0)
            tf = run_analysis(model, [ball], AnalysisConfig(t_end=120, dt=30, output_dt=60))
            assert tf.T_centroid[-1, 0] > 21.0

    def test_point_and_line_sources_3d(self):
        from fahts.core.heat.sources.concentrated_source import ConcentratedSource
        from fahts.core.heat.sources.line_source import LineSource
        model = _single_box_model()
        srcs = [ConcentratedSource(name="C", center=np.array([0.5, 0.0, 0.6]), power=2e5),
                LineSource(name="L", start=np.array([0.0, 0.0, -0.6]),
                           end=np.array([1.0, 0.0, -0.6]), power=2e5)]
        tf = run_analysis(model, srcs, AnalysisConfig(t_end=120, dt=30, output_dt=60))
        T = tf.T_section[101][-1]
        nodes = tf.nodal_geometry[101][0]
        side = np.abs(nodes[:, 1]) > 0.099
        assert np.all(np.isfinite(T)) and tf.T_centroid[-1, 0] > 20.5
        assert T[nodes[:, 2] > 0.099].mean() > T[side].mean()

    def test_3d_vs_2d_thin_box_mean(self):
        model = _single_box_model(t=0.006)
        zone = _zone([0.5, 0, 0], [3, 3, 3])
        res = {}
        for sd in ("2d", "3d"):
            cfg = AnalysisConfig(t_end=900.0, dt=30.0, output_dt=300.0, solver_dim=sd)
            res[sd] = run_analysis(model, [zone], cfg).T_centroid[-1, 0]
        rise2, rise3 = res["2d"] - 20.0, res["3d"] - 20.0
        assert abs(rise3 - rise2) / rise2 < 0.15

    def test_cg_matches_direct(self):
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        model = _frame_model({10: sec}, [(1, 1, 2, 10), (2, 2, 3, 10)],
                             {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0)})
        zone = _zone([0.5, 0, 0], [1.2, 1.0, 1.0])   # covers member 1 only (+ joint)
        out = {}
        for ls in ("direct", "cg"):
            cfg = AnalysisConfig(t_end=300.0, dt=30.0, output_dt=300.0, linear_solver=ls,
                                 element_ids=[1, 2])
            out[ls] = run_analysis(model, [zone], cfg)
        # GlobalThermalSolver's CG runs to a 1e-8 relative residual (~1e-6 K solution
        # error); 1e-4 K is still far below the ~1e-3 K Picard tolerance.
        for e in out["direct"].element_ids:
            np.testing.assert_allclose(out["cg"].T_section[e], out["direct"].T_section[e],
                                       rtol=0.0, atol=1e-4)


def _solid_solvers(model: FEMModel, q_hot: float, cfg: AnalysisConfig):
    eids = sorted(model.elements)
    meshes = {e: si.build_beam_solid_mesh(model.sections[model.elements[e].geom_id],
                                          model.elements[e].length, cfg) for e in eids}
    solvers = {}
    for k, e in enumerate(eids):
        q = q_hot if k == 0 else 0.0
        solvers[e] = SolidTransientSolver(meshes[e], _mat(), fire_temp=lambda t: 20.0,
                                          epsilon_m=0.0, h_conv=0.0,
                                          q_prescribed_fn=lambda t, q=q: q,
                                          epsilon_steel=1e-9)
    pos = _compute_global_node_positions(eids, meshes, model, {e: True for e in eids})
    return eids, meshes, solvers, pos


class TestJointCoupling:
    @pytest.mark.parametrize("layout", ["collinear", "perpendicular", "perpendicular_offset"])
    def test_heat_crosses_joint(self, layout):
        """
        collinear: end faces coincide → merged DOFs.  perpendicular: only the wall
        mid-nodes coincide (merged).  perpendicular_offset: member 2 is 16 mm taller,
        so NO node coincides — coupling must come from the joint links alone.
        """
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        sec2 = BoxSection(sid=11, H=0.216, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        p3 = (2, 0, 0) if layout == "collinear" else (1, 1, 0)
        g2 = 11 if layout == "perpendicular_offset" else 10
        model = _frame_model({10: sec, 11: sec2}, [(1, 1, 2, 10), (2, 2, 3, g2)],
                             {1: (0, 0, 0), 2: (1, 0, 0), 3: p3})
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        eids, meshes, solvers, pos = _solid_solvers(model, 20_000.0, cfg)
        gmap, n = si.build_solid_dof_map(eids, meshes, pos)
        for e in eids:
            assert len(np.unique(gmap[e])) == meshes[e].n_nodes
        k_link = si.build_joint_links(eids, meshes, pos, gmap, n, model)
        if layout == "perpendicular_offset":
            assert n == meshes[1].n_nodes + meshes[2].n_nodes     # nothing merged
            assert k_link is not None and k_link.nnz > 0
            # graph Laplacian: symmetric, zero row sums
            assert abs(k_link - k_link.T).max() < 1e-12
            assert np.abs(np.asarray(k_link.sum(axis=1))).max() < 1e-9
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=k_link)
        T = np.full(n, 20.0)
        for i in range(20):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        T_hot = T[gmap[1]].mean()
        T_cold = T[gmap[2]].mean()
        assert T_hot > 30.0
        assert T_cold > 20.5          # heat conducted into the unheated member
        assert T_cold < T_hot

    def test_no_link_no_heat_transfer(self):
        """Control: members far apart stay thermally isolated."""
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        model = _frame_model({10: sec}, [(1, 1, 2, 10), (2, 3, 4, 10)],
                             {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 5, 0), 4: (1, 5, 0)})
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        eids, meshes, solvers, pos = _solid_solvers(model, 20_000.0, cfg)
        gmap, n = si.build_solid_dof_map(eids, meshes, pos)
        assert si.build_joint_links(eids, meshes, pos, gmap, n, model) is None
        gs = GlobalThermalSolver(eids, solvers, gmap, n)
        T = np.full(n, 20.0)
        for i in range(5):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        np.testing.assert_allclose(T[gmap[2]], 20.0, atol=1e-8)

    def test_thin_wall_many_layers_not_collapsed(self):
        """2 mm wall with 4 layers → 0.5 mm node spacing < 1 mm merge tolerance."""
        sec = BoxSection(sid=10, H=0.1, W=0.1, T_side=0.002, T_bot=0.002, T_top=0.002)
        model = _frame_model({10: sec}, [(1, 1, 2, 10), (2, 2, 3, 10)],
                             {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)})
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30, n_layers_3d=4)
        eids, meshes, _solvers, pos = _solid_solvers(model, 0.0, cfg)
        gmap, n = si.build_solid_dof_map(eids, meshes, pos)
        n_end = int(np.sum(np.abs(meshes[1].nodes[:, 0]) < 1e-12))
        assert n == meshes[1].n_nodes + meshes[2].n_nodes - n_end   # exactly the end face
        for e in eids:
            assert len(np.unique(gmap[e])) == meshes[e].n_nodes


# ── helpers: view factors, M_extra, TemperatureField ──────────────────────────

class TestHelpers:
    def test_vectorised_view_factor_matches_reference(self):
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        mesh = BoxSolidMesher(sec, 1.0, 2, 3, 4, 2).build()
        model = _single_box_model()
        R = _beam_local_to_global(model.elements[101])
        corners, cents, normals = si.outer_face_geometry(mesh, R, np.zeros(3))
        zone = _zone([0.5, 0.0, -0.8], [1.0, 1.0, 1.0])       # below the beam
        F = si.face_view_factors(corners, normals, [zone])
        patches = zone.face_patches(4)
        ref = np.array([geometric_view_factor_double_area(corners[i], normals[i], patches, 2)
                        for i in range(len(corners))])
        np.testing.assert_allclose(F, ref, rtol=1e-10, atol=1e-14)
        assert F.max() > 0.0 and F.min() == 0.0                # top faces see nothing

    def test_ball_flux_matches_scalar(self):
        ball = RadiationBall(name="B", center=np.array([0.0, 0.0, 2.0]), radius=0.5,
                             flux=1e5)
        pts = np.array([[0, 0, 0], [0, 0, 1.8], [1, 0, 0]], dtype=float)
        nrm = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1]], dtype=float)
        q = si.ball_face_flux(ball, pts, nrm)
        ref = [ball.incident_flux(p, n) for p, n in zip(pts, nrm)]
        np.testing.assert_allclose(q, ref)

    def test_M_extra_on_inner_nodes(self):
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        mesh = BoxSolidMesher(sec, 1.0, 2, 3, 4, 2).build()
        M = si.solid_M_extra(sec, mesh, 1.0, 1200.0)
        assert M is not None
        assert M.sum() == pytest.approx(sec.inner_height * sec.inner_width * 1200.0)
        assert set(np.flatnonzero(M)) <= set(mesh.inner_node_indices.tolist())

    def test_from_solid_solver_run_and_gradient(self):
        sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
        mesh: SolidMesh = BoxSolidMesher(sec, 1.0, 2, 3, 4, 2).build()
        y, z = mesh.nodes[:, 1], mesh.nodes[:, 2]
        T1 = 100.0 + 50.0 * y + 30.0 * z
        hist = np.vstack([np.full(mesh.n_nodes, 20.0), T1])
        tf = TemperatureField.from_solid_solver_run(
            7, np.array([0.0, 60.0]), hist, mesh, nodes_global=mesh.nodes)
        assert tf.T_centroid.shape == (2, 1)
        assert tf.T_centroid[0, 0] == pytest.approx(20.0)
        assert tf.T_centroid[1, 0] == pytest.approx(100.0, abs=1e-8)   # symmetric section
        assert tf.nodal_geometry[7][1].shape == mesh.faces.shape
        by, bz = tf.section_gradient(7, 1, mesh)
        assert by == pytest.approx(30.0) and bz == pytest.approx(50.0)
        merged = TemperatureField.merge([tf, tf.__class__(
            times=tf.times, element_ids=[8], T_centroid=tf.T_centroid.copy(),
            T_section={8: hist})])
        assert merged.element_ids == [7, 8] and 7 in merged.nodal_geometry
