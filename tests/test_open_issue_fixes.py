"""
Regression tests for the 2026-09-27 open-issue fixes:

1. GlobalThermalSolver enforces PrescribedNodeBC (symmetric elimination, CG-safe).
2. Point/line source power E(t) is re-evaluated every step (TimeVaryingFaceFlux).
3. Joint surface ties couple T-joints, plate edges along beams and angled plates.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.bc.face_flux import TimeVaryingFaceFlux
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.solid_mesh import FACE_END, PlateSolidMesher
from fahts.core.heat.solver import solid_integration as si
from fahts.core.heat.solver.analysis_runner import (
    GlobalThermalSolver,
    _compute_global_node_positions,
    run_analysis,
)
from fahts.core.heat.solver.joint_ties import build_joint_ties, closest_point_on_triangles
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.sources.concentrated_source import ConcentratedSource
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

def _mat() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _model(sections, beams, coords, shells=()) -> FEMModel:
    nodes = {nid: Node(nid=nid, x=x, y=y, z=z) for nid, (x, y, z) in coords.items()}
    elements = {}
    for eid, a, b, gid in beams:
        pa, pb = nodes[a].xyz, nodes[b].xyz
        L = float(np.linalg.norm(pb - pa))
        d = (pb - pa) / L
        lz = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        lz = lz - d * np.dot(lz, d)
        lz /= np.linalg.norm(lz)
        elements[eid] = BeamElement(eid=eid, n1=a, n2=b, mat_id=1, geom_id=gid, lcoor_id=0,
                                    length=L, direction=d, local_z=lz)
    shell_elements = {eid: ShellElement(eid=eid, nodes=tuple(ns), mat_id=1, geom_id=gid)
                      for eid, ns, gid in shells}
    return FEMModel(nodes=nodes, elements=elements, shell_elements=shell_elements,
                    sections=sections, materials={1: _mat()}, groups={}, unitvecs={},
                    source_file=Path("synthetic.fem"))


def _meshes(model: FEMModel, cfg: AnalysisConfig) -> dict:
    meshes = {}
    for e, el in model.elements.items():
        meshes[e] = si.build_beam_solid_mesh(model.sections[el.geom_id], el.length, cfg)
    for e, sh in model.shell_elements.items():
        corners = np.array([model.nodes[n].xyz for n in sh.nodes])
        meshes[e] = PlateSolidMesher(section=model.sections[sh.geom_id], corners=corners,
                                     mesh_12=4, mesh_14=4, n_layers=cfg.n_layers_3d).build()
    return meshes


def _coupled(model: FEMModel, cfg: AnalysisConfig, hot: int, q: float = 30_000.0):
    """Global solver with only member *hot* heated by a uniform flux."""
    eids = sorted(model.elements) + sorted(model.shell_elements)
    meshes = _meshes(model, cfg)
    solvers = {
        e: SolidTransientSolver(meshes[e], _mat(), fire_temp=lambda t: 20.0, epsilon_m=0.0,
                                h_conv=0.0, q_prescribed_fn=lambda t, qq=(q if e == hot else 0.0):
                                qq, epsilon_steel=1e-9)
        for e in eids
    }
    is_surf = {e: True for e in eids}
    pos = _compute_global_node_positions(eids, meshes, model, is_surf)
    gmap, n = si.build_solid_dof_map(eids, meshes, pos)
    K = build_joint_ties(eids, meshes, pos, gmap, n, model)
    return eids, meshes, solvers, pos, gmap, n, K


def _check_laplacian(K) -> None:
    assert K is not None and K.nnz > 0
    assert abs(K - K.T).max() < 1e-9 * abs(K).max()
    assert np.abs(np.asarray(K.sum(axis=1))).max() < 1e-9 * abs(K).max()
    idx = np.unique(K.nonzero()[0])
    ev = np.linalg.eigvalsh(K[idx][:, idx].toarray())
    assert ev.min() > -1e-9 * abs(ev).max()          # positive semi-definite


BOX = BoxSection(sid=10, H=0.3, W=0.3, T_side=0.012, T_bot=0.012, T_top=0.012)
SMALL = BoxSection(sid=11, H=0.15, W=0.15, T_side=0.008, T_bot=0.008, T_top=0.008)


# ── 1. prescribed nodes in the global solver ─────────────────────────────────

class TestGlobalDirichlet:
    @pytest.mark.parametrize("solver_dim", ["2d", "3d"])
    @pytest.mark.parametrize("linear_solver", ["direct", "cg"])
    @pytest.mark.parametrize("mass", ["lumped", "consistent"])
    def test_pinned_value_held(self, solver_dim, linear_solver, mass):
        model = _model({10: SMALL}, [(101, 1, 2, 10)], {1: (0, 0, 0), 2: (1, 0, 0)})
        zone = FireZone(name="Z", center=np.array([0.5, 0, 0]), dims=np.array([2.0, 2, 2]),
                        curve=FireCurve(curve_type=FireCurveType.ISO_834), h_conv=25.0,
                        epsilon_fire=1.0, active=True)
        bc = PrescribedNodeBC(node_indices=[0, 1], temperature=lambda t: 20.0 + t)
        cfg = AnalysisConfig(t_end=120.0, dt=30.0, output_dt=30.0, element_ids=[101],
                             solver_dim=solver_dim, linear_solver=linear_solver,
                             mass_matrix=mass, prescribed_node_bcs=[bc])
        res = run_analysis(model=model, fire_zones=[zone], config=cfg)
        T = res.T_section[101]
        np.testing.assert_allclose(T[:, 0], 20.0 + res.times, atol=1e-8)
        np.testing.assert_allclose(T[:, 1], 20.0 + res.times, atol=1e-8)
        assert np.all(np.isfinite(T))
        assert T[-1, 2:].max() > 20.0                     # the rest still heats


# ── 2. time-varying point-source power ───────────────────────────────────────

class TestTimeVaryingSource:
    def test_schedule_scales_with_power(self):
        src = ConcentratedSource(name="S", center=np.array([0.0, 0.0, 1.0]),
                                 power=lambda t: 1e5 * t)
        cents = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        sched = TimeVaryingFaceFlux(2)
        sched.add_source(src, lambda u: u.per_quad_flux(cents, normals))
        assert sched.lit
        np.testing.assert_allclose(sched(0.0), 0.0)
        np.testing.assert_allclose(sched(10.0), src.per_quad_flux(cents, normals, t=10.0))
        assert sched(10.0)[1] == 0.0                      # back face never lit

    @pytest.mark.parametrize("solver_dim", ["2d", "3d"])
    def test_ramped_source_heats_member(self, solver_dim):
        """E(0) = 0: the member must not be skipped, and heats as E grows."""
        model = _model({10: SMALL}, [(101, 1, 2, 10)], {1: (0, 0, 0), 2: (1, 0, 0)})
        ramp = ConcentratedSource(name="S", center=np.array([0.5, 0.0, 1.0]),
                                  power=lambda t: 2e4 * t)
        const = ConcentratedSource(name="S", center=np.array([0.5, 0.0, 1.0]),
                                   power=2e4 * 300.0)
        out = {}
        for tag, src in (("ramp", ramp), ("const", const)):
            cfg = AnalysisConfig(t_end=300.0, dt=30.0, output_dt=30.0, element_ids=[101],
                                 solver_dim=solver_dim, concentrated_sources=[src])
            out[tag] = run_analysis(model=model, fire_zones=[], config=cfg).T_centroid[:, 0]
        assert out["ramp"][-1] > 25.0
        # linear ramp to the constant power: early heating is much slower
        assert out["ramp"][2] - 20.0 < 0.5 * (out["const"][2] - 20.0)
        assert out["ramp"][-1] < out["const"][-1]


# ── 3. joint surface ties ────────────────────────────────────────────────────

class TestJointTies:
    def test_closest_point_on_triangle(self):
        a, b, c = (np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]]))
        for p, q_exp in [((0.2, 0.2, 1.0), (0.2, 0.2, 0.0)), ((-1, -1, 0), (0, 0, 0)),
                         ((1, 1, 0), (0.5, 0.5, 0)), ((0.5, -1, 0.3), (0.5, 0, 0))]:
            q, lam = closest_point_on_triangles(np.array([p], float), a, b, c)
            np.testing.assert_allclose(q[0], q_exp, atol=1e-12)
            assert abs(lam.sum() - 1.0) < 1e-12 and lam.min() >= -1e-12

    def test_t_joint_every_brace_end_node_tied(self):
        """Brace end sits in the chord's hollow cavity — no node is near — yet all tie."""
        model = _model({10: BOX, 11: SMALL},
                       [(1, 1, 2, 10), (2, 2, 3, 10), (3, 2, 4, 11)],
                       {1: (-1, 0, 0), 2: (0, 0, 0), 3: (1, 0, 0), 4: (0, 0, 1.5)})
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        eids, meshes, solvers, pos, gmap, n, K = _coupled(model, cfg, hot=1)
        _check_laplacian(K)
        # brace end face at node 2 (x=0 of member 3): every node carries a tie row
        m3 = meshes[3]
        end_nodes = np.unique(m3.faces[m3.face_group == FACE_END])
        end0 = end_nodes[np.abs(m3.nodes[end_nodes, 0]) < 1e-9]
        rows = set(np.unique(K.nonzero()[0]).tolist())
        assert all(int(gmap[3][i]) in rows for i in end0)
        # chord through-ends (merged 1↔2) are not sources: their end nodes tie only as
        # targets, so heat flows chord → brace
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K)
        T = np.full(n, 20.0)
        for i in range(20):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        assert T[gmap[3]].max() > 25.0

    def test_plate_edge_along_beam_is_tied(self):
        """QUADSHEL whose edge lies on a beam's centre-line (deck plate on a girder)."""
        plate = PlateSection(sid=20, thickness=0.01)
        model = _model({10: BOX, 20: plate}, [(1, 1, 2, 10)],
                       {1: (0, 0, 0), 2: (2, 0, 0), 3: (2, 1, 0), 4: (0, 1, 0)},
                       shells=[(50, (1, 2, 3, 4), 20)])
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        eids, meshes, solvers, pos, gmap, n, K = _coupled(model, cfg, hot=1)
        _check_laplacian(K)
        edge = np.flatnonzero(np.abs(pos[50][:, 1]) < 1e-9)       # nodes on the y=0 edge
        rows = set(np.unique(K.nonzero()[0]).tolist())
        tied = [int(gmap[50][i]) in rows for i in edge]
        assert np.mean(tied) > 0.9                                # (not just the corners)
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K)
        T = np.full(n, 20.0)
        for i in range(20):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        T_edge = T[gmap[50][edge]].mean()
        T_far = T[gmap[50][np.abs(pos[50][:, 1] - 1.0) < 1e-9]].mean()
        assert T_edge > 25.0 and T_far < T_edge

    def test_angled_plates_odd_layers_coupled(self):
        """Two plates meeting at 90° with 3 layers: no mid-surface node to merge."""
        plate = PlateSection(sid=20, thickness=0.02)
        model = _model({20: plate}, [],
                       {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
                        5: (1, 0, 1), 6: (0, 0, 1)},
                       shells=[(50, (1, 2, 3, 4), 20), (51, (1, 2, 5, 6), 20)])
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30, n_layers_3d=3)
        eids, meshes, solvers, pos, gmap, n, K = _coupled(model, cfg, hot=50)
        _check_laplacian(K)
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K)
        T = np.full(n, 20.0)
        for i in range(20):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        assert T[gmap[51]].max() > 25.0

    def test_ties_conserve_energy(self):
        """Adiabatic coupled system: Σ M·ΔT equals the heat injected (ties add none)."""
        model = _model({10: BOX, 11: SMALL},
                       [(1, 1, 2, 10), (2, 2, 3, 10), (3, 2, 4, 11)],
                       {1: (-1, 0, 0), 2: (0, 0, 0), 3: (1, 0, 0), 4: (0, 0, 1.5)})
        cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
        eids, meshes, solvers, pos, gmap, n, K = _coupled(model, cfg, hot=1, q=0.0)
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K)
        T0 = np.full(n, 20.0)
        T0[gmap[1]] = 400.0                                       # hot chord segment
        with ThreadPoolExecutor(1) as pool:
            _, M0, _ = gs._assemble_global(T0, 0.0, pool)
        T = T0.copy()
        for i in range(10):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        # capacity frozen at T0 (c(T) varies, hence the loose tolerance): the energy
        # change must be small compared with the excess heat that was redistributed
        E0 = float((M0 @ T0).sum())
        E1 = float((M0 @ T).sum())
        assert T[gmap[3]].max() > 21.0
        assert abs(E1 - E0) < 0.02 * abs(float((M0 @ (T0 - 20.0)).sum()))
