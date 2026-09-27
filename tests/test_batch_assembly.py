"""
SolidBatch (vectorised multi-member assembly) must reproduce per-member assembly
(SolidTransientSolver.assemble_raw scattered through the DOF map) to round-off, for a
mix of boundary conditions, and GlobalThermalSolver results must be unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.bc.face_flux import TimeVaryingFaceFlux
from fahts.core.heat.bc.inner_robin_bc import InnerRobinBC
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.solid_mesh import PlateSolidMesher
from fahts.core.heat.solver import solid_integration as si
from fahts.core.heat.solver.analysis_runner import (
    GlobalThermalSolver,
    _compute_global_node_positions,
)
from fahts.core.heat.solver.joint_ties import build_joint_ties
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.sources.concentrated_source import ConcentratedSource
from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig


def _mat(**kw) -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6, **kw)


def _model() -> FEMModel:
    coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (1, 0, 1.2), 5: (2, 1, 0),
              6: (1, 1, 0)}
    nodes = {k: Node(nid=k, x=x, y=y, z=z) for k, (x, y, z) in coords.items()}
    secs = {10: BoxSection(sid=10, H=0.3, W=0.3, T_side=0.012, T_bot=0.012, T_top=0.012),
            11: PipeSection(sid=11, outer_diameter=0.2, thickness=0.01),
            12: ISection(sid=12, h=0.3, tw=0.0071, bf_top=0.15, tf_top=0.0107,
                         bf_bot=0.15, tf_bot=0.0107),
            20: PlateSection(sid=20, thickness=0.01)}
    elems = {}
    for eid, a, b, g in [(1, 1, 2, 10), (2, 2, 3, 12), (3, 2, 4, 11)]:
        pa, pb = nodes[a].xyz, nodes[b].xyz
        L = float(np.linalg.norm(pb - pa))
        d = (pb - pa) / L
        lz = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        lz = lz - d * np.dot(lz, d)
        lz /= np.linalg.norm(lz)
        elems[eid] = BeamElement(eid=eid, n1=a, n2=b, mat_id=1, geom_id=g, lcoor_id=0,
                                 length=L, direction=d, local_z=lz)
    shells = {50: ShellElement(eid=50, nodes=(2, 3, 5, 6), mat_id=1, geom_id=20)}
    return FEMModel(nodes=nodes, elements=elems, shell_elements=shells, sections=secs,
                    materials={1: _mat()}, groups={}, unitvecs={},
                    source_file=Path("mix.fem"))


def _build(model: FEMModel):
    cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
    meshes = {e: si.build_beam_solid_mesh(model.sections[el.geom_id], el.length, cfg)
              for e, el in model.elements.items()}
    sh = model.shell_elements[50]
    meshes[50] = PlateSolidMesher(model.sections[20],
                                  np.array([model.nodes[n].xyz for n in sh.nodes]),
                                  mesh_12=3, mesh_14=3, n_layers=2).build()
    eids = [1, 2, 3, 50]
    fire = lambda t: 20.0 + 900.0 * (1 - np.exp(-t / 300.0))  # noqa: E731
    rng = np.random.default_rng(0)
    solvers = {}
    n_o = {e: len(meshes[e].outer_face_indices) for e in eids}
    # 1: fire zone with view factors + enclosed gas + prescribed nodes
    solvers[1] = SolidTransientSolver(
        meshes[1], _mat(), fire_temp=fire, epsilon_m=0.7, h_conv=25.0,
        epsilon_steel=0.7, epsilon_fire=0.9, view_factors=rng.uniform(0.2, 1, n_o[1]),
        face_exposure=(rng.uniform(size=n_o[1]) > 0.3).astype(float),
        M_extra=np.full(meshes[1].n_nodes, 3.0),
        prescribed_node_bcs=[PrescribedNodeBC([0, 1], 150.0)])
    # 2: radiation-ball style static flux with re-radiation (epsilon_m = 0)
    solvers[2] = SolidTransientSolver(
        meshes[2], _mat(usfos_mode=True), fire_temp=lambda t: 20.0, epsilon_m=0.0,
        h_conv=0.0, epsilon_steel=0.85, q_per_face=rng.uniform(0, 5e4, n_o[2]))
    # 3: pipe, plain fire + contents Robin + time-varying point source
    sched = TimeVaryingFaceFlux(n_o[3])
    cents = meshes[3].face_centroids()[meshes[3].outer_face_indices]
    norms = meshes[3].face_normals()[meshes[3].outer_face_indices]
    sched.add_source(ConcentratedSource(name="s", center=np.array([0.0, 0.5, 0.5]),
                                        power=lambda t: 1e4 * t),
                     lambda u: u.per_quad_flux(cents, norms))
    solvers[3] = SolidTransientSolver(
        meshes[3], _mat(), fire_temp=fire, epsilon_m=0.56, h_conv=50.0, q_per_face=sched,
        inner_bc=InnerRobinBC(h=400.0, T_fluid=lambda t: 20.0 + 0.1 * t))
    # 50: plate, uniform prescribed flux
    solvers[50] = SolidTransientSolver(
        meshes[50], _mat(), fire_temp=lambda t: 20.0, epsilon_m=0.0, h_conv=10.0,
        q_prescribed_fn=lambda t: 2e4)
    pos = _compute_global_node_positions(eids, meshes, model, {e: True for e in eids})
    gmap, n = si.build_solid_dof_map(eids, meshes, pos)
    K_tie = build_joint_ties(eids, meshes, pos, gmap, n, model)
    return eids, solvers, gmap, n, K_tie


@pytest.fixture(scope="module")
def built():
    return _build(_model())


def test_batch_assembly_matches_per_member(built):
    eids, solvers, gmap, n, K_tie = built
    gb = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K_tie, batch=True)
    gm = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K_tie, batch=False)
    assert gb._batch is not None and not gb._other_eids
    assert gm._batch is None
    rng = np.random.default_rng(1)
    T = rng.uniform(20.0, 800.0, n)
    with ThreadPoolExecutor(1) as pool:
        for t in (0.0, 45.0, 600.0):
            Kb, Mb, Qb = gb._assemble_global(T, t, pool)
            Km, Mm, Qm = gm._assemble_global(T, t, pool)
            scale = abs(Km).max()
            assert abs(Kb - Km).max() <= 1e-12 * scale
            assert abs(Mb - Mm).max() <= 1e-12 * abs(Mm).max()
            np.testing.assert_allclose(Qb, Qm, rtol=1e-12, atol=1e-9 * np.abs(Qm).max())


@pytest.mark.parametrize("linear_solver", ["cg", "direct"])
def test_batch_run_matches_per_member(built, linear_solver):
    eids, solvers, gmap, n, K_tie = built
    out = []
    for batch in (True, False):
        gs = GlobalThermalSolver(eids, solvers, gmap, n, k_link=K_tie, batch=batch,
                                 linear_solver=linear_solver)
        T = np.full(n, 20.0)
        gs.apply_prescribed(T, 0.0)
        for i in range(6):
            T = gs.step(T, 30.0, 30.0 * (i + 1))
        out.append(T)
    np.testing.assert_allclose(out[0], out[1], rtol=1e-9, atol=1e-7)
    assert out[0].max() > 100.0
