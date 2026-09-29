"""
Radiation geometry (2026-09-27): ray casting, source shielding and surface-to-surface
exchange.  Exact checks: brute-force ray casting, analytic view factor of two parallel
squares, isothermal-enclosure zero exchange, reciprocity, and shielding of the two-plate
model (parallel_plates.fem).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.radiation.exchange import RadiationExchange
from fahts.core.heat.radiation.raycast import TriangleScene, brute_force_first_hit
from fahts.core.heat.radiation.shielding import (
    ball_visibility,
    build_scene,
    point_visibility,
)
from fahts.core.heat.solid_mesh import PlateSolidMesher, PipeSolidMesher
from fahts.core.heat.solver import solid_integration as si
from fahts.core.heat.solver.analysis_runner import (
    _beam_local_to_global,
    _compute_global_node_positions,
    run_analysis,
)
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.sources.concentrated_source import ConcentratedSource
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import PipeSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig

ROOT = Path(__file__).resolve().parents[1]


def _plates(gap: float = 1.0, side: float = 2.0, n: int = 4):
    sec = PlateSection(sid=1, thickness=0.02)
    meshes, pos = {}, {}
    for k, z in enumerate((0.0, gap)):
        c = np.array([[0, 0, z], [side, 0, z], [side, side, z], [0, side, z]], float)
        m = PlateSolidMesher(sec, c, mesh_12=n, mesh_14=n, n_layers=1).build()
        meshes[k + 1], pos[k + 1] = m, np.asarray(m.nodes)
    return meshes, pos


def _mat() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


# ── ray casting ───────────────────────────────────────────────────────────────

def test_grid_matches_brute_force():
    rng = np.random.default_rng(3)
    c = rng.uniform(0, 5, (800, 3))
    tris = c[:, None, :] + rng.normal(0, 0.25, (800, 3, 3))
    sc = TriangleScene(tris)
    O = rng.uniform(-1, 6, (300, 3))
    D = rng.normal(size=(300, 3))
    hit, t = sc.trace(O, D, np.full(300, np.inf))
    for i in range(300):
        bh, bt = brute_force_first_hit(tris, O[i], D[i])
        assert (bh == hit[i]) or (bh >= 0 and abs(bt - t[i]) < 1e-9)
    T = rng.uniform(0, 5, (300, 3))
    ref = [brute_force_first_hit(tris, O[i], T[i] - O[i], 1 - 1e-9, 1e-9)[0] >= 0
           for i in range(300)]
    np.testing.assert_array_equal(sc.blocked(O, T), ref)


def test_scene_normals_outward_for_mirrored_beam_frame():
    """The beam frame is left-handed (mirror); scene normals must still point outward."""
    model = read_usfos_fem(ROOT / "examples" / "models" / "tank_horizontal.fem")
    cfg = AnalysisConfig(t_end=60, dt=30, output_dt=30)
    e = 1
    el = model.elements[e]
    mesh = si.build_beam_solid_mesh(model.sections[el.geom_id], el.length, cfg)
    pos = _compute_global_node_positions([e], {e: mesh}, model, {e: True})
    _, members = build_scene([e], {e: mesh}, pos)
    mf = members[e]
    axis_pt = model.nodes[el.n1].xyz
    radial = mf.centroids - axis_pt
    radial -= np.outer(radial @ el.direction, el.direction)
    assert np.all(np.einsum("ij,ij->i", radial, mf.normals) > 0.0)
    # and they agree with the solver's rotated local normals
    R = _beam_local_to_global(el)
    _, _, n_ref = si.outer_face_geometry(mesh, R, axis_pt)
    np.testing.assert_allclose(mf.normals, n_ref, atol=1e-9)


# ── shielding ─────────────────────────────────────────────────────────────────

def test_ball_and_point_shielded_by_plate():
    meshes, pos = _plates()
    scene, mem = build_scene([1, 2], meshes, pos)
    ball = RadiationBall(name="B", center=np.array([1.0, 1.0, -1.0]), radius=0.5,
                         flux=1e5, active=True)
    for k, expect in ((1, 1.0), (2, 0.0)):
        mf = mem[k]
        down = mf.normals[:, 2] < -0.5                 # faces looking at the ball
        centre = (np.abs(mf.centroids[:, 0] - 1) < 0.6) & (np.abs(mf.centroids[:, 1] - 1) < 0.6)
        sel = down & centre
        vis = ball_visibility(scene, mf.centroids[sel], mf.normals[sel], ball)
        np.testing.assert_allclose(vis, expect)
        pv = point_visibility(scene, mf.centroids[sel], mf.normals[sel], ball.center)
        np.testing.assert_allclose(pv, expect)
    # ball below and just beyond the near plate's edge: from the far plate's underside
    # part of the ball is hidden by the near plate (partial shading)
    ball2 = RadiationBall(name="B2", center=np.array([2.3, 1.0, -1.0]), radius=0.45,
                          flux=1e5, active=True)
    mf = mem[2]
    sel = mf.normals[:, 2] < -0.5
    vis = ball_visibility(scene, mf.centroids[sel], mf.normals[sel], ball2)
    assert np.any((vis > 0.0) & (vis < 1.0))


# ── surface-to-surface exchange ───────────────────────────────────────────────

def _exchange(meshes, pos, rays=4000, patch=10.0, T_amb=20.0):
    solvers = {k: SolidTransientSolver(meshes[k], _mat(), fire_temp=lambda t: 20.0,
                                       epsilon_m=0.0, h_conv=0.0, epsilon_steel=0.7,
                                       q_prescribed_fn=lambda t: 1.0, T0=T_amb,
                                       conduction="monotone")
               for k in meshes}
    gd, off = {}, 0
    for k in meshes:
        gd[k] = off + np.arange(meshes[k].n_nodes)
        off += meshes[k].n_nodes
    scene, mem = build_scene(list(meshes), meshes, pos)
    ex = RadiationExchange(scene, mem, solvers, gd, off, patch_size=patch, rays=rays)
    return ex, mem, gd, off


def _parallel_squares_F(a: float, c: float) -> float:
    X = Y = a / c
    t1 = math.log(math.sqrt((1 + X * X) * (1 + Y * Y) / (1 + X * X + Y * Y)))
    t2 = X * math.sqrt(1 + Y * Y) * math.atan(X / math.sqrt(1 + Y * Y))
    t3 = Y * math.sqrt(1 + X * X) * math.atan(Y / math.sqrt(1 + X * X))
    return 2.0 / (math.pi * X * Y) * (t1 + t2 + t3 - X * math.atan(X) - Y * math.atan(Y))


def test_view_factor_parallel_squares_matches_analytic():
    meshes, pos = _plates(gap=1.0, side=2.0, n=4)
    ex, mem, *_ = _exchange(meshes, pos, rays=20000, patch=10.0)
    # patch of plate 1 facing up (+z) → patch of plate 2 facing down
    fp = ex.fpatch
    nz = np.concatenate([mem[k].normals[:, 2] for k in (1, 2)])
    zc = np.concatenate([mem[k].centroids[:, 2] for k in (1, 2)])
    p_up = np.unique(fp[(nz > 0.5) & (zc < 0.5)])
    p_dn = np.unique(fp[(nz < -0.5) & (zc > 0.5)])
    assert len(p_up) == 1 and len(p_dn) == 1
    F = ex.F[p_up[0], p_dn[0]]
    assert F == pytest.approx(_parallel_squares_F(2.0, 1.0), rel=0.03)   # 0.4152


def test_exchange_zero_in_isothermal_ambient_and_reciprocal():
    meshes, pos = _plates()
    ex, mem, gd, n = _exchange(meshes, pos, rays=500, patch=0.5)
    Q = ex.load(np.full(n, 20.0), 0.0)
    assert np.abs(Q).max() < 1e-9
    AF = ex.parea[:, None] * ex.F.toarray()
    np.testing.assert_allclose(AF, AF.T, atol=1e-12)
    # hot lower plate heats the upper one.  The base model already makes plate 1 emit
    # εσT⁴ into its whole hemisphere; exchange only lets plate 2 absorb the part that
    # reaches it (≤ ε1 ε2 σ (T1⁴ − Ta⁴) A1 F12), and plate 1 sees plate 2 at ambient → 0.
    T = np.full(n, 20.0)
    T[gd[1]] = 600.0
    Q = ex.load(T, 0.0)
    gain = Q[gd[2]].sum()
    F12 = _parallel_squares_F(2.0, 1.0)
    bound = 0.7 * 0.7 * 5.67e-8 * (873.15 ** 4 - 293.15 ** 4) * 4.0 * F12
    assert 0.0 < gain <= 1.05 * bound and gain > 0.7 * bound
    assert abs(Q[gd[1]].sum()) < 1e-6 * gain


# ── end to end ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source", ["ball", "point"])
def test_two_plates_shielding_and_exchange(source):
    model = read_usfos_fem(ROOT / "examples" / "models" / "parallel_plates.fem")
    if source == "ball":
        src = RadiationBall(name="B", center=np.array([1.0, 1.0, -1.0]), radius=0.5,
                            flux=2e5, active=True)
        cfg_kw = {}
    else:
        src = ConcentratedSource(name="S", center=np.array([1.0, 1.0, -0.8]), power=2e6)
        cfg_kw = {"concentrated_sources": [src]}
    z = {e: np.mean([model.nodes[n].z for n in sh.nodes])
         for e, sh in model.shell_elements.items()}

    def run(sh, ex):
        r = run_analysis(model, [src] if source == "ball" else [],
                         AnalysisConfig(t_end=600.0, dt=30.0, output_dt=600.0, shielding=sh,
                                        radiation_exchange=ex, **cfg_kw))
        T = r.T_centroid[-1]
        near = np.mean([T[i] for i, e in enumerate(r.element_ids) if z[e] < 0.5])
        far = np.mean([T[i] for i, e in enumerate(r.element_ids) if z[e] > 0.5])
        return near, far

    n0, f0 = run(False, False)
    n1, f1 = run(True, False)
    n2, f2 = run(True, True)
    assert f0 > 40.0                          # unshielded: far plate wrongly heated
    assert f1 == pytest.approx(20.0, abs=1e-6)   # shielded: far plate stays at ambient
    assert n1 == pytest.approx(n0, rel=1e-9)     # near plate unaffected by shielding
    # exchange: the near plate's emission now partly lands on the far plate, which
    # warms and re-radiates a little back
    assert f2 > f1 + 0.5 and n2 >= n1 - 1e-6


def test_fire_zone_with_exchange_stays_physical():
    """Bundle of BOX members, only partly inside a FireZone: with exchange on, no node may
    drop below ambient (regression: −219 °C when unexposed faces joined the exchange)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_solid_integration import _frame_model, _zone

    from fahts.core.model.section import BoxSection

    sec = BoxSection(sid=10, H=0.2, W=0.2, T_side=0.01, T_bot=0.01, T_top=0.01)
    coords = {}
    elems = []
    for k in range(3):                               # three parallel members 0.35 m apart
        coords[2 * k + 1] = (0.0, 0.35 * k, 0.0)
        coords[2 * k + 2] = (3.0, 0.35 * k, 0.0)
        elems.append((k + 1, 2 * k + 1, 2 * k + 2, 10))
    model = _frame_model({10: sec}, elems, coords)
    zone = _zone([0.8, 0.35, 0.0], [1.6, 2.0, 1.0])  # covers only the first ~1.6 m
    out = {}
    for ex in (False, True):
        r = run_analysis(model, [zone], AnalysisConfig(t_end=600.0, dt=30.0, output_dt=600.0,
                                                       radiation_exchange=ex))
        allT = np.concatenate([r.T_section[e][-1] for e in r.element_ids])
        out[ex] = allT
        assert allT.min() >= 20.0 - 1e-6
        assert allT.max() <= 1100.0
    # mutual shading inside the fire: the middle member runs cooler with exchange on
    assert out[True].mean() <= out[False].mean() + 1e-6
