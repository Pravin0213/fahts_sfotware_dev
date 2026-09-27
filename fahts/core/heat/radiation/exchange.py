"""
Surface-to-surface radiation exchange between solid members.

Patches
    FACE_OUTER faces of every member are grouped into patches of about ``patch_size``
    (same member, same outward-normal direction bin, same spatial bin).

View factors
    Monte Carlo: ``rays`` cosine-weighted rays per patch, from area-weighted random points
    on its faces; the first surface hit (any member, any face) gives F_ij.  Obstruction is
    therefore included automatically.  Reciprocity is enforced by symmetrising A_i·F_ij.
    Hits on faces that are not exposed patches (end caps, cavities) count as steel that
    does not take part in the exchange.

Exchange load (grey, diffuse, single reflection)
    Each member's own boundary model already lets every face absorb a default background
    irradiance a·G over its whole hemisphere and emit ε σ T⁴:
        ambient / prescribed-flux mode   G = σ T_amb⁴,              a = ε
        engulfed in a RadiationBall      G = ball flux,              a = 1  (ball model)
        FireZone (resultant ε_m)         G = σ T_fire⁴,              a = ε_m
        FireZone with view factors       G = σ F_zone ε_f T_fire⁴,   a = ε_s
    The part F_ij of the hemisphere that actually sees steel patch j receives j's
    radiosity J_j = ε_j σ T_j⁴ + (1 − ε_j) G_j instead:

        q_i = Σ_j F_ij [ ε_i J_j − a_i G_i ]

    Faces are weighted by their FireZone exposure fraction (faces outside the zone are
    adiabatic in the base model and do not take part).
    An isothermal enclosure at the background temperature gives q = 0 exactly, and the
    steel–steel part ε_i ε_j σ A_i F_ij (T_j⁴ − T_i⁴) is antisymmetric (energy-conserving).
    Applied explicitly (Picard-lagged) as an extra nodal load in GlobalThermalSolver.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from fahts.core.heat.radiation.raycast import TriangleScene
from fahts.core.heat.radiation.shielding import MemberFaces, _offset_origins

log = logging.getLogger(__name__)

_SIGMA = 5.67e-8
_K0 = 273.15
_DIRS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) != (0, 0, 0)], dtype=float)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def _member_radiation(sv) -> tuple[float, str]:
    """(emissivity, background mode) of a member solver: 'amb' | 'vf' | 'fire' | 'none'."""
    if getattr(sv, "_insulation", None) is not None:
        return 0.0, "none"
    if getattr(sv, "_eps_rerad", 0.0) > 0.0:
        return float(sv._eps_rerad), "amb"
    if getattr(sv, "_use_vf", False):
        return float(sv._eps_steel), "vf"
    if getattr(sv, "_eps", 0.0) != 0.0:
        return float(sv._eps), "fire"
    return 0.0, "none"


class RadiationExchange:
    """Precomputed patch view factors + per-assembly exchange load."""

    def __init__(self, scene: TriangleScene, members: dict[int, MemberFaces],
                 solvers: dict, gdof_map: dict[int, np.ndarray], n_global: int,
                 patch_size: float = 0.5, rays: int = 256, seed: int = 0,
                 balls: dict | None = None) -> None:
        rng = np.random.default_rng(seed)
        self._n = n_global
        face_nodes, face_row, face_area, face_patch = [], [], [], []
        face_eps, face_mem, face_F, face_eng, face_exp = [], [], [], [], []
        global_face_to_patch = np.full(scene.owner.max() + 1 if len(scene.owner) else 0,
                                       -1, np.int64)
        self._sv: list = []
        self._mode: list[str] = []
        self._T0K: list[float] = []
        norms_all, corners_all = [], []
        n_patch = 0
        for m_idx, (eid, mf) in enumerate(members.items()):
            sv = solvers.get(eid)
            if sv is None or not hasattr(sv, "_fo"):
                continue
            eps, mode = _member_radiation(sv)
            k = len(self._sv)
            self._sv.append(sv)
            self._mode.append(mode)
            self._T0K.append(float(getattr(sv, "_T0", 20.0)) + _K0)
            # patches: normal bin × spatial bin, per member
            nb = np.argmax(mf.normals @ _DIRS.T, axis=1)
            sb = np.floor(mf.centroids / patch_size).astype(np.int64)
            key = np.column_stack([nb, sb])
            _, pid = np.unique(key, axis=0, return_inverse=True)
            pid = np.asarray(pid).ravel() + n_patch
            n_patch = int(pid.max()) + 1
            global_face_to_patch[mf.face_offset + mf.outer] = pid
            gd = np.asarray(gdof_map[eid], dtype=np.intp)
            face_nodes.append(gd[sv._fo])
            face_row.append(sv._Fo_row)
            face_area.append(sv._Fo_row.sum(axis=1))
            face_patch.append(pid)
            face_eps.append(np.full(len(pid), eps))
            face_mem.append(np.full(len(pid), k))
            face_F.append(sv._F_arr if getattr(sv, "_use_vf", False) else np.ones(len(pid)))
            ball = (balls or {}).get(eid)
            eng = np.zeros(len(pid))                         # engulfed: ball flux, else 0
            if ball is not None:
                inside = np.linalg.norm(mf.centroids - np.asarray(ball.center), axis=1) \
                    <= float(ball.radius)
                eng[inside] = float(ball.flux)
            face_eng.append(eng)
            face_exp.append(np.asarray(getattr(sv, "_exp", np.ones(len(pid))), dtype=float))
            norms_all.append(mf.normals)
            corners_all.append(mf.corners)
        self.n_patches = n_patch
        if n_patch == 0:
            self.F = sp.csr_matrix((0, 0))
            self._empty = True
            return
        self._empty = False
        self.fnodes = np.concatenate(face_nodes)
        self.frow = np.concatenate(face_row)
        area = np.concatenate(face_area)
        self.fpatch = np.concatenate(face_patch)
        self.feps = np.concatenate(face_eps)
        self.fmem = np.concatenate(face_mem)
        self.fF = np.concatenate(face_F)
        self.feng = np.concatenate(face_eng)
        # FireZone partial exposure: faces outside the zone are adiabatic in the base
        # model (no fire, no emission), so they must not take part in the exchange either
        self.fexp = np.concatenate(face_exp)
        norms = np.concatenate(norms_all)
        corners = np.concatenate(corners_all)
        P = n_patch
        self.parea = np.bincount(self.fpatch, weights=area, minlength=P)
        self.peps = np.bincount(self.fpatch, weights=area * self.feps, minlength=P) / self.parea

        # ── Monte Carlo view factors ─────────────────────────────────────────
        order = np.argsort(self.fpatch, kind="stable")
        starts = np.searchsorted(self.fpatch[order], np.arange(P + 1))
        src_face = np.empty(P * rays, np.int64)
        for p in range(P):
            idx = order[starts[p]:starts[p + 1]]
            w = area[idx] / area[idx].sum()
            src_face[p * rays:(p + 1) * rays] = rng.choice(idx, size=rays, p=w)
        u, v = rng.random((2, len(src_face)))
        C = corners[src_face]                                   # bilinear point on the quad
        pts = ((1 - u)[:, None] * ((1 - v)[:, None] * C[:, 0] + v[:, None] * C[:, 3])
               + u[:, None] * ((1 - v)[:, None] * C[:, 1] + v[:, None] * C[:, 2]))
        n = norms[src_face]
        helper = np.where(np.abs(n[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        t1 = np.cross(n, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)
        r1, r2 = rng.random((2, len(src_face)))
        sr = np.sqrt(r1)
        phi = 2 * np.pi * r2
        dirs = (sr * np.cos(phi))[:, None] * t1 + (sr * np.sin(phi))[:, None] * t2 \
            + np.sqrt(1.0 - r1)[:, None] * n                   # cosine-weighted
        O = _offset_origins(scene, pts, n)
        hit, _ = scene.trace(O, dirs, np.inf, tmin=0.0)
        tgt = np.full(len(hit), -1, np.int64)
        ok = hit >= 0
        tgt[ok] = global_face_to_patch[scene.owner[hit[ok]]]
        src_p = np.repeat(np.arange(P), rays)
        keep = tgt >= 0
        F = sp.csr_matrix((np.full(int(keep.sum()), 1.0 / rays), (src_p[keep], tgt[keep])),
                          shape=(P, P))
        # reciprocity: A_i F_ij = A_j F_ji  (symmetrise the exchange areas)
        AF = sp.diags(self.parea) @ F
        AF = 0.5 * (AF + AF.T)
        F = (sp.diags(1.0 / self.parea) @ AF).tocsr()
        # symmetrised Monte Carlo estimates can exceed 1 per row for small patches
        rs = np.asarray(F.sum(axis=1)).ravel()
        self.F = (sp.diags(1.0 / np.maximum(rs, 1.0)) @ F).tocsr()
        self.Fsum = np.asarray(self.F.sum(axis=1)).ravel()
        log.info("Radiation exchange: %d patches, %d rays, %.1f %% of rays hit steel.",
                 P, len(hit), 100.0 * keep.mean())
        self._t_cache: float | None = None
        self._b_cache: np.ndarray | None = None

    def _background(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(G, a): background irradiance [W/m²] and its absorptivity per face at time t."""
        if self._t_cache == t and self._b_cache is not None:
            return self._b_cache
        G_mem = np.empty(len(self._sv))
        vf_mem = np.zeros(len(self._sv), dtype=bool)
        for k, (sv, mode) in enumerate(zip(self._sv, self._mode)):
            if mode in ("amb", "none"):
                G_mem[k] = _SIGMA * self._T0K[k] ** 4
            else:
                TfK4 = (float(sv._fire_temp(t)) + _K0) ** 4
                G_mem[k] = _SIGMA * TfK4 * (float(sv._eps_fire) if mode == "vf" else 1.0)
                vf_mem[k] = mode == "vf"
        G = G_mem[self.fmem] * np.where(vf_mem[self.fmem], self.fF, 1.0)
        a = self.feps.copy()
        eng = self.feng > 0.0
        G[eng] = self.feng[eng]
        a[eng] = 1.0
        self._t_cache, self._b_cache = t, (G, a)
        return G, a

    def load(self, T_global: np.ndarray, t: float) -> np.ndarray:
        """Nodal exchange load Q (n_global,) at temperature iterate T and time t."""
        Q = np.zeros(self._n)
        if self._empty:
            return Q
        G, a = self._background(t)
        T4 = ((T_global[self.fnodes] + _K0) ** 4).mean(axis=1)          # per face
        area = self.frow.sum(axis=1)
        # patch radiosity J = ε σ T⁴ + (1 − ε) G   (area-weighted over the patch's faces)
        J_face = self.feps * _SIGMA * T4 + (1.0 - self.feps) * G
        pJ = np.bincount(self.fpatch, weights=area * J_face, minlength=self.n_patches) \
            / self.parea
        recv = self.F @ pJ                                               # Σ_j F_ij J_j
        q = self.fexp * (self.feps * recv[self.fpatch] - self.Fsum[self.fpatch] * a * G)
        np.add.at(Q, self.fnodes.ravel(), (q[:, None] * self.frow).ravel())
        return Q
