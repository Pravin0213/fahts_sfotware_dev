#!/usr/bin/env python3
"""
===========================================================================
 FAHTS-3D: 3D FINITE ELEMENT TRANSIENT HEAT TRANSFER SOLVER

 Governing equation:  ρ cₚ ∂T/∂t = ∇·(k ∇T)   in Ω

 Mesh input  : Nastran BDF (GRID, CTETRA, CHEXA, CPENTA, PSOLID, MAT4)
 Elements    : Tet4 (analytical), Hex8 (2³ Gauss), Penta6 (3×2 Gauss)
 Time scheme : Implicit backward Euler (θ=1), unconditionally stable
 Nonlinearity: Fixed-point iteration (Picard), 3 iterations/step
 BCs         : Dirichlet (penalty), Convection, Radiation, Heat-flux, Fire
 Materials   : Temperature-dependent k/ρ/cₚ; char-in-place decomposition
 Output      : VTK (.vtu) for ParaView, CSV summary, NumPy binary

 Usage:
   python fem_heat_3d.py --mesh model.bdf --bc bc_config.json \
                         --tend 14400 --dt 1.0 --out_interval 60

 Or call programmatically:
   from fem_heat_3d import FEMHeatSolver3D
   solver = FEMHeatSolver3D(mesh_file="model.bdf", bc_file="bc.json")
   solver.run(t_end=14400, dt=1.0)
===========================================================================
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from collections import defaultdict
import os, sys, json, argparse, time as _wt

# ─── Physical constants ─────────────────────────────────────────────────────
SIGMA = 5.67e-8          # Stefan-Boltzmann  [W/(m²·K⁴)]
T_LNG = -163.0           # LNG saturation temperature  [°C]
T_AMBIENT = 20.0
H_FG_LNG = 511e3         # Latent heat  [J/kg]
RHO_LNG = 425.0          # Liquid density  [kg/m³]
PENALTY = 1.0e20         # Dirichlet penalty coefficient

# ─── Nastran node ordering for element faces ─────────────────────────────────
# Each face is a tuple of LOCAL node indices (0-based within element)
TET4_FACES  = [(0,1,2), (0,1,3), (1,2,3), (0,2,3)]           # 4 tri faces
HEX8_FACES  = [(0,1,2,3),(4,5,6,7),(0,1,5,4),                 # 6 quad faces
                (1,2,6,5),(2,3,7,6),(3,0,4,7)]
PENTA6_FACES = [(0,1,2),(3,4,5),(0,1,4,3),(1,2,5,4),(0,2,5,3)] # 2 tri + 3 quad

# Gauss 2×2×2 for Hex8
_G = 1.0 / np.sqrt(3.0)
HEX8_GP  = np.array([[-_G,-_G,-_G],[ _G,-_G,-_G],[ _G, _G,-_G],[-_G, _G,-_G],
                      [-_G,-_G, _G],[ _G,-_G, _G],[ _G, _G, _G],[-_G, _G, _G]])
HEX8_W   = np.ones(8)

# Nastran Hex8 natural node coords
HEX8_NODES = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)

# Gauss 3×2 for Penta6: 3-pt tri rule × 2-pt Gauss in ζ
_P3 = np.array([[1/6,1/6],[2/3,1/6],[1/6,2/3]])
_W3 = np.array([1/6, 1/6, 1/6])          # weights for triangle integral on [0,1]²
_Z2 = np.array([-1/_G, 1/_G])            # ζ Gauss points (reuse _G = 1/√3)
_WZ = np.array([1.0, 1.0])               # ζ weights

# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

def _plin(T, pts):
    """Piecewise linear interpolation from list of (T, val) pairs."""
    if T <= pts[0][0]:  return pts[0][1]
    for i in range(len(pts)-1):
        T1, v1 = pts[i]; T2, v2 = pts[i+1]
        if T1 <= T <= T2:
            return v1 + (T-T1)*(v2-v1)/(T2-T1)
    return pts[-1][1]


def k_alpet(T, **_):
    pts = [(-163,260),(20,235),(400,220),(600,210),(650,205)]
    if T <= 650: return _plin(T, pts)
    if T < 700:  return 205 + (T-650)/50*(95-205)
    return 95 + (T-700)/200*5

def cp_alpet(T, **_):
    cp0 = 984.1
    if T < -100: return cp0*0.75
    if T < 0:    return cp0*(0.75+(T+100)*0.0025)
    if T < 240:  return cp0
    if T < 320:  return cp0+(T-240)/80*(cp0*0.9-cp0)
    if T < 650:  return cp0*0.9
    if T < 700:  return cp0*0.9+(T-650)/50*(800-cp0*0.9)
    return 800.0

def rho_alpet(T, alpha=0):
    if T < 260:  return 1177.5
    if T < 350:  return 1177.5*0.60
    if T < 660:  return 1177.5*0.50
    return 1177.5*0.20


def _kchar_puf(T):
    if T <= 600:  return 0.17
    if T <= 1200: return 0.9
    return 2.7

def k_puf(T, alpha=0, **_):
    T_K = max(T+273.15, 1.0)
    pts = [(-165,0.0145),(24,0.0204),(300,0.0612)]
    ki  = _plin(T, pts)
    if alpha <= 0: return ki
    k_rad  = 4*SIGMA*0.9*T_K**3*0.002
    k_char = _kchar_puf(T) + k_rad
    return ki*(1-alpha) + k_char*alpha

def cp_puf(T, **_):
    pts = [(-163,450),(20,1400),(300,1400)]
    cp = _plin(T, pts)
    if T > 300: return 1400*0.8
    return cp

def rho_puf(T, alpha=0):
    return 41.4*(1-alpha) + 41.4*0.30*alpha


def _kchar_prf(T): return _kchar_puf(T)

def k_prf(T, alpha=0, **_):
    T_K = max(T+273.15, 1.0)
    pts = [(-175,0.0116),(23,0.0349),(425,0.12)]
    ki  = _plin(T, pts)
    if alpha <= 0: return ki
    k_rad  = 4*SIGMA*0.9*T_K**3*0.002
    k_char = _kchar_prf(T) + k_rad
    return ki*(1-alpha) + k_char*alpha

def cp_prf(T, **_):
    pts = [(-163,400),(20,1500),(400,1500)]
    cp = _plin(T, pts)
    if T > 400: return 1500*0.75
    return cp

def rho_prf(T, alpha=0):
    return 25.0*(1-alpha) + 25.0*0.30*alpha


def k_aluminum(T, **_):
    k0 = 120.0
    if T < -200:  return k0*0.58
    if T < -163:  return k0*(0.58+(T+200)/37*0.07)
    if T < 0:     return k0*(0.65+(T+163)/163*0.35)
    if T < 300:   return k0*(1.0+T/300*0.08)
    return k0*1.08

def cp_aluminum(T, **_):
    pts = [(-163,472),(20,900),(660,1200),(800,1100)]
    return _plin(T, pts)

def rho_aluminum(T, alpha=0): return 2650.0


# Built-in material registry (name → (k_func, cp_func, rho_func, T_decomp, T_service))
BUILTIN_MATERIALS = {
    "ALPET":    (k_alpet,    cp_alpet,    rho_alpet,    None,  None),
    "PUF":      (k_puf,      cp_puf,      rho_puf,      300.0, 80.0),
    "PRF":      (k_prf,      cp_prf,      rho_prf,      425.0, 150.0),
    "ALUMINUM": (k_aluminum, cp_aluminum, rho_aluminum, None,  None),
    "AL":       (k_aluminum, cp_aluminum, rho_aluminum, None,  None),
}


class MAT4Props:
    """Constant-property material loaded from MAT4 BDF card."""
    def __init__(self, k, cp, rho):
        self.k = k; self.cp = cp; self.rho = rho

    def k_func(self, T, alpha=0, **_):  return self.k
    def cp_func(self, T, **_):          return self.cp
    def rho_func(self, T, alpha=0):     return self.rho


# ─────────────────────────────────────────────────────────────────────────────
# NASTRAN BDF READER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bdf_line(line):
    """Return list of stripped field strings from a BDF line (fixed or free)."""
    if ',' in line:                         # free-field format
        parts = line.split(',')
        return [p.strip() for p in parts]
    # Fixed format: 8-char fields
    fields = []
    fields.append(line[:8].strip())         # card name (field 1)
    for start in range(8, min(len(line), 80), 8):
        fields.append(line[start:start+8].strip())
    return fields


def _float(s):
    """Convert BDF float string (handles Nastran 1.0-3 → 1.0e-3 notation)."""
    s = s.strip()
    if not s: return 0.0
    import re
    s = re.sub(r'([0-9])([\+\-])([0-9])', r'\1e\2\3', s)
    try: return float(s)
    except ValueError: return 0.0


def _int(s):
    s = s.strip()
    if not s: return 0
    try: return int(s)
    except ValueError: return 0


def read_nastran_bdf(filepath):
    """
    Parse a Nastran BDF file.

    Returns:
        nodes   : dict {node_id: np.array([x,y,z])}
        elements: list of dicts with keys:
                    eid, type ('TET4','HEX8','PENTA6'), pid, conn (list of node ids)
        psolids : dict {pid: mid}
        mat4s   : dict {mid: MAT4Props}
        set3s   : dict {sid: list of ids}
        bsurfs  : dict {sid: list of (eid, g1, g2, g3)}
    """
    nodes   = {}
    elements = []
    psolids  = {}
    mat4s    = {}
    mat1s    = {}
    set3s    = {}
    bsurfs   = {}

    with open(filepath, 'r', errors='replace') as f:
        raw_lines = f.readlines()

    # First pass: strip comments, merge continuation lines
    merged = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip('\n').rstrip('\r')
        # Skip comment and empty
        if not line.strip() or line.strip().startswith('$') or line.strip().startswith('//'):
            i += 1; continue
        # Continuation lines (start with + or ,)
        buf = line
        while i+1 < len(raw_lines):
            nxt = raw_lines[i+1].rstrip('\n').rstrip('\r')
            if nxt and nxt[0] in ('+', ',', '*'):
                buf += ' ' + nxt[1:] if nxt else buf
                i += 1
            else:
                break
        merged.append(buf)
        i += 1

    # Second pass: parse merged lines
    for line in merged:
        f = _parse_bdf_line(line)
        if not f: continue
        card = f[0].upper().split('*')[0]

        if card == 'GRID':
            nid = _int(f[1])
            x   = _float(f[3]) if len(f) > 3 else 0.0
            y   = _float(f[4]) if len(f) > 4 else 0.0
            z   = _float(f[5]) if len(f) > 5 else 0.0
            nodes[nid] = np.array([x, y, z])

        elif card == 'CTETRA':
            eid = _int(f[1]); pid = _int(f[2])
            conn = [_int(f[k]) for k in range(3, min(7, len(f)))]
            if len(conn) == 4:
                elements.append({'eid': eid, 'type': 'TET4', 'pid': pid, 'conn': conn})

        elif card == 'CHEXA':
            eid = _int(f[1]); pid = _int(f[2])
            conn = [_int(f[k]) for k in range(3, min(11, len(f)))]
            if len(conn) == 8:
                elements.append({'eid': eid, 'type': 'HEX8', 'pid': pid, 'conn': conn})

        elif card == 'CPENTA':
            eid = _int(f[1]); pid = _int(f[2])
            conn = [_int(f[k]) for k in range(3, min(9, len(f)))]
            if len(conn) == 6:
                elements.append({'eid': eid, 'type': 'PENTA6', 'pid': pid, 'conn': conn})

        elif card == 'PSOLID':
            pid = _int(f[1]); mid = _int(f[2])
            psolids[pid] = mid

        elif card == 'MAT4':
            mid = _int(f[1])
            k   = _float(f[2]) if len(f) > 2 else 1.0
            cp  = _float(f[3]) if len(f) > 3 else 1000.0
            rho = _float(f[4]) if len(f) > 4 else 1000.0
            mat4s[mid] = MAT4Props(k, cp, rho)

        elif card == 'MAT1':
            mid = _int(f[1])
            rho = _float(f[5]) if len(f) > 5 else 1000.0
            mat1s[mid] = {'rho': rho}

        elif card == 'SET3':
            sid = _int(f[1])
            ids = [_int(f[k]) for k in range(3, len(f)) if f[k].strip()]
            set3s[sid] = ids

        elif card == 'BSURFS':
            sid = _int(f[1])
            for j in range(2, len(f)-2, 4):
                eid = _int(f[j])
                g1  = _int(f[j+1])
                g2  = _int(f[j+2])
                g3  = _int(f[j+3])
                bsurfs.setdefault(sid, []).append((eid, g1, g2, g3))

    print(f"  BDF: {len(nodes)} nodes, {len(elements)} elements "
          f"({sum(1 for e in elements if e['type']=='TET4')} Tet4, "
          f"{sum(1 for e in elements if e['type']=='HEX8')} Hex8, "
          f"{sum(1 for e in elements if e['type']=='PENTA6')} Penta6)")

    return nodes, elements, psolids, mat4s, mat1s, set3s, bsurfs


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT FORMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Tet4 natural derivatives (constant — key advantage)
_TET4_dNdxi = np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def tet4_matrices(X, k_val, rho, cp):
    """
    Analytical stiffness K (4×4) and lumped mass M (4×4) for linear Tet4.

    X     : (4, 3) node coordinates
    k_val : scalar conductivity (isotropic) or (3,3) tensor (anisotropic)
    rho   : density [kg/m³]
    cp    : specific heat [J/(kg·K)]

    Returns K_e, M_e, V
    """
    J     = X.T @ _TET4_dNdxi      # (3,3): J[i,j] = dx_i/dxi_j
    det_J = np.linalg.det(J)
    V     = abs(det_J) / 6.0

    if V < 1e-30:
        return np.zeros((4,4)), np.zeros((4,4)), 0.0

    J_inv = np.linalg.inv(J)
    B     = ((_TET4_dNdxi @ J_inv.T)).T   # (3,4): B[i,k] = dN_k/dx_i

    if np.isscalar(k_val):
        K_e = V * k_val * (B.T @ B)
    else:
        K_e = V * (B.T @ k_val @ B)

    M_e = np.diag(np.full(4, rho * cp * V / 4.0))
    return K_e, M_e, V


def _hex8_dNdxi(xi, eta, zeta):
    """Shape function natural derivatives for Hex8, shape (8,3)."""
    xn = HEX8_NODES
    dN = np.zeros((8, 3))
    dN[:,0] = xn[:,0] * (1 + xn[:,1]*eta)  * (1 + xn[:,2]*zeta) / 8
    dN[:,1] = xn[:,1] * (1 + xn[:,0]*xi)   * (1 + xn[:,2]*zeta) / 8
    dN[:,2] = xn[:,2] * (1 + xn[:,0]*xi)   * (1 + xn[:,1]*eta)  / 8
    return dN


def hex8_matrices(X, k_val, rho, cp):
    """
    2³ Gauss-point stiffness K (8×8) and lumped mass M (8×8) for Hex8.

    X : (8, 3) node coordinates
    """
    K_e = np.zeros((8,8))
    V_tot = 0.0

    for gp, w in zip(HEX8_GP, HEX8_W):
        xi, eta, zeta = gp
        dN = _hex8_dNdxi(xi, eta, zeta)    # (8,3)
        J  = X.T @ dN                       # (3,3)
        dJ = np.linalg.det(J)
        if abs(dJ) < 1e-30: continue
        J_inv = np.linalg.inv(J)
        B  = (dN @ J_inv.T).T               # (3,8)
        if np.isscalar(k_val):
            K_e += w * dJ * k_val * (B.T @ B)
        else:
            K_e += w * dJ * (B.T @ k_val @ B)
        V_tot += w * dJ

    M_e = np.diag(np.full(8, rho * cp * V_tot / 8.0))
    return K_e, M_e, V_tot


def _penta6_dNdxi(xi, eta, zeta):
    """Shape function natural derivatives for Penta6, shape (6,3).
    Natural coords: xi,eta on triangle, zeta in [-1,1].
    """
    # N = [L1*(1-z)/2, L2*(1-z)/2, L3*(1-z)/2, L1*(1+z)/2, L2*(1+z)/2, L3*(1+z)/2]
    # where L1=1-xi-eta, L2=xi, L3=eta
    z = zeta
    L1 = 1 - xi - eta
    dN = np.zeros((6, 3))
    # dN/dxi
    dN[0,0] = -(1-z)/2;  dN[1,0] = (1-z)/2;  dN[2,0] = 0
    dN[3,0] = -(1+z)/2;  dN[4,0] = (1+z)/2;  dN[5,0] = 0
    # dN/deta
    dN[0,1] = -(1-z)/2;  dN[1,1] = 0;  dN[2,1] = (1-z)/2
    dN[3,1] = -(1+z)/2;  dN[4,1] = 0;  dN[5,1] = (1+z)/2
    # dN/dzeta
    dN[0,2] = -L1/2;  dN[1,2] = -xi/2;  dN[2,2] = -eta/2
    dN[3,2] =  L1/2;  dN[4,2] =  xi/2;  dN[5,2] =  eta/2
    return dN


def penta6_matrices(X, k_val, rho, cp):
    """
    3×2 Gauss-point stiffness K (6×6) and lumped mass M (6×6) for Penta6.

    X : (6, 3) node coordinates.
    Triangle rule: 3 points with w=1/6 on [0,1]² simplex.
    Zeta rule    : 2 Gauss points at ±1/√3.
    """
    K_e   = np.zeros((6,6))
    V_tot = 0.0

    for (xi, eta), w_tri in zip(_P3, _W3):
        for z, w_z in zip(_Z2, _WZ):
            dN = _penta6_dNdxi(xi, eta, z)  # (6,3)
            J  = X.T @ dN                    # (3,3)
            dJ = np.linalg.det(J)
            if abs(dJ) < 1e-30: continue
            J_inv = np.linalg.inv(J)
            B  = (dN @ J_inv.T).T            # (3,6)
            # _W3 weights (1/6 each) already include the triangle area factor (sum=1/2).
            # _WZ weights (1 each) span [-1,1]. No additional area pre-factor needed.
            w = w_tri * w_z
            if np.isscalar(k_val):
                K_e += w * abs(dJ) * k_val * (B.T @ B)
            else:
                K_e += w * abs(dJ) * (B.T @ k_val @ B)
            V_tot += w * abs(dJ)

    M_e = np.diag(np.full(6, rho * cp * V_tot / 6.0))
    return K_e, M_e, V_tot


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE FACE INTEGRATION (for convection / radiation BCs)
# ─────────────────────────────────────────────────────────────────────────────

def tri_face_matrices(X_face, h, T_inf):
    """
    Convection K (3×3) and load F (3,) for a 3-node triangular face.

    q = h (T∞ − T)   →   K_face += h·A/12·[2 1 1;1 2 1;1 1 2],  F += h·T∞·A/3
    """
    e1 = X_face[1] - X_face[0]
    e2 = X_face[2] - X_face[0]
    A  = 0.5 * np.linalg.norm(np.cross(e1, e2))
    K_face = h * A / 12.0 * np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
    F_face = h * T_inf * A / 3.0 * np.ones(3)
    return K_face, F_face, A


def quad_face_matrices(X_face, h, T_inf):
    """
    Convection K (4×4) and load F (4,) for a 4-node quadrilateral face.
    2×2 Gauss integration in natural (s, t) ∈ [−1,1]².
    """
    quad_nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=float)
    gp = np.array([-_G, _G])
    K_face = np.zeros((4,4)); F_face = np.zeros(4); A_tot = 0.0

    for s in gp:
        for t in gp:
            N  = (1 + quad_nodes[:,0]*s) * (1 + quad_nodes[:,1]*t) / 4
            dNs = quad_nodes[:,0] * (1 + quad_nodes[:,1]*t) / 4
            dNt = quad_nodes[:,1] * (1 + quad_nodes[:,0]*s) / 4
            dxs = X_face.T @ dNs   # (3,)
            dxt = X_face.T @ dNt   # (3,)
            Jac = np.linalg.norm(np.cross(dxs, dxt))
            K_face += Jac * h * np.outer(N, N)
            F_face += Jac * h * T_inf * N
            A_tot  += Jac

    return K_face, F_face, A_tot


# ─────────────────────────────────────────────────────────────────────────────
# MESH CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Mesh:
    """
    Container for the FEM mesh: nodes, elements, connectivity, and derived data.

    After construction, the mesh builds:
      - node_ids  : list of node IDs in the BDF (ordered by insertion)
      - node_index: dict {bdf_nid → local index 0..N-1}
      - X         : (N, 3) nodal coordinates
      - elems     : list of dicts with 'type', 'eid', 'conn_local' (local indices)
      - boundary_faces: dict {frozenset(local_nodes): (elem_idx, face_idx)}
    """

    def __init__(self, bdf_nodes, bdf_elements, psolids, mat4s, mat1s):
        # Build ordered node list
        self.bdf_node_ids = sorted(bdf_nodes.keys())
        self.node_index   = {nid: i for i, nid in enumerate(self.bdf_node_ids)}
        self.X = np.array([bdf_nodes[nid] for nid in self.bdf_node_ids])
        self.N = len(self.X)

        # Convert element connectivity to local indices
        self.elems = []
        for e in bdf_elements:
            conn_local = [self.node_index[nid] for nid in e['conn'] if nid in self.node_index]
            n_expected = {'TET4': 4, 'HEX8': 8, 'PENTA6': 6}[e['type']]
            if len(conn_local) != n_expected: continue
            self.elems.append({
                'type': e['type'],
                'eid':  e['eid'],
                'pid':  e['pid'],
                'mid':  psolids.get(e['pid'], 0),
                'conn': conn_local,
            })

        self.n_elem = len(self.elems)
        print(f"  Mesh: {self.N} DOFs, {self.n_elem} active elements")

        # Build face connectivity for boundary detection
        self._build_boundary_faces()

    def _build_boundary_faces(self):
        """
        Identify boundary faces (shared by exactly one element).
        self.boundary_faces: dict {frozenset(local_node_ids) → (elem_idx, face_local_idx)}
        """
        face_count = defaultdict(list)
        face_defs  = {'TET4': TET4_FACES, 'HEX8': HEX8_FACES, 'PENTA6': PENTA6_FACES}

        for ei, e in enumerate(self.elems):
            conn = e['conn']
            for fi, face_nodes in enumerate(face_defs[e['type']]):
                global_face = frozenset(conn[n] for n in face_nodes)
                face_count[global_face].append((ei, fi))

        self.boundary_faces = {k: v[0] for k, v in face_count.items() if len(v) == 1}
        print(f"  Boundary faces: {len(self.boundary_faces)}")

    def node_coords(self, local_ids):
        """Return (n, 3) coords for list of local node indices."""
        return self.X[local_ids]

    def face_node_coords(self, elem_idx, face_local_idx):
        """Return (n_face, 3) coordinates of a specific face."""
        e    = self.elems[elem_idx]
        fd   = {'TET4': TET4_FACES, 'HEX8': HEX8_FACES, 'PENTA6': PENTA6_FACES}
        face = fd[e['type']][face_local_idx]
        return self.X[[e['conn'][i] for i in face]], [e['conn'][i] for i in face]


# ─────────────────────────────────────────────────────────────────────────────
# BELTEMP READER (steel temperature history — reused from 1D solver)
# ─────────────────────────────────────────────────────────────────────────────

def load_beltemp(filepath, element_id):
    """Load cumulative steel temperature from a Nastran BELTEMP file."""
    load_cases = []; data = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('LCASETIM'):
                p = s.split()
                load_cases.append((int(p[1]), float(p[2])))
            elif s.startswith('BELTEMP'):
                p = s.split()
                data[int(p[1])][int(p[2])] = float(p[3])

    load_cases.sort(key=lambda x: x[1])
    lc_nums = [lc[0] for lc in load_cases]
    times_min = [lc[1] for lc in load_cases]

    times_sec = [0.0]; temps = [T_AMBIENT]; T = T_AMBIENT
    for i, lc in enumerate(lc_nums):
        T += data[lc].get(element_id, 0.0)
        times_sec.append(times_min[i]*60.0)
        temps.append(T)

    return np.array(times_sec), np.array(temps)


# ─────────────────────────────────────────────────────────────────────────────
# AIR-GAP FIRE FLUX (same physics as 1D solver)
# ─────────────────────────────────────────────────────────────────────────────

def air_gap_flux(T_steel, T_surface, gap_width=0.0017, eps_steel=0.8, eps_surf=0.9):
    """
    Total heat flux across air gap from steel at T_steel to surface at T_surface [°C].
    Returns q [W/m²].
    """
    Ts_K = T_steel  + 273.15
    Ta_K = T_surface + 273.15
    denom = 1.0/eps_steel + 1.0/eps_surf - 1.0
    q_rad = SIGMA * (Ts_K**4 - Ta_K**4) / denom

    T_mean = 0.5*(T_steel + T_surface)
    k_air  = 0.026 + 6e-5*max(T_mean, 0)
    dT     = abs(T_steel - T_surface)
    beta   = 1.0 / max(T_mean + 273.15, 1.0)
    Ra     = max(9.81*beta*dT*gap_width**3 / (15e-6*22e-6), 1.0)
    Nu     = max(1.0, 0.18*Ra**0.29)
    q_conv = Nu * k_air / gap_width * (T_steel - T_surface)

    return q_rad + q_conv


def h_boiling_lng(T_wall, T_sat=T_LNG):
    """Fixed nucleate boiling h for LNG (conservative)."""
    return 2000.0


# ─────────────────────────────────────────────────────────────────────────────
# FEM ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

class Assembler:
    """
    Assembles global sparse K and M from element contributions.

    Material assignment per element:
      If elem['mid'] is in mat4_lib → use constant MAT4Props
      Otherwise map pid to a built-in material name via mat_map dict
        e.g., mat_map = {1: 'ALPET', 2: 'PUF', 3: 'PRF', 4: 'ALUMINUM'}
    """

    def __init__(self, mesh, mat4_lib, mat_map=None,
                 decomp_info=None):
        """
        mesh       : Mesh instance
        mat4_lib   : dict {mid: MAT4Props}
        mat_map    : dict {pid_or_mid: builtin_name_str}  (fallback)
        decomp_info: dict {pid: (T_decomp, T_service)} for decomposable layers
        """
        self.mesh    = mesh
        self.mat4    = mat4_lib
        self.mat_map = mat_map or {}
        self.decomp  = decomp_info or {}
        self.N       = mesh.N

    def _mat_props(self, elem, T_elem, alpha_elem):
        """Return (k, cp, rho) for an element given current T and alpha."""
        mid = elem['mid']
        pid = elem['pid']

        # Check MAT4 library first
        if mid in self.mat4:
            m = self.mat4[mid]
            return m.k_func(T_elem), m.cp_func(T_elem), m.rho_func(T_elem)

        # Fall back to built-in by pid or mid name mapping
        name = self.mat_map.get(pid) or self.mat_map.get(mid)
        if name and name.upper() in BUILTIN_MATERIALS:
            kf, cpf, rf, _, _ = BUILTIN_MATERIALS[name.upper()]
            k   = kf(T_elem,  alpha=alpha_elem)
            cp  = cpf(T_elem)
            rho = rf(T_elem,  alpha=alpha_elem)
            return k, cp, rho

        # Default: steel-like
        return 50.0, 500.0, 7850.0

    def assemble(self, T_nodes, alpha_elem):
        """
        Build sparse K (N×N) and diagonal M (N,) for current temperature field.

        T_nodes   : (N,) nodal temperatures
        alpha_elem: (n_elem,) decomposition fraction per element
        """
        N = self.N
        # Use lil_matrix for efficient incremental assembly
        K_lil = lil_matrix((N, N))
        M_vec = np.zeros(N)           # lumped mass: diagonal only

        for ei, e in enumerate(self.mesh.elems):
            conn = e['conn']
            X_e  = self.mesh.X[conn]
            T_e  = float(np.mean(T_nodes[conn]))   # element-average temperature
            al   = alpha_elem[ei]

            k_val, cp_val, rho_val = self._mat_props(e, T_e, al)

            if e['type'] == 'TET4':
                K_e, M_e, _ = tet4_matrices(X_e, k_val, rho_val, cp_val)
            elif e['type'] == 'HEX8':
                K_e, M_e, _ = hex8_matrices(X_e, k_val, rho_val, cp_val)
            else:
                K_e, M_e, _ = penta6_matrices(X_e, k_val, rho_val, cp_val)

            n = len(conn)
            for a in range(n):
                ia = conn[a]
                M_vec[ia] += M_e[a, a]
                for b in range(n):
                    K_lil[ia, conn[b]] += K_e[a, b]

        return csr_matrix(K_lil), M_vec


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY CONDITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class BCManager:
    """
    Manages all boundary conditions for the 3D heat transfer problem.

    BC types supported:
      - 'dirichlet'  : prescribed T at nodes (penalty method)
      - 'convection' : h, T_inf on surface faces
      - 'radiation'  : epsilon, T_inf on surface faces (linearised)
      - 'heat_flux'  : q [W/m²] on surface faces (Neumann)
      - 'fire'       : air-gap flux from BELTEMP steel temperature
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.bcs  = []         # list of BC dicts

    # ── Node selection utilities ──────────────────────────────────────────

    def select_nodes_bbox(self, axis, xmin, xmax):
        """Return sorted list of local node indices within a bounding box slice."""
        ax = {'x':0, 'y':1, 'z':2}[axis.lower()]
        return sorted(i for i, x in enumerate(self.mesh.X)
                      if xmin - 1e-8 <= x[ax] <= xmax + 1e-8)

    def select_faces_bbox(self, axis, xmin, xmax):
        """
        Return list of (elem_idx, face_local_idx) for boundary faces where
        ALL face nodes fall within the bounding box.
        """
        ax = {'x':0, 'y':1, 'z':2}[axis.lower()]
        result = []
        face_defs = {'TET4': TET4_FACES, 'HEX8': HEX8_FACES, 'PENTA6': PENTA6_FACES}
        for face_nodes_set, (ei, fi) in self.mesh.boundary_faces.items():
            face_nodes = sorted(face_nodes_set)
            coords = self.mesh.X[face_nodes, ax]
            if np.all(coords >= xmin - 1e-8) and np.all(coords <= xmax + 1e-8):
                result.append((ei, fi))
        return result

    # ── BC registration ───────────────────────────────────────────────────

    def add_dirichlet(self, nodes, T_value):
        """Add prescribed temperature BC. nodes: list of local indices."""
        self.bcs.append({'type': 'dirichlet', 'nodes': nodes, 'T': T_value})

    def add_convection(self, faces, h, T_inf):
        """
        Add convection BC. faces: list of (elem_idx, face_local_idx).
        h [W/(m²·K)], T_inf [°C].
        """
        self.bcs.append({'type': 'convection', 'faces': faces, 'h': h, 'T_inf': T_inf})

    def add_radiation(self, faces, epsilon, T_inf):
        """Add radiation BC. epsilon: surface emissivity, T_inf [°C]."""
        self.bcs.append({'type': 'radiation', 'faces': faces, 'eps': epsilon, 'T_inf': T_inf})

    def add_heat_flux(self, faces, q):
        """Add prescribed heat flux BC. q [W/m²] positive = into domain."""
        self.bcs.append({'type': 'heat_flux', 'faces': faces, 'q': q})

    def add_fire(self, faces, steel_times, steel_temps,
                 gap_width=0.0017, eps_steel=0.8, eps_surf=0.9):
        """
        Add fire air-gap BC. The heat flux depends on local surface temperature
        and the interpolated steel temperature at each time step.
        """
        self.bcs.append({'type': 'fire', 'faces': faces,
                         'steel_times': steel_times, 'steel_temps': steel_temps,
                         'gap': gap_width, 'eps_s': eps_steel, 'eps_a': eps_surf})

    # ── BC application ────────────────────────────────────────────────────

    def apply(self, K_csr, M_vec, F, T_nodes, t):
        """
        Modify K (csr), M_vec (diagonal mass), and F (load vector) in-place
        for all registered BCs at time t [s].

        Returns K_csr, M_vec, F (possibly new objects for penalty addition).
        """
        face_defs = {'TET4': TET4_FACES, 'HEX8': HEX8_FACES, 'PENTA6': PENTA6_FACES}

        # Convert K to lil for efficient modification of surface entries
        K_lil = lil_matrix(K_csr)

        for bc in self.bcs:
            btype = bc['type']

            # ── Dirichlet ─────────────────────────────────────────────
            if btype == 'dirichlet':
                T_val = bc['T'] if np.isscalar(bc['T']) else bc['T'](t)
                for i in bc['nodes']:
                    K_lil[i, i] += PENALTY
                    F[i]        += PENALTY * T_val

            # ── Convection / Radiation / Heat-flux / Fire ─────────────
            else:
                faces = bc['faces']
                for ei, fi in faces:
                    e    = self.mesh.elems[ei]
                    etype = e['type']
                    conn  = e['conn']
                    face_local = face_defs[etype][fi]
                    face_global = [conn[k] for k in face_local]
                    X_face = self.mesh.X[face_global]

                    if btype == 'convection':
                        h    = bc['h'] if np.isscalar(bc['h']) else bc['h'](t)
                        Tinf = bc['T_inf'] if np.isscalar(bc['T_inf']) else bc['T_inf'](t)
                        self._apply_convection(K_lil, F, face_global, X_face, h, Tinf)

                    elif btype == 'radiation':
                        eps  = bc['eps']
                        Tinf = bc['T_inf'] if np.isscalar(bc['T_inf']) else bc['T_inf'](t)
                        T_surf = float(np.mean(T_nodes[face_global]))
                        Tinf_K = Tinf + 273.15
                        Ts_K   = T_surf + 273.15
                        h_rad  = SIGMA * eps * (Tinf_K**2 + Ts_K**2) * (Tinf_K + Ts_K)
                        self._apply_convection(K_lil, F, face_global, X_face, h_rad, Tinf)

                    elif btype == 'heat_flux':
                        q = bc['q'] if np.isscalar(bc['q']) else bc['q'](t)
                        self._apply_neumann(F, face_global, X_face, q)

                    elif btype == 'fire':
                        T_steel = float(np.interp(t, bc['steel_times'], bc['steel_temps']))
                        T_surf  = float(np.mean(T_nodes[face_global]))
                        q = air_gap_flux(T_steel, T_surf,
                                         bc['gap'], bc['eps_s'], bc['eps_a'])
                        self._apply_neumann(F, face_global, X_face, q)

        return csr_matrix(K_lil), M_vec, F

    def _apply_convection(self, K_lil, F, face_global, X_face, h, T_inf):
        """Add convection contribution to K and F."""
        n = len(face_global)
        if n == 3:
            Kf, Ff, _ = tri_face_matrices(X_face, h, T_inf)
        else:
            Kf, Ff, _ = quad_face_matrices(X_face, h, T_inf)
        for a, ia in enumerate(face_global):
            F[ia] += Ff[a]
            for b, ib in enumerate(face_global):
                K_lil[ia, ib] += Kf[a, b]

    def _apply_neumann(self, F, face_global, X_face, q):
        """Add Neumann (heat flux) contribution to F only."""
        n = len(face_global)
        if n == 3:
            e1 = X_face[1]-X_face[0]; e2 = X_face[2]-X_face[0]
            A  = 0.5*np.linalg.norm(np.cross(e1, e2))
            Ff = q * A / 3.0 * np.ones(3)
        else:
            Kf, Ff_full, _ = quad_face_matrices(X_face, 0.0, 0.0)
            # for pure Neumann: F += q * ∫N dA
            e1 = X_face[1]-X_face[0]; e2 = X_face[3]-X_face[0]
            A  = np.linalg.norm(np.cross(e1, e2))
            Ff = q * A / 4.0 * np.ones(4)
        for a, ia in enumerate(face_global):
            F[ia] += Ff[a]


# ─────────────────────────────────────────────────────────────────────────────
# VTK WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_vtu(filepath, mesh, T, alpha_elem, t, extra_point=None, extra_cell=None):
    """
    Write a VTK XML UnstructuredGrid file (.vtu) for ParaView/VisIt.

    T         : (N,) nodal temperature array
    alpha_elem: (n_elem,) per-element decomposition fraction
    extra_point: dict of {name: array(N,)} additional point data
    extra_cell : dict of {name: array(n_elem,)} additional cell data
    """
    VTK_TYPE = {'TET4': 10, 'HEX8': 12, 'PENTA6': 13}

    n_pts = mesh.N
    n_cls = mesh.n_elem

    with open(filepath, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cls}">\n')

        # --- Points ---
        f.write('<Points>\n')
        f.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x in mesh.X:
            f.write(f'  {x[0]:.8e} {x[1]:.8e} {x[2]:.8e}\n')
        f.write('</DataArray>\n</Points>\n')

        # --- Cells ---
        f.write('<Cells>\n')
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for e in mesh.elems:
            f.write('  ' + ' '.join(str(n) for n in e['conn']) + '\n')
        f.write('</DataArray>\n')

        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n')
        off = 0
        for e in mesh.elems:
            off += len(e['conn'])
            f.write(f'  {off}\n')
        f.write('</DataArray>\n')

        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        for e in mesh.elems:
            f.write(f'  {VTK_TYPE[e["type"]]}\n')
        f.write('</DataArray>\n')
        f.write('</Cells>\n')

        # --- Point data ---
        f.write('<PointData Scalars="Temperature">\n')
        f.write('<DataArray type="Float64" Name="Temperature_C" format="ascii">\n')
        for v in T: f.write(f'  {v:.4f}\n')
        f.write('</DataArray>\n')

        if extra_point:
            for name, arr in extra_point.items():
                f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
                for v in arr: f.write(f'  {v:.6g}\n')
                f.write('</DataArray>\n')
        f.write('</PointData>\n')

        # --- Cell data ---
        f.write('<CellData Scalars="Decomposition">\n')
        f.write('<DataArray type="Float64" Name="Decomposition_alpha" format="ascii">\n')
        for v in alpha_elem: f.write(f'  {v:.4f}\n')
        f.write('</DataArray>\n')

        f.write('<DataArray type="Int32" Name="Element_PID" format="ascii">\n')
        for e in mesh.elems: f.write(f'  {e["pid"]}\n')
        f.write('</DataArray>\n')

        if extra_cell:
            for name, arr in extra_cell.items():
                f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
                for v in arr: f.write(f'  {v:.6g}\n')
                f.write('</DataArray>\n')
        f.write('</CellData>\n')

        f.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


def write_pvd(pvd_path, vtu_files, times):
    """Write ParaView PVD collection file linking all VTU snapshots with time."""
    with open(pvd_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for vtu, t in zip(vtu_files, times):
            basename = os.path.basename(vtu)
            f.write(f'  <DataSet timestep="{t:.2f}" group="" part="0" file="{basename}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPOSITION UPDATER
# ─────────────────────────────────────────────────────────────────────────────

def update_decomposition(alpha, T_nodes, elems, mat_map, decomp_info, dt):
    """
    Update per-element decomposition fraction α ∈ [0,1].

    decomp_info: dict {pid: {'T_decomp': float, 'T_service': float,
                              'burn_rate': float [fraction/s per mm cell]}}
    """
    for ei, e in enumerate(elems):
        pid = e['pid']
        if pid not in decomp_info: continue
        info = decomp_info[pid]
        T_d  = info['T_decomp']
        T_s  = info['T_service']
        R_max = info['burn_rate']

        T_e = float(np.mean(T_nodes[e['conn']]))
        if T_e <= T_s or alpha[ei] >= 1.0: continue

        if T_e >= T_d:
            rate = R_max
        else:
            frac = (T_e - T_s) / (T_d - T_s)
            rate = R_max * frac**2

        alpha[ei] = min(1.0, alpha[ei] + rate * dt)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC MESH GENERATOR (for testing without a BDF file)
# ─────────────────────────────────────────────────────────────────────────────

def generate_box_mesh_tet4(Lx, Ly, Lz, nx, ny, nz):
    """
    Generate a structured Tet4 mesh on a box [0,Lx]×[0,Ly]×[0,Lz].
    Each hex cell is divided into 6 tetrahedra (standard Kuhn decomposition).

    Returns bdf_nodes, bdf_elements in the format expected by read_nastran_bdf.
    """
    dx = Lx/nx; dy = Ly/ny; dz = Lz/nz
    bdf_nodes = {}
    nid = 1
    idx_map = {}
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                bdf_nodes[nid] = np.array([i*dx, j*dy, k*dz])
                idx_map[(i,j,k)] = nid
                nid += 1

    def n(i,j,k): return idx_map[(i,j,k)]

    bdf_elements = []
    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # 8 corners of current hex cell
                c = [n(i,j,k), n(i+1,j,k), n(i+1,j+1,k), n(i,j+1,k),
                     n(i,j,k+1), n(i+1,j,k+1), n(i+1,j+1,k+1), n(i,j+1,k+1)]
                # 6 tets (Kuhn decomposition of a cube)
                tets = [
                    [c[0],c[1],c[3],c[4]],
                    [c[1],c[2],c[3],c[6]],
                    [c[1],c[4],c[6],c[5]],
                    [c[3],c[4],c[6],c[7]],
                    [c[1],c[3],c[4],c[6]],
                    [c[1],c[2],c[6],c[3]],   # degenerate for some — use 5
                ]
                # Standard 5-tet decomposition (avoids degenerate)
                tets5 = [
                    [c[0],c[1],c[3],c[4]],
                    [c[1],c[2],c[3],c[6]],
                    [c[4],c[5],c[6],c[1]],
                    [c[4],c[6],c[7],c[3]],
                    [c[1],c[3],c[4],c[6]],
                ]
                for tet in tets5:
                    bdf_elements.append({
                        'eid': eid, 'type': 'TET4',
                        'pid': 1,   'conn': tet
                    })
                    eid += 1

    return bdf_nodes, bdf_elements


def generate_layered_hex_mesh(layer_defs, Ly=0.1, Lz=0.1, ny=2, nz=2):
    """
    Generate a structured Hex8 mesh for a layered insulation stack.

    layer_defs: list of (thickness [m], n_cells, pid)
    Ly, Lz    : lateral dimensions
    ny, nz    : lateral subdivisions (2 is sufficient for 1D-like behaviour)

    Returns bdf_nodes, bdf_elements suitable for read_nastran_bdf.
    """
    bdf_nodes    = {}
    bdf_elements = []
    nid = 1; eid = 1

    # Build node grid
    x_coords = [0.0]
    pid_per_xi = []
    for (thick, ncells, pid) in layer_defs:
        dx = thick / ncells
        for _ in range(ncells):
            x_coords.append(x_coords[-1] + dx)
            pid_per_xi.append(pid)

    nx = len(x_coords) - 1
    y_arr = np.linspace(0, Ly, ny+1)
    z_arr = np.linspace(0, Lz, nz+1)

    idx_map = {}
    for ix, x in enumerate(x_coords):
        for iy, y in enumerate(y_arr):
            for iz, z in enumerate(z_arr):
                bdf_nodes[nid] = np.array([x, y, z])
                idx_map[(ix,iy,iz)] = nid
                nid += 1

    def n(ix,iy,iz): return idx_map[(ix,iy,iz)]

    for ix in range(nx):
        pid = pid_per_xi[ix]
        for iy in range(ny):
            for iz in range(nz):
                conn = [
                    n(ix,  iy,  iz),   n(ix+1,iy,  iz),
                    n(ix+1,iy+1,iz),   n(ix,  iy+1,iz),
                    n(ix,  iy,  iz+1), n(ix+1,iy,  iz+1),
                    n(ix+1,iy+1,iz+1), n(ix,  iy+1,iz+1),
                ]
                bdf_elements.append({
                    'eid': eid, 'type': 'HEX8',
                    'pid': pid, 'conn': conn
                })
                eid += 1

    return bdf_nodes, bdf_elements


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SOLVER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FEMHeatSolver3D:
    """
    3D transient heat transfer FEM solver.

    Minimal usage (layered insulation, LNG tank scenario):
        solver = FEMHeatSolver3D()
        solver.setup_lng_insulation_demo()
        results = solver.run(t_end=14400, dt=1.0, out_interval=60)

    Full usage with BDF mesh:
        solver = FEMHeatSolver3D(mesh_file='model.bdf', bc_file='bc.json')
        results = solver.run(t_end=14400, dt=1.0, out_interval=60)
    """

    def __init__(self, mesh_file=None, bc_file=None, output_dir='fem3d_results'):
        self.mesh_file  = mesh_file
        self.bc_file    = bc_file
        self.out_dir    = output_dir
        self.mesh       = None
        self.assembler  = None
        self.bc_mgr     = None
        self.T0         = None           # initial temperature field
        self.decomp_info = {}            # {pid: {T_decomp, T_service, burn_rate}}
        self.mat_map    = {}             # {pid: builtin_material_name}
        os.makedirs(output_dir, exist_ok=True)

    # ── Setup: LNG insulation demo (no external BDF needed) ──────────────

    def setup_lng_insulation_demo(self, beltemp_file=None, element_id=475,
                                  ny=2, nz=2):
        """
        Build a structured Hex8 mesh matching the 1D insulation stack:
          Layer 1 (pid=1): AL-PET   0.2 mm  4 cells
          Layer 2 (pid=2): PUF    100.0 mm 20 cells
          Layer 3 (pid=3): PRF    100.0 mm 20 cells
          Layer 4 (pid=4): Al-alloy 30.0 mm 6 cells

        ny/nz : lateral resolution (use 1–2 for fast 1D-equivalent runs)
        """
        print("\n  Building layered Hex8 mesh (LNG insulation demo)...")

        layer_defs = [
            (0.0002, 4,  1),    # AL-PET
            (0.100,  20, 2),    # PUF
            (0.100,  20, 3),    # PRF
            (0.030,  6,  4),    # Aluminum
        ]
        bdf_nodes, bdf_elements = generate_layered_hex_mesh(
            layer_defs, Ly=0.10, Lz=0.10, ny=ny, nz=nz)

        psolids = {1:1, 2:2, 3:3, 4:4}
        mat4s   = {}
        mat1s   = {}
        set3s   = {}
        bsurfs  = {}

        self.mesh = Mesh(bdf_nodes, bdf_elements, psolids, mat4s, mat1s)

        self.mat_map = {1:'ALPET', 2:'PUF', 3:'PRF', 4:'ALUMINUM'}

        self.decomp_info = {
            2: {'T_decomp': 300.0,  'T_service': 80.0,  'burn_rate': 0.000467},
            3: {'T_decomp': 425.0,  'T_service': 150.0, 'burn_rate': 0.000367},
        }

        self.assembler = Assembler(self.mesh, mat4s, self.mat_map, self.decomp_info)

        # Initial temperature: steady-state gradient (ambient → LNG)
        x_min = self.mesh.X[:,0].min()
        x_max = self.mesh.X[:,0].max()
        x_span = max(x_max - x_min, 1e-10)
        self.T0 = T_AMBIENT + (T_LNG - T_AMBIENT) * (self.mesh.X[:,0] - x_min) / x_span

        # Build BC manager
        self.bc_mgr = BCManager(self.mesh)

        # Fire BC: outer face (x ≈ x_min)
        fire_faces = self.bc_mgr.select_faces_bbox('x', x_min-1e-6, x_min+1e-4)
        print(f"    Fire BC faces: {len(fire_faces)}")

        if beltemp_file and os.path.isfile(beltemp_file):
            steel_times, steel_temps = load_beltemp(beltemp_file, element_id)
            print(f"    Loaded BELTEMP: {len(steel_times)} time points, "
                  f"peak {np.max(steel_temps):.0f}°C")
        else:
            # Synthetic fire curve: ramp to 800°C over 30 min, hold
            steel_times = np.array([0, 1800, 14400], dtype=float)
            steel_temps = np.array([20., 800., 800.], dtype=float)
            print("    Using synthetic fire curve (no BELTEMP file)")

        self.bc_mgr.add_fire(fire_faces, steel_times, steel_temps)

        # LNG BC: inner face (x ≈ x_max)
        lng_faces = self.bc_mgr.select_faces_bbox('x', x_max-1e-4, x_max+1e-6)
        print(f"    LNG BC faces: {len(lng_faces)}")
        self.bc_mgr.add_convection(lng_faces, h=2000.0, T_inf=T_LNG)

        return self

    # ── Setup: from BDF + JSON config ────────────────────────────────────

    def setup_from_files(self):
        """Load mesh from BDF and BCs from JSON configuration file."""
        print(f"\n  Reading BDF: {self.mesh_file}")
        nodes, elements, psolids, mat4s, mat1s, set3s, bsurfs = \
            read_nastran_bdf(self.mesh_file)

        self.mesh      = Mesh(nodes, elements, psolids, mat4s, mat1s)
        self.assembler = Assembler(self.mesh, mat4s, self.mat_map, self.decomp_info)

        # Initial temperature: uniform ambient
        self.T0 = np.full(self.mesh.N, T_AMBIENT)

        self.bc_mgr = BCManager(self.mesh)

        if self.bc_file:
            self._load_bc_json(self.bc_file, set3s, bsurfs, nodes)

        return self

    def _load_bc_json(self, bc_file, set3s, bsurfs, bdf_nodes):
        """Parse BC configuration from JSON file."""
        with open(bc_file) as f:
            cfg = json.load(f)

        def resolve_faces(sel):
            if 'axis' in sel:
                return self.bc_mgr.select_faces_bbox(
                    sel['axis'], sel.get('min', -1e9), sel.get('max', 1e9))
            if 'node_ids' in sel:
                return []   # node-based BCs handled separately
            return []

        def resolve_nodes(sel):
            if 'axis' in sel:
                return self.bc_mgr.select_nodes_bbox(
                    sel['axis'], sel.get('min', -1e9), sel.get('max', 1e9))
            if 'node_ids' in sel:
                return [self.mesh.node_index[n] for n in sel['node_ids']
                        if n in self.mesh.node_index]
            if 'set3_id' in sel:
                return [self.mesh.node_index[n] for n in set3s.get(sel['set3_id'], [])
                        if n in self.mesh.node_index]
            return []

        # Fire BC
        fbc = cfg.get('fire_bc')
        if fbc:
            faces = resolve_faces(fbc.get('node_select', {}))
            bf = fbc.get('beltemp_file')
            eid = fbc.get('element_id', 475)
            if bf and os.path.isfile(bf):
                st, sT = load_beltemp(bf, eid)
            else:
                st = np.array([0, 1800, 14400], dtype=float)
                sT = np.array([20., 800., 800.], dtype=float)
            self.bc_mgr.add_fire(faces, st, sT,
                                  fbc.get('gap_width', 0.0017),
                                  fbc.get('eps_steel', 0.8),
                                  1.0 - 0.1)   # eps_surface default

        # Convection BCs
        for bc in cfg.get('convection_bcs', []) + ([cfg['lng_bc']] if 'lng_bc' in cfg else []):
            faces = resolve_faces(bc.get('node_select', {}))
            self.bc_mgr.add_convection(faces, bc['h'], bc['T_inf'])

        # Dirichlet BCs
        for bc in cfg.get('dirichlet_bcs', []):
            nodes = resolve_nodes(bc.get('node_select', {}))
            self.bc_mgr.add_dirichlet(nodes, bc['T'])

        # Heat flux BCs
        for bc in cfg.get('heat_flux_bcs', []):
            faces = resolve_faces(bc.get('node_select', {}))
            self.bc_mgr.add_heat_flux(faces, bc['q'])

        # Radiation BCs
        for bc in cfg.get('radiation_bcs', []):
            faces = resolve_faces(bc.get('node_select', {}))
            self.bc_mgr.add_radiation(faces, bc['epsilon'], bc['T_inf'])

    # ── Main transient solver ─────────────────────────────────────────────

    def run(self, t_end=14400.0, dt=1.0, out_interval=60.0,
            n_picard=3, tol_picard=0.01, theta=1.0):
        """
        Run the transient FEM solver.

        t_end       : end time [s]
        dt          : time step [s]
        out_interval: VTK output every this many seconds
        n_picard    : max Picard iterations per step
        tol_picard  : convergence tolerance [°C]
        theta       : time integration parameter (1=backward Euler)

        Returns dict with result arrays.
        """
        if self.mesh is None:
            raise RuntimeError("Call setup_lng_insulation_demo() or setup_from_files() first.")

        N      = self.mesh.N
        n_elem = self.mesh.n_elem
        T      = self.T0.copy()
        alpha  = np.zeros(n_elem)
        t      = 0.0

        n_steps    = int(np.ceil(t_end / dt))
        next_out_t = 0.0
        vtu_files  = []
        out_times  = []

        # Summary storage
        summary_times = []
        summary_Tfire = []
        summary_TAlIn = []
        summary_alpha = []

        # Node indices for monitoring
        x_min = self.mesh.X[:,0].min()
        x_max = self.mesh.X[:,0].max()
        fire_nodes = [i for i, x in enumerate(self.mesh.X) if x[0] < x_min + 1e-4]
        al_nodes   = [i for i, x in enumerate(self.mesh.X) if x[0] > x_max - 1e-3]

        print(f"\n  Solving {n_steps} steps (dt={dt}s, t_end={t_end}s)")
        print(f"  Picard iterations: {n_picard}, tol={tol_picard}°C")
        print(f"  Output every {out_interval}s → VTU files in '{self.out_dir}/'")
        print(f"  {'Time':>8s} {'T_fire':>10s} {'T_Al_in':>10s} "
              f"{'alpha_PUF':>10s} {'alpha_PRF':>10s} {'wall_s':>8s}")
        print("  " + "─"*60)

        wall0 = _wt.time()

        for step in range(1, n_steps+1):
            t_new = min(t + dt, t_end)
            dt_   = t_new - t

            T_iter = T.copy()

            # ── Picard iteration ──────────────────────────────────────
            for _it in range(n_picard):
                # Assemble K and M with properties at T_iter
                K, M_vec = self.assembler.assemble(T_iter, alpha)

                # Backward Euler system: A = M/dt + theta*K
                # b = M/dt * T_old + (1-theta)*K*T_old + F
                from scipy.sparse import diags
                M_sp = diags(M_vec, format='csr')
                A    = M_sp * (1.0/dt_) + theta * K
                b    = (M_vec / dt_) * T + (1.0 - theta) * K.dot(T)

                F    = np.zeros(N)
                A, M_vec_bc, F = self.bc_mgr.apply(A, M_vec, F, T_iter, t_new)
                b   += F

                T_new_iter = spsolve(A, b)

                # Convergence check
                err = np.max(np.abs(T_new_iter - T_iter))
                T_iter = T_new_iter
                if err < tol_picard: break

            T = T_iter

            # ── Update decomposition ──────────────────────────────────
            update_decomposition(alpha, T, self.mesh.elems,
                                 self.mat_map, self.decomp_info, dt_)

            t = t_new

            # ── Output ───────────────────────────────────────────────
            if t >= next_out_t - 1e-9:
                vtu_name = os.path.join(self.out_dir, f'heat_{int(t):06d}.vtu')
                write_vtu(vtu_name, self.mesh, T, alpha, t)
                vtu_files.append(vtu_name)
                out_times.append(t)
                next_out_t += out_interval

            # ── Summary storage ───────────────────────────────────────
            T_fire_now = float(np.mean(T[fire_nodes])) if fire_nodes else float(T[0])
            T_al_in    = float(np.mean(T[al_nodes]))   if al_nodes  else float(T[-1])
            alpha_puf  = float(np.mean([alpha[ei] for ei, e in enumerate(self.mesh.elems)
                                        if e['pid'] == 2])) if any(e['pid']==2 for e in self.mesh.elems) else 0.0
            alpha_prf  = float(np.mean([alpha[ei] for ei, e in enumerate(self.mesh.elems)
                                        if e['pid'] == 3])) if any(e['pid']==3 for e in self.mesh.elems) else 0.0
            summary_times.append(t)
            summary_Tfire.append(T_fire_now)
            summary_TAlIn.append(T_al_in)
            summary_alpha.append((alpha_puf, alpha_prf))

            if step % max(1, int(60/dt)) == 0 or step == n_steps:
                wall_s = _wt.time() - wall0
                print(f"  t={t:7.0f}s ({t/60:5.1f}min) "
                      f"T_fire={T_fire_now:7.1f}°C "
                      f"T_Al_in={T_al_in:7.1f}°C "
                      f"α_PUF={alpha_puf:.3f} "
                      f"α_PRF={alpha_prf:.3f} "
                      f"wall={wall_s:6.1f}s")

        # ── Write PVD collection ──────────────────────────────────────
        pvd_path = os.path.join(self.out_dir, 'heat_results.pvd')
        write_pvd(pvd_path, vtu_files, out_times)
        print(f"\n  PVD collection: {pvd_path}")

        # ── Write CSV summary ─────────────────────────────────────────
        csv_path = os.path.join(self.out_dir, 'summary.csv')
        with open(csv_path, 'w') as f:
            f.write("time_s,time_min,T_fire_surface_C,T_Al_inner_C,"
                    "alpha_PUF_mean,alpha_PRF_mean\n")
            for i, t_ in enumerate(summary_times):
                ap, ar = summary_alpha[i]
                f.write(f"{t_:.1f},{t_/60:.3f},"
                        f"{summary_Tfire[i]:.2f},{summary_TAlIn[i]:.2f},"
                        f"{ap:.4f},{ar:.4f}\n")
        print(f"  CSV summary    : {csv_path}")

        # ── Write NumPy binary ────────────────────────────────────────
        npz_path = os.path.join(self.out_dir, 'results.npz')
        np.savez_compressed(npz_path,
                            times=np.array(summary_times),
                            T_fire=np.array(summary_Tfire),
                            T_al_inner=np.array(summary_TAlIn),
                            alpha=np.array(summary_alpha),
                            T_final=T,
                            alpha_final=alpha)
        print(f"  NumPy binary   : {npz_path}")

        total_wall = _wt.time() - wall0
        print(f"\n  Done. Total wall time: {total_wall:.1f}s")

        return {
            'T_final': T,
            'alpha_final': alpha,
            'times': np.array(summary_times),
            'T_fire': np.array(summary_Tfire),
            'T_al_inner': np.array(summary_TAlIn),
            'alpha_arr': np.array(summary_alpha),
            'vtu_files': vtu_files,
            'pvd': pvd_path,
            'csv': csv_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(
        description='FAHTS-3D: 3D Finite Element Transient Heat Transfer Solver')
    p.add_argument('--mesh',        type=str,   default=None,
                   help='Nastran BDF mesh file. Omit to run built-in LNG demo.')
    p.add_argument('--bc',          type=str,   default=None,
                   help='JSON boundary condition configuration file.')
    p.add_argument('--beltemp',     type=str,   default='FAHTS_outer_dome_beltemp.fem',
                   help='BELTEMP file for steel temperature history (demo mode).')
    p.add_argument('--element_id',  type=int,   default=475,
                   help='Nastran element ID in BELTEMP for steel temperature.')
    p.add_argument('--tend',        type=float, default=14400.0,
                   help='End time [s] (default: 14400 = 4 hours).')
    p.add_argument('--dt',          type=float, default=1.0,
                   help='Time step [s] (default: 1.0).')
    p.add_argument('--out_interval',type=float, default=60.0,
                   help='VTK output interval [s] (default: 60).')
    p.add_argument('--n_picard',    type=int,   default=3,
                   help='Max Picard iterations per time step (default: 3).')
    p.add_argument('--outdir',      type=str,   default='fem3d_results',
                   help='Output directory for VTU/CSV/NPZ files.')
    p.add_argument('--ny',          type=int,   default=2,
                   help='Lateral mesh divisions (demo mode, default: 2).')
    p.add_argument('--nz',          type=int,   default=2,
                   help='Lateral mesh divisions (demo mode, default: 2).')
    return p


def main():
    args = build_arg_parser().parse_args()

    print("=" * 70)
    print("  FAHTS-3D: 3D Finite Element Transient Heat Transfer Solver")
    print("  ρ cₚ ∂T/∂t = ∇·(k ∇T)   [implicit backward Euler, Picard NL]")
    print("=" * 70)

    solver = FEMHeatSolver3D(
        mesh_file  = args.mesh,
        bc_file    = args.bc,
        output_dir = args.outdir,
    )

    if args.mesh:
        solver.setup_from_files()
    else:
        print("\n  No --mesh supplied → running LNG insulation demo (Hex8 mesh)")
        solver.setup_lng_insulation_demo(
            beltemp_file = args.beltemp,
            element_id   = args.element_id,
            ny = args.ny, nz = args.nz,
        )

    results = solver.run(
        t_end        = args.tend,
        dt           = args.dt,
        out_interval = args.out_interval,
        n_picard     = args.n_picard,
    )

    print("\n" + "=" * 70)
    print(f"  Peak fire-surface temperature : {np.max(results['T_fire']):.1f}°C")
    print(f"  Peak Al-inner temperature     : {np.max(results['T_al_inner']):.1f}°C")
    final_alpha = results['alpha_arr']
    if len(final_alpha):
        puf_f = final_alpha[-1, 0]; prf_f = final_alpha[-1, 1]
        print(f"  Final PUF decomposition (mean): {puf_f*100:.1f}%")
        print(f"  Final PRF decomposition (mean): {prf_f*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
