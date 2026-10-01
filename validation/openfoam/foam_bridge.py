"""
Bridge between FAHTS Hex8 SolidMesh and an OpenFOAM (v2412, ESI) solidFoam case.

The SAME hexahedral mesh is written as an OpenFOAM polyMesh (one FV cell per Hex8),
the same boundary conditions are expressed with OpenFOAM patch conditions, the case is
run with ``solidFoam`` and the cell temperatures are read back.

Boundary-condition mapping (per patch):
    "adiabatic"                          → zeroGradient
    ("fixed", T_C)                       → fixedValue
    ("robin", h, Ta(t) table [°C], eps)  → externalWallHeatFluxTemperature, mode coefficient
                                           q = h (Ta − T) + ε σ (Ta⁴ − T⁴)   (exact in OF)
    ("flux", q)                          → externalWallHeatFluxTemperature, mode flux

OpenFOAM is run through ``conda run -n <env>`` so no global install is needed.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fahts.core.heat.solid_mesh import SolidMesh
from fahts.core.heat.solid_mesh._hex_topology import HEX_FACES

log = logging.getLogger(__name__)

CONDA = Path.home() / "anaconda3" / "bin" / "conda"
FOAM_ENV = "fahts-openfoam"
K0 = 273.15

_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
"""


def _write(path: Path, cls: str, obj: str, body: str, loc: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER.format(cls=cls, obj=obj, loc=loc) + body)


# ── mesh ──────────────────────────────────────────────────────────────────────

def write_polymesh(mesh: SolidMesh, patches: dict[str, np.ndarray], case: Path) -> None:
    """
    Write *mesh* as constant/polyMesh.  ``patches`` maps patch name → indices into
    ``mesh.faces``; together they must cover every boundary face exactly once.
    Cell i of the polyMesh is hex i of the SolidMesh.
    """
    hexes = np.asarray(mesh.hexes)
    cell_faces = hexes[:, HEX_FACES]                          # (n_h, 6, 4) outward
    flat = cell_faces.reshape(-1, 4)
    cell_of = np.repeat(np.arange(len(hexes)), 6)
    keys = np.sort(flat, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    # internal faces: pairs sharing a key; owner = lower cell, orientation from owner
    order = np.argsort(inv, kind="stable")
    internal: list[tuple[int, int, np.ndarray]] = []
    k = 0
    while k < len(order):
        i = order[k]
        if counts[inv[i]] == 2:
            j = order[k + 1]
            a, b = (i, j) if cell_of[i] < cell_of[j] else (j, i)
            internal.append((int(cell_of[a]), int(cell_of[b]), flat[a]))
            k += 2
        else:
            k += 1
    internal.sort(key=lambda x: (x[0], x[1]))

    # boundary: map every mesh.faces row to its owning cell via the sorted key
    bkey_to_row = {tuple(r): n for n, r in enumerate(np.sort(np.asarray(mesh.faces), axis=1))}
    b_owner = np.full(len(mesh.faces), -1)
    b_nodes = np.zeros((len(mesh.faces), 4), dtype=np.intp)
    for n in np.flatnonzero(counts[inv] == 1):
        row = bkey_to_row[tuple(keys[n])]
        b_owner[row] = cell_of[n]
        b_nodes[row] = flat[n]                                # outward from the cell
    covered = np.concatenate([np.asarray(v) for v in patches.values()])
    if sorted(covered.tolist()) != list(range(len(mesh.faces))):
        raise ValueError("patches must cover every boundary face exactly once")

    faces: list[np.ndarray] = [f for _, _, f in internal]
    owner: list[int] = [o for o, _, _ in internal]
    neigh: list[int] = [nb for _, nb, _ in internal]
    bnd_entries = []
    start = len(faces)
    for name, idx in patches.items():
        idx = np.asarray(idx, dtype=np.intp)
        faces.extend(b_nodes[idx])
        owner.extend(b_owner[idx].tolist())
        bnd_entries.append((name, len(idx), start))
        start += len(idx)

    pm = case / "constant" / "polyMesh"
    pts = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in np.asarray(mesh.nodes))
    _write(pm / "points", "vectorField", "points",
           f"{len(mesh.nodes)}\n(\n{pts}\n)\n", "constant/polyMesh")
    fl = "\n".join("4(" + " ".join(map(str, f)) + ")" for f in faces)
    _write(pm / "faces", "faceList", "faces", f"{len(faces)}\n(\n{fl}\n)\n", "constant/polyMesh")
    note = (f"note \"nPoints:{len(mesh.nodes)} nCells:{len(hexes)} nFaces:{len(faces)} "
            f"nInternalFaces:{len(internal)}\";\n")
    for obj, arr in (("owner", owner), ("neighbour", neigh)):
        body = f"{len(arr)}\n(\n" + "\n".join(map(str, arr)) + "\n)\n"
        _write(pm / obj, "labelList", obj, body, "constant/polyMesh")
        txt = (pm / obj).read_text().replace("    object", note + "    object", 1)
        (pm / obj).write_text(txt)
    bl = "".join(f"    {n}\n    {{\n        type wall;\n        nFaces {c};\n        startFace {s};\n"
                 f"    }}\n" for n, c, s in bnd_entries)
    _write(pm / "boundary", "polyBoundaryMesh", "boundary",
           f"{len(bnd_entries)}\n(\n{bl})\n", "constant/polyMesh")


# ── case ──────────────────────────────────────────────────────────────────────

@dataclass
class FoamCase:
    """Everything needed to write and run one solidFoam case."""
    path: Path
    mesh: SolidMesh
    patches: dict[str, np.ndarray]
    bcs: dict[str, object]                 # patch → bc spec (see module docstring)
    rho: float
    k_table: np.ndarray                    # (n, 2) [°C, W/mK]
    cp_table: np.ndarray                   # (n, 2) [°C, J/kgK]
    T0_C: float
    t_end: float
    dt: float
    write_dt: float
    ddt: str = "CrankNicolson 0.9"
    # True: impose the Robin condition as a flux (gradient) q = h(Ta−T) + εσ(Ta⁴−T⁴)
    # evaluated from the current face temperature (codedMixed, valueFraction 0).
    # The built-in externalWallHeatFluxTemperature is converted to enthalpy with
    # refValue = h(Ta); with a temperature-dependent Cp that scales the wall flux by
    # c̄p(T_wall→Ta)/cp(T_wall) (≈ +20 % for EN 1993 steel in a fire) — see report.
    robin_as_gradient: bool = True
    n_outer: int = 4
    extra: dict = field(default_factory=dict)

    def _table(self, tab: np.ndarray) -> str:
        return "(\n" + "\n".join(f"            ({T + K0:.6f} {v:.10g})" for T, v in tab) + "\n        )"

    def _bc(self, spec) -> str:
        if spec == "adiabatic":
            return "type zeroGradient;"
        kind = spec[0]
        if kind == "fixed":
            return f"type fixedValue; value uniform {spec[1] + K0:.6f};"
        if kind == "robin" and self.robin_as_gradient:
            return self._robin_gradient(spec)
        if kind == "robin":
            _, h, Ta_tab, eps = spec
            ta = " ".join(f"({t:.6g} {T + K0:.6f})" for t, T in Ta_tab)
            return ("type externalWallHeatFluxTemperature; mode coefficient; kappaMethod solidThermo;"
                    f" h uniform {h}; Ta table ({ta}); emissivity {eps};"
                    f" value uniform {self.T0_C + K0:.6f};")
        if kind == "flux":
            return ("type externalWallHeatFluxTemperature; mode flux; kappaMethod solidThermo;"
                    f" q uniform {spec[1]}; value uniform {self.T0_C + K0:.6f};")
        raise ValueError(f"unknown bc {spec!r}")

    def _robin_gradient(self, spec) -> str:
        _, h, Ta_tab, eps = spec
        n_bc = getattr(self, "_n_coded", 0)
        self._n_coded = n_bc + 1
        ta_t = ", ".join(f"{t:.10g}" for t, _ in Ta_tab)
        ta_v = ", ".join(f"{T + K0:.10g}" for _, T in Ta_tab)
        k_t = ", ".join(f"{T + K0:.10g}" for T, _ in self.k_table)
        k_v = ", ".join(f"{v:.10g}" for _, v in self.k_table)
        return f"""type codedMixed;
        refValue uniform {self.T0_C + K0:.6f}; refGradient uniform 0; valueFraction uniform 0;
        value uniform {self.T0_C + K0:.6f};
        name robinGrad{n_bc};
        code
        #{{
            static const double taT[] = {{{ta_t}}};
            static const double taV[] = {{{ta_v}}};
            static const double kT[] = {{{k_t}}};
            static const double kV[] = {{{k_v}}};
            const int nTa = sizeof(taT)/sizeof(double);
            const int nK = sizeof(kT)/sizeof(double);
            auto lin = [](const double* x, const double* y, int n, double v)
            {{
                if (v <= x[0]) return y[0];
                if (v >= x[n-1]) return y[n-1];
                int lo = 0, hi = n - 1;
                while (hi - lo > 1) {{ int m = (lo + hi)/2; if (x[m] <= v) lo = m; else hi = m; }}
                return y[lo] + (y[hi] - y[lo])*(v - x[lo])/(x[hi] - x[lo]);
            }};
            const scalar t = this->db().time().value();
            const scalar Ta = lin(taT, taV, nTa, t);
            const scalar sigma = 5.67e-8;
            const scalarField& Tw = *this;
            scalarField grad(Tw.size());
            forAll(Tw, i)
            {{
                const scalar q = {h}*(Ta - Tw[i])
                    + {eps}*sigma*(pow4(Ta) - pow4(Tw[i]));
                grad[i] = q/lin(kT, kV, nK, Tw[i]);
            }}
            this->refGrad() = grad;
            this->refValue() = Tw;
            this->valueFraction() = 0.0;
        #}};"""

    def write(self) -> None:
        self._n_coded = 0
        if self.path.exists():
            shutil.rmtree(self.path)
        write_polymesh(self.mesh, self.patches, self.path)
        c = self.path
        _write(c / "constant" / "thermophysicalProperties", "dictionary",
               "thermophysicalProperties", f"""
thermoType
{{
    type heSolidThermo; mixture pureMixture; transport tabulated;
    thermo hTabulated; equationOfState icoPolynomial; specie specie; energy sensibleEnthalpy;
}}
mixture
{{
    specie {{ molWeight 55.85; }}
    transport {{ kappa {self._table(self.k_table)}; }}
    thermodynamics {{ Hf 0; Sf 0; Cp {self._table(self.cp_table)}; }}
    equationOfState {{ rhoCoeffs<8> ({self.rho} 0 0 0 0 0 0 0); }}
}}
""", "constant")
        _write(c / "constant" / "radiationProperties", "dictionary", "radiationProperties",
               "radiation off;\nradiationModel none;\n", "constant")
        bf = "\n".join(f"    {n} {{ {self._bc(self.bcs[n])} }}" for n in self.patches)
        _write(c / "0" / "T", "volScalarField", "T",
               f"dimensions [0 0 0 1 0 0 0];\ninternalField uniform {self.T0_C + K0:.6f};\n"
               f"boundaryField\n{{\n{bf}\n}}\n", "0")
        # p is required by heSolidThermo
        bp = "\n".join(f"    {n} {{ type calculated; value uniform 1e5; }}" for n in self.patches)
        _write(c / "0" / "p", "volScalarField", "p",
               f"dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 1e5;\n"
               f"boundaryField\n{{\n{bp}\n}}\n", "0")
        _write(c / "system" / "controlDict", "dictionary", "controlDict", f"""
application solidFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime {self.t_end}; deltaT {self.dt}; writeControl adjustableRunTime;
writeInterval {self.write_dt}; writeFormat ascii; writePrecision 12;
timeFormat general; timePrecision 8; runTimeModifiable false; adjustTimeStep no;
""", "system")
        _write(c / "system" / "fvSchemes", "dictionary", "fvSchemes", f"""
ddtSchemes {{ default {self.ddt}; }}
gradSchemes {{ default Gauss linear; }}
divSchemes {{ default none; }}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
""", "system")
        _write(c / "system" / "fvSolution", "dictionary", "fvSolution", f"""
solvers
{{
    "h.*" {{ solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }}
}}
PIMPLE {{ nOuterCorrectors {self.n_outer}; nNonOrthogonalCorrectors 2; }}
relaxationFactors {{ equations {{ "h.*" 1; }} }}
""", "system")

    def run(self, timeout: float = 3600.0) -> None:
        ensure_foam_src()
        self.write()
        cmd = [str(CONDA), "run", "-n", FOAM_ENV, "bash", "-c",
               # the conda build's wmake rules read the compiler from GCC/GXX; codedMixed
               # also needs $WM_PROJECT_DIR/src → include/OpenFOAM-2412/src (symlink, see
               # ensure_foam_src()).
               "export GCC=gcc GXX=g++; "
               f"cd '{self.path}' && checkMesh > log.checkMesh 2>&1; "
               f"solidFoam > log.solidFoam 2>&1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log_txt = (self.path / "log.solidFoam").read_text() if (
            self.path / "log.solidFoam").exists() else res.stderr
        if res.returncode != 0 or "\nEnd" not in log_txt:
            raise RuntimeError(f"solidFoam failed in {self.path}:\n{log_txt[-3000:]}")
        chk = (self.path / "log.checkMesh").read_text()
        if "Mesh OK." not in chk:
            raise RuntimeError(f"checkMesh failed in {self.path}:\n{chk[-3000:]}")

    def results(self) -> tuple[np.ndarray, np.ndarray]:
        """(times (n_t,), T_cells (n_t, n_cells) [°C]) incl. t=0."""
        times, fields = [0.0], [np.full(self.mesh.n_hexes, self.T0_C)]
        dirs = []
        for d in self.path.iterdir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            if t > 0 and (d / "T").exists():
                dirs.append((t, d))
        for t, d in sorted(dirs):
            times.append(t)
            fields.append(read_internal_field(d / "T", self.mesh.n_hexes) - K0)
        return np.array(times), np.array(fields)


def ensure_foam_src() -> None:
    """The conda OpenFOAM keeps headers in include/OpenFOAM-2412/src but wmake expects
    $WM_PROJECT_DIR/src (needed to compile codedMixed BCs) — create that symlink."""
    env = CONDA.parent.parent / "envs" / FOAM_ENV
    link, target = env / "src", env / "include" / "OpenFOAM-2412" / "src"
    if not link.exists() and target.is_dir():
        link.symlink_to(target)


def read_internal_field(path: Path, n: int) -> np.ndarray:
    """Read a volScalarField internalField (uniform or nonuniform ascii)."""
    txt = path.read_text()
    i = txt.index("internalField")
    head = txt[i:i + 200]
    if "nonuniform" not in head:
        val = float(head.split("uniform")[1].split(";")[0])
        return np.full(n, val)
    j = txt.index("(", i)
    k = txt.index(")", j)
    arr = np.array(txt[j + 1:k].split(), dtype=float)
    if len(arr) != n:
        raise ValueError(f"{path}: expected {n} values, got {len(arr)}")
    return arr
