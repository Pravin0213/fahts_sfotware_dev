"""
fahts/core/io/usfos_reader.py
─────────────────────────────
USFOS .fem file reader.

Parses the USFOS bulk-data text format into an FEMModel.

Cards handled
─────────────
  NODE       — node ID, X Y Z, optional BC flags
  BEAM       — element ID, n1, n2, mat_id, geom_id, lcoor_id
  BOX        — hollow rectangular cross-section geometry
  PIPE       — circular hollow pipe cross-section
  IHPROFIL   — I-beam cross-section
  QUADSHEL   — 4-node quadrilateral shell element
  TRISHELL   — 3-node triangular shell element
  MISOIEP    — isotropic elastic-plastic steel material
  UNITVEC    — local z-axis direction vectors (used for beam orientation)
  BEAMHING   — moment release flags (stored but not used in thermal)
  PLTHICK    — plate thickness (used by QUADSHEL/TRISHELL)
  Name Group / GroupDef  — named element sets

Format rules
────────────
  - Comment lines: first non-space character is '  (single quote) → skip
  - Free-space delimited (NOT fixed-width Nastran)
  - GroupDef cards continue across multiple lines (all continuation lines
    start with whitespace after the GroupDef header)

Usage
─────
    from pathlib import Path
    from fahts.core.io.usfos_reader import read_usfos_fem

    model = read_usfos_fem(Path("model_t1.fem"))
    print(f"Nodes: {len(model.nodes)}, Elements: {model.n_elements}")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fahts.core.model.fem_model import FEMModel
from fahts.core.model.node import Node
from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.section import BoxSection, PipeSection, ISection, PlateSection
from fahts.core.model.material import SteelMaterial
from fahts.core.model.group import Group

log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def read_usfos_fem(filepath: Path | str) -> FEMModel:
    """
    Parse a USFOS .fem file and return a fully populated FEMModel.

    Parameters
    ----------
    filepath : path to the .fem file

    Returns
    -------
    FEMModel

    Raises
    ------
    FileNotFoundError  — if the file does not exist
    ValueError         — if required cards are missing or malformed
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FEM file not found: {filepath}")

    log.info("Reading USFOS .fem: %s", filepath)

    parser = _UsfosParser()
    parser.parse(filepath)
    model = parser.build_model(filepath)

    log.info(
        "Loaded: %d nodes, %d beams, %d shells, %d sections, %d materials, %d groups",
        len(model.nodes), len(model.elements), len(model.shell_elements),
        len(model.sections), len(model.materials), len(model.groups),
    )
    return model


# ── Internal parser ───────────────────────────────────────────────────────────

class _UsfosParser:
    """
    Single-pass parser that accumulates raw data from each card type,
    then assembles them into the FEMModel.
    """

    def __init__(self) -> None:
        self._nodes:    dict[int, list] = {}   # nid → [x, y, z, bc_flags]
        self._beams:    dict[int, list] = {}   # eid → [n1, n2, mat, geom, lcoor, ecc1_id, ecc2_id]
        self._shells:   dict[int, list] = {}   # eid → [nodes_tuple, mat, geom]
        self._boxes:    dict[int, list] = {}   # gid → [H, T_side, T_bot, T_top, W]
        self._pipes:    dict[int, list] = {}   # gid → [outer_diameter, thickness]
        self._isects:   dict[int, list] = {}   # gid → [h, tw, bf1, tf1, bf2, tf2]
        self._mats:     dict[int, list] = {}   # mid → [E, nu, fy, rho, alpha]
        self._unitvecs: dict[int, list] = {}   # vid → [dx, dy, dz]
        self._eccents:  dict[int, list] = {}   # eid → [Ex, Ey, Ez] global offset [m]
        self._beamhinge: list[dict]     = []
        self._plthick:  dict[int, float] = {}
        # Groups: two-pass (Name Group registers name, GroupDef populates members)
        self._group_names: dict[int, str] = {}   # gid → name
        self._group_elems: dict[int, list[int]] = {}  # gid → [eid, ...]
        self._current_groupdef_id: int | None = None

    # ── File parsing ──────────────────────────────────────────────────────────

    def parse(self, filepath: Path) -> None:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n").rstrip("\r")
            stripped = line.strip()

            # Skip blank lines and comment lines (start with ')
            if not stripped or stripped.startswith("'"):
                if self._current_groupdef_id is not None and not stripped:
                    self._current_groupdef_id = None
                i += 1
                continue

            tokens = stripped.split()
            keyword = tokens[0].upper()

            if keyword == "NODE":
                self._parse_node(tokens)
                self._current_groupdef_id = None

            elif keyword == "BEAM":
                self._parse_beam(tokens)
                self._current_groupdef_id = None

            elif keyword == "BOX":
                self._parse_box(tokens)
                self._current_groupdef_id = None

            elif keyword == "PIPE":
                self._parse_pipe(tokens)
                self._current_groupdef_id = None

            elif keyword == "IHPROFIL":
                self._parse_ihprofil(tokens)
                self._current_groupdef_id = None

            elif keyword == "QUADSHEL":
                self._parse_quadshel(tokens)
                self._current_groupdef_id = None

            elif keyword == "TRISHELL":
                self._parse_trishell(tokens)
                self._current_groupdef_id = None

            elif keyword == "MISOIEP":
                self._parse_material(tokens)
                self._current_groupdef_id = None

            elif keyword == "UNITVEC":
                self._parse_unitvec(tokens)
                self._current_groupdef_id = None

            elif keyword == "ECCENT":
                self._parse_eccent(tokens)
                self._current_groupdef_id = None

            elif keyword == "BEAMHING":
                self._parse_beamhing(tokens)
                self._current_groupdef_id = None

            elif keyword == "PLTHICK":
                self._parse_plthick(tokens)
                self._current_groupdef_id = None

            elif keyword == "NAME":
                self._parse_name_group(tokens)
                self._current_groupdef_id = None

            elif keyword == "GROUPDEF":
                self._parse_groupdef_header(tokens)
                i += 1
                while i < len(lines):
                    cont = lines[i].rstrip("\n").rstrip("\r")
                    cont_stripped = cont.strip()
                    if cont_stripped and cont[0] in (" ", "\t"):
                        self._parse_groupdef_continuation(cont_stripped.split())
                        i += 1
                    else:
                        break
                continue  # skip the i += 1 at bottom

            elif keyword in ("HEAD", "GRAVITY"):
                pass

            else:
                log.debug("Unrecognised card (ignored): %s", keyword)
                self._current_groupdef_id = None

            i += 1

    # ── Card parsers ──────────────────────────────────────────────────────────

    def _parse_node(self, tok: list[str]) -> None:
        # NODE  id  X  Y  Z  [BC_x  BC_y  BC_z  ...]
        nid = int(tok[1])
        x   = float(tok[2])
        y   = float(tok[3])
        z   = float(tok[4])
        bc  = tuple(int(v) for v in tok[5:11]) if len(tok) > 5 else ()
        self._nodes[nid] = [x, y, z, bc]

    def _parse_beam(self, tok: list[str]) -> None:
        # BEAM  eid  n1  n2  mat_id  geom_id  lcoor_id  [ecc1_id  ecc2_id]
        eid     = int(tok[1])
        n1      = int(tok[2])
        n2      = int(tok[3])
        mat_id  = int(tok[4])
        geom_id = int(tok[5])
        lcoor   = int(tok[6]) if len(tok) > 6 else 1
        ecc1_id = int(tok[7]) if len(tok) > 7 else 0
        ecc2_id = int(tok[8]) if len(tok) > 8 else 0
        self._beams[eid] = [n1, n2, mat_id, geom_id, lcoor, ecc1_id, ecc2_id]

    def _parse_box(self, tok: list[str]) -> None:
        # BOX  id  H  T_side  T_bot  T_top  Width  [Sh_y  Sh_z]
        gid    = int(tok[1])
        H      = float(tok[2])
        T_side = float(tok[3])
        T_bot  = float(tok[4])
        T_top  = float(tok[5])
        W      = float(tok[6]) if len(tok) > 6 else H
        self._boxes[gid] = [H, T_side, T_bot, T_top, W]

    def _parse_pipe(self, tok: list[str]) -> None:
        # PIPE  id  outer_diameter  wall_thickness
        gid = int(tok[1])
        od  = float(tok[2])
        t   = float(tok[3])
        self._pipes[gid] = [od, t]

    def _parse_ihprofil(self, tok: list[str]) -> None:
        # IHPROFIL  id  h  tw  bf_top  tf_top  bf_bot  tf_bot
        gid    = int(tok[1])
        h      = float(tok[2])
        tw     = float(tok[3])
        bf_top = float(tok[4])
        tf_top = float(tok[5])
        bf_bot = float(tok[6]) if len(tok) > 6 else bf_top
        tf_bot = float(tok[7]) if len(tok) > 7 else tf_top
        self._isects[gid] = [h, tw, bf_top, tf_top, bf_bot, tf_bot]

    def _parse_quadshel(self, tok: list[str]) -> None:
        # QUADSHEL  eid  n1  n2  n3  n4  mat_id  geom_id
        eid    = int(tok[1])
        nodes  = (int(tok[2]), int(tok[3]), int(tok[4]), int(tok[5]))
        mat_id = int(tok[6])
        geom_id = int(tok[7]) if len(tok) > 7 else 0
        self._shells[eid] = [nodes, mat_id, geom_id]

    def _parse_trishell(self, tok: list[str]) -> None:
        # TRISHELL  eid  n1  n2  n3  mat_id  geom_id
        eid    = int(tok[1])
        nodes  = (int(tok[2]), int(tok[3]), int(tok[4]))
        mat_id = int(tok[5])
        geom_id = int(tok[6]) if len(tok) > 6 else 0
        self._shells[eid] = [nodes, mat_id, geom_id]

    def _parse_material(self, tok: list[str]) -> None:
        # MISOIEP  id  E  nu  fy  rho  alpha_T
        mid   = int(tok[1])
        E     = float(tok[2])
        nu    = float(tok[3])
        fy    = float(tok[4])
        rho   = float(tok[5]) if len(tok) > 5 else 7850.0
        alpha = float(tok[6]) if len(tok) > 6 else 1.2e-5
        self._mats[mid] = [E, nu, fy, rho, alpha]

    def _parse_unitvec(self, tok: list[str]) -> None:
        # UNITVEC  id  dx  dy  dz
        vid = int(tok[1])
        vec = [float(tok[2]), float(tok[3]), float(tok[4])]
        self._unitvecs[vid] = vec

    def _parse_eccent(self, tok: list[str]) -> None:
        # ECCENT  id  Ex  Ey  Ez  — offset in global coordinates [m]
        eid = int(tok[1])
        vec = [float(tok[2]), float(tok[3]), float(tok[4])]
        self._eccents[eid] = vec

    def _parse_beamhing(self, tok: list[str]) -> None:
        # BEAMHING  f1 f2 f3 f4 f5 f6  f1 f2 f3 f4 f5 f6  eid
        try:
            eid    = int(tok[-1])
            end1   = tuple(int(v) for v in tok[1:7])
            end2   = tuple(int(v) for v in tok[7:13])
            self._beamhinge.append({"eid": eid, "end1": end1, "end2": end2})
        except (ValueError, IndexError):
            log.debug("Malformed BEAMHING card: %s", " ".join(tok))

    def _parse_plthick(self, tok: list[str]) -> None:
        # PLTHICK  id  thickness
        pid = int(tok[1])
        t   = float(tok[2])
        self._plthick[pid] = t

    def _parse_name_group(self, tok: list[str]) -> None:
        # Name  Group  id  name_string
        if len(tok) >= 4:
            gid  = int(tok[2])
            name = tok[3]
            self._group_names[gid]  = name
            self._group_elems[gid]  = []

    def _parse_groupdef_header(self, tok: list[str]) -> None:
        # GroupDef  id  Elem  id1  id2  ...
        if len(tok) >= 2:
            gid = int(tok[1])
            if gid not in self._group_elems:
                self._group_elems[gid] = []
            self._current_groupdef_id = gid
            for v in tok[3:]:
                try:
                    self._group_elems[gid].append(int(v))
                except ValueError:
                    pass  # "Elem" keyword

    def _parse_groupdef_continuation(self, tok: list[str]) -> None:
        gid = self._current_groupdef_id
        if gid is None:
            return
        for v in tok:
            try:
                self._group_elems[gid].append(int(v))
            except ValueError:
                pass

    # ── Model builder ─────────────────────────────────────────────────────────

    def build_model(self, source_file: Path) -> FEMModel:
        """Assemble FEMModel from parsed raw data."""

        # Nodes
        nodes: dict[int, Node] = {}
        for nid, (x, y, z, bc) in self._nodes.items():
            nodes[nid] = Node(nid=nid, x=x, y=y, z=z, bc=bc)

        # Materials
        materials: dict[int, SteelMaterial] = {}
        for mid, (E, nu, fy, rho, alpha) in self._mats.items():
            materials[mid] = SteelMaterial(
                mid=mid, E=E, nu=nu, fy=fy, rho=rho, alpha_T=alpha
            )

        # Sections
        sections: dict = {}
        for gid, (H, T_side, T_bot, T_top, W) in self._boxes.items():
            sections[gid] = BoxSection(
                sid=gid, H=H, T_side=T_side, T_bot=T_bot, T_top=T_top, W=W
            )
        for gid, (od, t) in self._pipes.items():
            sections[gid] = PipeSection(sid=gid, outer_diameter=od, thickness=t)
        for gid, (h, tw, bf_top, tf_top, bf_bot, tf_bot) in self._isects.items():
            sections[gid] = ISection(
                sid=gid, h=h, tw=tw,
                bf_top=bf_top, tf_top=tf_top,
                bf_bot=bf_bot, tf_bot=tf_bot,
            )
        for pid, t in self._plthick.items():
            sections[pid] = PlateSection(sid=pid, thickness=t)

        # Unit vectors
        unitvecs: dict[int, np.ndarray] = {
            vid: np.array(v, dtype=float)
            for vid, v in self._unitvecs.items()
        }

        # Eccentricity vectors (global coordinates)
        eccents: dict[int, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in self._eccents.items()
        }

        # Beam elements (compute derived geometry)
        elements: dict[int, BeamElement] = {}
        for eid, (n1, n2, mat_id, geom_id, lcoor, ecc1_id, ecc2_id) in self._beams.items():
            elem = BeamElement(
                eid=eid, n1=n1, n2=n2,
                mat_id=mat_id, geom_id=geom_id, lcoor_id=lcoor,
            )
            # Eccentricity offsets (0 means no eccentricity)
            if ecc1_id and ecc1_id in eccents:
                elem.ecc1 = eccents[ecc1_id]
            if ecc2_id and ecc2_id in eccents:
                elem.ecc2 = eccents[ecc2_id]
            if n1 in nodes and n2 in nodes:
                p1 = np.array([nodes[n1].x, nodes[n1].y, nodes[n1].z])
                p2 = np.array([nodes[n2].x, nodes[n2].y, nodes[n2].z])
                # Direction uses eccentric endpoints when available
                p1_eff = p1 + elem.ecc1 if elem.ecc1 is not None else p1
                p2_eff = p2 + elem.ecc2 if elem.ecc2 is not None else p2
                vec = p2_eff - p1_eff
                L   = float(np.linalg.norm(vec))
                elem.length    = L
                elem.direction = vec / L if L > 1e-12 else vec
            if lcoor in unitvecs:
                elem.local_z = unitvecs[lcoor]
            elements[eid] = elem

        # Shell elements
        shell_elements: dict[int, ShellElement] = {}
        for eid, (node_ids, mat_id, geom_id) in self._shells.items():
            shell_elements[eid] = ShellElement(
                eid=eid, nodes=node_ids, mat_id=mat_id, geom_id=geom_id
            )

        # Groups
        groups: dict[str, Group] = {}
        for gid, name in self._group_names.items():
            eids = set(self._group_elems.get(gid, []))
            groups[name] = Group(gid=gid, name=name, element_ids=eids)

        if not nodes:
            raise ValueError("No NODE cards found — check the .fem file.")
        if not elements and not shell_elements:
            raise ValueError("No BEAM or shell element cards found — check the .fem file.")

        return FEMModel(
            nodes=nodes,
            elements=elements,
            sections=sections,
            materials=materials,
            groups=groups,
            unitvecs=unitvecs,
            source_file=source_file,
            shell_elements=shell_elements,
        )
