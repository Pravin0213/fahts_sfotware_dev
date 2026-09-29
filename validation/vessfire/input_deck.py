"""Reader for VessFire input decks (Admin.brl, Segment.brl, Scenario.brl, heatload.scn).

Interoperability for validation: turns a case folder into the ``case`` dict the process
model takes (``seg``, ``hl``, ``admin``). Units as in the deck except: pressures in Pa,
temperatures in K. From vfpy's vessfire_io.read_case (unchanged).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

P_ATM = 101325.0


def _lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(errors="replace").splitlines()]


def read_case(case_dir: Path) -> dict:
    """Parse the input deck in ``case_dir`` into a case dict."""
    case_dir = Path(case_dir)
    admin = {}
    for ln in _lines(case_dir / "Admin.brl"):
        p = ln.split()
        if len(p) >= 2 and p[0].startswith("#"):
            admin[p[0][1:].lower()] = p[1]

    seg = {"fluid": {}, "bdv": None, "psv": None, "back_pressure": P_ATM}
    in_fluid = False
    for ln in _lines(case_dir / "Segment.brl"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "#Fluid":
            in_fluid = True
            continue
        if in_fluid and not p[0].startswith("#"):
            if len(p) >= 4:                                   # pseudo: frac, rel. density/API, Tb
                seg.setdefault("pseudo", {})[p[0].upper()] = dict(rd=float(p[2]), Tb=float(p[3]))
            seg["fluid"][p[0].upper()] = float(p[1])
            continue
        in_fluid = False
        key = p[0]
        if key == "#Vessel":
            seg.update(tag=p[1], strength_mpa=float(p[2]), material=p[3],
                       D=float(p[4]), t=float(p[5]), L=float(p[6]))
        elif key == "#Vessel_conditions":
            seg.update(P0=float(p[1]) * 1e3, T0=float(p[2]), hc_level=float(p[3]),
                       water_level=float(p[4]), T_shell=float(p[p.index("%Shell") + 1]))
        elif key == "#Vessel_Outside_Conditions":
            seg.update(T_env=float(p[1]), h_out=float(p[2]), eps_surf=float(p[3]))
        elif key == "#Vessel_Orientation":
            seg["orientation"] = p[1].upper()
        elif key == "#StressType":
            seg["stress_type"] = p[1].upper()
        elif key == "#StressFactor":
            seg["stress_factor"] = float(p[1])
        elif key == "#External_Longitudinal_Stress":
            seg["ext_long_mpa"] = float(p[1])
        elif key == "#Blowdown_valve":
            seg["bdv"] = dict(d=float(p[1]), cd=float(p[2]), delay=float(p[3]))
        elif key == "#Blowdown_line":
            seg["bdv_line"] = dict(d=float(p[1]), t=float(p[2]), L=float(p[3]))
        elif key == "#BDV_Valve_location":
            seg["bdv_loc"] = (float(p[1]), float(p[2]))
        elif key == "#Process_safety_valve":
            seg["psv"] = dict(d=float(p[1]), cd=float(p[2]), p_set=float(p[3]) * 1e3,
                              p_full=float(p[4]) * 1e3, p_reseat=float(p[5]) * 1e3,
                              type=int(p[6]))
        elif key == "#Back_pressure":
            seg["back_pressure"] = float(p[1]) * 1e3

    hl = {}
    txt = _lines(case_dir / "heatload.scn")
    num = lambda s: float(s.split(":")[-1])  # noqa: E731
    for ln in txt:
        if ln.startswith("Longitudinal direction, start"):
            hl["xi_start"] = num(ln)
        elif ln.startswith("end [0 - 1]"):
            hl["xi_end"] = num(ln)
        elif ln.startswith("Circumferential length"):
            hl["circ_deg"] = num(ln)
        elif ln.startswith("Angle of attack"):
            hl["attack_deg"] = num(ln)
    rows = []
    for ln in txt:
        p = ln.split()
        if len(p) == 3:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    hl["series"] = np.array(rows) if rows else np.zeros((1, 3))
    return dict(admin=admin, seg=seg, hl=hl, name=case_dir.name)




def parse_fluid_lines(lines) -> tuple[dict, dict]:
    """Parse '#Fluid' lines ('NAME frac' or 'NAME frac SG Tb') -> (composition, pseudo)."""
    comp, pseudo = {}, {}
    for ln in lines:
        t = ln.split("%")[0].split()
        if len(t) < 2 or t[0].startswith("#"):
            continue
        comp[t[0].upper()] = float(t[1])
        if len(t) >= 4:
            pseudo[t[0].upper()] = {"sg": float(t[2]), "Tb": float(t[3])}
    return comp, pseudo
