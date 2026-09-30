"""Import VessFire input decks (Admin.brl, Segment.brl, Scenario.brl, heatload.scn).

Interoperability: reads the input files a user wrote for VessFire and turns them into a
``VesselCase``. Nothing of VessFire's code or data is used; VessFire material grade names
cannot be resolved without its database, so the built-in EN 1993-1-2 carbon steel is used
and a warning is returned (supply the grade's data as a CSV table).

    case, warnings = case_from_vessfire_deck(folder)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fahts.coupling.vessel_case import (
    PSV_TYPES,
    AmbientSpec,
    BlowdownLineSpec,
    BlowdownSpec,
    ContentsSpec,
    HeatLoadSpec,
    MaterialSpec,
    PeakZoneSpec,
    PSVSpec,
    RunSpec,
    StressSpec,
    VesselCase,
    VesselSpec,
)

P_ATM = 101325.0


def _lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(errors="replace").splitlines()]


def read_deck(case_dir: Path) -> dict:
    """Parse the input deck in ``case_dir`` into the model's case dict (SI units)."""
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
            if len(p) >= 4:  # pseudo: frac, rel. density/API, Tb
                seg.setdefault("pseudo", {})[p[0].upper()] = dict(rd=float(p[2]), Tb=float(p[3]))
            seg["fluid"][p[0].upper()] = float(p[1])
            continue
        in_fluid = False
        key = p[0]
        if key == "#Vessel":
            seg.update(
                tag=p[1],
                strength_mpa=float(p[2]),
                material=p[3],
                D=float(p[4]),
                t=float(p[5]),
                L=float(p[6]),
            )
        elif key == "#Vessel_conditions":
            seg.update(
                P0=float(p[1]) * 1e3,
                T0=float(p[2]),
                hc_level=float(p[3]),
                water_level=float(p[4]),
                T_shell=float(p[p.index("%Shell") + 1]),
            )
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
            seg["psv"] = dict(
                d=float(p[1]),
                cd=float(p[2]),
                p_set=float(p[3]) * 1e3,
                p_full=float(p[4]) * 1e3,
                p_reseat=float(p[5]) * 1e3,
                type=int(p[6]),
            )
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


def case_from_vessfire_deck(case_dir: Path | str) -> tuple[VesselCase, list[str]]:
    """Read a VessFire input deck folder -> (VesselCase, warnings)."""
    case_dir = Path(case_dir)
    d = read_deck(case_dir)
    s, hl, adm = d["seg"], d["hl"], d["admin"]
    warn = []
    if s.get("material"):
        warn.append(
            f"VessFire material {s['material']!r} replaced by the built-in EN 1993-1-2 "
            "carbon steel: import the grade's data as a CSV table for design work"
        )
    if s.get("orientation", "H") != "H":
        warn.append("vertical vessel: the model treats every vessel as horizontal (known issue #2)")
    h_out = s.get("h_out", -0.5)
    ambient = AmbientSpec(
        T_amb_C=s.get("T_env", 293.15) - 273.15,
        convection="fixed" if h_out > 0 else "wind",
        wind_m_s=abs(h_out) if h_out <= 0 else 0.5,
        h_W_m2K=h_out if h_out > 0 else 10.0,
        emissivity=s.get("eps_surf", 0.85),
    )
    ser = np.asarray(hl["series"], float)
    if ser.ndim != 2 or ser.shape[1] < 3:
        ser = np.zeros((1, 3))
    heat = HeatLoadSpec(
        times_s=ser[:, 0].tolist(),
        q_background_kW_m2=ser[:, 1].tolist(),
        q_peak_kW_m2=ser[:, 2].tolist(),
        peak=PeakZoneSpec(
            hl.get("xi_start", 0.45),
            hl.get("xi_end", 0.55),
            hl.get("circ_deg", 40.0),
            hl.get("attack_deg", 180.0),
        ),
    )
    bdv = BlowdownSpec()
    if s.get("bdv"):
        b = s["bdv"]
        line = s.get("bdv_line")
        bdv = BlowdownSpec(
            enabled=True,
            orifice_mm=b["d"] * 1e3,
            cd=b["cd"],
            delay_s=b["delay"],
            line=BlowdownLineSpec(line["d"] * 1e3, line["t"] * 1e3, line["L"]) if line else None,
        )
    psv = PSVSpec()
    if s.get("psv"):
        p = s["psv"]
        psv = PSVSpec(
            enabled=True,
            orifice_mm=p["d"] * 1e3,
            cd=p["cd"],
            set_bara=p["p_set"] / 1e5,
            full_open_bara=p["p_full"] / 1e5,
            reseat_bara=p["p_reseat"] / 1e5,
            characteristic=PSV_TYPES[min(max(int(p["type"]), 0), 2)],
        )
    case = VesselCase(
        name=case_dir.name if case_dir.name != "inputs" else case_dir.parent.name,
        description=f"Imported from VessFire input deck {case_dir}",
        vessel=VesselSpec(
            tag=s.get("tag", "V-001"),
            inner_diameter_m=s["D"],
            wall_m=s["t"],
            length_m=s["L"],
            strength_mpa=s["strength_mpa"],
            material=MaterialSpec(),
        ),
        contents=ContentsSpec(
            composition=dict(s["fluid"]),
            pseudo={k: {"sg": v["rd"], "Tb_K": v["Tb"]} for k, v in s.get("pseudo", {}).items()},
            P0_bara=s["P0"] / 1e5,
            T0_C=s["T0"] - 273.15,
            hc_liquid_depth_m=s["hc_level"],
            water_depth_m=s["water_level"],
            T_shell_C=s["T_shell"] - 273.15,
        ),
        ambient=ambient,
        heat_load=heat,
        back_pressure_bara=s["back_pressure"] / 1e5,
        blowdown=bdv,
        psv=psv,
        stress=StressSpec(
            basis="UTS" if s.get("stress_type", "U") == "U" else "yield",
            factor=s.get("stress_factor", 1.0),
            ext_long_mpa=s.get("ext_long_mpa", 0.0),
        ),
        run=RunSpec(
            t_end_s=float(adm.get("maxtime", 3600)),
            dt_s=float(adm.get("max_timestep", 1)),
            output_s=float(adm.get("output_frequence", 10)),
        ),
    )
    total = sum(case.contents.composition.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        warn.append(f"mole fractions summed to {total:.8f}; normalised")
        case.contents.composition = {k: v / total for k, v in case.contents.composition.items()}
    return case, warn
