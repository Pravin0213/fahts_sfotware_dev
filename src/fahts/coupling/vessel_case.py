"""Input of one vessel-in-fire analysis (the "process side" of a study).

``VesselCase`` holds every input in engineering units (bara, degC, m, mm, kW/m2), validates it,
saves/loads it as JSON (``*.vcase.json``) and converts it to the model's internal case dict
(SI, see ``to_model_case``). The GUI edits a ``VesselCase``; ``fahts.coupling.runner`` runs it.

Conventions
- Liquid inventory: ``water_depth_m`` of free water at the bottom, ``hc_liquid_depth_m`` of
  hydrocarbon liquid on top of it (both measured as depths, horizontal vessel).
- Pressures are absolute (bara).
- Peak (jet) fire zone: length fraction ``xi_start..xi_end`` and arc ``attack_deg +- circ_deg/2``
  measured from the top (180 = bottom).
- Ambient convection: fixed coefficient, or wind speed (forced + natural convection).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from fahts.coupling.options import VesselFireOptions
from fahts.materials import SteelTable, en1993_carbon_steel
from fahts.thermo.component_data import COMPONENTS
from fahts.thermo.constants import ALIASES

SCHEMA = "fahts.vessel_case"
SCHEMA_VERSION = 1
# options of the removed 1-D wall model: ignored (with a warning) in older case files
REMOVED_OPTIONS = ("wall_model", "wall_cells", "wall_nodes")

log = logging.getLogger(__name__)
P_ATM_BAR = 1.01325

PSV_TYPES = ("trapezoidal", "triangular", "square")  # model type index 0 / 1 / 2


@dataclass
class MaterialSpec:
    kind: str = "en1993_carbon"  # "en1993_carbon" (built-in) | "table" (embedded user table)
    table: dict | None = None  # SteelTable.to_dict() when kind == "table"

    def steel_table(self) -> SteelTable:
        if self.kind == "en1993_carbon":
            return en1993_carbon_steel()
        if self.kind == "table" and self.table:
            return SteelTable.from_dict(self.table)
        raise ValueError(f"material: unknown kind {self.kind!r} or missing table")

    @property
    def label(self) -> str:
        return (
            "EN 1993-1-2 carbon steel"
            if self.kind == "en1993_carbon"
            else (self.table or {}).get("name", "user table")
        )


@dataclass
class VesselSpec:
    tag: str = "V-001"
    inner_diameter_m: float = 2.0
    wall_m: float = 0.06
    length_m: float = 6.0  # shell (tangent-tangent) length
    strength_mpa: float = 490.0  # reference strength for the allowable (UTS or yield, see stress)
    material: MaterialSpec = field(default_factory=MaterialSpec)


@dataclass
class ContentsSpec:
    composition: dict[str, float] = field(default_factory=lambda: {"C1": 1.0})  # mole fractions
    pseudo: dict[str, dict] = field(default_factory=dict)  # name -> {"sg": SG or API, "Tb_K": ..}
    P0_bara: float = 100.0
    T0_C: float = 20.0
    hc_liquid_depth_m: float = 0.0
    water_depth_m: float = 0.0
    T_shell_C: float = 20.0


@dataclass
class AmbientSpec:
    T_amb_C: float = 20.0
    convection: str = "wind"  # "wind" | "fixed"
    wind_m_s: float = 0.5
    h_W_m2K: float = 10.0
    emissivity: float = 0.85


@dataclass
class PeakZoneSpec:
    xi_start: float = 0.45
    xi_end: float = 0.55
    circ_deg: float = 40.0
    attack_deg: float = 180.0


@dataclass
class HeatLoadSpec:
    times_s: list[float] = field(default_factory=lambda: [0.0, 3600.0])
    q_background_kW_m2: list[float] = field(default_factory=lambda: [100.0, 100.0])
    q_peak_kW_m2: list[float] = field(default_factory=lambda: [100.0, 100.0])
    peak: PeakZoneSpec = field(default_factory=PeakZoneSpec)

    @property
    def has_fire(self) -> bool:
        return any(q > 0 for q in self.q_background_kW_m2 + self.q_peak_kW_m2)


@dataclass
class BlowdownLineSpec:
    outer_diameter_mm: float = 114.3
    wall_mm: float = 6.02
    length_m: float = 20.0


@dataclass
class BlowdownSpec:
    enabled: bool = False
    orifice_mm: float = 20.0
    cd: float = 0.84
    delay_s: float = 0.0
    line: BlowdownLineSpec | None = None


@dataclass
class PSVSpec:
    enabled: bool = False
    orifice_mm: float = 20.0
    cd: float = 0.975
    set_bara: float = 110.0
    full_open_bara: float = 121.0
    reseat_bara: float = 100.0
    characteristic: str = "trapezoidal"  # trapezoidal | triangular | square


@dataclass
class StressSpec:
    basis: str = "UTS"  # "UTS" | "yield": which strength retention the allowable follows
    factor: float = 1.0  # allowable = strength_mpa * factor * retention(T)
    ext_long_mpa: float = 0.0  # external longitudinal membrane stress


@dataclass
class RunSpec:
    t_end_s: float = 3600.0
    dt_s: float = 1.0
    output_s: float = 10.0


@dataclass
class VesselCase:
    name: str = "New case"
    description: str = ""
    vessel: VesselSpec = field(default_factory=VesselSpec)
    contents: ContentsSpec = field(default_factory=ContentsSpec)
    ambient: AmbientSpec = field(default_factory=AmbientSpec)
    heat_load: HeatLoadSpec = field(default_factory=HeatLoadSpec)
    back_pressure_bara: float = P_ATM_BAR
    blowdown: BlowdownSpec = field(default_factory=BlowdownSpec)
    psv: PSVSpec = field(default_factory=PSVSpec)
    stress: StressSpec = field(default_factory=StressSpec)
    run: RunSpec = field(default_factory=RunSpec)
    options: dict = field(default_factory=dict)  # VesselFireOptions overrides (model choices)

    # ================================================================== validation
    def validate(self) -> list[str]:
        """Errors that prevent a run (empty list = OK)."""
        e = []
        v, c, h = self.vessel, self.contents, self.heat_load
        for label, val in (
            ("inner diameter", v.inner_diameter_m),
            ("wall thickness", v.wall_m),
            ("length", v.length_m),
            ("strength", v.strength_mpa),
            ("initial pressure", c.P0_bara),
            ("end time", self.run.t_end_s),
            ("time step", self.run.dt_s),
            ("output interval", self.run.output_s),
        ):
            if not val > 0:
                e.append(f"{label} must be > 0")
        if c.hc_liquid_depth_m < 0 or c.water_depth_m < 0:
            e.append("liquid depths must be >= 0")
        if c.hc_liquid_depth_m + c.water_depth_m >= v.inner_diameter_m:
            e.append("liquid depths fill the vessel: a gas space is needed")
        if not c.composition:
            e.append("composition is empty")
        elif abs(sum(c.composition.values()) - 1.0) > 1e-6:
            e.append(f"mole fractions sum to {sum(c.composition.values()):.6f}, not 1")
        for name, x in c.composition.items():
            if x < 0:
                e.append(f"{name}: negative mole fraction")
            key = ALIASES.get(name.upper(), name.upper())
            if key not in COMPONENTS and name not in c.pseudo:
                e.append(f"{name}: unknown component (define it as a pseudo-component)")
        for name, p in c.pseudo.items():
            if not (p.get("sg", 0) > 0 and p.get("Tb_K", 0) > 0):
                e.append(f"pseudo-component {name}: SG/API and Tb must be > 0")
        n = len(h.times_s)
        if not all(math.isfinite(v) for v in h.times_s + h.q_background_kW_m2 + h.q_peak_kW_m2):
            e.append("heat load: every time and flux entry must be a number")
        if n < 1 or len(h.q_background_kW_m2) != n or len(h.q_peak_kW_m2) != n:
            e.append("heat load: time, background and peak columns must have equal length >= 1")
        elif any(b <= a for a, b in zip(h.times_s, h.times_s[1:])):
            e.append("heat load: times must be increasing")
        if any(q < 0 for q in h.q_background_kW_m2 + h.q_peak_kW_m2):
            e.append("heat load: fluxes must be >= 0")
        p = h.peak
        if not (0 <= p.xi_start <= p.xi_end <= 1):
            e.append("peak zone: need 0 <= xi_start <= xi_end <= 1")
        if not (0 <= p.circ_deg <= 360):
            e.append("peak zone: circumferential extent must be 0-360 deg")
        if self.blowdown.enabled and not (self.blowdown.orifice_mm > 0 and self.blowdown.cd > 0):
            e.append("blowdown valve: orifice and Cd must be > 0")
        if self.blowdown.enabled and self.blowdown.line is not None:
            ln = self.blowdown.line
            if not (ln.length_m > 0 and ln.outer_diameter_mm > 2 * ln.wall_mm):
                e.append("blowdown line: length > 0 and wall thinner than half the diameter")
        s = self.psv
        if s.enabled:
            if not (s.orifice_mm > 0 and s.cd > 0):
                e.append("PSV: orifice and Cd must be > 0")
            if not (s.reseat_bara <= s.set_bara <= s.full_open_bara):
                e.append("PSV: need reseat <= set <= full-open pressure")
            if s.characteristic not in PSV_TYPES:
                e.append(f"PSV: characteristic must be one of {PSV_TYPES}")
        if self.stress.basis not in ("UTS", "yield"):
            e.append("stress basis must be 'UTS' or 'yield'")
        unknown = set(self.options) - {f.name for f in fields(VesselFireOptions)}
        if unknown:
            e.append(f"unknown model options: {sorted(unknown)}")
        try:
            self.vessel.material.steel_table()
        except (ValueError, KeyError) as exc:
            e.append(f"material: {exc}")
        return e

    # ================================================================== model input
    def to_model_case(self) -> dict:
        """The model's case dict (``seg`` / ``hl`` / ``admin``, SI units)."""
        v, c, a, h = self.vessel, self.contents, self.ambient, self.heat_load
        seg = dict(
            tag=v.tag,
            strength_mpa=v.strength_mpa,
            material=v.material.label,
            D=v.inner_diameter_m,
            t=v.wall_m,
            L=v.length_m,
            P0=c.P0_bara * 1e5,
            T0=c.T0_C + 273.15,
            hc_level=c.hc_liquid_depth_m,
            water_level=c.water_depth_m,
            T_shell=c.T_shell_C + 273.15,
            T_env=a.T_amb_C + 273.15,
            # model convention: h_out > 0 fixed coefficient, < 0 wind speed
            h_out=a.h_W_m2K if a.convection == "fixed" else -abs(a.wind_m_s),
            eps_surf=a.emissivity,
            orientation="H",
            stress_type="U" if self.stress.basis == "UTS" else "Y",
            stress_factor=self.stress.factor,
            ext_long_mpa=self.stress.ext_long_mpa,
            fluid={k.upper(): float(x) for k, x in c.composition.items()},
            back_pressure=self.back_pressure_bara * 1e5,
            bdv=None,
            psv=None,
        )
        if c.pseudo:
            seg["pseudo"] = {k.upper(): dict(rd=p["sg"], Tb=p["Tb_K"]) for k, p in c.pseudo.items()}
        b = self.blowdown
        if b.enabled:
            seg["bdv"] = dict(d=b.orifice_mm / 1e3, cd=b.cd, delay=b.delay_s)
            if b.line is not None:
                seg["bdv_line"] = dict(
                    d=b.line.outer_diameter_mm / 1e3, t=b.line.wall_mm / 1e3, L=b.line.length_m
                )
        s = self.psv
        if s.enabled:
            seg["psv"] = dict(
                d=s.orifice_mm / 1e3,
                cd=s.cd,
                p_set=s.set_bara * 1e5,
                p_full=s.full_open_bara * 1e5,
                p_reseat=s.reseat_bara * 1e5,
                type=PSV_TYPES.index(s.characteristic),
            )
        hl = dict(
            xi_start=h.peak.xi_start,
            xi_end=h.peak.xi_end,
            circ_deg=h.peak.circ_deg,
            attack_deg=h.peak.attack_deg,
            series=np.column_stack([h.times_s, h.q_background_kW_m2, h.q_peak_kW_m2]).astype(float),
        )
        admin = dict(
            maxtime=str(self.run.t_end_s),
            max_timestep=str(self.run.dt_s),
            output_frequence=str(self.run.output_s),
        )
        return dict(seg=seg, hl=hl, admin=admin, name=self.name)

    def model_options(self) -> VesselFireOptions:
        return VesselFireOptions(
            **dict(
                self.options, t_end=self.run.t_end_s, dt=self.run.dt_s, out_every=self.run.output_s
            )
        )

    # ================================================================== JSON
    def to_dict(self) -> dict:
        return dict(schema=SCHEMA, schema_version=SCHEMA_VERSION, **asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "VesselCase":
        if d.get("schema") != SCHEMA:
            raise ValueError("not a fahts vessel case file")
        if d.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError(f"case file version {d['schema_version']} is newer than this program")
        v = dict(d["vessel"])
        v["material"] = MaterialSpec(**v.get("material", {}))
        h = dict(d["heat_load"])
        h["peak"] = PeakZoneSpec(**h.get("peak", {}))
        b = dict(d["blowdown"])
        b["line"] = BlowdownLineSpec(**b["line"]) if b.get("line") else None
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            vessel=VesselSpec(**v),
            contents=ContentsSpec(**d["contents"]),
            ambient=AmbientSpec(**d["ambient"]),
            heat_load=HeatLoadSpec(**h),
            back_pressure_bara=d.get("back_pressure_bara", P_ATM_BAR),
            blowdown=BlowdownSpec(**b),
            psv=PSVSpec(**d["psv"]),
            stress=StressSpec(**d["stress"]),
            run=RunSpec(**d["run"]),
            options=_current_options(d.get("options", {})),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "VesselCase":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ================================================================== conveniences
    @property
    def volume_m3(self) -> float:
        """Shell volume (flat heads, as in the model)."""
        return math.pi / 4 * self.vessel.inner_diameter_m**2 * self.vessel.length_m


def _current_options(options: dict) -> dict:
    old = [k for k in options if k in REMOVED_OPTIONS]
    if old:
        log.warning("ignoring options of the removed 1-D wall model: %s", ", ".join(old))
    return {k: v for k, v in options.items() if k not in REMOVED_OPTIONS}
