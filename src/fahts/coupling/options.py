"""Options of the coupled vessel-in-fire model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VesselFireOptions:
    """Model options of the vessel-in-fire simulation. Defaults = VessFire-matching
    profile (legacy/vfpy/MODEL_CHOICES.md, N5); see that file for the evidence."""

    flux: str = "blackbody"  # blackbody | balance | none
    eps_surf_fire: float = 0.7
    h_fire: float = 25.0
    h_corr: str = "evans_stefany"  # wall -> gas natural convection. VessFire-matching
    # default with the nucleate-only wet wall (study 25);
    # "churchill_chu" (horizontal cylinder) is the alternative
    h_corr_liq: str | None = "churchill_chu"  # wall -> single-phase (dense) liquid
    wet_above_crit: str = "boiling"  # wetted-wall heat transfer when the pool cannot
    # boil (P above its cricondenbar / latent heat collapsed):
    #  "single-phase"  natural convection (physics default)
    #  "supercritical" natural convection with integrated cp
    #                  (pseudo-critical enhancement; Jackson &
    #                  Hall 1979, Pioro & Duffey 2005)
    #  "boiling"       keep nucleate-boiling heat transfer with
    #                  the pool at saturation (VessFire-like,
    #                  see MODEL_CHOICES.md)
    psv_liquid: str = "liquid"  # PSV flow when it draws from the liquid/dense zone:
    # "gas" (API 520 vapour equation, real Z, k) or "liquid"
    # (API 520 liquid equation, Cd A sqrt(2 rho dP), Kw=Kv=1;
    # VessFire-matching: its dense-phase PSV rates equal this
    # within 0.3 %, studies/25_top_fixes/dense_relief.csv)
    water_mode: str = "vf"  # free-water pool heat: "physical" (IAPWS pool heats and
    # boils), "sink" (pool keeps wall contact, heat it receives
    # is removed), "vf" (sink only while the pool holds < 2 kg
    # hydrocarbon liquid; VessFire-matching, study 23)
    wet_boiling: str = "nucleate_only"  # wetted-wall boiling curve: "full" (nucleate -> CHF ->
    # transition -> film; physics) or "nucleate_only" (Cooper
    # nucleate boiling, no CHF cap; VessFire-matching: its wet
    # wall stays 2-4 K above the liquid all hour, study 22)
    rad_internal: bool = True  # radiation inside the vapour space (dry wall, liquid
    # surface, grey gas) - radiosity network
    eps_wall_in: float = 0.8  # inner (oxidised steel) wall emissivity
    eps_liq: float = 0.95  # liquid free-surface emissivity
    eps_gas: float = 0.0  # grey-gas emissivity of the vapour (0 = transparent)
    liq_grashof: str = "drho"  # "beta" (g beta dT) or "drho" (density difference across
    # the film, Jackson & Hall 1979) for single-phase liquid
    interface: bool = True  # interface heat exchange
    interface_mass: bool = False  # interface evaporation/condensation (T_i = T_bubble);
    # False = sensible exchange, phase change by zone flashes
    boiling: str = "cooper"  # nucleate boiling correlation
    dt: float | None = None
    out_every: float | None = None
    t_end: float | None = None
    sat_every: float = 10.0  # s between saturation-property refreshes
    max_steps_warn: int = 0
    line_diameter: str = "outer"  # blowdown-line diameter input: "outer" (inner = d - 2t,
    # default) or "inner" (d is the bore). The manual does not
    # say; published line physics gives 6-9 % vs VessFire on the
    # LPG cases with d - 2t and 20-37 % with d (study 14).
    use_line: bool = True  # friction in the blowdown line
    valve_model: str = "b1"  # "b1": relief.blowdown (API 520 orifice + Borda-Carnot
    # expansion + real-gas PR Fanno line); "simple": relief.ideal_gas
    wall_cells: int = 10  # radial cells through the steel wall (wall_column.radial_nodes)
    peak_zone: bool = True  # model the heat load's local peak zone (jet fire) as separate wall
    # regions; False = background flux everywhere (vfpy vessel2 behaviour)
    wall_nodes: tuple[float, ...] | None = None  # explicit node positions [m] from the inner
    # surface; overrides wall_cells (e.g. to reproduce another code's grid)
