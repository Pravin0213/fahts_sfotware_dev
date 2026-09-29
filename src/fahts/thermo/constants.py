"""Constants, component aliases and binary interaction parameters for the PR model.

Sources: see the docstring of ``fahts.thermo.pr_mixture``.
"""

from __future__ import annotations

import math


R = 8.314462618
SQ2 = math.sqrt(2.0)
D1 = 1.0 + SQ2
D2 = 1.0 - SQ2
T_REF = 298.15
P_REF = 1.0e5
ZFLOOR = 1e-30          # floor on mole fractions inside the flash algorithms
B_VC_PR = 0.25308       # b/Vc at the PR critical point (0.07780/0.3074)

ALIASES = {"NC4": "C4", "NC5": "C5", "NC6": "C6", "NC7": "C7", "NC8": "C8",
           "METHANE": "C1", "ETHANE": "C2", "PROPANE": "C3", "WATER": "H2O",
           "ARGON": "AR", "HELIUM": "HE", "OXYGEN": "O2", "NITROGEN": "N2"}

# Dipole moments [debye], Poling et al. (2001) Appendix A; others zero.
DIPOLE = {"H2O": 1.8, "H2S": 0.9}
# Chung et al. (1988) association factor (Poling 2001 Table 9-1)
KAPPA = {"H2O": 0.075908}
# Parachors: Weinaug & Katz (1943) / Pedersen et al. (2015) Table 10.? (hydrocarbons,
# N2, CO2, H2S); water 52.0 (Firoozabadi & Ramey 1988); H2, He, Ar, O2 from Sugden's
# atomic constants (approximate - these components are rarely in a liquid).
PARACHOR = {"N2": 41.0, "CO2": 78.0, "H2S": 80.1, "C1": 77.0, "C2": 108.0, "C3": 150.3,
            "IC4": 181.5, "C4": 189.9, "IC5": 225.0, "C5": 231.5, "C6": 271.0,
            "C7": 312.5, "C8": 351.5, "H2O": 52.0, "H2": 34.0, "HE": 20.5, "AR": 54.0,
            "O2": 54.0}

HEAVY_HC = ("C7", "C8")          # treated with the "C7+" k_ij column
HC = ("C1", "C2", "C3", "IC4", "C4", "IC5", "C5", "C6", "C7", "C8")

# Binary interaction parameters for PR.
# - N2, CO2, H2S with hydrocarbons and among themselves: Knapp, Doring, Oellrich,
#   Plocker & Prausnitz (1982) "Vapor-liquid equilibria for mixtures of low boiling
#   substances", DECHEMA Chem. Data Ser. VI (PR regressions, as reproduced in the
#   common PR data sets, e.g. ChemSep/DWSIM "pr_ip" table).  "C7+" values are used
#   for C7, C8 and pseudo-components.  Transcribed values; differences between
#   published tables are typically <= 0.02.
# - H2O with gases: Soreide & Whitson (1992) Fluid Phase Equilib. 77, 217
#   (k_ij for the non-aqueous phase), C5+ and other gases 0.5 (common default for
#   the classical mixing rule, Pedersen et al. 2015 Ch. 16).
# - hydrocarbon-hydrocarbon 0, except C1 with C7+ (Chueh-Prausnitz, below);
#   H2, He, Ar, O2 pairs 0 unless listed.
KIJ = {
    ("N2", "C1"): 0.0311, ("N2", "C2"): 0.0515, ("N2", "C3"): 0.0852, ("N2", "IC4"): 0.1033,
    ("N2", "C4"): 0.0800, ("N2", "IC5"): 0.0922, ("N2", "C5"): 0.1000, ("N2", "C6"): 0.1496,
    ("N2", "C7+"): 0.1441,
    ("CO2", "C1"): 0.0919, ("CO2", "C2"): 0.1322, ("CO2", "C3"): 0.1241, ("CO2", "IC4"): 0.1200,
    ("CO2", "C4"): 0.1333, ("CO2", "IC5"): 0.1219, ("CO2", "C5"): 0.1222, ("CO2", "C6"): 0.1100,
    ("CO2", "C7+"): 0.1100,
    ("H2S", "C1"): 0.0888, ("H2S", "C2"): 0.0862, ("H2S", "C3"): 0.0925, ("H2S", "IC4"): 0.0474,
    ("H2S", "C4"): 0.0633, ("H2S", "IC5"): 0.0600, ("H2S", "C5"): 0.0633, ("H2S", "C6"): 0.0500,
    ("H2S", "C7+"): 0.0500,
    ("N2", "CO2"): -0.0170, ("N2", "H2S"): 0.1767, ("CO2", "H2S"): 0.0974,
    ("H2O", "C1"): 0.4850, ("H2O", "C2"): 0.4920, ("H2O", "C3"): 0.5525, ("H2O", "IC4"): 0.5091,
    ("H2O", "C4"): 0.5091, ("H2O", "N2"): 0.4778, ("H2O", "CO2"): 0.1896, ("H2O", "H2S"): 0.19031,
}
