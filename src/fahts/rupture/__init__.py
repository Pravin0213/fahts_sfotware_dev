"""
Rupture: through-wall pressure + thermal stress in the cylindrical shell and
failure-time evaluation for several stress/strength variants.

Written only from published mechanics and material data (no VessFire internals).

Geometry and loads
------------------
Long hollow cylinder, inner radius a = D/2, outer radius b = a + t, gauge
internal pressure p (outside = atmosphere), closed ends, plus an externally
applied longitudinal membrane stress sigma_ext (case field ext_long_mpa) acting
on the shell cross-section.  Axial resultant per the free-body of a closed
vessel:  F_z = p*pi*a^2 + sigma_ext*pi*(b^2 - a^2).

Radial temperature T(r) from the model's wall nodes (column {col}_T1_C ...
{col}_T12_C at r_i = a + x_i, x_i = the wall node layout),
linearly interpolated in r.  Stress-free reference state = initial shell
temperature (first output row).

Stress solutions
----------------
1. Membrane (thin shell, as membrane_stresses): hoop = p*rm/t,
   long = p*rm/(2t) + sigma_ext, radial = -p/2, rm = a + t/2.

2. Lame thick-wall pressure solution (Timoshenko & Goodier, Theory of
   Elasticity, 3rd ed. 1970, sec. 28 "Thick-walled cylinder"):
       sigma_r = p a^2/(b^2-a^2) (1 - b^2/r^2)
       sigma_t = p a^2/(b^2-a^2) (1 + b^2/r^2)
       sigma_z = F_z / (pi (b^2-a^2))

3. Thermal stress, generalized plane strain (uniform axial strain eps_z,
   zero thermal axial resultant = "free ends" far from the ends), Timoshenko &
   Goodier sec. 150 (long hollow circular cylinder, eqs. 244-246 in 3rd ed.):
       sigma_r = E/((1-nu) r^2) [ (r^2-a^2)/(b^2-a^2) I(b) - I(r) ]
       sigma_t = E/((1-nu) r^2) [ (r^2+a^2)/(b^2-a^2) I(b) + I(r) - e(r) r^2 ]
       sigma_z = E/(1-nu) [ 2 I(b)/(b^2-a^2) - e(r) ]
   with e(r) = thermal strain (alpha*T for constant alpha) and
   I(r) = int_a^r e(s) s ds.  Valid for constant E, nu: implemented in
   thermal_tg() with E at the through-wall mean temperature (check solution).

   Because E(T) drops by >50 % between 400 and 700 C, the main solution is a
   1-D axisymmetric generalized-plane-strain finite-element model (linear
   elements, 2-point Gauss) with E(r) = E(T(r)), thermal strain eps_th(T(r)),
   internal pressure on r = a and axial force F_z conjugate to eps_z
   (``GeneralizedPlaneStrainFE``).  With constant E it reproduces 2.+3.
   (tests/unit/rupture).

Material data (carbon steel)
----------------------------
EN 1993-1-2:2005 (Eurocode 3, structural fire design):
  * E(T) = 210 GPa * k_E,theta, Table 3.1 (reduction factor for the slope of
    the linear elastic range).
  * thermal elongation Delta l / l, clause 3.4.1.1 eq. (3.1a-c):
      20 <= T < 750 C : 1.2e-5 T + 0.4e-8 T^2 - 2.416e-4
      750 <= T <= 860 : 1.1e-2
      860 < T <= 1200 : 2e-5 T - 6.2e-3
  * nu = 0.3 (EN 1993-1-1 cl. 3.2.6; taken temperature independent).

Strength (shared case input, not VessFire internals): sigma_f(T) =
strength_mpa * stress_factor * F(T), F = F_UTS (stress type U) or F_Yield,
table from ``fahts.materials.SteelTable``.

Equivalent stresses: von Mises  sqrt(0.5*((s1-s2)^2+(s2-s3)^2+(s3-s1)^2)),
Tresca  max(s) - min(s)  (both compared with sigma_f).

Variants (see evaluate()):
  membrane        membrane stresses; allowable at T_mean ("mean") and at the
                  hottest (outer) surface temperature ("local").  "mean" is the
                  check in the process model output.
  lame            Lame pressure stresses only, at inner / mid / outer surface.
  thermal_elastic Lame + elastic thermal stress (FE, E(T)), inner/mid/outer.
  thermal_tg      as thermal_elastic but closed-form T&G with E at T_mean.
  thermal_2x      thermal_elastic P+Q equivalent vs 2*sigma_f: the classic
                  primary+secondary (shakedown) limit, ASME VIII-2 5.5.6
                  (S_PS) / EN 13445-3 Annex C C.7.  A code design check, not a
                  rupture criterion; reported for orientation only.
  limit           membrane stresses vs the through-thickness average flow
                  stress (1/t) int_a^b sigma_f(T(r)) dr.  Justification:
                  thermal stress is self-equilibrating (secondary, ASME
                  VIII-2 5.2.2 / EN 13445-3 Annex C) and does not change the
                  plastic collapse load of a ductile elastic-perfectly-plastic
                  structure (limit theorems; e.g. Lubliner, Plasticity Theory,
                  1990, sec. 4.4/5.2; EN 13445-3 Annex B, B.8.2 GPD check uses
                  mechanical actions only), while the membrane (collapse)
                  capacity of a section with through-thickness varying flow
                  stress is N0 = int sigma_f dz (Ilyushin's generalized
                  stress; Hodge, Plastic Analysis of Structures, 1959).  For
                  Tresca this is Hill's thick-cylinder limit pressure
                  p_L = int sigma_f/r dr (Hill, Math. Theory of Plasticity,
                  1950, ch. V) to within (b-a)/a ~ 1 %.

Allowable basis for the through-wall variants: "local" = sigma_f(T at the
evaluation radius), "mean" = sigma_f(T_mean).  Failure time = first time the
equivalent stress reaches the allowable at any of the evaluation points
(linear interpolation between output rows).
"""

from fahts.rupture.failure import evaluate, failure_times, stress_series
from fahts.rupture.stress_solutions import membrane_stresses

__all__ = ["evaluate", "failure_times", "membrane_stresses", "stress_series"]
