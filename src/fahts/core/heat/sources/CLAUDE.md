# Heat Sources Domain — FAHTS

## Files

| File | Role |
|------|------|
| `base_source.py` | `HeatSource` ABC — interface all sources must implement |
| `fire_zone.py` | `FireCurve` + `FireZone` — rectangular compartment fire |
| `rad_ball.py` | `RadiationBall` — spherical prescribed-flux source (USFOS USERFLUX type 0) |

---

## HeatSource (base_source.py)

Abstract base class. Two required methods:

```python
@abstractmethod
def temperature(self, t: float) -> float:
    """Gas temperature [°C] at time t [s]."""

@abstractmethod
def bounds(self) -> tuple[np.ndarray, np.ndarray]:
    """(min_xyz, max_xyz) bounding box in global coords [m]."""
```

All fire source lists in the codebase are typed `list[FireZone | RadiationBall]`.

---

## FireZone (fire_zone.py)

Rectangular box heat source. Key fields:

```python
@dataclass
class FireZone(HeatSource):
    name: str
    center: np.ndarray   # (3,) global coords [m]  — NOTE: 'center' not 'centre'
    dims: np.ndarray     # (3,) full extents [m]    — NOTE: 'dims' not 'dimensions'
    curve: FireCurve
    epsilon_fire: float  # default 1.0
    h_conv: float        # default 25 W/(m²·K); use 50 for hydrocarbon
    active: bool         # default True
```

`contains_midpoint(pt, tolerance=0.0)` — bounding-box check used by `view_factor.py`.

**Critical field names:** `center` (not `centre`), `dims` (not `dimensions`). The GUI
and tests use these names directly.

### FireCurve (fire_zone.py)

```python
@dataclass
class FireCurve:
    curve_type: FireCurveType   # ISO_834 | HYDROCARBON | USER_DEFINED
    user_points: list[tuple[float, float]]  # (t [s], T [°C]) pairs for USER_DEFINED
    T_ambient: float = 20.0
```

`FireCurveType` values: `ISO_834`, `HYDROCARBON`, `USER_DEFINED` (exact enum names — used in
dialogs and tests). Formula reference:
```
ISO 834:     T = T₀ + 345·log₁₀(8t_min + 1)
Hydrocarbon: T = T₀ + 1080·(1 − 0.325·e^{−0.167t_min} − 0.675·e^{−2.5t_min})
USER_DEFINED: piecewise-linear interpolation of user_points
```

---

## RadiationBall (rad_ball.py)

Single-zone spherical prescribed-flux source: a sphere of `radius` [m] whose surface
radiates uniformly at `flux` [W/m²] (Lambertian/diffuse emitter). No calibration curve —
incident flux follows the exact point-to-sphere view factor:

```
d <= radius : q = flux                                (engulfed — all faces, no cos weighting)
d >  radius : q = flux * (radius / d)**2 * cos(θ)      (exterior — classical "differential
                                                          area to sphere" configuration factor,
                                                          e.g. Incropera Table 13.2)
```

θ = angle between the target's outward normal and the direction from the target toward the
ball centre. The engulfed case is not a special-cased hack — it's the d→radius limit of the
same physics (a point fully enclosed by a uniform-exitance surface sees irradiance = that
exitance, isotropically, regardless of its own orientation). See `rad_ball.py` module
docstring for the full derivation. **Breaking change (2026-08-24):** replaced the old
two-zone `r1/flux1/r2/flux2` piecewise-linear calibration law — see
`heat/solver/CLAUDE.md` for the migration rationale.

```python
@dataclass
class RadiationBall(HeatSource):
    name: str
    center: np.ndarray   # (3,) global coords [m]
    radius: float        # ball radius [m]
    flux: float          # uniform surface exitance [W/m²]
    active: bool = True
```

Key methods: `max_flux_at_distance(d)` (direction-agnostic upper bound, used for coarse
element screening / `exposed_element_ids`), `incident_flux(point, normal=None)` (exact
directional value — the one the solver uses per-quad).

**Solver interaction:** flux is fully prescribed — do **not** use Stefan-Boltzmann or
convective terms. Set `epsilon_m=0`, `h_conv=0` in the solver when processing this source.
`temperature(t)` returns ambient 20°C (not meaningful for this source type).

Benchmark config (USFOS verification, `radius=r1`/`flux=flux1` of the old calibration —
outer-zone quantitative match against USFOS is not yet re-tuned for the new model):
```
center=(343, 484, 64)m,  radius=5m,  flux=350,000 W/m²
```
