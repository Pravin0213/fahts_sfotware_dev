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

Spherical two-zone prescribed-flux source. Maps to `USERFLUX 0 set cx cy cz r1 flux1 r2 flux2`
in the USFOS .fem file.

```python
@dataclass
class RadiationBall(HeatSource):
    name: str
    center: np.ndarray   # (3,) global coords [m]
    r1: float            # inner zone radius [m]
    flux1: float         # inner zone irradiance [W/m²]
    r2: float            # outer zone radius [m]
    flux2: float         # outer zone irradiance [W/m²]
    active: bool = True
```

**Solver interaction:** flux is fully prescribed — do **not** use Stefan-Boltzmann or
convective terms. Set `epsilon_m=0`, `h_conv=0` in the solver when processing this source.
`temperature(t)` returns ambient 20°C (not meaningful for this source type).

Benchmark config (USFOS verification):
```
center=(343, 484, 64)m,  r1=5m / flux1=350,000 W/m²,  r2=100m / flux2=1,500 W/m²
```
