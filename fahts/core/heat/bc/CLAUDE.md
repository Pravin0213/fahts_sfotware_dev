# Boundary Conditions Domain — FAHTS

## Files

| File | Role |
|------|------|
| `net_flux.py` | EN 1993-1-2 Eq. (3.1) net heat flux formula (pure function) |
| `view_factor.py` | Exposure detection and per-element BC assembly |

---

## net_flux.py

```python
net_heat_flux(T_fire, T_steel, epsilon_m, h_conv, sigma=5.67e-8) → float
```

Computes combined radiative + convective flux [W/m²] into a steel surface.

```
q_net = ε_m · σ · (T_fire_K⁴ − T_steel_K⁴) + h_conv · (T_fire − T_steel)
```

- Temperatures in **°C** (function converts internally to K).
- `epsilon_m = epsilon_fire × epsilon_steel` (resultant emissivity).
- Standard convective coefficient: 25 W/(m²·K); hydrocarbon: 50 W/(m²·K).
- **RadiationBall sources**: pass `epsilon_m=0`, `h_conv=0` — flux is fully prescribed by the ball, not computed here.

---

## view_factor.py

### ElementHeatBC (dataclass)

```python
@dataclass
class ElementHeatBC:
    eid: int
    face_fluxes: dict[str, float]  # {'top': q, 'bot': q, 'left': q, 'right': q} [W/m²]
    t: float
```

Face labels follow the **BOX convention**: `top`/`bot` in local-z, `left`/`right` in local-y.
This is a snapshot at one time step; the solver iterates over time internally.

### Key functions

```python
exposure_flags(beam, fire_zones, nodes, tolerance=0.0) → dict[str, float]
```
Phase 2 implementation: binary check on beam midpoint. Returns 1.0 for all faces if inside
any active FireZone, else 0.0. Phase 5 will replace with fractional view-factor geometry.

```python
exposed_element_ids(elements, fire_zones, nodes, tolerance=0.0) → set[int]
```
Set of element IDs whose midpoints fall inside any active FireZone. Used by `analysis_runner`
to skip elements with no fire exposure before entering the solver loop.

```python
compute_element_bc(beam, fire_zones, nodes, t, T_steel=20.0,
                   epsilon_steel=0.7, tolerance=0.0) → ElementHeatBC | None
```
Returns `None` if not exposed. Multi-zone rule: uses the **hottest covering zone** at time t.

```python
compute_all_bcs(elements, fire_zones, nodes, t, ...) → list[ElementHeatBC]
```
Convenience wrapper over all elements; returns only exposed ones.

---

## Phase 5 Notes

`view_factor.py` is the extension point for Phase 5 insulation + advanced exposure.
The `exposure_flags()` function signature is stable — only the implementation changes.
`ElementHeatBC.face_fluxes` already supports per-face values for partial exposure.
