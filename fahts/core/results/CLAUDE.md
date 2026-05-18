# Results Domain — FAHTS

## TemperatureField — temperature_field.py

The central results container. Returned by `run_analysis()`, consumed by GUI, PostProcessor, and BELTEMP exporter.

```python
@dataclass
class TemperatureField:
    times:       np.ndarray              # (n_steps,) in seconds
    element_ids: list[int]
    T_centroid:  np.ndarray              # (n_steps, n_elems) area-weighted mean T per element
    T_section:   dict[int, np.ndarray]  # {eid: (n_steps, n_nodes)} nodal temperatures
```

### Key methods

```python
tf.centroid_temperature(eid, t_idx=-1)   # → float, last step by default
tf.section_temperatures(eid, t_idx=-1)   # → np.ndarray (n_nodes,)
tf.peak_centroid_temperature(eid)         # → float
tf.time_to_critical(eid, T_crit=600.0)   # → float (seconds), linear interp, inf if not reached
tf.peak_temperatures                      # property → dict[eid, float]
tf.critical_elements(T_crit=600.0)        # → list[int]
tf.section_gradient(eid, t_idx, mesh)     # → (beta_y, beta_z) for BELTEMP export
```

### Factory methods

```python
# From surface solver (all beam types — primary path)
TemperatureField.from_surface_solver_run(eid, times, T_history, mesh)

# From 1-D shell solver (TRISHELL fallback)
TemperatureField.from_shell_solver_run(eid, times, T_history, mesh)

# Merge multiple single-element results into one
TemperatureField.merge(fields: list[TemperatureField])
```

`T_centroid` is area-weighted mean: `Σ(T_k · A_k) / Σ(A_k)` where A_k = element area.

---

## AnalysisConfig — analysis_config.py

```python
@dataclass
class AnalysisConfig:
    t_end: float          # analysis duration [s]
    dt: float             # time step [s]
    output_dt: float      # output interval [s] (≥ dt)
    n_layers: int = 1     # legacy cross-section mesh layers (kept for compatibility)
    elem_size: float | None = None   # legacy element size

    element_ids: list[int] = field(default_factory=list)  # empty = all exposed elements

    # BOX surface mesh
    n_top: int = 2; n_side: int = 3; n_length: int = 4
    # IHPROFIL surface mesh
    n_top_i: int = 4; n_side_i: int = 2; n_bottom_i: int = 2; n_length_i: int = 2
    # PIPE surface mesh
    c_circ: int = 8; n_length_p: int = 4
    # Shell plate mesh
    mesh_12: int = 4; mesh_14: int = 2
```

`validate()` raises `ValueError` on invalid params (dt > t_end, output_dt < dt, etc.).
`n_output_steps` property: `round(t_end / output_dt) + 1`.

---

## PostProcessor — post_processor.py

```python
PostProcessor(result: TemperatureField)
```

### ElementSummary (frozen dataclass)

```python
@dataclass(frozen=True)
class ElementSummary:
    eid: int
    T_initial: float
    T_peak_centroid: float
    T_peak_nodal: float
    t_crit_500: float    # seconds to 500°C, inf if not reached
    t_crit_600: float    # seconds to 600°C, inf if not reached

    # Properties
    fire_resistance_min: float   # t_crit_600 / 60
    passes_600: bool             # T_peak_centroid < 600
```

### PostProcessor methods

```python
pp.element_summary(eid)          # → ElementSummary
pp.element_summaries()           # → list[ElementSummary]
pp.fire_resistance_minutes(eid, T_crit=600)
pp.minimum_fire_resistance()     # → float (minutes)
pp.critical_elements()           # elements that exceed 600°C
pp.passing_elements()            # elements that stay below 600°C
pp.centroid_history(eid)         # → (times, T_centroid) arrays
pp.all_centroid_history()        # → dict[eid, np.ndarray]
pp.summary_text()                # → formatted PASS/FAIL table string
pp.to_csv(path)                  # peak summary CSV
pp.to_temperature_csv(path)      # full time history CSV
pp.to_dataframe()                # pandas DataFrame of ElementSummary
pp.to_history_dataframe()        # pandas DataFrame of full history
```

---

## How results flow through the system

```
run_analysis()
    └── _solve_beam_element() × n_elements
            └── SurfaceTransientSolver.run()
                    └── TemperatureField.from_surface_solver_run()
    └── TemperatureField.merge(all_fields)
            ↓
    GUI: ResultsPanel.show_section(eid, model, result, t_idx)
    GUI: SceneManager.colour_by_temperature(T_per_element)
    IO:  export_beltemp(result, path, meshes=meshes)
    IO:  export_peak_temperature_csv(result, path)
    Results: PostProcessor(result).summary_text()
```
