# FAHTS — Fire Analysis and Heat Transfer Software
## Product Roadmap & Architecture Reference

**Version:** 0.1-planning  
**Started:** 2026-05-16  
**Target platform:** Desktop (Linux / Windows / macOS)  
**Language:** Python 3.11+

---

## 1. Vision

FAHTS is an industry-level software for 3D heat transfer analysis of structural
steel frameworks under fire scenarios. It reads structural models in USFOS `.fem`
format, renders them interactively in 3D, and lets the engineer define heat
sources (fire zones, point radiators) directly in the 3D scene. It then solves
the heat equation within each structural member and visualises the temperature
distribution on the structure in real time.

**Analogy:** Like USFOS for non-linear structural analysis, FAHTS is the
dedicated thermal companion — same model format, same visual feel, but focused
on resolving temperature fields needed for fire-code compliance and LNG tank
thermal assessment.

---

## 2. Technology Stack

| Concern               | Library               | Version target |
|-----------------------|-----------------------|----------------|
| GUI framework         | PyQt6                 | ≥ 6.6          |
| 3D rendering          | PyVista + pyvistaqt   | ≥ 0.44         |
| Numerics              | NumPy + SciPy         | ≥ 2.0 / 1.14   |
| Results tables        | Pandas                | ≥ 2.2          |
| 2-D cross-section plots | Matplotlib (Qt backend) | ≥ 3.9      |
| Package management    | pip / conda           | —              |

```
pip install PyQt6 pyvista pyvistaqt numpy scipy pandas matplotlib
```

---

## 3. File Format Reference (USFOS .fem)

The primary input format is USFOS bulk-data text. Key cards parsed:

| Card       | Fields                                         | Notes                                 |
|------------|------------------------------------------------|---------------------------------------|
| `NODE`     | id  X  Y  Z  [BC_x BC_y BC_z ...]             | Coords in **metres** (global vessel)  |
| `BEAM`     | id  n1  n2  mat_id  geom_id  lcoor_id  [ecc1  ecc2] | 1-D line element; ecc IDs are ECCENT refs (0=none) |
| `BOX`      | id  H  T_side  T_bot  T_top  Width             | Hollow rectangular section; H in local-z (UNITVEC), Width in local-y |
| `IHPROFIL` | id  h  tw  bf_top  tf_top  bf_bot  tf_bot      | I-beam; h in local-z (UNITVEC), bf in local-y |
| `PIPE`     | id  outer_diameter  wall_thickness             | Circular hollow section               |
| `MISOIEP`  | id  E  ν  σ_y  ρ  α_T                         | Isotropic steel material              |
| `UNITVEC`  | id  dx  dy  dz                                 | Local **z-axis** of beam cross-section |
| `ECCENT`   | id  Ex  Ey  Ez                                 | Beam-end offset in **global** coords [m] |
| `QUADSHEL` | id  n1  n2  n3  n4  mat  geom                 | 4-node shell element                  |
| `TRISHELL` | id  n1  n2  n3  mat  geom                     | 3-node shell element                  |
| `Name Group` / `GroupDef` | id  name  elem_ids...            | Named element sets (ELEV_A … MAIN_STEEL_BEAMS) |
| `BEAMHING` | release flags end1/end2  elem_id               | Moment releases (stored, not used thermally) |
| `PLTHICK`  | id  thickness                                  | Plate thickness (used by shells)      |

**model_file.fem statistics:**
- 659 nodes, ~700 BEAM elements, 5 BOX section types, 1 steel material (S355)
- Groups: ELEV_A through ELEV_K (floor rings), MAIN_STEEL_BEAMS
- Coordinates: global vessel frame (X≈347–353 m, Y≈524–534 m, Z≈54–73 m)

---

## 4. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                            FAHTS Desktop App                          │
│                                                                       │
│  ┌─────────────────────┐     ┌─────────────────────────────────────┐  │
│  │     PyQt6 GUI        │     │        PyVista 3D Viewport          │  │
│  │  ┌───────────────┐  │     │  (pyvistaqt.BackgroundPlotter)      │  │
│  │  │  Model Tree   │  │ ◄──►│  - Beam geometry (extruded BOX)     │  │
│  │  │  Properties   │  │     │  - Fire zone boxes                  │  │
│  │  │  Heat Sources │  │     │  - Temperature colormap             │  │
│  │  │  Results      │  │     │  - Axes, legend, time slider        │  │
│  │  └───────────────┘  │     └─────────────────────────────────────┘  │
│  └──────────┬──────────┘                    ▲                         │
│             │ user events                   │ render calls            │
│             ▼                               │                         │
│  ┌──────────────────────────────────────────┴────────────────────┐    │
│  │                     Application Controller                     │    │
│  │   (orchestrates IO → model → heat sources → solver → results)  │    │
│  └────────┬────────────────────┬────────────────────┬────────────┘    │
│           │                    │                    │                 │
│  ┌────────▼───────┐  ┌─────────▼──────────┐  ┌────▼──────────────┐   │
│  │   core/io      │  │   core/heat        │  │  core/results     │   │
│  │                │  │                    │  │                   │   │
│  │ usfos_reader   │  │ sources/           │  │ temperature_field │   │
│  │ results_writer │  │   fire_zone        │  │ post_processor    │   │
│  └────────┬───────┘  │   rad_point        │  └───────────────────┘   │
│           │          │ bc/                │                           │
│  ┌────────▼───────┐  │   view_factor      │                           │
│  │  core/model    │  │   net_flux         │                           │
│  │                │  │ section_mesh/      │                           │
│  │ FEMModel       │  │   box_mesher       │                           │
│  │ Node           │  │ solver/            │                           │
│  │ BeamElement    │  │   fem_2d_section   │                           │
│  │ BoxSection     │  │   fem_3d_volume    │                           │
│  │ Material       │  │   time_integrator  │                           │
│  │ Group          │  └────────────────────┘                           │
│  └────────────────┘                                                   │
└───────────────────────────────────────────────────────────────────────┘
```

### Data Flow (one analysis run)

```
model_file.fem
      │
      ▼
 usfos_reader.py ──► FEMModel (nodes, beams, sections, materials, groups)
      │
      ▼
 beam_geometry.py ──► 3-D VTK mesh per beam (extruded BOX section)
      │                    │
      ▼                    ▼
 fire_zone.py          rendered in PyVista viewport
      │
      ▼
 view_factor.py + net_flux.py ──► q_net per beam face per time step
      │
      ▼
 fem_2d_section.py (or fem_3d_volume.py)
 time_integrator.py ──► T(x, y, t) per beam cross-section
      │
      ▼
 temperature_field.py ──► nodal temperature arrays
      │
      ▼
 colormap + PyVista ──► live colored 3-D rendering
      │
      ▼
 results_writer.py ──► CSV / VTK output
```

---

## 5. Module Specifications

### 5.1  `fahts/core/model/`

**`fem_model.py`** — Top-level container
```python
@dataclass
class FEMModel:
    nodes:     dict[int, Node]          # {nid: Node}
    elements:  dict[int, BeamElement]   # {eid: BeamElement}
    sections:  dict[int, Section]       # {geom_id: BoxSection | ...}
    materials: dict[int, Material]      # {mat_id: Material}
    groups:    dict[str, Group]         # {name: Group}
    source_file: Path
    coord_unit: str = "m"
```

**`node.py`**
```python
@dataclass
class Node:
    nid: int
    x: float; y: float; z: float
    bc: tuple[int,...] = ()     # boundary code flags
```

**`element.py`**
```python
@dataclass
class BeamElement:
    eid: int
    n1: int; n2: int            # node IDs
    mat_id: int
    geom_id: int
    lcoor_id: int               # local coordinate reference
    # Derived (computed after read):
    length: float = 0.0
    direction: np.ndarray = None   # unit vector n2-n1
    local_z: np.ndarray = None     # from UNITVEC
```

**`section.py`**
```python
@dataclass
class BoxSection:
    sid: int
    H: float        # height [m]
    T_side: float   # side wall thickness
    T_bot: float    # bottom wall thickness
    T_top: float    # top wall thickness
    W: float        # width [m]

    @property
    def perimeter(self) -> float: ...
    @property
    def area(self) -> float: ...
    @property
    def section_factor_Am_V(self) -> float: ...  # exposed perimeter / area
```

**`material.py`**
```python
@dataclass
class SteelMaterial:
    mid: int
    E: float       # Young's modulus [Pa]
    nu: float      # Poisson's ratio
    fy: float      # yield stress [Pa]
    rho: float     # density [kg/m³]
    alpha_T: float # thermal expansion [1/K]
    name: str = "S355"

    # Temperature-dependent thermal properties (EN 1993-1-2)
    def conductivity(self, T_C: float) -> float: ...   # [W/(m·K)]
    def specific_heat(self, T_C: float) -> float: ...  # [J/(kg·K)]
    def density(self, T_C: float) -> float: ...        # ≈ const for steel
```

---

### 5.2  `fahts/core/io/`

**`usfos_reader.py`** — USFOS .fem parser
```python
def read_usfos_fem(filepath: Path) -> FEMModel:
    """
    Parse USFOS .fem file and return a fully populated FEMModel.

    Cards handled: HEAD, NODE, BEAM, BOX, MISOIEP, UNITVEC,
                   BEAMHING, Name/GroupDef, GRAVITY, PLTHICK
    """
```

Key parsing rules:
- Lines starting with `'` are comments → skip
- Free-space delimited (not fixed-width like Nastran)
- All coordinates and dimensions already in metres
- Groups are defined after elements — do a two-pass parse

---

### 5.3  `fahts/core/heat/sources/`

**`base_source.py`**
```python
class HeatSource(ABC):
    name: str
    active: bool = True

    @abstractmethod
    def temperature(self, t: float) -> float:
        """Return fire/source temperature [°C] at time t [s]."""

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box."""
```

**`fire_zone.py`** — Rectangular fire zone
```python
class FireZone(HeatSource):
    """
    A rectangular box region that emits heat.

    Attributes:
        center: (x, y, z) [m]   — geometric centre of fire box
        dims:   (dx, dy, dz) [m] — half-extents
        curve:  FireCurve        — T_fire(t) time history
        epsilon_fire: float      — emissivity of fire / hot gas (default 1.0)
        h_conv: float            — convective coefficient [W/(m²·K)] (default 25)

    Exposure logic:
        A beam element is 'inside' the fire zone if its midpoint lies
        within the bounding box + a tolerance margin.
        All four BOX faces of an inside beam are fully exposed.
        Edge/partial exposure is handled in Phase 5 via view factors.
    """
```

**Fire curves** (implemented in `fire_zone.py`):
```python
class FireCurve:
    ISO_834       # Standard cellulosic: T = 20 + 345·log10(8t+1)
    HYDROCARBON   # Offshore: T = 20 + 1080·(1 - 0.325e^{-0.167t} - 0.675e^{-2.5t})
    USER_DEFINED  # Piecewise linear list of (t[s], T[°C]) points
```

---

### 5.4  `fahts/core/heat/bc/`

**`net_flux.py`** — EN 1993-1-2 net heat flux to exposed steel surface
```python
def net_heat_flux(
    T_fire: float,    # fire compartment temperature [°C]
    T_steel: float,   # current steel surface temperature [°C]
    epsilon_m: float, # resultant emissivity (fire × surface)
    h_conv: float,    # convective coefficient [W/(m²·K)]
    sigma: float = 5.67e-8,
) -> float:
    """
    EN 1993-1-2 Eq. (3.1):
        q_net = q_rad + q_conv
        q_rad = ε_m · σ · (T_fire_K⁴ - T_steel_K⁴)
        q_conv = h · (T_fire - T_steel)
    Returns q_net [W/m²].
    """
```

**`view_factor.py`** — Geometric view factors (Phase 2, simple; Phase 5, full)
```python
def exposure_flags(
    beam: BeamElement,
    fire_zones: list[FireZone],
) -> dict[str, float]:
    """
    Returns per-face exposure fraction {face_id: f} in [0, 1].
    Phase 2: binary (1.0 inside zone, 0.0 outside).
    Phase 5: analytical view-factor geometry.
    """
```

---

### 5.5  `fahts/core/heat/section_mesh/`

**`box_mesher.py`** — 2-D FEM mesh of a BOX cross-section

For a BOX(H=0.20, W=0.20, T=0.008):
```
  ╔═══════════════════╗   ← top plate (T_top = 8mm)
  ║  ┌─────────────┐  ║   ← left wall (T_side = 8mm) | inner hollow | right wall
  ║  │             │  ║
  ║  │   hollow    │  ║
  ║  │             │  ║
  ║  └─────────────┘  ║
  ╚═══════════════════╝   ← bottom plate (T_bot = 8mm)
```

Each wall is meshed with n_layers Quad4 elements through the thickness.
Total DOFs ≈ 4 × (perimeter/elem_size) × n_layers nodes.

**Output:** `SectionMesh(nodes_2d, quads, face_labels)` where `face_labels` marks
which boundary edges are "outer exposed" vs "inner hollow" vs "corner".

---

### 5.6  `fahts/core/heat/solver/`

**`fem_2d_section.py`** — 2-D FEM heat transfer in cross-section plane

Governing equation (per section at axial position z):
```
ρ·cₚ·∂T/∂t = ∂/∂x(k·∂T/∂x) + ∂/∂y(k·∂T/∂y)
```

Boundary conditions:
- Outer faces: Robin BC   q·n̂ = q_net(T_fire, T_surface)
- Inner (hollow): Adiabatic or free convection (negligible)

Assembly: sparse 2-D stiffness + mass matrices on section mesh
Time integration: backward Euler (same as legacy `fem_heat_3d.py`)

**`fem_3d_volume.py`** (Phase 5) — reuses `legacy/fem_heat_3d.py` with adaptors:
- Accepts a 3-D hex mesh generated from extruded section mesh
- For beams with varying fire exposure along length

**`time_integrator.py`**
```python
class TransientSolver:
    def step(self, dt: float, t: float) -> np.ndarray:
        """One backward-Euler step. Returns T_new (all beam sections)."""

    def run(self, t_end: float, dt: float, callback=None) -> Results:
        """Full transient run. Calls callback(t, T) at each output step."""
```

---

### 5.7  `fahts/renderer/`

**`beam_geometry.py`** — Generates 3-D VTK mesh for rendering

For each BEAM element, creates a `pyvista.PolyData` box:
```
Input:  n1_pos, n2_pos, BoxSection, local_z
Output: Hollow box mesh (8 faces × 2 triangles = 16 triangles) extruded
        from node1 to node2 with proper orientation from local z-axis
```

Rendering modes:
1. **Wire** — just the beam centreline
2. **Section** — extruded 3-D box at true scale
3. **Scaled** — exaggerated thickness for visibility

**`scene_manager.py`** — Manages the PyVista plotter
```python
class SceneManager:
    plotter: BackgroundPlotter

    def load_model(self, model: FEMModel) -> None
    def show_fire_zones(self, zones: list[FireZone]) -> None
    def update_temperature(self, t: float, T_field: TemperatureField) -> None
    def set_render_mode(self, mode: str) -> None       # 'wire'|'section'|'scaled'
    def animate(self, results: Results, fps: int) -> None
```

**`colormap.py`** — Temperature → colour mapping
- Range: T_min → T_max (auto or user-set)
- Colormaps: jet, inferno, cool-warm, custom
- Annotations: colourbar with °C labels

---

### 5.8  `fahts/gui/`

**`main_window.py`** — QMainWindow layout:
```
┌──────────────────────────────────────────────────────────────────┐
│ File  View  Model  Heat  Analysis  Results  Help                 │  ← MenuBar
├──────────────────────────────────────────────────────────────────┤
│ [Open] [Save] [Run] [Animate] [Screenshot] │ t=   s  [▶][■]    │  ← Toolbar
├─────────────────────┬────────────────────────────────────────────┤
│  Model Tree Panel   │                                            │
│  ┌───────────────┐  │        PyVista 3D Viewport                 │
│  │ ▼ Structure   │  │        (pyvistaqt.BackgroundPlotter)       │
│  │   ▶ Nodes     │  │                                            │
│  │   ▶ Elements  │  │                                            │
│  │   ▶ Groups    │  │                                            │
│  │ ▼ Heat Sources│  │                                            │
│  │   + Fire Zone │  │                                            │
│  │ ▼ Materials   │  │                                            │
│  │   ▶ S355      │  │                                            │
│  └───────────────┘  │                                            │
├─────────────────────┤                                            │
│  Properties Panel   │                                            │
│  (shows selected    │                                            │
│   element info)     │                                            │
├─────────────────────┴────────────────────────────────────────────┤
│  Status bar: "Model loaded: 659 nodes, 700 elements"             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Development Phases

---

### PHASE 1 — Foundation ✅ COMPLETE
**Goal:** Working 3-D viewer that opens model_file.fem and displays the structure interactively.

| # | Task | Module | Status |
|---|------|--------|--------|
| 1.1 | Project structure & requirements.txt | root | ✅ Done |
| 1.2 | USFOS .fem reader | `core/io/usfos_reader.py` | ✅ Done |
| 1.3 | Data model dataclasses | `core/model/` | ✅ Done |
| 1.4 | Beam geometry (extruded sections) for rendering | `renderer/beam_geometry.py` | ✅ Done |
| 1.5 | PyVista scene manager (basic render) | `renderer/scene_manager.py` | ✅ Done |
| 1.6 | PyQt6 main window + PyVista embed | `gui/main_window.py` | ✅ Done |
| 1.7 | Model tree panel (groups, elements) | `gui/panels/model_tree_panel.py` | ✅ Done |
| 1.8 | Properties panel (click element → see dimensions) | `gui/panels/properties_panel.py` | ✅ Done |
| 1.9 | File → Open dialog + recent files | `gui/main_window.py` | ✅ Done |
| 1.10 | Element colour by group | `renderer/colormap.py` | ✅ Done |
| B.1 | Section orientation bug fix (H↔W swap in beam_geometry) | `renderer/beam_geometry.py` | ✅ Done |
| B.2 | ECCENT parsing + eccentricity applied to beam endpoints | `core/io/usfos_reader.py`, `renderer/beam_geometry.py` | ✅ Done |
| B.3 | Hollow pipe rendering (annular end caps + inner surface) | `renderer/beam_geometry.py` | ✅ Done |

**Phase 1 acceptance criteria — all met:**
- `python main.py` opens a window ✅
- File → Open loads `model_file.fem` within 3 seconds ✅
- 3-D structure renders with correct section orientation (BOX H in UNITVEC direction) ✅
- IHPROFIL (I-beams) and PIPE (hollow) sections render correctly ✅
- Node eccentricities applied — beam endpoints offset from nodes where specified ✅
- Mouse: rotate, zoom, pan ✅
- Click on a beam → Properties panel shows EID, section dimensions, material ✅
- Model tree lists all groups; toggling visibility shows/hides group elements ✅

---

### PHASE 2 — Heat Source Engine ✅ COMPLETE
**Goal:** User can place a fire zone in the 3D scene; the software translates it to per-element heat flux BCs.

| # | Task | Module | Status |
|---|------|--------|--------|
| 2.1 | FireCurve (ISO 834, hydrocarbon, user) | `heat/sources/fire_zone.py` | ✅ Done |
| 2.2 | FireZone class (box position + curve) | `heat/sources/fire_zone.py` | ✅ Done |
| 2.3 | Fire zone dialog (position, dims, curve type) | `gui/dialogs/fire_zone_dialog.py` | ✅ Done |
| 2.4 | Render fire zone as transparent box in viewport | `renderer/scene_manager.py` | ✅ Done |
| 2.5 | Exposure logic: which beams are inside fire zone | `heat/bc/view_factor.py` | ✅ Done |
| 2.6 | Net heat flux calculator (EN 1993-1-2) | `heat/bc/net_flux.py` | ✅ Done |
| 2.7 | Heat sources panel (list, add, edit, delete) | `gui/panels/heat_source_panel.py` | ✅ Done |
| 2.8 | Highlight exposed beams in viewport (red tint) | `renderer/scene_manager.py` | ✅ Done |
| 2.9 | Export BC summary to CSV | `core/io/results_writer.py` | ✅ Done |

**Phase 2 acceptance criteria — all met:**
- User can add a fire zone via dialog (center X/Y/Z, W/H/D, fire curve type) ✅
- Fire zone rendered as a semi-transparent orange/red box in the 3D scene ✅
- Exposed beams highlighted in fire-orange tint in viewport ✅
- ISO 834 and hydrocarbon fire curves working correctly (44 unit tests passing) ✅
- EN 1993-1-2 q_net BC computed for all exposed elements ✅
- BC summary exported to CSV (Heat → Export BC Summary…) ✅

---

### PHASE 3 — Heat Transfer Solver (Months 3–4)
**Goal:** Solve transient heat equation in steel cross-sections; store temperature vs time.

| # | Task | Module | Status |
|---|------|--------|--------|
| 3.1 | Temperature-dep. steel properties (EN 1993-1-2) | `core/model/material.py` | ✅ Done |
| 3.2 | BOX cross-section 2-D mesh generator | `heat/section_mesh/box_mesher.py` | ✅ Done |
| 3.3 | Quad4 element stiffness + mass matrix | `heat/solver/fem_2d_section.py` | ✅ Done |
| 3.4 | Assembly + BC application (Robin on outer faces) | `heat/solver/fem_2d_section.py` | ✅ Done |
| 3.5 | Backward Euler time integrator | `heat/solver/time_integrator.py` | ✅ Done |
| 3.6 | TemperatureField results container | `core/results/temperature_field.py` | ✅ Done |
| 3.7 | Analysis run dialog (dt, t_end, output interval) | `gui/dialogs/run_analysis_dialog.py` | ✅ Done |
| 3.8 | Progress bar + cancelable run thread | `gui/main_window.py` | ✅ Done |
| 3.9 | Post-processor: peak T per beam, T at centroid | `core/results/post_processor.py` | ✅ Done |
| 3.10 | Validation: compare results against USFOS benchmark | `tests/test_heat_solver.py` | ⬜ Deferred |

**Phase 3E — Multi-section solver extension (all ✅ Done as of 2026-05-17)**

| # | Task | Module | Status |
|---|------|--------|--------|
| 3E.1 | IHPROFIL 2-D mesh generator (I/H-sections) | `heat/section_mesh/ihprofil_mesher.py` | ✅ Done |
| 3E.2 | PIPE annular 2-D mesh generator | `heat/section_mesh/pipe_mesher.py` | ✅ Done |
| 3E.3 | Shell 1-D through-thickness mesh | `heat/section_mesh/shell_mesh.py` | ✅ Done |
| 3E.4 | Shell1DSolver — 1-D backward Euler rod FEM | `heat/solver/shell_1d_solver.py` | ✅ Done |
| 3E.5 | RadiationBall fire source (USERFLUX type 0) — two-zone model; GUI dialog | `heat/sources/rad_ball.py`, `gui/dialogs/rad_ball_dialog.py` | ✅ Done |
| 3E.6 | analysis_runner extended to BOX + IHPROFIL + PIPE + shells + RadiationBall | `heat/solver/analysis_runner.py` | ✅ Done |
| 3E.7 | BELTEMP parser + export (USFOS benchmark I/O) | `core/io/beltemp_parser.py`, `results_writer.py` | ✅ Done |
| 3E.8 | USFOS benchmark comparison test (automated) | `tests/` | ⬜ Deferred until after Phase 3F |

---

**Phase 3F — Theory-Compliant Solver Refactor (added 2026-05-18)**
**Goal:** Align the core solver with the original SINTEF FAHTS theory
(Crank-Nicolson θ=1/2, heat accumulation element, temperature gradients).
**Reference:** `docs/3D_FEM_heat_transfer_theory.txt` and `FAHTS_theory/Fahts_Theory_Manual.pdf`

| # | Task | Module | Status | Priority |
|---|------|--------|--------|----------|
| 3F.1 | Replace backward Euler (θ=1) with Crank-Nicolson (θ=1/2) in TransientSolver | `heat/solver/time_integrator.py` | ✅ Done | Critical |
| 3F.2 | Replace backward Euler (θ=1) with Crank-Nicolson (θ=1/2) in Shell1DSolver | `heat/solver/shell_1d_solver.py` | ✅ Done | Critical |
| 3F.3 | Add Heat Accumulation Element for BOX and PIPE hollow profiles | `heat/solver/analysis_runner.py`, `heat/solver/fem_2d_section.py` | ✅ Done | Critical |
| 3F.4 | Add `inner_node_indices` property to SectionMesh | `heat/section_mesh/section_mesh.py` | ✅ Done | Supporting |
| 3F.5 | Add temperature gradient (βy, βz) computation for BELTEMP linearization | `core/io/results_writer.py`, `core/results/temperature_field.py` | ✅ Done | Moderate |
| 3F.6 | Update all solver tests for Crank-Nicolson results (tolerances may change) | `tests/test_heat_solver.py`, etc. | ✅ Done | Supporting |

**Phase 3F detailed specifications:**

### 3F.1 — Crank-Nicolson in TransientSolver

File: `fahts/core/heat/solver/time_integrator.py`

**Current `step()` signature and logic (backward Euler, θ=1):**
```python
def step(self, T_prev, dt, t):
    K = assemble_K(self.mesh, k_val)
    C = assemble_C_lumped(self.mesh, rho, cp)
    A = lil_matrix(n); A.setdiag(C / dt); A += K
    add_robin_bc(A, b, ...)
    return spsolve(A.tocsr(), b)
```

**Required Crank-Nicolson `step()` logic (θ=1/2):**
```python
# TransientSolver needs new state fields: K_prev, M_prev, T_dot_prev
# Initialise at t=0: T_dot_prev = M_0^{-1} * (Q_0 - K_0 * T_0)
#
# Each step:
def step(self, T_prev, dt, t):
    T_mean = 0.5 * (T_prev + T_guess)  # T_guess = T_prev + T_dot_prev*dt
    k_val = material.conductivity(T_mean.mean())
    rho   = material.density(T_mean.mean())
    cp    = material.specific_heat(T_mean.mean())
    K_i   = assemble_K(mesh, k_val)           # conductivity at step i
    M_i   = assemble_C_lumped(mesh, rho, cp)  # mass at step i (as array)
    # Build Q_i (fire load vector at time i):
    Q_i   = build_fire_load_vector(...)
    # Crank-Nicolson system:
    A     = K_i + (2.0/dt) * diag(M_i)        # (sparse + scaled diagonal)
    B     = Q_i - self.K_prev @ T_prev + self.M_prev * self.T_dot_prev
    dT    = spsolve(A, B)
    T_new = T_prev + dT
    T_dot_new = (2.0/dt) * dT - self.T_dot_prev
    # Save for next step:
    self.K_prev    = K_i
    self.M_prev    = M_i
    self.T_dot_prev = T_dot_new
    return T_new
```

Note: K_prev and M_prev are stored between steps. On the FIRST step (i=1),
K_prev = K_0 and M_prev = M_0, computed at T_initial.

### 3F.2 — Crank-Nicolson in Shell1DSolver

File: `fahts/core/heat/solver/shell_1d_solver.py`

Same pattern as 3F.1 applied to the 1D rod system. The 1D system has:
- `_assemble_K_1d(mesh, k)` → K (tridiagonal)
- `_assemble_C_1d(mesh, rho, cp)` → M (diagonal, lumped)

Replace the backward Euler solve with the CN incremental form, tracking
K_prev, M_prev, T_dot_prev.

### 3F.3 — Heat Accumulation Element

File: `fahts/core/heat/solver/analysis_runner.py`

In the function that solves each beam element (currently `run_analysis` or
a helper it calls), after building the section mesh for BOX or PIPE:

```python
def _add_heat_accumulation(mesh, section, element, M_lumped):
    """Add trapped-air thermal mass to inner surface nodes."""
    inner_nodes = list({n for pair in mesh.inner_edge_pairs for n in pair})
    if not inner_nodes:
        return  # Open profile — no accumulation
    n_inner = len(inner_nodes)
    if isinstance(section, BoxSection):
        A_inner = (section.W - 2*section.T_side) * (section.H - section.T_top - section.T_bot)
    elif isinstance(section, PipeSection):
        import math
        R_in = section.outer_diameter/2 - section.wall_thickness
        A_inner = math.pi * R_in**2
    else:
        return
    rho_c_air = 1200.0  # J/(m³·K) for air at ~300°C
    m_acc = A_inner * element.length * rho_c_air / n_inner
    for idx in inner_nodes:
        M_lumped[idx] += m_acc
```

This function is called after `assemble_C_lumped()` returns M_lumped (ndarray),
before passing M_lumped to the time integrator.

### 3F.5 — Temperature Gradient Linearization

File: `fahts/core/io/results_writer.py`, function `export_beltemp()`

After the 2D FEM gives nodal temperatures T_section[eid] at each step,
compute the equivalent linear temperature state:

```
T_mean = Σ(T_k · A_k) / A_total            (area-weighted mean)
β_z    = Σ(T_k · y_k · A_k) / Iy          (bending gradient about z)
β_y    = Σ(T_k · z_k · A_k) / Iz          (bending gradient about y)
```

Where (y_k, z_k) are node coordinates in the section local frame,
A_k is the area share of node k (from lumped mass with ρ=1, cp=1),
Iy and Iz are second moments of area of the cross-section.

Currently BELTEMP writes zeros for columns 4 and 5 (gradients).
Update to write the computed β_z and β_y instead.

**Phase 3 acceptance criteria:**
- Run → analysis completes for model_file.fem with fire zone, 4-hour simulation
- Temperature at steel centroid matches USFOS reference results within agreed tolerance
- Results stored in TemperatureField object keyed by (element_id, time_step)
- All section types (BOX, IHPROFIL, PIPE, shells) produce results

---

### PHASE 4 — Results Visualisation (Months 4–5)
**Goal:** See temperatures animated on the 3D structure; export results.

| # | Task | Module | Status |
|---|------|--------|--------|
| 4.1 | Map nodal temperatures → VTK point/cell data | `renderer/scene_manager.py` | ✅ Done |
| 4.2 | Colour beams by temperature (jet/inferno LUT) | `renderer/colormap.py` | ✅ Done |
| 4.3 | Colorbar with °C scale + time label | `renderer/scene_manager.py` | ✅ Done |
| 4.4 | Time slider + play/pause/step animation | `gui/main_window.py` | ✅ Done |
| 4.5 | Cross-section temperature plot (2-D) for selected beam | `gui/panels/results_panel.py` | ✅ Done |
| 4.6 | Temperature-time graph for selected element | `gui/panels/results_panel.py` | ✅ Done |
| 4.7 | Export: CSV (peak T per element), VTK, Excel | `core/io/results_writer.py` | ✅ Done |
| 4.8 | Screenshot / save animation (GIF/MP4) | `renderer/scene_manager.py` | ✅ Done |
| 4.9 | Critical temperature threshold overlay (660°C steel) | `renderer/colormap.py` | ✅ Done |

**Phase 4 acceptance criteria:**
- Animated temperature field plays in real time at ≥ 10 fps for model_file.fem
- Clicking a beam shows its temperature-time curve and cross-section plot
- Export to CSV and VTK works; VTK opens correctly in ParaView

---

### PHASE 5 — Advanced Features (Months 5–6)
**Goal:** Insulation layers, proper view factors, 3-D volume solver, additional heat source types.

| # | Task | Module | Status |
|---|------|--------|--------|
| 5.1 | Insulation layer around beam (material + thickness) | `core/model/section.py` | ⬜ |
| 5.2 | Multi-layer 1-D through-insulation (reuse legacy 1D solver) | `heat/solver/insulation_solver.py` | ⬜ |
| 5.3 | Point radiation source (rad_ball) | `heat/sources/rad_point.py` | ⬜ |
| 5.4 | Jet fire (directional cone) | `heat/sources/jet_fire.py` | ⬜ |
| 5.5 | Shadow / obstruction detection (ray casting) | `heat/bc/view_factor.py` | ⬜ |
| 5.6 | Proper geometric view factor calculation | `heat/bc/view_factor.py` | ⬜ |
| 5.7 | 3-D volume solver option (hex mesh per beam) | `heat/solver/fem_3d_volume.py` | ⬜ |
| 5.8 | Nastran BDF reader (from legacy code) | `core/io/nastran_reader.py` | ⬜ |
| 5.9 | Multiple simultaneous fire zones | `heat/sources/` | ⬜ |
| 5.10 | Project save / load (.fahts JSON project file) | `core/io/project_file.py` | ⬜ |
| 5.11 | **Fix: cross-element thermal coupling.** Currently each beam solves its own K/M/Q in total isolation — no conduction between adjacent members through shared structural joints, unlike real SINTEF FAHTS (which merges coincident mesh nodes across elements into one global assembled system). See `heat/solver/CLAUDE.md` "Known Limitation" for full analysis. Assigned to Fable. | `heat/solver/` | ⬜ |

---

## 7. Key Design Decisions & Rationale

| Decision | Choice | Reason |
|----------|--------|--------|
| Heat solver default | 2-D FEM per cross-section | Industry standard (SAFIR, Abaqus); balances accuracy and speed. Full 3-D option in Phase 5. |
| Time integration | Implicit backward Euler (θ=1) | Unconditionally stable; critical for long fire analyses (4 h). |
| Mass lumping | Row-sum lumped | Avoids oscillations; diagonal system → faster per step. |
| Section mesh | Quad4 elements | Simplest 2-D element; sufficient for thin-walled BOX. Upgrade to Quad8 in Phase 5 for curved sections. |
| GUI framework | PyQt6 + pyvistaqt | Native cross-platform; pyvistaqt proven for scientific VTK apps. |
| Fire exposure (Phase 2) | Binary (in/out box) | Simple, fast, testable. Real view factors in Phase 5. |
| Steel properties | EN 1993-1-2 Annex C | Eurocode standard; table-based piecewise linear k(T) and cp(T). |
| Coordinate centering | Auto-center + normalize | Model lives at X≈350m, Y≈530m — must translate to origin for rendering. |

---

## 8. Data Contracts Between Modules

These are the interfaces that must remain stable as modules are developed independently.

### A. FEMModel (produced by IO, consumed by renderer + solver)
- `model.nodes[nid]` → `Node(nid, x, y, z, bc)`
- `model.elements[eid]` → `BeamElement(eid, n1, n2, mat_id, geom_id, lcoor_id)`
- `model.sections[sid]` → `BoxSection(sid, H, T_side, T_bot, T_top, W)`
- `model.materials[mid]` → `SteelMaterial(mid, E, nu, fy, rho, alpha_T)`
- `model.groups[name]` → `Group(name, element_ids: set[int])`

### B. HeatBC (produced by heat/bc, consumed by solver)
```python
@dataclass
class ElementHeatBC:
    eid: int
    face_fluxes: dict[str, float]   # {'top': q, 'bot': q, 'left': q, 'right': q}  [W/m²]
    t: float                         # time this BC applies [s]
```

### C. TemperatureField (produced by solver, consumed by renderer + post-processor)
```python
@dataclass
class TemperatureField:
    times: np.ndarray              # shape (n_steps,)
    element_ids: list[int]
    T_centroid: np.ndarray         # shape (n_steps, n_elems) — centroid temperature
    T_section: dict[int, np.ndarray]  # {eid: (n_steps, n_section_nodes)}
```

---

## 9. Testing Strategy

| Level | Tool | What to test |
|-------|------|--------------|
| Unit | pytest | Reader correctness (node count, group membership), element matrix values, fire curve formulas |
| Integration | pytest | Full pipeline: read → mesh → apply BC → solve 1 step → check energy balance |
| Regression | pytest | Validation: FAHTS results must match industry-recognised USFOS software output within agreed tolerance |
| Performance | cProfile | Analysis of model_file.fem (700 beams, 4h, dt=1s) should complete in < 120s |

---

## 10. File & Folder Map

```
FAHTS_solver/
├── CLAUDE.md                        ← AI agent context (READ THIS FIRST)
├── ROADMAP.md                       ← This file
├── requirements.txt
├── main.py                          ← Entry point: python main.py [file.fem]
│
├── model_file.fem                   ← Primary test model (USFOS format)
│
├── fahts/                           ← Main package
│   ├── core/
│   │   ├── model/
│   │   │   ├── fem_model.py         ← FEMModel container
│   │   │   ├── node.py
│   │   │   ├── element.py
│   │   │   ├── section.py           ← BoxSection (and future section types)
│   │   │   ├── material.py          ← SteelMaterial with EN 1993-1-2 props
│   │   │   └── group.py
│   │   ├── io/
│   │   │   ├── usfos_reader.py      ← PRIMARY reader for .fem files
│   │   │   ├── nastran_reader.py    ← Phase 5
│   │   │   ├── project_file.py      ← Phase 5 (.fahts save/load)
│   │   │   └── results_writer.py
│   │   ├── heat/
│   │   │   ├── sources/
│   │   │   │   ├── base_source.py
│   │   │   │   ├── fire_zone.py     ← Phase 2 primary source
│   │   │   │   ├── rad_point.py     ← Phase 5
│   │   │   │   └── jet_fire.py      ← Phase 5
│   │   │   ├── bc/
│   │   │   │   ├── view_factor.py
│   │   │   │   └── net_flux.py      ← EN 1993-1-2 q_net formula
│   │   │   ├── section_mesh/
│   │   │   │   └── box_mesher.py    ← Phase 3
│   │   │   └── solver/
│   │   │       ├── fem_2d_section.py   ← Phase 3 primary solver
│   │   │       ├── fem_3d_volume.py    ← Phase 5 (wraps legacy fem_heat_3d.py)
│   │   │       └── time_integrator.py
│   │   └── results/
│   │       ├── temperature_field.py
│   │       └── post_processor.py
│   ├── renderer/
│   │   ├── scene_manager.py
│   │   ├── beam_geometry.py         ← Extruded BOX mesh from BeamElement
│   │   ├── heat_source_vis.py
│   │   └── colormap.py
│   └── gui/
│       ├── main_window.py
│       ├── pyvista_widget.py        ← pyvistaqt wrapper
│       ├── panels/
│       │   ├── model_tree_panel.py
│       │   ├── properties_panel.py
│       │   ├── heat_source_panel.py
│       │   └── results_panel.py
│       ├── dialogs/
│       │   ├── fire_zone_dialog.py
│       │   ├── material_dialog.py
│       │   └── run_analysis_dialog.py
│       └── actions.py
│
├── legacy/                          ← Keep for reference; do not import directly
│   ├── 3D_heat_solver.py            ← Original 1D insulation solver
│   └── fem_heat_3d.py               ← 3D FEM solver (solid elements)
│
├── docs/
│   └── 3D_FEM_heat_transfer_theory.txt
│
├── tests/
│   ├── test_usfos_reader.py
│   ├── test_section_mesh.py
│   ├── test_heat_solver.py
│   └── test_net_flux.py
│
└── assets/
    └── icons/
```

---

## 11. Current Status (2026-05-16)

| Item | Status |
|------|--------|
| Phase 1 — Foundation (10 tasks + 3 bug fixes) | ✅ COMPLETE |
| USFOS reader: NODE, BEAM, BOX, IHPROFIL, PIPE, QUADSHEL, TRISHELL, MISOIEP, UNITVEC, ECCENT, BEAMHING, PLTHICK, Name/GroupDef | ✅ |
| Data model: FEMModel, Node, BeamElement (ecc1/ecc2), BoxSection, ISection, PipeSection, PlateSection, SteelMaterial, Group, ShellElement | ✅ |
| Renderer: beam_geometry (BOX+IHPROFIL+PIPE correct orientation + hollow + eccentricity), scene_manager, colormap | ✅ |
| GUI: main_window, model_tree_panel, properties_panel, recent files | ✅ |
| Test suite: 292 passing, 1 skipped | ✅ |
| Phase 2 — Heat Source Engine (9 tasks + GUI enhancements) | ✅ COMPLETE |
| Non-modal fire zone dialog + viewport point-picking for centre | ✅ |
| KFXView-style axis marker (cross + arrowheads + sphere, screen-centred via 1 ms QTimer) | ✅ |
| Live world-coordinate status bar display | ✅ |
| Phase 3 — Heat Transfer Solver | 🔄 IN PROGRESS (3.1–3.9 + 3E.1–3E.7 + 3F.1–3F.6 + full surface mesh refactor done; 3E.8 next) |
| Phase 4 — Results Visualisation | ✅ COMPLETE |
| Phase 5 — Advanced Features | ⬜ NOT STARTED |

**Full surface mesh refactor ✅ COMPLETE (2026-05-18):**
ALL section types now use FAHTS axial × hoop surface-shell approach (SINTEF FAHTS §3.2.2).

- BOX: `BoxSurfaceMesher` — n_top=2, n_side=3, n_length=4 (unchanged)
- IHPROFIL: `IProfileSurfaceMesher` — n_top=4, n_side=2, n_bottom=2, n_length=2; 8 outer faces
- PIPE: `PipeSurfaceMesher` — c_circ=8, n_length=4; unrolled cylinder coords in local_coords_2d
- QUADSHEL: `PlateSurfaceMesher` — mesh_12=4, mesh_14=2; bilinear plate grid in local_coords_2d
- TRISHELL: Shell1DSolver 1-D fallback retained (3 nodes — no surface mesh)
- `BeamSurfaceMesh` extended with optional `local_coords_2d` for curved/tilted elements
- `SurfaceTransientSolver` updated to use `element_coords_2d()` uniformly
- `AnalysisConfig` extended: n_top_i, n_side_i, n_bottom_i, n_length_i, c_circ, n_length_p, mesh_12, mesh_14
- `RunAnalysisDialog`: 4 Default/Custom mesh groups (BOX, IHPROFIL, PIPE, Shell)
- 72 new tests across test_iprofil_surface_mesher.py, test_pipe_surface_mesher.py, test_plate_surface_mesher.py

Test suite: 1008 passing, 1 skipped.

**Next immediate task:** Phase 3E.8 — Automated USFOS benchmark comparison test.
Compare FAHTS output against `validation/usfos/reference/fahts_beltemp.fem` using the Crank-Nicolson solver.
After 3E.8: Phase 5 (advanced features).

**USFOS Benchmark (validation/usfos/reference/):**
- Model: `model_t1.fem` (same as main test model — IHPROFIL + PIPE + BOX + shells)
- Heat source: two-zone RadiationBall at center=(343,484,64)m
  - Inner zone: r1=5m, flux1=350,000 W/m²
  - Outer zone: r2=100m, flux2=1,500 W/m²
- Reference output: `fahts_beltemp.fem` — BELTEMP format, 2056 elements, 15 min, 1-min output steps
- BELTEMP values are INCREMENTAL (not cumulative); accumulate from T_initial=20°C
- Mesh config: MESHBOX 8×4×4×4, MESHIPRO 8×4×4×4
- Material: EN 1993-1-2 Annex C tables (rho=7850, cp=510 baseline, k=50 baseline with T-dep multipliers, emiss=0.85)

**Key differences FAHTS vs USFOS:**
| Aspect | FAHTS | USFOS |
|--------|-------|-------|
| Beam FEM | 2-D cross-section FEM | Full 3-D beam FEM |
| Heat equation | 2-D plane (y-z) | 3-D volume |
| Axial conduction | Ignored | Included |
| Material props | EN 1993-1-2 Annex C piecewise | Table with multipliers |
| cp reference | Variable full table | 510 J/kg·K × T-dep factor |
| k reference | Variable full table | 50 W/m·K × T-dep factor |
| Output | Centroid T + section nodal T | Mean T + Y/Z gradients per element |

Note: Tasks 3.1–3.9 complete.
Note: Phase 3E tasks 3E.1–3E.7 complete. RadiationBall corrected to two-zone model.
Note: Phase 3F complete (CN solver, heat accumulation, βy/βz gradients).
Note: Full surface mesh refactor complete 2026-05-18 (all section types → FAHTS surface-shell approach). 1008 tests passing.

Phase 3 implements the 2-D FEM heat transfer solver in the steel cross-sections.
Start with the BOX cross-section mesh generator (Quad4 elements through the wall
thickness), then assemble stiffness + mass matrices, apply Robin BC on outer faces,
and run backward Euler time integration.

See `CLAUDE.md` for the full agent handover brief.
