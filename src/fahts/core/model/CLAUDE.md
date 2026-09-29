# Data Model Domain — FAHTS

## Section Types and Field Names

These are the exact field names used in Python dataclasses AND parsed from USFOS .fem files.
**Do not rename them** — they map directly to USFOS card fields.

### BoxSection — hollow rectangular (USFOS card: BOX)
```
BOX  id  H  T_side  T_bot  T_top  W
```
```python
@dataclass
class BoxSection:
    geom_id: int
    H: float        # height in local-z direction (UNITVEC direction)
    T_side: float   # wall thickness of left/right webs
    T_bot: float    # bottom flange thickness
    T_top: float    # top flange thickness
    W: float        # width in local-y direction
```
Inner cavity dimensions: `(W - 2·T_side) × (H - T_top - T_bot)`

### ISection — I/H profile (USFOS card: IHPROFIL)
```
IHPROFIL  id  h  tw  bf_top  tf_top  bf_bot  tf_bot
```
```python
@dataclass
class ISection:
    geom_id: int
    h: float        # total height in local-z direction
    tw: float       # web thickness
    bf_top: float   # top flange width
    tf_top: float   # top flange thickness
    bf_bot: float   # bottom flange width
    tf_bot: float   # bottom flange thickness
```

### PipeSection — circular hollow (USFOS card: PIPE)
```
PIPE  id  outer_diameter  wall_thickness
```
```python
@dataclass
class PipeSection:
    geom_id: int
    D: float    # outer diameter
    t: float    # wall thickness
    # Derived: R_outer = D/2, R_inner = D/2 - t
```

### PlateSection — shell plate (USFOS card: PLTHICK)
```
PLTHICK  id  thickness
```
```python
@dataclass
class PlateSection:
    geom_id: int
    thickness: float
```

---

## UNITVEC Convention (CRITICAL)

`UNITVEC` defines the **local z-axis** of the beam cross-section.

```python
local_x = normalize(node2 - node1)          # axial direction
local_y = normalize(cross(local_x, UNITVEC)) # flange/width direction
local_z = cross(local_y, local_x)            # height direction (UNITVEC approx)
```

- For **BOX**: `local_z` → H dimension; `local_y` → W dimension
- For **IHPROFIL**: `local_z` → web/height direction; `local_y` → flange direction
- Stored on `BeamElement.local_z` (np.ndarray shape (3,))

---

## FEMModel

```python
@dataclass
class FEMModel:
    nodes:          dict[int, Node]           # keyed by node id
    elements:       dict[int, BeamElement]    # keyed by eid
    shell_elements: dict[int, ShellElement]   # keyed by eid
    sections:       dict[int, Section]        # keyed by geom_id
    materials:      dict[int, SteelMaterial]  # keyed by mat_id
    groups:         dict[str, Group]          # keyed by group name
    unitvecs:       dict[int, np.ndarray]     # keyed by lcoor_id, shape (3,)
    source_file:    Path
```

## BeamElement

```python
@dataclass
class BeamElement:
    eid: int
    n1: int; n2: int
    mat_id: int; geom_id: int; lcoor_id: int
    length: float
    direction: np.ndarray   # unit vector n1→n2, shape (3,)
    local_z: np.ndarray     # cross-section z-axis, shape (3,)
    ecc1: np.ndarray | None  # eccentricity at n1 in global coords [m]
    ecc2: np.ndarray | None  # eccentricity at n2 in global coords [m]
```

## SteelMaterial (USFOS card: MISOIEP)

```python
@dataclass
class SteelMaterial:
    mat_id: int
    E: float        # Young's modulus [Pa]
    nu: float       # Poisson's ratio
    sigma_y: float  # yield stress [Pa]
    rho: float      # density [kg/m³] — typically 7850
    alpha_T: float  # thermal expansion coefficient [1/K]
```

Thermal properties (k, cp) come from EN 1993-1-2 Annex C tables in `material.py`,
NOT from MISOIEP fields. The Annex C tables override rho and give T-dependent k/cp.

---

## Node

```python
@dataclass
class Node:
    nid: int
    x: float; y: float; z: float   # global coords [m], X≈350m for model_file.fem
```

## ShellElement

```python
@dataclass
class ShellElement:
    eid: int
    node_ids: list[int]   # 4 nodes for QUADSHEL, 3 for TRISHELL
    mat_id: int; geom_id: int
    element_type: str     # "QUADSHEL" or "TRISHELL"
```

---

## Phase 5 notes (insulation)

When adding insulation layers in Phase 5, new fields will be added to section types
or a wrapper `InsulatedSection` dataclass. Do not add insulation fields directly to
BoxSection/ISection/PipeSection — the USFOS .fem format does not carry them.
