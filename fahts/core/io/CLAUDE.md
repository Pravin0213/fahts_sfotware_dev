# I/O Domain — FAHTS

## USFOS .fem Format

NOT Nastran BDF. Key differences:
- Comments: lines starting with `'` (single quote)
- Free-space delimited (not fixed 8-char fields)

### Key Cards

| Card | Fields | Notes |
|------|--------|-------|
| `NODE` | id X Y Z [BC...] | Coords in metres (global vessel frame) |
| `BEAM` | id n1 n2 mat geom lcoor [ecc1 ecc2] | ecc1/ecc2 are ECCENT IDs (0=none) |
| `BOX` | id H T_side T_bot T_top W | H in local-z (UNITVEC); W in local-y |
| `IHPROFIL` | id h tw bf_top tf_top bf_bot tf_bot | h in local-z; bf in local-y |
| `PIPE` | id outer_diameter wall_thickness | |
| `QUADSHEL` | id n1 n2 n3 n4 mat geom | 4-node shell |
| `TRISHELL` | id n1 n2 n3 mat geom | 3-node shell |
| `MISOIEP` | id E ν σ_y ρ α_T | Isotropic steel material |
| `UNITVEC` | id dx dy dz | Local **z-axis** of beam cross-section |
| `ECCENT` | id Ex Ey Ez | Node offset in **global** coords [m] |
| `Name Group` / `GroupDef` | id name / id Elem id1 id2... | Named element sets |
| `BEAMHING` | release flags eid | Moment releases (stored, not used thermally) |
| `PLTHICK` | id thickness | Plate thickness for QUADSHEL/TRISHELL |
| `USERFLUX` | 0 set cx cy cz r1 flux1 r2 flux2 | RadiationBall (type 0) |

### UNITVEC Convention
`UNITVEC` defines the local z-axis of the beam.
- `local_y = cross(local_x, UNITVEC)`
- `local_z = cross(local_y, local_x)`
- For IHPROFIL: lz → web/height direction; ly → flange direction
- For BOX: lz → H dimension; ly → W dimension

### Test Models
- `model_file.fem`: 659 nodes, ~783 beams, 0 shells, 5 BOX sections, 1 material (S355), coords X≈350m
- `model_t1.fem`: ~1040 nodes, ~1800 beams, shells, IHPROFIL+PIPE+BOX+ECCENT records
- `model_t3.fem`: asymmetric BOX sections

---

## BELTEMP Format

USFOS BELTEMP file format for thermal results exchange.

**CRITICAL: Values are INCREMENTAL temperature changes**, not absolute temperatures.
Accumulate from T_initial=20°C using `beltemp_parser.cumulative_temperatures()`.

### Parser — beltemp_parser.py
- `parse_beltemp(path) → list[BeltempRecord]`
- `cumulative_temperatures(records, T_initial=20) → dict[eid, (times, T_mean, T_grad_y, T_grad_z)]`

### Exporter — results_writer.export_beltemp()
```python
export_beltemp(result, path, T_initial=20, time_unit="s", meshes=None)
```
When `meshes={eid: SectionMesh}` supplied, columns 4+5 carry real βy/βz increments.

### Temperature Gradients (SINTEF §3.4.2)
```
βz = Σ(T_k·y_k·A_k) / Iz   [gradient about z-axis → bending in y]
βy = Σ(T_k·z_k·A_k) / Iy   [gradient about y-axis → bending in z]
Iz = Σ(y_k²·A_k),  Iy = Σ(z_k²·A_k)
```
Implemented in `TemperatureField.section_gradient(eid, t_idx, mesh)`.

---

## USFOS Benchmark

Location: `usfos_verification_results/`

| File | Contents |
|------|----------|
| `fahts_beltemp.fem` | Reference output: 2056 elements, 15 time steps (1–15 min), incremental |
| `fahts.fem` | Material config: rho=7850, c_ref=510 J/kg·K, k_ref=50 W/m·K, emiss=0.85 |

**Heat source config:**
```
USERFLUX 0 1 343 484 64 5 350000 100 1500
```
- Type 0 = RadiationBall; center=(343, 484, 64) m
- Inner zone: r1=5 m, flux1=350,000 W/m²
- Outer zone: r2=100 m, flux2=1,500 W/m²
- Model used: `model_t1.fem`

**Material difference:** FAHTS uses EN 1993-1-2 Annex C tables; USFOS uses
`thermpar × tempdepy` multiplier tables. Do NOT expect exact numerical agreement.

---

## Other Export Functions — results_writer.py

- `export_peak_temperature_csv(result, path)` — peak T per element
- `export_temperature_history_csv(result, path)` — full time history
- `export_results_vtk(result, model, path)` — VTK for ParaView
- `export_results_excel(result, path)` — Excel workbook
- `export_bc_summary_csv(bcs, path)` — boundary condition summary
