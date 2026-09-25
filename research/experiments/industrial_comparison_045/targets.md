# External industrial-model benchmark targets

Date: 2026-09-26  
Status: Tier 2 evidence table

Purpose: keep industrial comparison explicit instead of replacing it with
internal baseline chasing.

## Model families

| Model | Family | First comparison role |
|---|---|---|
| MOM6 | structured generalized-coordinate OGCM | first direct rerun target |
| NEMO | structured/curvilinear operational climate OGCM | operational physics and tracer-scheme reference |
| MITgcm | flexible finite-volume framework | nonhydrostatic/adjoint comparison only if needed |
| MPAS-Ocean | variable-resolution unstructured C-grid | scalability and variable-resolution reference |
| FESOM2 | unstructured finite-volume with 3D FCT | transport/vertical-advection robustness reference |
| ICON-Ocean | icosahedral C-grid coupled system | mimetic discretization and GPU reference |
| HYCOM | operational hybrid-coordinate system | operational forecast and DA reference |

## Evidence type

Use three labels for every external entry:

- `rerun`: same protocol, same hardware, same reference, run by us.
- `public_output`: public model output regridded/scored by us.
- `literature`: published metric only; never used as the primary winner table.

The current ocean_solver annual candidate is `internal_diagnostic`, not
directly comparable to mature-model OMIP/CORE-II runs.

## External references

- OMIP physical protocol:
  Griffies et al., 2016, DOI `10.5194/gmd-9-3231-2016`.
- OMIP-2 resolution study:
  Tsujino et al., 2020, DOI `10.5194/gmd-13-4595-2020`.
- MOM6: https://mom6.readthedocs.io/
- NEMO: https://sites.nemo-ocean.io/user-guide/
- MITgcm: https://mitgcm.readthedocs.io/
- MPAS-Ocean: https://mpas-dev.github.io/
- FESOM2: https://fesom2.readthedocs.io/
- ICON-Ocean: https://icon-model.org/

## Immediate action

Move from qualitative comparison to one direct slice:

1. Pick MOM6 as the first external rerun target.
2. Reuse the current WOA-based error definitions.
3. Fix forcing, grid, bathymetry, initial state, duration, and diagnostics.
4. Run both ocean_solver and MOM6 on that slice.
5. Publish the comparison only with all configuration fields attached.

Until then, report industrial comparison as architecture parity/gap, not as a
numerical win or loss.
