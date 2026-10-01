# Full candidate safety contracts — 2026-10-01

Base: `3ff4198fa108092336655990ecbd741bc3ad1356` on
`codex/ocean-full-candidate-interface`. These are safety contracts only, not
complete historical failure-step repair or industrial/scientific qualification.

## Independent reproduction

Using the existing production-grid `configured()` synthetic fixture, refreshing
the parameter digest after each coefficient mutation, the initial regression run
had 11 failures and 20 passes. All six negative coefficients (`kappa_h`,
`kappa_v`, `kappa_conv`, `nu_h`, `nu_v`, `r_bot`) were accepted at -0.001;
positive-infinite `r_bot` was accepted. Other tested nonfinite values were already
rejected downstream and must not be described as accepted. Doubling only the
grid's `dx_2d`, `dy` or `cos_lat` was accepted. Restart loading also accepted
mutually matching changed parameter/grid metrics without binding the saved
parameter digest. A direct linear-stage control demonstrates negative drag
increases kinetic energy, while positive drag dissipates it.

## Minimal change

`advance` now requires the six diffusion/viscosity/drag coefficients to be real,
finite, nonnegative scalars before entering numerical updates, including inactive
convection. It returns a deep copy of the complete entry state on rejection;
tests cover all five arrays, identity, step and unchanged input. No clipping,
threshold relaxation or physical algorithm change is introduced.

Geometry binding compares all horizontal metrics consumed by this candidate:
`dx_2d` (nx, ny), scalar `dy`, and `cos_lat` (ny). Both representations must be
finite and positive with the exact contracted shapes and identical values.
These checks run on advance and restart loading. Loading additionally binds
the complete parameter digest and validates the dissipative coefficients.
The rotation parameter remains independently configurable; this is not a claim
that every unused production-grid field is bound. The low-level `linear` helper
is an internal numerical operator, not the validated full-step entry point.

## Limits

Tests are small synthetic safety regressions, not a new standing-wave integration
or a real archived-state replay. Existing unsupported-process rejection, stage
schedule mismatch, incomplete original solver equivalence, historical-input
provenance gaps and physical validity questions remain. Digests are consistency
identities, not signatures against deliberate coordinated rewriting. Original
user workspace/data and the candidate base branch were not modified.

Validation: 106 focused candidate/geometry/static-bridge tests passed in 5.31s;
repository ruff and diff whitespace checks passed. Independent inspection caught
NumPy masked-array coercion dropping unknown values; both gates now reject any
mask before conversion, with nine additional regressions. No private arrays,
input files, credentials or solver integrations are included.
