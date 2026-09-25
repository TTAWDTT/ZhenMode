# Industrial-mode external benchmark protocol

Status: active
Date: 2026-09-26

## Why this is required

The current `candidate_65n_045_icefloor` metrics are internal. They are useful
for ocean_solver development, but they are not a claim that ocean_solver beats
MOM6, NEMO, MITgcm, MPAS-Ocean, FESOM2, ICON-Ocean, or HYCOM.

The current limitation is not that the solver lacks an RMSE; it is that the
present comparison does not yet satisfy an external-model benchmark contract.

## Non-negotiable comparability conditions

An external comparison may only be called apples-to-apples when these are fixed:

1. Grid: same horizontal grid family and nominal resolution.
2. Vertical grid: same vertical coordinate definition or a documented remap.
3. Bathymetry and masks: same source and processing rules.
4. Initial state: same observed temperature/salinity initial condition.
5. Forcing: same wind stress, 2m temperature/humidity, radiation, precipitation,
   and runoff if runoff is enabled.
6. Sea ice: either the same ice component or the same ice proxy treatment.
7. Integration length and spin-up state: same duration and restart policy.
8. Reference: same SST/S/MLD observational reference and masking.
9. Metrics: global A1/A2, raw bias/RMSE, regional 40--60N Atlantic and near-wall
   55--60N, heat/salt drift, ice extent, MLD, and wall time per simulated year.
10. Verdict: stability watchdog result and reproducibility repeat.

## Benchmark ladder

### Tier 0 — internal diagnostic benchmark (done)

Current 0.45-degree annual candidate uses WOA SST as the same-grid reference.
It is not directly comparable with industrial models because the forcing,
spin-up, resolution, and physics differ.

### Tier 1 — capability and architecture comparison (done)

`research/to_human/industrial_positioning_zh.md` compares ocean_solver with
MOM6, NEMO, MITgcm, FESOM2, ICON-Ocean, ROMS/FVCOM, and related systems.
This is architecture-level evidence, not a performance claim.

### Tier 2 — published protocol alignment (active)

Use the public OMIP/OMIP-2 definitions to select forcing, diagnostics, and
evaluation fields. This does not require running a mature model first, but it
makes later comparison interpretable.

Primary protocol references:

- OMIP: Griffies et al., 2016,
  [10.5194/gmd-9-3231-2016](https://doi.org/10.5194/gmd-9-3231-2016).
- OMIP-2 resolution study: Tsujino et al., 2020,
  [10.5194/gmd-13-4595-2020](https://doi.org/10.5194/gmd-13-4595-2020).

### Tier 3 — one-model direct comparison (next)

Pick one mature model as the first apples-to-apples external target. The
preferred target is MOM6 because:

- it is a widely used structured global ocean component;
- its configuration and diagnostic ecosystem are public;
- it has strong OMIP-style usage and a generalized-coordinate path;
- it avoids an immediate unstructured-mesh rewrite.

The first direct slice should be:

1. A 1-degree or 0.5-degree ocean-only 365d integration.
2. The same initial SST/S field used by ocean_solver.
3. The same annual/seasonal atmospheric forcing, or an explicit OMIP forcing
   run for both models.
4. The same WOA-based SST error metrics.
5. The same stability and heat/salt-drift gates.

### Tier 4 — multi-model external table (later)

Only after one model comparison passes should the table be extended to NEMO,
MITgcm, MPAS-Ocean, FESOM2, ICON-Ocean, or HYCOM. Each row must state whether
it is a rerun, public output, or literature-derived number.

## Output contract

Create one table per run with the following fields:

| Field | Required |
|---|---|
| model | yes |
| repository/version | yes |
| grid | yes |
| vertical grid | yes |
| bathymetry | yes |
| initial state | yes |
| forcing | yes |
| sea-ice treatment | yes |
| days | yes |
| global A2 RMSE | yes |
| NA 40--60N A2 RMSE | yes |
| near-wall raw bias | yes |
| raw global bias/RMSE | yes |
| heat drift | yes |
| salt drift | yes |
| verdict | yes |
| reproducible repeat | yes |
| wall time / sim-year | if hardware is compared |
| source | URL, config, or internal run path |

## Decision rule

1. Do not put ocean_solver and an external model in the same "winner" table
   unless every field above is matched.
2. Report capability gaps separately from metric gaps.
3. Treat a public model output as secondary unless we can reproduce its setup.
4. Treat a literature-derived number as context, not as the primary benchmark.
5. Promote ocean_solver changes only through the internal benchmark gates;
   external comparison is validation, not an excuse to retune ad hoc.
