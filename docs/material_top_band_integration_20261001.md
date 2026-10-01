# Original full-stage top-band seam — 2026-10-01

Base `0e1daf95257606b431fd6ebf50fd70fcc48cb801` (main). Research only;
production solver defaults and numerical formulas are unchanged. This is the
first integration seam, **not a moving top-band implementation or historical
failure repair**. Do not run the actual dt300 archive through it as a proof.

Dependency update: main was normally fast-forwarded to
`0e4dd0fd6c2577e091260b7fa62bc70c9590063b` to include reviewed PR14. Its inventory
kernel still has a two-column, fixed -22.5 m contract and rejects pressure/deep
transport; this entry does not pretend that it is a whole-grid momentum adapter.
The existing general column P1 integration function is reused directly. PR14's
boolean-wet-prefix and missing explicit water-residual gate review notes remain
tracked requirements before using that kernel in a future numerical adapter;
no PR14 kernel operation is called by this restricted full-stage seam.

## Callable contract

From the checkout with its `src` on the Python path:

```python
from research.experiments.material_top_band.integration import advance, coverage
capability = coverage(original_fd_state, frozen_params, grid, max_subcycles=256)
result = advance(original_fd_state, frozen_params, grid, max_subcycles=256)
```

`result` contains selected `state`, an optional diagnostic `band` export,
`accepted` and `report`. On rejection, all six original fields are deep copies
of the entry snapshot. A numerical rejection can happen after stages execute;
executed and validated stage lists are separate. A successful call requires
every declared stage to execute and validate, the original complete-step gates,
parameter/source/input-state snapshot checks and independent heat/salt budgets.
No checkpoint or second inventory authority is introduced by this first seam.

## Deliberately narrow executable domain

All-wet, uniform bathymetry, horizontally identical resting columns, eta=0,
ice=0, no wind, horizontally uniform heat/atmosphere. At least four and at most
64 nodes, no more than 1024 horizontal cells. Top-band control intervals are
the original first-three nodal-dual intervals; their actual masses coincide
with reference masses. The verified P1 bridge integrates four stocks over the
identical intervals. It is a diagnostic view of the original stock authority,
not a remap pretending to solve changing geometry.

Within this domain actual-mass velocity is exactly original velocity (zero).
At common physical depths all columns have identical pressure profiles, so
both original and band horizontal pressure forces/work are zero even though
their vertical interpolation profiles need not coincide. The observer measures
original 3D and column pressure force and power at each stage; nonzero values
reject. The shared original layer/column faces and bottom-up vertical inverse
are inspected before tracer replay; measured cross-band transport, face mismatch
or any nonzero face flux rejects. This zero-transport domain does **not** validate
nonzero pressure work, transport or momentum migration.

Nonzero diffusion/biharmonic, convection coefficients, bulk heat and linear drag
are retained; FCT and original nonlinear momentum filtering are dispatched by
the original operators. Some tendencies vanish in resting identical columns.
The successful prototype therefore proves dispatch plus changing tracer stocks,
not dynamic correctness of those zero-tendency processes.

## One stage sequence, one stock contract

The entry calls the existing `material_top._material_step`; it does not copy the
solver or substitute inventory KDK:

1. Exact bottom drag half-step.
2. Actual-capacity subcycled material linear half-step, including active tracer
   biharmonic and joint momentum diffusion.
3. Original symmetric-fast nonlinear predictor with its Heun/FCT/filtered
   momentum operators and source feedback.
4. Predictor's original second linear half-step.
5. Original 12-substep symmetric fast surface update.
6. Original layer/column transport match and vertical cross-band diagnostics.
7. Accepted material tracer replay with original Heun weights, FCT, convection
   and source inventory.
8. Second actual-capacity material linear half-step.
9. Final bottom drag half-step.
10. Closed-wall normal momentum condition.

`src/material_top.py` adds an optional **eager** observer at these existing
dispatch points; default `None` is a no-op, including existing JIT entry points.
An observer is for this eager research entry, not a promise of JIT-compatible
callbacks or per-fast-substep instrumentation.

The gate preserves `symmetric_fast_v3`, `match_barotropic_transport=True`, 12 fast
steps filling dt, `adv_nsub=conv_nsub=nu_nsub=2`, FCT and linear bottom drag.
Original active coefficients/forcing are never edited to fit this domain.
Capacity plans use the original row-rate bounds and a maximum of 256; unsupported
plans reject before any numerical stage. The archived 42/47 linear substep
requirements are **not replaced by a fixed count** or claimed demonstrated here:
moving geometry still prevents using the actual archived inputs.

Parameter metrics bind to the existing nodal-dual geometry builder, including
inverse, vertical, surface/bottom/wall fields and reference-depth weights.
Unknown masks and nonfinite inputs reject. Heat/salt inventory differences are
computed directly from NumPy before/returned fields, areas and coincident masses,
not read from the solver's observed-change budget. Reported source terms retain
the original accepted replay's weights. Digests record the input six fields,
parameter snapshot and listed local source files; they are not signed provenance
or a complete dependency/environment attestation.

## Explicit missing adapters

Moving eta, velocity, horizontal density gradients, wind, coastline/shallow
columns and variable bathymetry are rejected **before** entering the original
numerical stages. General integration still needs a single authoritative moving
stock representation across actual-mass momentum, common-depth pressure force
and work, deep/top flux exchange, biharmonic/vertical diffusion, FCT and source
deposition. Preserving old reference-mass dynamics while replacing only tracer
capacity is not accepted as an adapter. Restoring/mixed-layer/coastal branches
outside this seam are also refused. Already disabled GM/Redi/ice/sponge are not
new targets, and no real input is edited to disable an active process.

Next vertical change should extend **one common geometry/mass contract** across
those coupled stages, prove nonzero-force/work/transport cases, and then pass
independent review before the single real dt300 call. Passing this resting-column
prototype cannot justify that call, a long integration, physical qualification,
industrial quality, temporal order or a speed claim.

## Tests and resource boundary

`tests/test_material_top_band_integration.py` runs the full original stage path
with nonzero mixing/biharmonic and bulk heat, checks heat stock/source closure,
stage coverage and unchanged inputs; compares the observer path byte-for-byte
with the original complete result. Preflight refusal covers moving/inhomogeneous
inputs, wind/coast, inconsistent consumed metrics, unknown masks and unsupported
capacity. Midstage geometry injection rolls back. A forged mutually matching
production heat/source budget is rejected by the independent inventory check.

Small CPU prototypes only, each bounded at 180 seconds and 4 GiB by the isolated
test runner; total test execution budget is 20 minutes. No archived-state solver
step or new standing-wave integration is run. Original workspace/data, Goals,
other sessions, B branch, system settings and reset cards are untouched.

## 2026-10-01 validation record

Final bounded batch: 56 tests passed in 51.33 seconds (child process wall
52.83 seconds), peak Windows working set 1,213,816,832 bytes and peak process
commit counter 1,418,964,992 bytes. Both are below 4 GiB. Repository ruff,
explicit research-module ruff and diff whitespace checks passed. All local
test executions in this iteration total less than five minutes, below the
20-minute budget. The existing default-JIT complete heat/capacity regressions,
joint momentum operator oracles and reviewed PR14 kernel regressions also pass.

Independent static inspection closed masked grid rotation and NaN source-budget
fail-open cases and accepted the final code gate. Invalid source shape/NaN/Inf
and a forged mutually matching production budget cannot manufacture independent
heat closure. The observer path matches the original complete selected state
byte-for-byte on the declared domain. This evidence remains prototype-level;
actual dt300/dt600 solver steps, moving-domain qualification and industrial
speed/quality comparisons were not attempted.

## 2026-10-01 correction: derived capacity and comparison scope

A finite coefficient (`kappa_bi=1e308`) can overflow the derived row bound and
produce an infinite required count. The former host integer conversion raised
`OverflowError` before structured refusal. Capability checks now validate scalar
plan fields and finite requirements first, inspect support before integer
conversion, and check supported counts against the capacity/integer contract.
Nonfinite plans record a null count and an explicit missing contract; finite
over-capacity requirements remain uncut finite values. No broad exception
swallowing or coefficient clipping is introduced. Direct coverage and full-entry
regressions require JSON-safe refusal, zero numerical stages and full snapshot
rollback; the existing finite over-capacity witness remains.

The byte comparison is specifically against
`material_top._material_step(actual_geometry_v2, joint_heun_v1)`, **not** the
production `_step_impl`. The resting domain's positive biharmonic, FCT, drag and
other coefficients often have zero tendencies. Stage dispatch and the observed
heat/vertical-mixing stock change do not establish nonzero dynamic physics.

Follow-up validation: 26 focused integration tests passed in 40.16 seconds;
bounded child wall 41.58 seconds, peak working set 1,072,394,240 bytes and process
commit counter 1,254,649,856 bytes. Both lint checks and independent static
inspection passed. No numerical scope or production exception policy changed.
