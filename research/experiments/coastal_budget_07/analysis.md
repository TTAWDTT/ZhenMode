# Coastal Surface Heat Budget Diagnostic

Status: complete
Date: 2026-09-23
Runs: 30d `coastal_budget_baseline_30d` and 30d hard coastal restore
`tau=0.5d, cells<=7`; both saved 3D tracer terms. Budgets below average the
last 10d snapshots (day 20 and 30) with cell-area weights.

## Near-wall 55--60N

| land-distance group | SST bias baseline -> restore | advection baseline -> restore | convection baseline -> restore | bulk baseline -> restore | SST restore | net baseline -> restore |
|---|---:|---:|---:|---:|---:|---:|
| 0--3 cells | -1.31 -> -0.44 C | 0.000 -> -0.027 K/d | +1.398 -> +0.236 K/d | ~0 K/d | +0.875 K/d | +1.399 -> +1.083 K/d |
| 4--7 cells | -0.96 -> -0.29 C | -0.023 -> -0.111 K/d | +0.321 -> -0.032 K/d | ~0 K/d | +0.573 K/d | +0.297 -> +0.429 K/d |
| 8--14 cells | -0.73 -> -0.36 C | -0.030 -> -0.059 K/d | +0.377 -> +0.174 K/d | ~0 K/d | 0 K/d | +0.347 -> +0.115 K/d |
| >=15 cells | -0.54 -> -0.53 C | -0.049 -> -0.048 K/d | +0.431 -> +0.447 K/d | ~0 K/d | 0 K/d | +0.383 -> +0.399 K/d |

The +0.875 and +0.573 K/d restore terms correspond to roughly 200 and 130
W/m2 in the 5 m top layer.

## Interpretation

1. Horizontal diffusion, vertical diffusion, and bulk heat exchange are almost
   zero in the top layer. The direct SST restoring term is therefore not
   compensating for a missing scalar local diffusion or bulk-exchange process.
2. Restore warming strongly reduces local convection: -1.163 K/d in the 0--3
   cell band and -0.352 K/d in the 4--7 cell band. The constraint acts as a
   boundary-value control on the coastal mixed layer, not as a simple heat
   addition.
3. The 8--14 cell band improves remotely even without direct restoring, while
   the >=15 cell interior is nearly unchanged. This confirms that the main
   benefit is localized to the coastal/transitional band.
4. A missing physical closure should supply the required coastal boundary heat
   or transport, not another scalar diffusion or bulk-flux multiplier.

## Decision

Keep the hard-band SST restore as a diagnostic upper bound. Do not promote it
to a production default. Next examine a boundary-value or lateral-transport
closure, with a narrowly scoped 30d probe before any 365d run.
