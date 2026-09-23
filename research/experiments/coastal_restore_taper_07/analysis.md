# Coastal T Restoring Taper Probe

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`
Control: hard coastal restore, `tau=0.5d`, cells<=7

## Question

Does weighting the coastal SST restoring away from land improve the 30d
coastal-band skill?

## Change

Add `--coastal-restore-taper {none,linear,cos}`. The cosine profile decays from
land to the edge of the 0..7-cell band. The runner stores the coastal mask as a
float-weighted field.

## Runs

1. `tau=0.5d, cells<=7, cos` (natural taper).
2. `tau=0.25d, cells<=7, cos` as an equal-total-strength probe. The cosine
   profile has roughly half the integrated weight of the hard band over integer
   cells, so halving tau approximately doubles the restoring rate.

## 30d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| hard tau=0.5d | 0.919 C | 0.763 C | 0.785 C |
| cos tau=0.5d | 0.935 C | 0.786 C | 0.823 C |
| cos tau=0.25d | 0.924 C | 0.786 C | 0.830 C |

## Decision

Reject the cosine taper. It reduces the control over the 4..7-cell transition
band and remains worse than the hard band even with approximately equal total
strength. Keep the hard-band SST constraint only as the diagnostic upper bound.
Do not spend a 365d run on this lever.
