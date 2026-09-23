# Boundary Transport Closure A/B at 0.45 Degree

Status: active
Date: 2026-09-24

## Question

After rejecting simple coastal diffusion, can a standard lateral
transport closure improve the North Atlantic/near-wall cold band without
degrading global A2?

## Fixed reference

Use the locked 0.45-degree lambda80 candidate. Keep `dt=1800s`, `nu_h=2e6`,
`area` remapping, annual real air forcing, and all other candidate flags
unchanged.

## 30d A/B result

| `kappa_gm` | global A2 | NA 40--60N A2 | near-wall raw bias | verdict |
|---:|---:|---:|---:|---|
| 0 | 0.9190 C | 0.8494 C | -0.7425 C | control |
| 500 | 0.9132 C | 0.8485 C | -0.7403 C | promote to 365d |
| 1000 | 0.9195 C | 0.8648 C | -0.7501 C | reject |

GM500 improves the global metric and is marginally better in both regional
metrics. GM1000 makes the North Atlantic and near-wall cells colder and is
rejected.

## Current test

Run the same GM500 setting for 365d. If it passes and preserves the 30d
all-around gain, consider it as the next candidate; otherwise retain the
0.45-degree GM0 candidate.

## Decision rules

- Reject a closure if global A2 worsens by more than 1% or near-wall bias becomes colder.
- Promote to 365d only if the 30d probe improves the all-around comparison.
- Keep any diagnosed coastal restore separate from the production candidate.
- Do not add a second new closure before interpreting the single-lever A/B result.
