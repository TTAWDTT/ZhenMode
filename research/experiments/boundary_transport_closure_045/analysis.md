# Boundary Transport Closure A/B at 0.45 Degree

Status: complete
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
| 500 | 0.9132 C | 0.8485 C | -0.7403 C | promoted to 365d |
| 1000 | 0.9195 C | 0.8648 C | -0.7501 C | reject |

GM500 improves the global metric and is marginally better in both regional
metrics. GM1000 makes the North Atlantic and near-wall cells colder and is
rejected.

## 365d validation

| `kappa_gm` | global A2 | NA 40--60N A2 | near-wall raw bias | verdict |
|---:|---:|---:|---:|---|
| 0 | 1.1488 C | 1.0105 C | -0.9175 C | locked candidate |
| 500 | 1.1718 C | 1.1086 C | -0.9430 C | reject |

The 30d GM500 gain does not survive the annual integration. The bolus
transport overcools the North Atlantic/near-wall band on climate timescales.

## Decision

Reject GM500. Retain `candidate_65n_045_gm0` (`kappa_gm=0`) as the
production-like baseline. The next closure must improve the boundary heat
budget without adding this sustained regional cooling.

## Decision rules

- Reject a closure if global A2 worsens by more than 1% or near-wall bias becomes colder.
- Promote to 365d only if the 30d probe improves the all-around comparison.
- Keep any diagnosed coastal restore separate from the production candidate.
- Do not add a second new closure before interpreting the single-lever A/B result.
