# Finer-Resolution Stability and Skill Probe

Status: active
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80` at 0.7 degree

## Question

Can the boundary/lateral-transport bottleneck indicated by the coastal heat
budget be improved by finer horizontal resolution, once the finer rungs are
stabilized?

## Stabilization

The finer probes use `dt=1800s` and `nu_h=2.0e6 m2/s`. Other physics remain the
all-around 0.7-degree candidate: `lambda_bulk=160`, `kappa_v=1e-6`,
`kappa_conv=0.01`, `kappa_gm=0`, localized convection, FCT transport, projected
advective velocity, `min_depth=500`, and `smooth_passes=80`.

## Stability

| resolution | remap | 10d | 30d | 365d |
|---:|---|---|---|---|
| 0.65 | area | PASS | PASS | PASS |
| 0.60 | legacy | PASS | PASS | PASS |
| 0.55 | area | PASS | PASS | pending |
| 0.50 | legacy | PASS | PASS | running |

The previous 0.65/0.6/0.5 failures used the 0.7-degree time step and automatic
viscosity scaling. The current probes show that a smaller baroclinic step plus a
modest viscosity floor is sufficient for these rungs.

## 30d climate metrics

| resolution | global A2 | NA 40--60N | near-wall 55--60N |
|---:|---:|---:|---:|
| 0.70 | 0.985 C | 0.794 C | 0.823 C |
| 0.65 | 0.948 C | 0.769 C | 0.799 C |
| 0.60 | 0.923 C | 0.765 C | 0.808 C |
| 0.55 | 0.909 C | 0.749 C | 0.799 C |
| 0.50 | 0.893 C | 0.734 C | 0.756 C |

## 365d climate metrics

| resolution | global A2 | NA 40--60N | near-wall 55--60N |
|---:|---:|---:|---:|
| 0.70 | 1.294 C | 0.997 C | 1.019 C |
| 0.65 | 1.265 C | 0.952 C | 0.998 C |
| 0.60 | 1.221 C | 0.940 C | 0.989 C |

## Interpretation

1. 0.65 and 0.60 improve the all-around 365d metrics over 0.70.
2. The 30d ladder continues to improve through 0.55 and 0.50.
3. 0.50 is now the leading short-run resolution, but it still needs a 365d
   stability and climate check before any promotion.

## Decision

Keep 0.7 as the locked production-like baseline for now. Promote 0.6 to the
finest validated all-around diagnostic resolution. Do not promote 0.5 until its
365d run and scored climate metrics complete.
