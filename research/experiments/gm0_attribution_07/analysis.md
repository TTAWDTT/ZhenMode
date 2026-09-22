# GM0 Attribution at 0.7 Degree

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

## Climate metrics

| Metric | GM0 | GM500 |
|---|---:|---:|
| Official A2 RMSE | 1.339 C | 1.339 C |
| North Atlantic 40--60N A2 RMSE | 1.141 C | 1.216 C |
| North Atlantic raw bias | -0.685 C | -0.760 C |
| Near-wall raw bias | -0.876 C | -0.936 C |

Global A2 is essentially unchanged, but the regional benefit of GM0 is
reproducible.

## Surface heat tendencies

| Term | GM0 North Atlantic | GM500 North Atlantic |
|---|---:|---:|
| Advection | -0.057 K/d | -0.072 K/d |
| Convection | +0.327 K/d | +0.207 K/d |
| GM bolus | 0 K/d | +0.001 K/d |
| Surface bulk flux | +0.166 K/d | +0.192 K/d |

Near the wall, the difference is larger:

| Term | GM0 | GM500 |
|---|---:|---:|
| Advection | -0.038 K/d | -0.075 K/d |
| Convection | +0.455 K/d | +0.281 K/d |
| GM bolus | 0 K/d | +0.055 K/d |
| Surface bulk flux | +0.134 K/d | +0.156 K/d |

Surface velocities are nearly identical between the two runs. Thus GM500 does
not materially change the large-scale horizontal current speed. Its main effect
is to strengthen advective cooling in the target sector and reduce compensating
convective warming.

## Decision

The GM0 benefit is physical in the sense that the current GM500 closure
over-cools the North Atlantic at 0.7 degree. It is not a velocity-field
artifact. Keep GM0 in the locked diagnostic baseline. The next step is to
directly diagnose the boundary-current and heat-transport structure.
