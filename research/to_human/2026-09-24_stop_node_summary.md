# Stop-Node Summary — 0.45-Degree Ice-Floor Baseline

Date: 2026-09-24  
Status: consolidate before the next model feature

## Production-like candidate

`candidate_65n_045_icefloor`

- 0.45 degree, `dt=1800s`, `nu_h=2e6`
- lambda80, GM0, localized convection, FCT transport
- annual real 2m air forcing with `-1.8 C` ice-air floor
- global A2 RMSE: `1.1126 C`
- North Atlantic A2 RMSE: `1.0096 C`
- near-wall raw bias: `-0.9121 C`
- sub-freezing SST cells: `0`
- 365d repeat: reproduced

## Diagnostic parameterizations

These are useful error-attribution upper bounds, not pure dynamical
production defaults.

| proxy | global A2 | NA A2 | near-wall bias |
|---|---:|---:|---:|
| ice floor only | 1.113 C | 1.010 C | -0.912 C |
| + coastal restore tau=10d | 1.060 C | 0.978 C | -0.834 C |
| + coastal restore tau=3d | 1.003 C | 0.939 C | -0.711 C |

tau=10d is the gentler compromise. tau=3d is the strongest validated
diagnostic rung. tau=1d was not promoted because it becomes a tighter data
constraint.

## Rejected branches

- GM500 / Redi500: 30d gain reverses by 365d.
- Wet-cell marine-air smoothing: regional/near-wall metrics worsen.
- Simple coastal vertical/horizontal diffusion: no improvement.
- `freeze_adv_vel`: 30d global/NA/near-wall metrics worsen.
- SSS restore 30d: negligible.
- lambda60/70/100: no all-around improvement over lambda80.

## Remaining error

After tau=3d, the 0..3-cell coastal band still contains `47.6%` of global A2
SSE. Latitude `0..40N` contributes `42.2%`, and the Southern mid-latitude band
contributes another `34.9%`. Thus the remaining problem is not only the North
Atlantic wall.

## Why stop here

The easy incremental levers have now been bracketed or rejected. The next
meaningful improvement requires either:

1. a physical boundary-current/lateral heat-transport closure, or
2. a full mixed-layer/sea-ice treatment, or
3. a new surface-forcing dataset or bulk-flux formulation.

These are larger changes and should not be mixed with the current locked
candidate. This is the right point to consolidate before opening a new
experiment branch.
