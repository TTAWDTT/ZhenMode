# Min-Depth 500 m Sensitivity Result

Date: 2026-09-22
Status: complete
Runs: `coastal_mindepth500_30d`, `coastal_mindepth500_365d`

The only change relative to the locked candidate is `--min-depth 500` instead
of `100`. The 30d probe was stable, and the 365d run passed the global A-class
criteria.

## Climate metrics

| Metric | baseline | min-depth 500 |
|---|---:|---:|
| Global A2 RMSE | 1.336 C | 1.282 C |
| North Atlantic 40--60N A2 RMSE | 1.131 C | 1.158 C |
| Near-wall 55--60N A2 RMSE | 1.153 C | 1.148 C |

Global A2 improves by about 4.1%. The target sector is essentially unchanged,
while the broader North Atlantic sector is slightly worse.

## Near-wall land-distance bands

| Run | Group | cells | RMSE | SSE share | cold >1C share |
|---|---|---:|---:|---:|---:|
| baseline | land 0--3 | 122 | 1.533 C | 0.372 | 0.828 |
| baseline | land 4--7 | 238 | 1.246 C | 0.482 | 0.697 |
| baseline | land >=8 | 188 | 0.764 C | 0.146 | 0.202 |
| min500 | land 0--3 | 110 | 1.401 C | 0.337 | 0.791 |
| min500 | land 4--7 | 225 | 1.190 C | 0.499 | 0.644 |
| min500 | land >=8 | 176 | 0.765 C | 0.164 | 0.222 |

The 500 m floor removes 14 near-wall cells and improves both coastal bands.
This supports the hypothesis that the near-wall error is partly a mask/geometry
issue.

## Decision

Promote `min-depth 500` cautiously as a new diagnostic candidate. It is not a
production default yet because the broader North Atlantic sector is slightly
worse and the improvement is only about 4% global. Next either run a
reproducibility repeat or test a gentler mask change.
