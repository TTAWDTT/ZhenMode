# Coastal Structural Levers Protocol

Status: complete
Date: 2026-09-22
Baseline: `lambda160 + min-depth500 + smooth80`
Question: Can a finer vertical grid or stronger bathymetry smoothing remove the
remaining 0--3-cell coastal cold bias?

## Runs

1. `z19` probe with 19 vertical levels: FAIL_BLOWUP at day 10.
2. `z18` probe with 18 vertical levels: PASS, but the 30d coastal 0--3-cell
   bias is nearly unchanged.
3. `smooth160` 30d probe: PASS and improves the global 0--3-cell bias slightly.
4. `smooth160` 365d run: PASS.

## 365d metrics

| smooth passes | global A2 | NA 40--60N | near-wall 55--60N |
|---:|---:|---:|---:|
| 80 | 1.294 C | 0.997 C | 1.019 C |
| 160 | 1.281 C | 1.000 C | 1.048 C |

## Decision

Keep `smooth80` as the all-around candidate. A finer vertical grid is not a
cheap coastal fix, and more bathymetry smoothing trades near-wall skill for a
small global gain. Next try a narrower coastal-ventilation/boundary experiment
rather than more resolution or more smoothing.
