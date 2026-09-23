# 0.5-Degree Resolution Plus Coastal Restore

Status: complete
Date: 2026-09-23
Question: Does the stabilized 0.5-degree run make the coastal SST constraint
redundant?

## Runs

- 0.50 degree, no restore, `dt=1800s`, `nu_h=2e6`.
- 0.50 degree plus hard coastal SST restore, `tau=0.5d`, cells<=7.

## 30d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| 0.50 no restore | 0.893 C | 0.734 C | 0.756 C |
| 0.50 + restore | 0.831 C | 0.700 C | 0.683 C |

## 365d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| 0.50 no restore | 1.204 C | 0.914 C | 0.976 C |
| 0.50 + restore | 1.078 C | 0.818 C | 0.749 C |

For comparison, the 0.70-degree hard restore diagnostic is `1.154 C / 0.880 C /
0.804 C`.

## Interpretation

1. Finer resolution improves the no-restore climate, but does not remove the
   coastal boundary-value error.
2. The coastal constraint remains complementary and produces the best 365d
   all-around diagnostic result so far.
3. This is still an assimilation-like upper bound, not a production default.

## Decision

Record `0.50 degree + tau=0.5d coastal restore` as the current diagnostic upper
bound. Keep the locked 0.70-degree lambda80 candidate as the production-like
baseline. Next test whether a gentler restore timescale can retain most of the
gain at 0.50 degree.
