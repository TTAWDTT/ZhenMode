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

## Gentler restore-timescale sweep

At 0.50 degree, the 30d timescale sweep in the same cells<=7 band is:

| tau | global A2 | NA 40--60N | near-wall 55--60N |
|---:|---:|---:|---:|
| no restore | 0.893 C | 0.734 C | 0.756 C |
| 3d | 0.866 C | 0.717 C | 0.718 C |
| 1d | 0.843 C | 0.704 C | 0.690 C |
| 0.5d | 0.831 C | 0.700 C | 0.683 C |

The 365d validation for the gentler 	au=1d setting is:

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| 0.50 no restore | 1.204 C | 0.914 C | 0.976 C |
| 0.50 + tau=1d | 1.100 C | 0.831 C | 0.776 C |
| 0.50 + tau=0.5d | 1.078 C | 0.818 C | 0.749 C |

Decision: keep 	au=0.5d as the diagnostic upper bound and 	au=1d as the gentler compromise. Both remain assimilation-like and are not production defaults.

