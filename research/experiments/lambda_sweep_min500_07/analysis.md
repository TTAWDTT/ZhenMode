# Bulk-Lambda Sweep Result

Date: 2026-09-22
Status: complete
Runs: 30d probes at lambda 120/160/240; 365d runs at 120/160/240; 365d repeat at lambda 160

The 30d probes were all stable. At 30d the cold-region error keeps improving
through lambda 240, while the global A2 metric is best near lambda 120.

## 365d climate metrics

| lambda | Global A2 RMSE | NA 40--60N A2 RMSE | near-wall A2 RMSE |
|---:|---:|---:|---:|
| 80 | 1.282 C | 1.158 C | 1.148 C |
| 120 | 1.292 C | 1.057 C | 1.057 C |
| 160 | 1.304 C | 1.009 C | 1.010 C |
| 160 repeat | 1.304 C | 1.009 C | 1.010 C |
| 240 | 1.321 C | 0.966 C | 0.964 C |

## Interpretation

1. The earlier monotonic regional improvement continues beyond lambda 80.
2. But the global A2 metric does not continue improving; it worsens slowly as
   lambda increases.
3. Therefore "continue up" is partly true, but only if the goal is the regional
   cold bias. It is not a global-metric optimization.
4. Lambda 120 is a reasonable compromise: regional and near-wall errors improve
   substantially while global A2 worsens by only about 0.8%.
5. Lambda 240 gives the best regional skill but costs about 3.1% global A2 and
   is more strongly constrained to the atmospheric state.

## Decision

The lambda 160 repeat reproduces the regional and near-wall metrics exactly.
Promote `lambda_bulk=160 + min-depth 500` as the preferred regional-bias
diagnostic candidate, while keeping `lambda_bulk=80 + min-depth 500` as the
global-A2-favored candidate. Neither is a production default. Future work
should separate physical atmosphere-ocean exchange calibration from metric
tuning.
''',encoding='utf-8')
