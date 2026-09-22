# Bulk-Lambda Sweep Result

Date: 2026-09-22
Status: complete
Runs: 30d probes at lambda 120/160/240; 365d runs at 120/160/240

The 30d probes were all stable. At 30d the cold-region error keeps improving
through lambda 240, while the global A2 metric is best near lambda 120.

## 365d climate metrics

| lambda | Global A2 RMSE | NA 40--60N A2 RMSE | near-wall A2 RMSE |
|---:|---:|---:|---:|
| 80 | 1.282 C | 1.158 C | 1.148 C |
| 120 | 1.292 C | 1.057 C | 1.057 C |
| 160 | 1.304 C | 1.009 C | 1.010 C |
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

Keep `lambda_bulk=80` as the global-A2-favored diagnostic candidate. Record
`lambda_bulk=160` or `120` as useful regional-bias variants, but do not promote
either as a production default yet. Future work should separate physical
atmosphere-ocean exchange calibration from metric tuning.
''',encoding='utf-8')
