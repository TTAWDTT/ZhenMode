# Stage-F 365d internal stratification comparison

| run | verdict | days | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_f_365d_no_ice | PASS | 365 | 1.297 C | 0.799 C | 0.788 C | -0.650 C | -0.962% | -0.000621% |
| stage_f_365d_mld20 | PASS | 365 | 1.406 C | 1.302 C | 1.026 C | -0.459 C | -0.276% | -0.000216% |
| stage_f_365d_strat_mld | PASS | 365 | 1.340 C | 1.230 C | 1.085 C | -0.489 C | -0.285% | -0.000063% |
| stage_f_365d_strat_mld_lat20_60_noice | PASS | 365 | 1.281 C | 1.605 C | 1.291 C | -0.472 C | -0.732% | -0.000420% |
| stage_f_365d_strat_mld_lat20_60_ice | PASS | 365 | 1.289 C | 1.606 C | 1.261 C | -0.489 C | -0.686% | +0.002192% |

Interpretation: regional stratification MLD improves the 30d probe and near-wall
bias, but neither the global nor the northern-only annual run beats the annual
Stage-F no-ice control on North Atlantic RMSE. Do not promote.
