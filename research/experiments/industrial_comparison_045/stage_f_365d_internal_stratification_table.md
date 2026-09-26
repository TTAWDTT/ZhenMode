# Stage-F 365d internal stratification comparison

| run | verdict | days | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_f_365d_no_ice | PASS | 365 | 1.297 C | 0.799 C | 0.788 C | -0.650 C | -0.962% | -0.000621% |
| stage_f_365d_mld20 | PASS | 365 | 1.406 C | 1.302 C | 1.026 C | -0.459 C | -0.276% | -0.000216% |
| stage_f_365d_strat_mld | PASS | 365 | 1.340 C | 1.230 C | 1.085 C | -0.489 C | -0.285% | -0.000063% |

Interpretation: stratification MLD improves the 30d probe and near-wall bias, but
does not beat the annual Stage-F no-ice control on global or North Atlantic
RMSE. Do not promote.
