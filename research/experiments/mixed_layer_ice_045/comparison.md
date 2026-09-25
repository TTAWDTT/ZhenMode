# Standardized benchmark comparison

Date: 2026-09-26  
Status: internal ladder only; not an industrial-model ranking

| run | verdict | days | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control 365d | PASS | 365 | 1.113 | 0.9119 | 1.071 | -0.5379 | -0.2833 | -0.0003596 |
| mixed-layer 30d | PASS | 30 | 0.7165 | 0.4282 | 0.2195 | -0.1443 | -0.04411 | -0.0001826 |

Do not interpret the 30d row as an annual replacement yet.  The 365d
mixed-layer/ice check is still running.

The 30d gate is recorded in `gate_30d.json` and passes.
