| run | verdict | days | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_050_icefloor | PASS | 365 | 1.128 | 0.9243 | 1.085 | -0.5342 | -0.2517 | -0.0002678 |
| stage_i_365d_dynamic_ice | PASS | 365 | 1.362 | 1.19 | 1.006 | -0.5006 | -0.2834 | -0.0001028 |
| stage_f_365d_no_ice | PASS | 365 | 1.297 | 0.7992 | 0.788 | -0.6496 | -0.962 | -0.0006209 |
| stage_f_365d_mld20 | PASS | 365 | 1.406 | 1.302 | 1.026 | -0.4591 | -0.2762 | -0.0002157 |
| stage_f_365d_constant_mld_lat40_60_noice | PASS | 365 | 1.343 | 1.451 | 1.209 | -0.611 | -0.854 | -0.000372 |
| stage_f_365d_dynamic_ice_no_mld | PASS | 365 | 1.213 | 0.983 | 1.095 | -0.642 | -0.913 | +0.0022461 |
| stage_f_365d_band_mld_ice_v2 | PASS* | 365 | 1.246 | 1.495 | 1.260 | -0.556 | -0.865 | -0.000360 |
| stage_f_365d_band_mld_ice_coolinggate_v2 | PASS* | 365 | 1.315 | 1.947 | 1.451 | -0.453 | -0.839 | -0.000558 |

Interpretation: the fixed 100m 40--60N MLD probe passes stability but reverses its 30d regional gain by 365d; the band-ice/fixed-MLD and cooling-gate annual candidates also fail the full climate/3D gate. Keep the Stage-F no-ice control as the current annual bulk control.

PASS* means stability passed, but the annual climate/3D gate rejected the candidate.
