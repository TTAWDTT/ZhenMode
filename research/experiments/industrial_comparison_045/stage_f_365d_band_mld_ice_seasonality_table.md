# Band-ice + fixed-MLD annual seasonality

Updated: 2026-09-27 15:55 local. Per-day values are instantaneous 3D snapshot
diagnostics on the shared 0.5-degree Stage-F slice; the official annual gate is
still the final-90d average.

| day | no-ice global 3D | candidate global 3D | no-ice NA 3D | candidate NA 3D | no-ice near-wall 3D | candidate near-wall 3D | candidate 40--60N MLD bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.861 | 0.777 | 1.175 | 0.580 | 1.125 | 0.311 | +17.0 m |
| 60 | 1.148 | 1.038 | 1.626 | 0.796 | 1.615 | 0.507 | +41.0 m |
| 90 | 1.224 | 1.132 | 1.587 | 0.901 | 1.325 | 0.509 | +37.6 m |
| 120 | 1.200 | 1.126 | 1.560 | 1.012 | 1.236 | 0.580 | +42.8 m |
| 150 | 1.150 | 1.113 | 1.273 | 1.087 | 1.019 | 0.639 | +40.1 m |
| 180 | 1.215 | 1.195 | 1.165 | 1.102 | 0.892 | 0.682 | +18.3 m |
| 210 | 1.426 | 1.379 | 1.539 | 1.160 | 1.114 | 0.766 | -5.2 m |
| 240 | 1.662 | 1.592 | 1.878 | 1.302 | 1.318 | 0.926 | -5.6 m |
| 270 | 1.738 | 1.687 | 1.802 | 1.430 | 1.118 | 1.048 | -4.9 m |
| 300 | 1.652 | 1.641 | 1.484 | 1.512 | 0.958 | 1.172 | -2.3 m |
| 330 | 1.554 | 1.568 | 1.248 | 1.581 | 1.001 | 1.294 | +8.5 m |
| 360 | 1.567 | 1.572 | 1.310 | 1.652 | 1.158 | 1.426 | +25.5 m |
| 365 | 1.597 | 1.597 | 1.364 | 1.664 | 1.251 | 1.449 | +29.5 m |

Interpretation:
- the candidate wins clearly through the first half of the year;
- the advantage fades near day 180--210;
- after day 300, North Atlantic and near-wall 3D errors become worse than the
  no-ice control;
- therefore this is a **seasonal closure failure**, not simply a wrong constant
  MLD choice.

Sources:
- `ocean_solver_stage_f_band_mld_ice_365d_v2_per_snapshot_3d.json`
- `ocean_solver_stage_f_band_mld_ice_365d_v2_3d_benchmark.json`
