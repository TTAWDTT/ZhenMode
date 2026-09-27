# Band-ice + fixed-MLD final-90d error difference

Updated: 2026-09-27 16:05 local. These are candidate-minus-control 3D
temperature differences for the final-90d window. Positive values mean the
candidate is warmer than the no-ice control.

| region/depth | candidate minus control bias | RMSE |
|---|---:|---:|
| global 3D | +0.055 C | 0.317 C |
| North Atlantic 40--60N | +0.692 C | 1.089 C |
| near-wall 55--60N | +0.719 C | 0.999 C |

The excess warming is concentrated in the North Atlantic upper ocean. In the
40--60N band, the candidate is warmer than the no-ice control by:

| depth | bias | RMSE |
|---|---:|---:|
| 0 m | +1.348 C | 1.765 C |
| 5 m | +1.374 C | 1.738 C |
| 15 m | +1.309 C | 1.653 C |
| 30 m | +1.231 C | 1.541 C |
| 50 m | +1.125 C | 1.382 C |
| 75 m | +0.917 C | 1.136 C |
| 100 m | +0.651 C | 0.856 C |
| 150 m | +0.408 C | 0.589 C |
| 200 m | +0.233 C | 0.367 C |
| 300 m | +0.116 C | 0.254 C |

This localizes the annual failure to upper-ocean North Atlantic warming, rather
than a broad global drift. The next closure should therefore control seasonal
upper-ocean heat capacity or ventilation, not add another global scalar.

Source:
- `ocean_solver_stage_f_band_mld_ice_365d_v2_vs_control_3d.json`
