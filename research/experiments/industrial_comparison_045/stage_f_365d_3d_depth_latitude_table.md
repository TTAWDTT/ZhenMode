# Stage-F annual 3D depth-latitude error

Mean of the final three 30-day annual 3D snapshots, shared WOA reference and
bathymetric mask. Values are temperature bias in `C`.

| depth | 60--40S | 40--20S | 20--0 | 0--20N | 20--40N | 40--60N |
|---:|---:|---:|---:|---:|---:|---:|
| 0m | -0.043 | -0.254 | -0.513 | -0.567 | -1.508 | -1.498 |
| 5m | -0.719 | -2.012 | -1.369 | -0.999 | -1.689 | -1.398 |
| 15m | -0.933 | -2.675 | -1.767 | -1.379 | -2.208 | -1.254 |
| 30m | -0.961 | -2.701 | -1.787 | -1.783 | -2.482 | -0.803 |
| 50m | -0.823 | -2.218 | -1.472 | -1.802 | -2.016 | +0.016 |
| 75m | -0.491 | -1.412 | -0.957 | -1.191 | -1.331 | +0.547 |
| 100m | -0.152 | -0.955 | -0.820 | -0.765 | -1.127 | +0.668 |
| 150m | +0.162 | -0.555 | -0.021 | +0.656 | -0.832 | +0.598 |
| 200m | +0.102 | -0.827 | -0.264 | +0.582 | -1.104 | +0.349 |
| 300m | -0.014 | -0.948 | -0.028 | +0.399 | -1.248 | +0.184 |
| 500m | -0.154 | -0.740 | +0.272 | +0.234 | -1.066 | +0.068 |
| 1000m | +0.006 | -0.063 | +0.254 | +0.165 | -0.114 | +0.186 |
| 2000m | -0.009 | +0.013 | +0.098 | +0.089 | +0.021 | +0.137 |
| 4000m | +0.068 | +0.030 | +0.019 | +0.014 | +0.021 | +0.037 |

Interpretation: the annual 3D loss is dominated by a broad upper-ocean cold
bias in the 40--20S and 20--40N bands, especially between 15 and 50 m. The
40--60N band is much less biased and even becomes slightly warm below 50m.
This argues against a wall-only fix and favors a seasonal upper-ocean heat
storage/ventilation diagnosis.

Source:
- `/root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz`
- `/root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d_3d/`
