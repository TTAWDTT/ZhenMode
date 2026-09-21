# Real 2-m Air Temperature Repeat

Status: PASS
Date: 2026-09-21
Purpose: reproduce the first real-air experiment with identical physics.

## Comparison

| metric | first run | repeat | difference |
|---|---:|---:|---:|
| A1 corr | 0.997 | 0.997 | 0.000 |
| A1 RMSE | 1.466 C | 1.469 C | +0.003 C |
| A2 corr | 0.988 | 0.988 | 0.000 |
| A2 RMSE | 1.883 C | 1.886 C | +0.003 C |
| heat drift | 0.651% | 0.656% | +0.005% |
| salt drift | 7.37e-6 | 8.13e-6 | +7.6e-7 |
| max|u| peak | 1.693 m/s | 1.693 m/s | stable |
| final max|eta| | 1.382 m | 1.382 m | stable |
| verdict | PASS | PASS | PASS |

The very small metric differences are consistent with GPU/JAX floating-point
non-reproducibility; the climate conclusion is unchanged.

## Conclusion

The real 2m-air result is reproducible. A2 remains below the 2 C criterion and
improves over the zonal WOA-SST baseline by about 10.8%. The next experiment
should use monthly-varying atmospheric temperature rather than changing scalar
lambda further.
