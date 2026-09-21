# Vertical Mixing Sensitivity Results

## Climate scores

| run | A1 corr | A1 RMSE | A2 corr | A2 RMSE | mean error |
|---|---:|---:|---:|---:|---:|
| annual real-air baseline | 0.997 | 1.469 C | 0.988 | 1.886 C | -1.314 C |
| kappa_v 1e-6 | 0.997 | 1.430 C | 0.988 | 1.851 C | -1.270 C |
| kappa_conv 0.01 | 0.997 | 1.447 C | 0.988 | 1.870 C | -1.297 C |
| both reduced | 0.997 | 1.427 C | 0.988 | 1.848 C | -1.263 C |
| both reduced repeat | 0.997 | 1.420 C | 0.988 | 1.844 C | -1.260 C |

## Bias by WOA surface-to-50m stratification quintile

| run | q1 | q2 | q3 | q4 | q5 |
|---|---:|---:|---:|---:|---:|
| baseline | -0.552 | -1.131 | -1.322 | -1.465 | -1.583 |
| kv1e-6_kconv005 | -0.525 | -1.094 | -1.300 | -1.410 | -1.493 |
| kv1e-5_kconv001 | -0.543 | -1.105 | -1.307 | -1.446 | -1.565 |
| kv1e-6_kconv001 | -0.519 | -1.082 | -1.283 | -1.403 | -1.500 |
| kv1e-6_kconv001_repeat | -0.518 | -1.081 | -1.288 | -1.402 | -1.484 |

## Reading

- The strongest-stratification cold bias improves from `-1.583 C`
  to `-1.500 C` in the combined
  reduced-mixing case, a change of `0.083 C`.
- The 365d repeat gives `-1.484 C`.
- The combined reduced-mixing case also has the best global A2 RMSE and
  remains reproducible.
- All reduced-mixing runs remain stable and have small heat/salt drift.
- A2 improvement is above the locked 2% gate, but the strongest-layer
  bias change is below the 0.2 C gate, so this is suggestive rather
  than decisive evidence of over-mixing.

## Next

Treat the combined reduced-mixing case as the preferred candidate
baseline. It has been reproduced. The next step is a regional error
audit, especially in strong-stratification and coastal regions, before
deciding whether to implement a mixed-layer closure or investigate the
bulk heat-flux formula further.
