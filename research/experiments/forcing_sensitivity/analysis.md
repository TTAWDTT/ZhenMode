# Surface Forcing Sensitivity Results

## Runs

| run | snaps | A1 corr | A1 RMSE | A2 corr | A2 RMSE | mean error | max-u | max-eta | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lambda1_baseline | 10 | 0.997 | 1.013 | 0.974 | 2.115 | -0.763 | 1.6934 | 1.3568 | 0.007058 | 0.00000683 |
| lambda0.25 | 10 | 0.977 | 3.104 | 0.960 | 3.387 | -2.342 | 1.6970 | 1.3417 | 0.006176 | 0.00001230 |
| lambda0.5 | 10 | 0.991 | 1.835 | 0.971 | 2.482 | -1.353 | 1.6953 | 1.3516 | 0.007131 | 0.00000920 |
| lambda2.0 | 10 | 0.999 | 0.504 | 0.974 | 2.031 | -0.450 | 1.6927 | 1.3600 | 0.006717 | 0.00000657 |

## Regional SSE shares

| run | near_wall | coast | shallow_interior | deep_interior |
|---|---:|---:|---:|---:|
| lambda1_baseline | 5.56% | 31.63% | 0.24% | 62.57% |
| lambda0.25 | 2.27% | 21.09% | 0.21% | 76.43% |
| lambda0.5 | 4.10% | 27.11% | 0.23% | 68.56% |
| lambda2.0 | 6.01% | 32.28% | 0.30% | 61.42% |

## Reading

- `lambda1` is the existing baseline (`bulk_lambda_mult=1`).
- Lower values weaken restoring; higher values strengthen restoring.
- The key criterion is global A2 RMSE, not only coastal RMSE.
- A full 2D WOA SST restoring target is deliberately not used because
  it would make the A2 score circular.

## Conclusion

The best scalar-restoring case is `lambda2.0`, but its A2 RMSE improvement is
only 4.0%, below the pre-registered 5% threshold. Therefore no default value
was changed. The next step was to test spatially varying atmospheric forcing.
