# Real 2-m Air Temperature Forcing Results

Status: PASS
Date: 2026-09-21

## Run

- Grid: 1 degree, 360 x 120 x 14
- Integration: 365 days
- Time stepping: split-explicit RK2 with `lax.scan`
- Transport: FCT/TVD
- GM: `kappa_gm = 1000 m^2/s`
- Wind: 12 monthly NCEP R1 snapshots for 2023
- Bulk target: annual-mean NCEP R1 2-m air temperature
- Flag: `--real-air-temp`
- Output: `results/real_air_temp/global_real_air_2m_1deg_365d.npz`

## Climate score

| run | A1 corr | A1 RMSE | A2 corr | A2 RMSE | verdict |
|---|---:|---:|---:|---:|---|
| zonal WOA SST baseline | 0.997 | 1.013 C | 0.974 | 2.115 C | FAIL |
| annual NCEP 2m air | 0.997 | 1.466 C | 0.988 | 1.883 C | PASS |

The A2 RMSE improvement is:

```text
(2.115 - 1.883) / 2.115 = 11.0%
```

This exceeds the pre-registered 5% threshold. A1 remains below the 2 C RMSE
limit, though its RMSE increases because the model no longer follows a single
meridional profile as closely.

## Stability and budgets

| metric | value |
|---|---:|
| verdict | PASS |
| max|u| peak | 1.693 m/s |
| final max|eta| | 1.382 m |
| heat-content drift | 0.651% |
| salt-content drift | 7.37e-6 |

## Regional A2 SSE shares

| region | zonal target | real 2m air |
|---|---:|---:|
| near wall | 5.56% | 7.45% |
| coast | 31.63% | 36.56% |
| shallow interior | 0.24% | 0.17% |
| deep interior | 62.57% | 55.82% |

Real atmospheric forcing reduces the deep-interior RMSE from `1.828 C` to
`1.538 C`. It does not solve the coastal error, but it addresses the largest
integrated error group.

## Conclusion

The improvement is above the locked threshold. The real-air result is a strong
candidate for the next baseline, but it should remain opt-in until the result
is confirmed and monthly-varying atmospheric temperature is tested. The
original zonal WOA target remains useful as a controlled non-circular baseline.
