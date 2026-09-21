# Monthly 2m Air Temperature Results

Status: PASS, but does not beat annual-mean target
Date: 2026-09-21

## Run

- Grid: 1 degree, 360 x 120 x 14
- Integration: 365 days
- Bulk target: twelve monthly-mean NCEP R1 2m air fields for 2023
- Month blending: same 5-day linear window as seasonal wind
- Flag: `--real-air-temp-monthly`
- Output: `results/monthly_air_temp/global_monthly_air_2m_1deg_365d.npz`

## Climate score

| run | A1 corr | A1 RMSE | A2 corr | A2 RMSE | verdict |
|---|---:|---:|---:|---:|---|
| zonal WOA SST baseline | 0.997 | 1.013 C | 0.974 | 2.115 C | FAIL |
| annual NCEP 2m air repeat | 0.997 | 1.469 C | 0.988 | 1.886 C | PASS |
| monthly NCEP 2m air | 0.998 | 1.435 C | 0.987 | 1.990 C | PASS |

Monthly forcing improves A1 slightly, from `1.469` to `1.435 C`, but degrades
A2 from `1.886` to `1.990 C` (about 5.5% worse). It therefore does not meet the
pre-registered 2% A2 improvement threshold.

## Stability and budgets

| metric | value |
|---|---:|
| verdict | PASS |
| max|u| peak | 1.696 m/s |
| final max|eta| | 1.372 m |
| heat-content drift | 0.559% |
| salt-content drift | 5.64e-6 |

It introduces strong seasonal SST variability (mean variance `0.981 C^2`), so
the dynamics are seasonally active, but the last-quarter mean pattern score is
slightly worse.

## Regional A2 SSE shares

| region | annual real air | monthly real air |
|---|---:|---:|
| near wall | 7.43% | 6.41% |
| coast | 36.52% | 35.62% |
| shallow interior | 0.17% | 0.18% |
| deep interior | 55.88% | 57.79% |

## Conclusion

Monthly atmospheric forcing is stable and physically more complete, but it is
not the next error-reduction lever in this configuration. The annual-mean
real-air target remains the best simple baseline. The next higher-leverage
target is coastal masking/mixing, not another scalar forcing tweak.
