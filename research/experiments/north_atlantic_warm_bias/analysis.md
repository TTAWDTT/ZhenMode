# North Atlantic Warm-Bias Audit

## Bottom line

- Region: `300..360E / 40..60N`.
- Wet cells: `1052`.
- Warm-cell share: `8.2%`.
- Mean bias: `-2.255 C`.
- Regional RMSE: `2.872 C`.
- Mean annual NCEP 2m air minus WOA SST: `-0.372 C`.

## Interpretation

- The coastal subset has `26.0%` warm cells and a mean bias of `-1.368 C`.
- The deep open-ocean subset has `1.1%` warm cells and a mean bias of `-2.200 C`.
- The warm bias is therefore not solely a coastal mask artifact; it also
  occurs in deep open-ocean cells.
- `T_atm - WOA SST` has the expected sign in many cells, but the SST
  error is not explained by forcing alone. The next diagnostic should
  inspect boundary/ice and bulk heat-flux completeness.
