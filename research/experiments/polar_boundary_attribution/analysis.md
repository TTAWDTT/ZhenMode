# Polar / Boundary Attribution Results

## Bottom line

- Global A2 RMSE: `2.115 C`.
- Wet cells: `30917`.
- The closed northern/southern wall is **not** the dominant error source.
- Most squared error is in the **deep open ocean**, not only in polar rows.
- Coastal cells are high-error in RMSE and account for roughly one-third of SSE.

## Exclusive region decomposition

| region | count | mean error | RMSE | max abs | SSE share | zero-group RMSE |
|---|---:|---:|---:|---:|---:|---:|
| near_wall | 1416 | 0.150 | 2.330 | 10.581 | 5.56% | 2.055 |
| coast | 3508 | -2.234 | 3.532 | 9.828 | 31.63% | 1.749 |
| shallow_interior | 97 | -0.570 | 1.835 | 4.297 | 0.24% | 2.113 |
| deep_interior | 25896 | -0.614 | 1.828 | 5.930 | 62.57% | 1.294 |

The last column asks: if the errors in that region were exactly zero,
what would the remaining global RMSE be? It is an attribution metric,
not a proposed result.

## Interpretation

1. The earlier maximum-error map correctly found high-latitude/coastal
   outliers, especially near 59.5N. Those cells are real diagnostic
   signals, but there are too few of them to dominate the global SSE.
2. The coastal group has the largest RMSE after shallow water and
   contributes about one-third of global A2 squared error.
3. The deep open ocean contributes the largest share of SSE because it
   contains most ocean cells. Its per-cell RMSE is lower than the coast,
   but the integrated bias is still the largest lever.
4. The model is closer to the zonally uniform bulk target than WOA is:
   homogeneous T_atm suppresses the observed zonal SST structure.
   The target-vs-WOA term explains most of the cell-wise SST error.
5. Therefore the next physical experiment should test surface forcing and
   vertical/large-scale closure before changing the polar cap or masks.

## Model-survey connection

MOM6/NEMO-style diagnostics separate masked regional error from global
score. This is exactly why we avoid tuning to one maximum-error cell.
ROMS/FESOM2 remind us that coastal robustness matters, but the next
highest-leverage fix is the global surface heat-forcing closure.

## Bulk-target attribution

The regression `model - WOA = a * (T_atm - WOA) + b` gives:

- correlation about `0.83` over all wet cells;
- slope about `0.89`;
- `R^2` about `0.69`.

The model's global RMSE to the zonally uniform target is about `1.35 C`,
while WOA's RMSE to that target is about `1.86 C`. This supports the
attribution that a zonally uniform atmospheric state is too smooth for
the A2 test. A short sensitivity sweep of surface flux/restoring should
come before polar-boundary changes.
