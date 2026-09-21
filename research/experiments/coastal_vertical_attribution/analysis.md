# Coastal / Vertical Attribution Diagnostic

## Bottom line

- Raw ocean SST RMSE: `1.665 C`; mean bias: `-1.211 C`.
- Official A2 RMSE: `1.886 C`.
- Deep open ocean dominates raw SSE (`73.9%`),
  simply because it contains most ocean area.
- The coast has much higher per-cell RMSE (`2.052 C`),
  but only `17.2%` of raw SSE.

## Interpretation

1. The global model is generally too cold, but the strongest individual
   errors are regional warm anomalies in western-boundary-current-like
   regions. This is not a single uniform cold bias.
2. Closer to land is colder on average, so coastal physics matters,
   but the deep open ocean still dominates total squared error.
3. Stronger annual wind stress does **not** produce a colder coastal SST;
   it is associated with a warmer model. Wind alone is therefore not a
   clean coastal-upwelling attribution variable here.
4. Stronger WOA surface-to-50m stratification coincides with a colder
   model surface. This is the clearest signal pointing toward vertical
   mixing / mixed-layer treatment, but it is a correlation, not proof.
5. The official A2 smoother amplifies the coastal contribution relative
   to raw ocean error, so mask-aware smoothing should be checked before
   tuning coastal physics.

## Next diagnostic conclusion

The next model experiment should be a vertical-mixing sensitivity test,
not another transport or scalar lambda change. Keep annual real 2m air
forcing as the preferred baseline.
