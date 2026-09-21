# GM/Bolus Transport Sensitivity on the 65N lambda80 Candidate

Date: 2026-09-21
Status: complete
Reference: `results/surface_heat_flux_65n/global_real_air_kv1e-6_kconv001_lat65_lambda80.npz`
Reference GM: `kappa_gm=1000`

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | NA 40--60N A2 bias | NA 40--60N A2 RMSE |
|---|---:|---:|---:|---:|---:|
| GM0 | PASS | 0.993 C | 1.419 C | -0.788 C | 1.338 C |
| GM500 | PASS | 1.029 C | 1.449 C | -0.883 C | 1.417 C |
| GM1000 | PASS | 1.130 C | 1.530 C | -1.052 C | 1.572 C |
| GM3000 | FAIL (A2) | 1.767 C | 2.089 C | -1.793 C | 2.346 C |

All runs were dynamically stable over 365d. GM3000 was the least stable in
tracer/climate terms and had the largest heat drift (`2.59%`).

## Interpretation

The response is monotonic over this range: stronger GM cools the North Atlantic
and worsens the A2 error. GM0 is slightly better than GM500, but a nonzero GM
closure is still scientifically preferable at 1 degree resolution. GM500 is
therefore the preferred non-zero diagnostic setting.

## Decision

1. Reduce the 65N diagnostic GM strength from 1000 to 500.
2. Do not use GM3000; it fails the A2 gate and increases heat drift.
3. GM strength alone reduces the cold bias but does not remove it; proceed to
   mixed-layer/convection closure diagnostics.
