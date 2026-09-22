# Coastal Mask and Ventilation Attribution

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

## Near-wall `55..60N` result

| Run | Group | cells | RMSE (C) | SSE share | cold >1C share | mean depth (m) | counterfactual RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| GM0 | land 0--3 cells | 122 | 1.539 | 0.371 | 0.836 | 1155 | 0.940 |
| GM0 | land 4--7 cells | 238 | 1.252 | 0.482 | 0.706 | 2506 | 0.853 |
| GM0 | land >=8 cells | 188 | 0.770 | 0.146 | 0.218 | 2334 | 1.096 |
| GM500 | land 0--3 cells | 122 | 1.537 | 0.335 | 0.852 | 1155 | 1.016 |
| GM500 | land 4--7 cells | 238 | 1.301 | 0.471 | 0.735 | 2506 | 0.907 |
| GM500 | land >=8 cells | 188 | 0.931 | 0.194 | 0.372 | 2334 | 1.119 |

## Key findings

1. The error is concentrated in the land-adjacent and transitional band within
   7 cells of land. In GM0 these two groups occupy about 65% of near-wall area
   but explain about 85% of the near-wall SSE.
2. If the `0--3`-cell band were perfect, the GM0 near-wall RMSE would drop from
   `1.186 C` to `0.940 C`. If the `4--7`-cell band were also perfect, it would
   drop to `0.853 C`.
3. This is not only a shallow-water problem. The land-adjacent group has mean
   depth `1155 m`, and the `>=1000 m` deep group still contributes `80%` of
   near-wall SSE in GM0.
4. The high-latitude interior cluster is secondary. The `>=8`-cell group is much
   better than the coastal/transitional groups.

## Interpretation

The dominant issue is the land-adjacent and transitional coastal band, not just
a small set of shallow cells. The next experiment should test whether this is a
mask/geometry issue or a ventilation/transport issue in the near-land band. A
narrow high-latitude ice-proxy experiment should wait until this is resolved.
