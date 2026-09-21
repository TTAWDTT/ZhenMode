# 3D Heat-Tendency Decomposition

Date: 2026-09-21
Status: complete
Run: `results/heat_tendency_decomposition/global_real_air_lambda80_gm500_localconv_3dterms.npz`
Verdict: PASS
Global A2 RMSE: `1.412 C`
Snapshots used: days `300, 330, 360, 365`

## Main result

The remaining North Atlantic cold bias is not caused by diffusion or by
insufficient surface heat input. It is dominated by the redistribution of heat
by ocean circulation and convection.

### Surface tendency means

| Term | NA 40--60N mean | NA mean share | Near-wall mean | Near-wall share |
|---|---:|---:|---:|---:|
| Convection | +0.231 K/d | 0.465 | +0.432 K/d | 0.560 |
| Surface bulk flux | +0.179 K/d | 0.360 | +0.202 K/d | 0.262 |
| Advection | -0.069 K/d | 0.139 | -0.112 K/d | 0.145 |
| GM bolus | +0.017 K/d | 0.034 | +0.023 K/d | 0.030 |
| Horizontal diffusion | ~0 K/d | 0.000 | ~0 K/d | 0.000 |
| Vertical diffusion | -0.001 K/d | 0.003 | -0.002 K/d | 0.003 |

The sign convention is warming positive.

### Vertical redistribution

At the surface node in the North Atlantic region:

- advection cools the surface by about `-0.071 K/d`
- convection warms the surface by about `+0.252 K/d`
- surface bulk flux warms it by about `+0.179 K/d`
- GM is weakly warming at the surface

At 5m depth, the signs reverse: convection is about `-0.053 K/d`, while
advection is weakly warming. Thus both advection and convection are moving
heat downward rather than simply destroying it.

### Local-error correlations

Correlation of time-mean local tendency with SST error:

| Term | NA 40--60N | Near wall 55--60N |
|---|---:|---:|
| Advection | +0.433 | +0.511 |
| Horizontal diffusion | +0.484 | +0.632 |
| Vertical diffusion | +0.462 | +0.706 |
| Convection | +0.251 | +0.350 |
| GM | +0.232 | +0.406 |
| Surface bulk flux | -0.539 | -0.728 |

The positive advection correlation means colder cells also experience stronger
advective cooling. The negative surface-bulk-flux correlation is expected
negative feedback: colder cells receive more warming.

## Decision

1. The remaining cold bias is primarily a heat-transport problem, not a
   surface-flux problem.
2. Diffusion is negligible in the regional surface budget.
3. Convection is a large term but its surface sign is warming; it is
   redistributing heat downward, not directly creating the cold bias.
4. The next target should be horizontal heat transport and boundary-current
   structure, not further scalar tuning of bulk flux, GM strength, or
   kappa_conv.
