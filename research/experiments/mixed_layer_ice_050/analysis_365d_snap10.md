# Mixed-layer / Sea-Ice 0.5-degree Annual A/B

Date: 2026-09-26  
Status: failed the pre-registered internal gates

## Runs

| run | snapshot cadence | verdict | global A2 | NA raw RMSE | near-wall raw bias | heat drift |
|---|---:|---:|---:|---:|---:|---:|
| 0.5° no-ice control | 10d | PASS | 1.171 | 0.9238 | -0.9229 | -0.3911% |
| 0.5° mixed-layer/ice | 10d | PASS | 1.530 | 1.2702 | -0.1724 | -0.0798% |

The first single-snapshot annual run gave nearly the same result and is retained
only as a preliminary check.

## Gate

`gate_365d_snap10.json` is `FAIL`:

- global A2 is worse than the no-ice control;
- NA raw RMSE is worse;
- near-wall bias improves substantially;
- heat and salt drift are bounded.

## Decision

Do **not** promote `--mixed-layer-depth 50 --ice-salt-flux 1e-7`.

The improvement is regional, but it is bought by a large global error.  The
next diagnostic is a same-resolution 0.5-degree ice-floor control with **no**
mixed-layer depth and **no** ice-salt flux, now running as
`candidate_65n_050_icefloor_365d`.

After that control finishes, repeat the gate.  If the mixed-layer/ice run still
fails, reject the current parameter pair and run only one closed-loop variant at
a time rather than changing two closures together.
