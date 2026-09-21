# Surface Heat-Flux Sensitivity on the 65N Diagnostic Candidate

Status: locked
Date: 2026-09-21
Baseline: `lat65 + reduced vertical mixing + annual real air`

## Question

Is the remaining North Atlantic cold bias controlled by the surface heat-flux
relaxation timescale, or by ocean heat transport / missing high-latitude
physics?

## Motivation

The 65N repeat has a `-1.70 C` North Atlantic bias. The strongest cold-error
cells also have the largest positive bulk heat input (`lambda * (T_atm-SST)`),
so the model is still equilibrating too cold despite strong surface warming.
A scalar flux sensitivity will separate surface damping from interior heat
transport before implementing a more complex sea-ice proxy.

## Runs

Keep unchanged:

- 65N boundary, `ny=130`
- annual-mean NCEP R1 2m air forcing
- reduced vertical mixing: `kappa_v=1e-6`, `kappa_conv=0.01`
- FCT/TVD transport
- GM `kappa_gm=1000`
- seasonal 2023 wind
- 365d integration

Change only the bulk heat-transfer coefficient:

1. `lat65_lambda20`: `--lambda-bulk 20`
2. `lat65_lambda80`: `--lambda-bulk 80`

The reference is `real_air_kv1e-6_kconv001_lat65_repeat` with
`--lambda-bulk 40`.

## Metrics

- Global A1/A2 correlations and RMSEs
- North Atlantic `300..360E / 40..60N` bias and RMSE
- near-wall `55..60N` bias and RMSE
- final bulk heat-flux quintiles versus SST error
- global heat drift and stability

## Decision rules

1. If `lambda80` materially warms and improves the target region, the surface
   heat-exchange timescale is a leading control.
2. If `lambda20` or `lambda80` worsens the region, the bulk coefficient is not
   the main missing process.
3. If target-region RMSE is insensitive to lambda, prioritize ocean heat
   transport / high-latitude closure rather than sea-ice flux tuning.
