# Stage-I minimal sea-ice / mixed-layer closed-loop status

Updated: 2026-09-27. This is the standardized, same-grid 0.5-degree slice
result. It is a thermodynamic proxy, not a full sea-ice model.

## Current manifest evidence

| duration | verdict | global A2 | NA RMSE | ice cells | ice extent fraction | mean MLD | heat-budget residual |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30d | PASS | 0.956 C | 0.874 C | 18 | 0.000105 | 31.5 m | -20.67 W/m2 |
| 365d | PASS | 1.362 C | 1.190 C | 77 | 0.000416 | 31.5 m | -5.92 W/m2 |

The 365d run also has 77 explicit ice cells, growth/melt latent heat, brine
salt flux, and a bounded stability verdict.

## What is already closed

- explicit ice thickness state;
- freezing/melting;
- ice-insulation heat flux;
- brine salt flux;
- mixed-layer heat capacity;
- density-threshold MLD diagnostic;
- final stability and climate scores.

## What is not yet closed

- no ice dynamics/advection;
- no full thermodynamic ice column;
- no coupled atmospheric humidity/radiation/precipitation/runoff;
- no matched MOM6 Stage-I run yet;
- the 365d Stage-F dynamic-ice/no-MLD ablation is stable but not promoted;
  it improves global A2 to 1.213 C but worsens North Atlantic and near-wall
  errors.

## Next gate

1. Keep the current 365d Stage-F no-ice control as the baseline.
2. Finish the fixed 100m 40--60N annual MLD probe now running.
3. Only after that gate, test the same regional closure with dynamic ice in a
   single pre-registered 365d run.
4. Do not promote any closure unless it improves global A2 without degrading
   North Atlantic/near-wall metrics or stability.

Future runs now save the solver-effective mixed-layer depth field; the scorer uses it when present instead of falling back to the initial-state MLD.

Latest gate: the fixed 100m 40--60N annual MLD probe passed stability but failed the annual climate gate. The 30d combined dynamic-ice/fixed-depth probe also did not improve the closure; therefore the Stage-I proxy remains a diagnostic, not a promoted production default.

A new standardized 3D MLD metric shows the annual Stage-F no-ice control reaches a mean MLD of 170.8m versus the 31.5m WOA reference, so the next mixed-layer target is excessive entrainment, not another fixed-depth regional band.

Band breakdown: high latitudes dominate the MLD over-deepening (40--60N +225m, 60--40S +176m), while subtropical bands are too shallow. This is a targeted sea-ice/convective closure signal, not another global scalar mixing issue.

A 30d Stage-F dynamic-ice diagnostic (165 ice cells) reduces the 3D global temperature RMSE to 0.707 C and the MLD bias to +22.6m versus the no-ice Stage-F control, so dynamic ice is a useful 3D upper-ocean diagnostic even when the surface metric is nearly unchanged. The 365d 3D rerun is running.

Annual dynamic-ice/no-MLD 3D rerun is now complete: global 3D RMSE is
`1.543 C` versus `1.547 C` for the no-ice control, so it does not improve the
annual 3D field. Global MLD worsens to `288.7 m`. Band diagnostics show
opposite effects: 40--60N improves to `153.8 m`, while 60--40S worsens to
`314.9 m`. The Stage-I proxy therefore stays diagnostic; the next closure must
be band/ice-state dependent rather than another global scalar.
A corrected 30d final-10d probe now combines dynamic ice in 40--65N with a
100m mixed-layer in 40--60N. It passes the 30d gate and improves global A2 to
`1.562 C`, NA RMSE to `0.848 C`, global 3D RMSE to `0.777 C`, and MLD bias to
`+16.3 m`. The matching 365d final-90d gate is running; promotion still waits
for that annual check.
The matching 365d final-90d check for the combined band-ice/fixed-depth probe
has completed and is **rejected**: global A2 rises to `4.759 C`, NA RMSE to
`8.434 C`, global 3D RMSE to `4.042 C`, and salt drift worsens to `-0.1497%`.
MLD bias improves to `+81.2 m`, but that is not enough because the North
Atlantic 3D temperature field is grossly too warm. This rules out another
fixed-depth band probe; the next closure must be seasonal/ice-state dependent.
