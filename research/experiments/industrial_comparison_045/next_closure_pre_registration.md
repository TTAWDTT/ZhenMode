# Next closure pre-registration

Updated: 2026-09-27. This is a plan, not a promoted result.

## Blocking gate

Do not launch this run until the annual MOM6 Stage-F v12 final-90d 3D/MLD
comparison is complete and scored.

## Candidate

cooling_ice mixed-layer gate:

- dynamic ice only in 40--65N, as in the rejected diagnostic probe;
- fixed 100m mixed-layer only in 40--60N;
- apply the 100m depth only when either
  1. atmosphere is cooling the live SST, or
  2. dynamic ice is present.

The scientific question is whether the first-half improvement from the
band-ice/fixed-depth probe can be retained without the late-season North
Atlantic/near-wall deterioration.

## Run

- duration: 365d;
- control: annual Stage-F no-ice control;
- reference: WOA on the shared 0.5-degree slice;
- scoring: final-90d surface, 3D temperature, and MLD metrics.

## Promotion gates

The candidate is promotable only if all of the following improve or do not
degrade versus the control:

1. stability watchdog passes;
2. heat/salt drift bounded;
3. global A2 not worse;
4. NA 40--60N surface RMSE not worse;
5. near-wall signed bias not worse;
6. global 3D temperature RMSE not worse;
7. NA 40--60N 3D temperature RMSE not worse;
8. MLD bias improves without creating a new opposite-sign band error.

If the candidate fails, the next step is a stratification/ventilation-based
closure, not another global scalar tuning round.

## Planned command

The same pre-registered command is also saved as
scripts/run_stage_f_cooling_ice_365d.sh.

~~~bash
/root/jax-gpu/bin/python src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 --lat-max 65 --resolution 0.5 \
  --resolution-remap area --seasonal-wind --wind-year 2023 \
  --real-air-temp-monthly --no-meridional-heat-flux --lambda-bulk 80 \
  --kappa-v 1e-6 --kappa-conv 0.01 --kappa-gm 0 --localize-conv --fct-adv \
  --project-adv-vel --min-depth 500 --smooth-passes 80 --nu-h 2e6 --dt 1800 \
  --dynamic-ice --dynamic-ice-lat-band 40 65 --mixed-layer-depth 100 \
  --mixed-layer-lat-band 40 60 --mixed-layer-mode constant \
  --mixed-layer-gate-mode cooling_ice --days 365 --snap-days 30 --save-3d \
  --tag global_stage_f_dyn_ice_band40_65_mld100_lat40_60_cooling_ice_gate_365d \
  --out-dir /root/external_models/results/industrial_comparison_045 \
  --log-dir /root/external_models/logs/industrial_comparison_045
~~~

Do not run until the blocking MOM6 gate is scored.
