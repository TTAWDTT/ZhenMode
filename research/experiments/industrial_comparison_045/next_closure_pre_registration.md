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
