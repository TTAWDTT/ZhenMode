# North Boundary / Polar-Cap Sensitivity Analysis

Date: 2026-09-21
Status: complete
Reference run: `results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz`

## What was tested

All runs used the reduced-mixing candidate physics and annual-mean NCEP R1 2m
air forcing. Only the northern-boundary geometry changed:

1. `candidate_baseline`: 60N wall, 2-row polar cap, 3-row taper.
2. `lat65`: extend the northern wall to 65N (`ny=130`).
3. `lat65_repeat`: reproducibility run for `lat65`.
4. `polarcap4_6`: 60N wall with a 4-row polar cap and 6-row taper.
5. `nocap90d`: 60N wall with no polar cap, 90d stability probe only.

The reference SST is the WOA surface field stored in each run's `T_init`.
Regional metrics use the common `300..360E / 40..60N` overlap. The A2 metric
uses the pre-registered 2-degree smoother over the final 90 days.

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | North Atlantic A2 bias | North Atlantic A2 RMSE |
|---|---:|---:|---:|---:|---:|
| candidate baseline | PASS | 1.420 C | 1.844 C | -2.044 C | 2.811 C |
| lat65 | PASS | 1.457 C | 1.770 C | -1.703 C | 2.177 C |
| lat65 repeat | PASS | 1.463 C | 1.773 C | -1.700 C | 2.175 C |
| polarcap4_6 | PASS | 1.435 C | 1.964 C | -2.572 C | 3.592 C |

For `55..60N / 300..360E`, the closed-wall cold bias falls from `-2.70 C` in
the candidate baseline to `-1.67 C` with the 65N domain. The regional
`40..60N` A2 RMSE improves by about `22.6%` and is reproducible. Global A2
improves by `3.85..4.01%`, just below the original `5%` global gate, but the
regional improvement is decisive relative to the targeted error sector.

The widened polar cap makes the target region worse: global A2 increases by
`6.50%` and the North Atlantic regional A2 RMSE by `27.8%`. Thus widening the
current zonal-averaging cap is not the right lever.

The 90d no-cap probe was stable: no NaN, `max|u|=0.927 m/s`, and
`max|eta|=1.997 m`. This supports future no-cap testing, but the 365d evidence
still favors the 65N extension for the current diagnostic.

Depth-binned regional errors were not computed because the run driver saved
surface snapshots only (`n_3d_snaps=0`) for these experiments.

## Decision

1. The 60N closed wall is a major contributor to the North Atlantic cold
   bias. The effect is reproduced.
2. Treat `lat65 + reduced vertical mixing + annual real air` as the new
   diagnostic candidate, not yet as a production default.
3. Do not use the widened polar cap as an error remedy.
4. The remaining high-latitude cold bias still points to missing high-latitude
   surface physics, with a sea-ice proxy or physically constrained winter heat
   flux being the next candidate.

## Next experiment

Freeze the 65N reduced-mixing candidate and diagnose high-latitude surface
heat flux. The first test should separate the effect of cold annual-mean air
from the missing winter/sea-ice constraint before changing any default.
