# Candidate Vertical-Mixing Regional Audit Protocol

Status: locked
Date: 2026-09-21
Baseline: reproduced annual real-air 365-day run
Candidate: combined reduced vertical mixing repeat

## Question

Where does the A2 improvement from reduced vertical mixing come from?

The candidate run reduces both `kappa_v` and `kappa_conv`. It improves the
global A2 RMSE from `1.886 C` to `1.844 C`. This audit checks whether the
improvement is:

- concentrated near the coast,
- concentrated in strong WOA stratification,
- broadly distributed,
- or just a uniform warming signal.

## Runs

- Baseline:
  `results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz`
- Candidate:
  `results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz`

No new model runs are performed.

## Metrics

Use the same last-90-day SST climatology convention. For the same wet cells
compare:

- raw mean bias;
- raw RMSE;
- global summed squared error;
- regional SSE change and RMSE change;
- WOA surface-to-50 m stratification quintiles;
- warming fraction;
- distribution of the 100 largest remaining candidate errors.

## Decision rule

If the candidate's improvement is concentrated in strong stratification and/or
coastal regions, vertical mixing remains a plausible physical lever. If it is
just a broad warming shift without regional structure, do not implement a
mixed-layer closure solely from this result.
