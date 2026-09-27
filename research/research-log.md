# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-09-21 | bootstrap | Started broad survey of widely used high-fidelity ocean models. Created research workspace. |
| 2 | 2026-09-21 | literature | Fetched structured docs/source material for MOM6, MITgcm, NEMO, ROMS, POP2, MPAS-Ocean, FESOM2, ICON-Ocean, HYCOM, and FVCOM. |
| 3 | 2026-09-21 | synthesis | Wrote `research/literature/comparative_notes.md` and expanded `research/findings.md` into a cross-model survey and transferable-lessons summary. |
| 4 | 2026-09-21 | next-step | Highest-leverage next experiment: prototype a full 3D flux-corrected-transport tracer option with a monotone low-order fallback and a higher-order base scheme. |
| 2 | 2026-09-21 | literature | Fetched structured docs/source material for MOM6, MITgcm, NEMO, ROMS, POP2, MPAS-Ocean, FESOM2, ICON-Ocean, HYCOM, and FVCOM. |
| 3 | 2026-09-21 | synthesis | Wrote `research/literature/comparative_notes.md` and expanded `research/findings.md` into a cross-model survey and transferable-lessons summary. |
| 4 | 2026-09-21 | next-step | Highest-leverage next experiment: prototype a full 3D flux-corrected-transport tracer option with a monotone low-order fallback and a higher-order base scheme. |
| 5 | 2026-09-21 | experiment-plan | Committed the FCT-transport protocol under `research/experiments/fct_transport/` before implementation. |
| 6 | 2026-09-21 | implementation | Added `--fct-adv`: a compact TVD/MUSCL flux limiter for horizontal tracer transport, with operator-level conservation and boundedness tests. |
| 7 | 2026-09-21 | test | `tests/test_fct_advection.py`: 4 tests pass; full suite 147 passed. |
| 8 | 2026-09-21 | experiment | Ran 30d/365d centered, monotone, and FCT/TVD transport comparisons. All passed; FCT stays non-default. |
| 9 | 2026-09-21 | analysis | Climatology A1/A2 scoring shows A2 RMSE is nearly identical across transport schemes, so transport is not the dominant climate-error lever at 1°. |
| 10 | 2026-09-21 | experiment-plan | Locked the next experiment: heat/salt/mass budget diagnostics, because transport is not the dominant 1° climate-error lever. |
| 11 | 2026-09-21 | implementation | Added `src/diagnostics.py`, snapshot-level heat/salt/volume diagnostics in the run driver, and two invariant tests. |
| 12 | 2026-09-21 | experiment | 365d centered/FCT budget runs show volume exactly conserved; heat drift ~0.7%, salt drift ~7e-6; schemes are nearly identical. |
| 13 | 2026-09-21 | analysis | A2 SST error map points to high-latitude/coastal rows, so the next diagnostic target is polar/boundary forcing and coast/topography structure. |
| 14 | 2026-09-21 | experiment-plan | Locked polar/boundary attribution on existing 365d climatologies before new long runs. |
| 15 | 2026-09-21 | analysis | Attribution found 5.6% of A2 SSE near walls, 31.6% at coasts, and 62.6% in deep interior; target-vs-WOA regression has R^2=0.695. |
| 16 | 2026-09-21 | experiment | Ran 365d bulk-restoring sensitivity lambda=0.25, 0.5, and 2.0. Only 2.0 improved A2 (2.115 to 2.031), below the 5% gate. |
| 17 | 2026-09-21 | implementation | Added opt-in annual-mean NCEP R1 2m air-temperature bulk target (`--real-air-temp`) with cache and tests; full suite 151 passed. |
| 18 | 2026-09-21 | experiment | The real 2m-air run passed 365d stability, A1 (RMSE 1.466), and A2 (RMSE 1.883). A2 improved 11.0% over the zonal-WOA baseline. |
| 19 | 2026-09-21 | reproduction | Repeated the annual NCEP 2m-air 365d run: A1 RMSE 1.469, A2 RMSE 1.886, PASS. Confirms the A2 improvement within GPU floating-point variability. |
| 20 | 2026-09-21 | implementation | Added opt-in monthly NCEP R1 2m air forcing with the same 5d seasonal blending as wind; runtime bulk target reuses one compiled dynamic-forcing graph. |
| 20 | 2026-09-21 | test | Added monthly-loader and dynamic-air-target tests; full suite 155 passed. |
| 21 | 2026-09-21 | experiment | The monthly 2m-air 365d run PASSed, but A2 RMSE was 1.990 vs 1.886 for annual forcing; kept annual mean as the better real-air baseline. |
| 22 | 2026-09-21 | analysis | Completed coastal/vertical attribution on annual real air. Deep ocean has 73.9% of raw SSE, coast 17.2%; coast RMSE 2.05 C. Stronger WOA surface stratification correlates with colder model SST, pointing to vertical mixing. |
| 23 | 2026-09-21 | experiment | Completed vertical-mixing sensitivity. Combined kappa_v 1e-6 and kappa_conv 0.01 improved A2 RMSE to 1.848 C and remained PASS. |
| 23 | 2026-09-21 | reproduction | Repeated the combined reduced-mixing run: A2 RMSE 1.844 C, confirming the improvement. Strongest-layer bias improvement remains below the 0.2 C gate. |
| 24 | 2026-09-21 | analysis | Regional audit of the reduced-mixing candidate: global RMSE 1.625 C, SSE improvement 4.74%, coast improves 6.46% and strongest-stratification quintile 5.98%. Remaining worst errors are warm biases in 300..360E / 40..60N. |
| 25 | 2026-09-21 | analysis | Audited the high-latitude North Atlantic warm-error sector: 83/100 largest global errors lie in 300..360E / 40..60N, with strong warm bias near the north wall and deep open ocean. |
| 26 | 2026-09-21 | experiment-plan | Locked a north-boundary/ice-proxy protocol: 60N baseline, 65N extension, widened polar cap, and a 90d no-cap stability probe. |
| 27 | 2026-09-21 | experiment | All four runs completed PASS. The two 365d 65N runs were stable; the wider polar cap was stable but less accurate; the 90d no-cap probe was stable. |
| 28 | 2026-09-21 | analysis | The 65N extension reduced North Atlantic 40..60N A2 RMSE from 2.811 to 2.175/2.175 C and cold bias from -2.044 to -1.703/-1.700 C. Global A2 improved from 1.844 to 1.770/1.773 C. |
| 29 | 2026-09-21 | reflection | The closed 60N wall is a major source of the targeted cold bias. Widen-cap is rejected. Freeze the 65N reduced-mixing run as a diagnostic candidate and next diagnose high-latitude surface heat flux or a sea-ice proxy. |
| 30 | 2026-09-21 | experiment-plan | Locked a 65N surface heat-flux sensitivity protocol at fixed boundary/forcing/physics, varying only bulk lambda. |
| 31 | 2026-09-21 | experiment | Ran 365d lambda20 and lambda80. Both were dynamically stable; lambda20 failed the A-class climate test and lambda80 passed. |
| 32 | 2026-09-21 | analysis | Lambda80 improved global A2 RMSE from 1.773 to 1.530 C and North Atlantic 40..60N A2 RMSE from 2.175 to 1.572 C. |
| 33 | 2026-09-21 | reflection | Surface heat exchange is a leading control, but residual cold errors remain concentrated in the strongest imposed-warming quintile. Keep lambda80 as a diagnostic candidate; next diagnose GM/bolus heat transport and mixed-layer closure rather than sea-ice flux cap. |
| 34 | 2026-09-21 | experiment | Ran GM0, GM500, and GM3000 on the 65N lambda80 candidate. All were dynamically stable; GM3000 failed the A2 climate gate. |
| 35 | 2026-09-21 | analysis | Stronger GM monotonically cooled the North Atlantic: A2 RMSE changed from 1.419 C at GM0 to 1.530 C at GM1000 and 2.089 C at GM3000. Selected GM500 as the preferred non-zero diagnostic setting. |
| 36 | 2026-09-21 | experiment | Ran localized convective adjustment plus kappa_conv 0.05 and 0.002 against the GM500 baseline. All three were stable. |
| 37 | 2026-09-21 | reproduction | Repeated the localized-conv run. Global A2 RMSE remained 1.410 C and North Atlantic A2 RMSE remained 1.306 C. |
| 38 | 2026-09-21 | reflection | Localized convection is a structural improvement: it improves global A2 by 2.71% and regional A2 by 7.84% over the GM500 baseline, while scalar kappa_conv changes are small. Freeze the combined candidate and next run a 3D heat-tendency decomposition. |
| 39 | 2026-09-21 | experiment | Ran the preferred 65N candidate with 30-day 3D state and heat-tendency snapshots for 365d. The run passed stability and produced a global A2 RMSE of 1.412 C. |
| 40 | 2026-09-21 | analysis | In the North Atlantic surface layer, convection and surface bulk flux warm while advection cools. Diffusion is negligible. |
| 41 | 2026-09-21 | reflection | Colder cells have stronger advective cooling (correlation about 0.43 regionally and 0.51 near the wall). The remaining cold bias is therefore primarily a heat-transport problem. Next target horizontal heat transport and boundary-current structure, not scalar closure tuning. |
| 42 | 2026-09-22 | experiment | Ran nu_h 2.5e6 and 1e6 sensitivity runs on the preferred 65N candidate. Both were stable but worsened the North Atlantic cold bias. |
| 43 | 2026-09-22 | analysis | Lower horizontal viscosity increased KE and max|u| but did not improve the target region. Viscosity is not the limiting factor. |
| 44 | 2026-09-22 | experiment | Ran 30d and 90d WOA SSS-restoring sensitivity runs. Both were stable and only slightly improved the global and regional metrics. |
| 45 | 2026-09-22 | reflection | SSS restoring is a minor refinement, not a structural fix. The remaining cold bias still points to horizontal heat transport / boundary-current geometry or resolution. |
| 46 | 2026-09-22 | experiment | Ran a 0.8-degree 365d candidate run and a 0.5-degree 30d stability probe. |
| 47 | 2026-09-22 | analysis | The 0.8-degree run improved global A2 RMSE from 1.412 to 1.360 C and the North Atlantic regional RMSE from 1.313 to 1.252 C. |
| 48 | 2026-09-22 | reflection | Resolution is a real lever for the remaining cold bias. Treat 0.8 degree as the new preferred diagnostic resolution and stabilize 0.5 degree before using it. |
| 49 | 2026-09-22 | experiment | Ran a 0.7-degree 365d candidate run after a stable 30d probe. |
| 50 | 2026-09-22 | analysis | The 0.7-degree run improved global A2 RMSE to 1.337 C and North Atlantic RMSE to 1.206 C. |
| 51 | 2026-09-22 | reflection | 0.7 degree is the new preferred diagnostic resolution. 0.5 and 0.6 degree remain unstable and need targeted stabilization. |
| 52 | 2026-09-22 | experiment | Tested a 0.65-degree 30d probe with area remapping. It blew up by day 10. |
| 53 | 2026-09-22 | experiment | Ran a 0.7-degree GM0 365d run. It improved global A2 RMSE to 1.336 C and North Atlantic RMSE to 1.043 C. |
| 54 | 2026-09-22 | reflection | At 0.7 degree, GM500 is no longer beneficial for the North Atlantic sector. Treat 0.7 degree + GM0 + localized convection as the new best diagnostic candidate. |
| 55 | 2026-09-22 | baseline | Locked the best diagnostic candidate as `candidate_65n_07_gm0` with a single reproducible runner. |
| 56 | 2026-09-22 | reproduction | Re-ran the locked candidate baseline for 365d. It passed and reproduced A1/A2 RMSE 0.973/1.336 C and North Atlantic A2 RMSE 1.043 C. |
| 57 | 2026-09-22 | attribution | Completed GM0 vs GM500 3D attribution at 0.7 degree. GM0 reduced regional A2 RMSE from 1.216 to 1.141 C, mainly by weakening near-wall advective cooling. |
| 58 | 2026-09-22 | analysis | Diagnosed 0.7-degree surface velocity and heat-transport structure. GM0 and GM500 have nearly identical mean/rms speeds and v*T, so the regional benefit is not from gross current speed. |
| 59 | 2026-09-22 | stability | A 30d 0.6-degree GM0 probe still produced NaNs by day 10. 0.7 degree remains the finest stable resolution. |
| 60 | 2026-09-22 | experiment | Ran the near-wall gradient diagnostic on the matched 0.7-degree GM0 and GM500 final-90d states. |
| 61 | 2026-09-22 | analysis | GM500 has stronger near-wall full-solver advective cooling (-0.075 versus -0.038 K/day), while mean current speed and SST gradients are nearly unchanged. |
| 62 | 2026-09-22 | reflection | The worst cold cells cluster in local wall sectors and often have weak mean currents. Next diagnose vertical heat redistribution and local geometry in these cells before changing transport or resolution. |
| 63 | 2026-09-22 | analysis | Decomposed heat tendencies for the 100 coldest near-wall cells. GM500 has -0.187 K/day local surface advection versus -0.058 K/day for GM0. |
| 64 | 2026-09-22 | reflection | The GM0 coldest cells still have positive net surface warming; the near-wall cold population is mixed between land-adjacent and weakly ventilated high-latitude cells. Next separate coastal geometry from high-latitude ventilation. |
| 65 | 2026-09-22 | analysis | Separated coastal and interior near-wall cells by land distance. Coastal cells remain the larger error source. |
| 66 | 2026-09-22 | reflection | The 59--60N interior cluster persists after excluding land-adjacent cells, but it is secondary. Next diagnose coastal masking/ventilation, then only consider a narrow high-latitude proxy experiment. |
| 67 | 2026-09-22 | analysis | Attributed near-wall error by land-distance and depth. The 0--3 and 4--7 cell bands explain about 85% of near-wall SSE. |
| 68 | 2026-09-22 | reflection | The error is a coastal/transitional-band issue, not only shallow water. Next design a narrowly scoped coastal mask/ventilation sensitivity test. |
| 69 | 2026-09-22 | experiment | Ran 30d and 365d min-depth 500 m probes on the locked candidate physics. Both were stable. |
| 70 | 2026-09-22 | analysis | Min-depth 500 improved global A2 from 1.336 to 1.282 C and improved the 0--3 and 4--7 coastal bands, while NA 40--60N was slightly worse. |
| 71 | 2026-09-22 | reflection | Promote min-depth 500 cautiously as a diagnostic candidate. Next validate reproducibility before changing the locked baseline. |
| 72 | 2026-09-22 | reproduction | Repeated the 365d min-depth 500 run. Global A2 RMSE reproduced exactly at 1.282 C. |
| 73 | 2026-09-22 | reflection | Keep min-depth 500 as the preferred diagnostic candidate. Next compare a gentler coastal mask or near-land ventilation test, changing only one lever. |
| 74 | 2026-09-22 | experiment | Ran 30d and 365d min-depth 300 m probes. Both were stable. |
| 75 | 2026-09-22 | analysis | The 300 m floor improved global A2 to 1.299 C but did not dominate min-depth 500. Keep min-depth 500 as the preferred diagnostic candidate. |
| 76 | 2026-09-22 | reflection | Stop scanning the mask floor. Next diagnose near-land ventilation/current structure in the 0--7-cell coastal band. |
| 77 | 2026-09-22 | experiment | Ran 30d bulk-lambda probes at 120, 160, and 240 on the min-depth 500 candidate. All were stable. |
| 78 | 2026-09-22 | experiment | Ran 365d lambda 120, 160, and 240. Regional cold bias improved monotonically, but global A2 worsened slowly. |
| 79 | 2026-09-22 | reflection | Higher lambda is not a global-metric win. Keep lambda80 for global A2; record lambda120/160 as regional-bias variants and diagnose near-land ventilation next. |
| 80 | 2026-09-22 | reproduction | Repeated the lambda 160 + min-depth 500 candidate. It reproduced global A2 1.304 C, North Atlantic 1.009 C, and near-wall 1.010 C. |
| 81 | 2026-09-22 | reflection | Promote lambda 160 + min-depth 500 as the preferred regional-bias diagnostic candidate; keep lambda 80 + min-depth 500 as the global-A2 candidate. Next diagnose near-land ventilation/current structure. |
| 82 | 2026-09-22 | experiment | Ran lambda80 + min-depth500 + smooth80 for 30d and 365d; both passed. |
| 83 | 2026-09-22 | analysis | Smooth80 improved global A2 from 1.282 to 1.269 C and NA 40--60N from 1.158 to 1.143 C, with near-wall unchanged. |
| 84 | 2026-09-22 | reflection | Promote lambda80 + smooth80 over lambda80 + smooth30 as the global-A2 diagnostic candidate; next test lambda120 + smooth80. |
| 85 | 2026-09-22 | experiment | Ran lambda120 + min-depth500 + smooth80 for 30d and 365d; both passed. |
| 86 | 2026-09-22 | analysis | Lambda120 + smooth80 gives global A2 1.281 C, NA 40--60N 1.044 C, and near-wall 1.063 C. |
| 87 | 2026-09-22 | reflection | Lambda120 is a good compromise but not the regional optimum. Next reproduce lambda160 + smooth80 before choosing the next diagnostic candidate. |
| 88 | 2026-09-22 | reproduction | Repeated lambda160 + min-depth500 + smooth80 for 365d. Global A2 1.294 C, NA 0.997 C, near-wall 1.019 C. |
| 89 | 2026-09-22 | reflection | Promote lambda160 + smooth80 as the all-around diagnostic candidate; keep lambda80 + smooth80 for pure global A2. Stop the lambda scan and isolate the 0--3-cell coastal band next. |
| 90 | 2026-09-22 | experiment | Ran lambda160 + smooth80 + min-depth1000 for 30d and 365d; both passed. |
| 91 | 2026-09-22 | analysis | Min-depth1000 improved global A2 to 1.276 C but worsened NA to 1.026 C and near-wall to 1.095 C. |
| 92 | 2026-09-22 | experiment | Ran lambda160 + smooth80 + min-depth750 for 30d and 365d; both passed. |
| 93 | 2026-09-22 | analysis | Min-depth750 gave global 1.282 C, NA 1.019 C, near-wall 1.061 C; min-depth500 remains the all-around candidate. |
| 94 | 2026-09-22 | reflection | Stop the mask-floor scan. Next test a structural coastal/vertical lever, starting with a finer vertical grid probe. |
| 95 | 2026-09-22 | experiment | Ran 30d probes with 19 and 18 vertical levels; the 19-level probe blew up and the 18-level probe did not improve the coastal band. |
| 96 | 2026-09-22 | experiment | Ran smooth160 for 30d and 365d; both passed. |
| 97 | 2026-09-22 | analysis | Smooth160 gave global 1.281 C, NA 1.000 C, and near-wall 1.048 C, trading a small global gain for worse near-wall skill. |
| 98 | 2026-09-22 | reflection | Keep smooth80 as the all-around candidate. Next design a narrowly scoped coastal-ventilation/boundary experiment. |
| 99 | 2026-09-22 | experiment | Ran a 30d local SSH relaxation probe in 300--320E/55--60N. |
| 100 | 2026-09-22 | analysis | The coastal-band metrics were essentially unchanged, so local SSH relaxation is not the missing coastal lever. |
| 101 | 2026-09-23 | experiment | Ran 30d coastal T restoring probes with tau=30/10/3 days in the 0--3-cell band. |
| 102 | 2026-09-23 | analysis | The 30d probes show monotonic improvement of the coastal band, with tau=3d giving global 0--3 RMSE 1.003 C versus 1.290 C baseline. |
| 103 | 2026-09-23 | experiment | Ran 365d coastal T restoring with tau=10d and tau=3d; both passed. |
| 104 | 2026-09-23 | analysis | The tau=3d run improves global A2 to 1.253 C, NA to 0.978 C, and near-wall to 0.979 C. |
| 105 | 2026-09-23 | reflection | Keep coastal restoring as a diagnostic branch, not a production default. Next translate it into a defensible coastal boundary or mixing closure. |
| 106 | 2026-09-23 | experiment | Ran 30d coastal extra bulk-flux probes with lambda=40 and 80 in the 0--3-cell band. |
| 107 | 2026-09-23 | analysis | Both probes worsened the near-wall 0--3-cell bias, so extra local bulk exchange is not the physical closure. |
| 108 | 2026-09-23 | experiment | Ran 30d coastal T restoring width probes with cells<=1/3/5/7/9 at tau=3d. |
| 109 | 2026-09-23 | analysis | The coastal-band improvement is monotonic with width, with most of the extra gain coming from the 0..7-cell coastal/transitional zone. |
| 110 | 2026-09-23 | experiment | Ran 365d coastal T restoring with tau=3d and cells<=7 and cells<=9; both passed. |
| 111 | 2026-09-23 | analysis | The tau=3d, cells<=7 run improves global A2 to 1.229 C, NA to 0.933 C, and near-wall to 0.885 C. |
| 112 | 2026-09-23 | reflection | Keep tau=3d, cells<=7 as the current diagnostic candidate; cells<=9 gives a small extra gain but covers a much larger ocean fraction. |
| 113 | 2026-09-23 | experiment | Ran 30d coastal T restoring strength probes with tau=1d, 0.5d, and 0.25d at cells<=7. |
| 114 | 2026-09-23 | analysis | The 30d probes show a monotonic upper-bound improvement, with tau=0.5d giving near-wall 0..7 RMSE 0.343 C. |
| 115 | 2026-09-23 | experiment | Ran 365d coastal T restoring with tau=1d and 0.5d at cells<=7; both passed. |
| 116 | 2026-09-23 | analysis | The tau=0.5d run improves global A2 to 1.154 C, NA to 0.880 C, and near-wall to 0.804 C. |
| 117 | 2026-09-23 | reflection | Keep tau=0.5d, cells<=7 as the diagnostic upper bound, not a production default. Next translate it into a defensible coastal boundary-layer or mixing closure. |
| 118 | 2026-09-23 | experiment | Ran 30d coastal T restoring width probes at tau=0.5d with cells<=3, 5, and 7. |
| 119 | 2026-09-23 | analysis | cells<=7 remains the best compromise; cells<=5 is close but leaves more error in the 4..7-cell band. |
| 120 | 2026-09-23 | experiment | Ran 30d coastal vertical-diffusion probes with kappa_v=1e-5 and 1e-4 in the 0--7-cell band. |
| 121 | 2026-09-23 | analysis | Both probes failed to improve the coastal band; the stronger case worsened it. |
| 122 | 2026-09-23 | reflection | Local enhanced diffusion and bulk exchange are not the missing physical closure. Next consider a more explicit coastal boundary-layer scheme or a high-latitude proxy. |
| 123 | 2026-09-23 | reproduction | Repeated the tau=0.5d, cells<=7 coastal T restore run for 365d. |
| 124 | 2026-09-23 | analysis | The repeat reproduces global A2 1.154 C, NA 0.880 C, and near-wall 0.804 C. |
| 125 | 2026-09-23 | experiment | Ran 30d cosine-taper coastal T restoring probes at tau=0.5d and an equal-total-strength tau=0.25d. |
| 126 | 2026-09-23 | analysis | Both taper variants are worse than the hard-band control; keep hard-band coastal restore as the diagnostic upper bound. |

| 127 | 2026-09-23 | experiment | Ran paired 30d baseline and hard coastal-restore runs with 3D tracer terms. |
| 128 | 2026-09-23 | analysis | Near-wall surface budgets show negligible diffusion and bulk terms, a strong direct SST restore term, and a large compensating reduction in convection. |
| 129 | 2026-09-23 | reflection | The missing coastal closure is boundary-value or lateral transport, not another scalar diffusion or bulk-exchange process. |

| 130 | 2026-09-23 | experiment | Stabilized 0.65/0.60/0.55/0.50-degree probes with dt=1800s and nu_h=2e6. |
| 131 | 2026-09-23 | analysis | The 0.65 and 0.60 degree 365d runs pass and improve global, NA, and near-wall metrics over 0.70 degree. |
| 132 | 2026-09-23 | reflection | Promote 0.60 degree as the finest validated all-around diagnostic resolution; wait for the 0.50 degree 365d check before further promotion. |

| 133 | 2026-09-23 | experiment | The 0.50-degree 365d run passed with dt=1800s and nu_h=2e6. |
| 134 | 2026-09-23 | analysis | The 0.50-degree run improves global A2 to 1.204 C, NA to 0.914 C, and near-wall to 0.976 C over 0.70 degree. |
| 134 | 2026-09-23 | reflection | Promote 0.50 degree as the finest validated all-around diagnostic resolution; keep the 0.70-degree lambda80 candidate as the locked production-like baseline. |

| 135 | 2026-09-23 | experiment | Ran 30d and 365d 0.50-degree runs with and without the hard coastal SST restore. |
| 136 | 2026-09-23 | analysis | The 365d combined run improves global A2 to 1.078 C, NA to 0.818 C, and near-wall to 0.749 C over the no-restore 0.50-degree run. |
| 136 | 2026-09-23 | reflection | Resolution and coastal restore are complementary; record the combined run as the diagnostic upper bound and test gentler restore strengths next. |

| 137 | 2026-09-23 | experiment | Ran 30d tau=3d and tau=1d restore probes at 0.50 degree and validated tau=1d for 365d. |
| 138 | 2026-09-23 | analysis | The tau=1d 365d run gives global A2 1.100 C, NA 0.831 C, and near-wall 0.776 C; tau=0.5d remains the diagnostic upper bound. |
| 138 | 2026-09-23 | reflection | Keep tau=1d as the gentler compromise and tau=0.5d as the upper bound; both remain assimilation-like diagnostic constraints. |

| 139 | 2026-09-23 | experiment | Ran 30d and 365d lambda80 candidates at 0.50 degree with dt=1800s and nu_h=2e6. |
| 140 | 2026-09-23 | analysis | The 0.50-degree lambda80 candidate improves global A2 to 1.171 C and North Atlantic A2 to 1.025 C over the 0.70-degree baseline. |
| 140 | 2026-09-23 | reflection | Promote candidate_65n_05_gm0 as the finer-resolution production-like candidate; keep the 0.70-degree baseline as fallback. |

| 141 | 2026-09-23 | experiment | Ran 30d and 365d lambda80 + 0.50-degree runs with tau=1d coastal restore. |
| 142 | 2026-09-23 | analysis | The 365d tau=1d run improves global A2 to 1.020 C, North Atlantic A2 to 0.911 C, and near-wall bias to -0.508 C. |
| 141 | 2026-09-23 | reflection | This is the best all-around diagnostic so far, but remains assimilation-like; the next step is to physicalize the coastal boundary closure. |

| 143 | 2026-09-23 | experiment | Ran a 30d lambda120 probe on the stabilized 0.50-degree grid. |
| 143 | 2026-09-23 | analysis | Lambda120 gives global A2 0.866 C, worse than lambda80 at 0.836 C; lambda80 remains the best production-like scalar choice. |


| 144 | 2026-09-24 | experiment | Extended the stabilized resolution ladder to 0.45/0.40/0.35 degree; all three 365d runs passed. |
| 145 | 2026-09-24 | analysis | 0.45 gives global/NA 1.149/1.011 C and near-wall bias -0.917 C; 0.40 and 0.35 cost more and 0.40/0.35 worsen near-wall bias. |
| 146 | 2026-09-24 | reflection | Resolution gains diminish near 0.45 degree; stop the finer-resolution scan and promote 0.45 only after reproducibility. |
| 147 | 2026-09-24 | experiment | Launched a 365d reproducibility run for the 0.45-degree lambda80 candidate. |

| 148 | 2026-09-24 | reproduction | The 0.45-degree 365d repeat passed in 39.6 min and reproduces global/NA A2 1.1488/1.0105 C and near-wall bias -0.9175 C. |
| 149 | 2026-09-24 | reflection | Lock candidate_65n_045_gm0 as the production-like baseline; keep the reproduced 0.50-degree candidate as fallback. |
| 149 | 2026-09-24 | experiment | Registered and prepared 30d GM500/GM1000 boundary-transport A/B probes at 0.45 degree. |

| 150 | 2026-09-24 | experiment | Ran 30d 0.45-degree GM500 and GM1000 probes against the locked GM0 candidate. |
| 150 | 2026-09-24 | analysis | GM500 gives global/NA A2 0.9132/0.8485 C and near-wall bias -0.7403 C, a small all-around improvement; GM1000 worsens regional/near-wall metrics. |
| 151 | 2026-09-24 | reflection | Reject GM1000; promote GM500 to a 365d stability and climate check. |

| 151 | 2026-09-24 | experiment | Ran the promoted 365d 0.45-degree GM500 check. |
| 152 | 2026-09-24 | analysis | GM500 passes but degrades to global/NA A2 1.1718/1.1086 C and near-wall bias -0.9430 C versus the GM0 candidate. |
| 152 | 2026-09-24 | reflection | Reject GM500; retain GM0 as the 0.45-degree production-like baseline. The standard GM bolus closure does not solve the remaining boundary bias at this grid. |

| 153 | 2026-09-24 | experiment | Ran a 30d Redi500 check. It nearly duplicates GM500 because both enable the same skew-flux operator. |
| 153 | 2026-09-24 | reflection | Do not run a separate Redi 365d check after the GM500 rejection; the closure family is already rejected. |
| 154 | 2026-09-24 | experiment | Added an opt-in freezing-point air-target floor and ran a 30d 0.45-degree ice-proxy probe. |
| 155 | 2026-09-24 | analysis | The ice floor removes all sub-freezing SST cells and improves 30d global A2 from 0.9190 to 0.9041 C while leaving NA metrics neutral. |

| 155 | 2026-09-24 | experiment | Ran the promoted 365d 0.45-degree ice-air-floor check. |
| 156 | 2026-09-24 | analysis | The ice floor passes and improves global/NA A2 to 1.1126/1.0096 C, near-wall bias to -0.9121 C, and removes all sub-freezing cells. |
| 156 | 2026-09-24 | reflection | This is the first physical annual-closure improvement; promote to reproducibility before replacing candidate_65n_045_gm0. |

| 157 | 2026-09-24 | reproduction | The 365d ice-floor repeat passed in 39.7 min and reproduces global/NA A2 1.1126/1.0096 C with near-wall bias -0.9121 C. |
| 157 | 2026-09-24 | reflection | Promote the -1.8 C ice-air floor to the production-like 0.45-degree candidate; retain the GM0 no-proxy candidate as fallback. |

| 157 | 2026-09-24 | experiment | Re-scored remaining error bands for the ice-floor candidate; the 0..3-cell coastal band holds 47.9% of global A2 SSE. |
| 158 | 2026-09-24 | experiment | Tested 30d wet-cell-only marine-air target smoothing at 5 and 20 passes. |
| 158 | 2026-09-24 | analysis | Both make global A2 slightly better but worsen NA/near-wall metrics; reject marine-air smoothing as a coastal closure. |

| 159 | 2026-09-24 | experiment | Ran a 30d tau=30d cells<=7 coastal SST constraint on top of the ice-floor candidate. |
| 159 | 2026-09-24 | analysis | The gentle constraint improves global/NA A2 to 0.8962/0.8438 C and near-wall bias to -0.7318 C; promote to a 365d check. |

| 160 | 2026-09-24 | experiment | Ran the 365d tau=30d cells<=7 coastal proxy on the ice-floor baseline. |
| 160 | 2026-09-24 | analysis | It passes and improves global/NA A2 to 1.0914/0.9973 C and near-wall bias to -0.8842 C versus the ice-floor candidate. |
| 160 | 2026-09-24 | reflection | Test tau=10d to bracket the weakest useful strength; do not promote until a repeat of the chosen strength passes. |

| 161 | 2026-09-24 | experiment | Ran the 30d tau=10d coastal proxy. It improves global/NA A2 to 0.8827/0.8344 C and near-wall bias to -0.7117 C. |
| 161 | 2026-09-24 | reflection | Promote tau=10d over tau=30d for a 365d check; keep the final choice contingent on annual reproducibility. |

| 162 | 2026-09-24 | experiment | Ran the 365d tau=10d coastal proxy. It improves global/NA A2 to 1.0597/0.9777 C and near-wall bias to -0.8345 C. |
| 162 | 2026-09-24 | reflection | Treat tau=10d as the best diagnostic parameterization and tau=30d as the gentler compromise; run a tau=10d repeat before recording it as validated. |

| 163 | 2026-09-24 | reproduction | The tau=10d coastal proxy repeat passed and matches: global/NA A2 1.0598/0.9777 C with near-wall bias -0.8344 C. |
| 163 | 2026-09-24 | reflection | Lock tau=10d as the validated diagnostic parameterization, not a pure dynamical production default; test tau=3d only as a bracketing probe. |

| 164 | 2026-09-24 | experiment | Bracketed the coastal proxy with 30d tau=3d and tau=1d probes; both improve all-around metrics. |
| 164 | 2026-09-24 | analysis | Tau=3d gives global/NA A2 0.8519/0.8139 C and near-wall bias -0.6519 C; tau=1d gives 0.8199/0.7929 C and -0.5458 C. |
| 164 | 2026-09-24 | reflection | Promote tau=3d to a 365d check as a less extreme strong-proxy rung; do not promote tau=1d because it is an increasingly tight data constraint. |

| 165 | 2026-09-24 | experiment | Ran the 365d tau=3d coastal proxy. It passes and improves global/NA A2 to 1.0033/0.9392 C with near-wall bias -0.7115 C. |
| 165 | 2026-09-24 | reflection | Record tau=3d as the strongest validated diagnostic rung; keep tau=10d as the gentler compromise and do not promote tau=1d. |
| 166 | 2026-09-26 | benchmark | Added an explicit industrial-comparison protocol and model-target table. The next external benchmark is a same-protocol MOM6 slice, not another internal tuning round. |
| 167 | 2026-09-26 | tooling | Added benchmark manifest generation and a reusable comparison-table CLI so internal and later external model runs share the same reproducible report format. |
| 168 | 2026-09-26 | tooling | Added the pre-registered benchmark gate CLI and recorded the 30d mixed-layer/ice gate as PASS. |
| 169 | 2026-09-26 | plan | Added the MOM6 direct-slice setup checklist without vendoring external source code. |
| 170 | 2026-09-26 | ops | Found the first 0.5-degree mixed-layer/ice annual run was CPU-bound because the wrong Windows JAX environment lacked CUDA. Stopped it and restarted through the WSL GPU environment with the same 365d snapshot cadence. |
| 171 | 2026-09-26 | protocol | Pre-registered the 0.5-degree annual mixed-layer/ice check with explicit gates and control files. |
| 172 | 2026-09-26 | analysis | The first 0.5-degree annual closure run had only a final snapshot, making the steady-window scoring degenerate. It failed the global/NA gates, so a 10-day-snapshot rerun is now running for a fair comparison. |
| 173 | 2026-09-26 | experiment | The 10-day-snapshot 0.5-degree mixed-layer/ice annual run improved near-wall bias but failed the global and North Atlantic gates. Rejected as a production candidate; launched the same-resolution ice-floor control. |
| 174 | 2026-09-26 | audit | Audited the MLD diagnostic: without 3D snapshots it is computed from the initial T/S state. Marked it as such in the scorer so it cannot be used as a post-integration climate metric. |
| 175 | 2026-09-26 | control | Added the same-resolution 0.5-degree ice-floor control. It improves global A2 to 1.128 C and confirms that the 50m mixed-layer/ice pair is the source of the global error, not the ice floor. |
| 176 | 2026-09-26 | experiment | A 20m mixed-layer/ice annual probe reduced the global penalty versus the 50m pair but still failed the global and NA gates. It remains diagnostic-only. |
| 176 | 2026-09-26 | diagnosis | The annual 20m closure hurts the 10S..40N and southern subtropical bands while improving the polar/near-wall band, so a spatially varying mixed-layer depth is the next targeted test rather than more uniform-depth tuning. |
| 177 | 2026-09-26 | feature | Added an optional mixed-layer latitude-band mask. A 30d 0.5-degree probe with 20m MLD only in 50-65N preserves global A2, improves NA RMSE from 0.5511 to 0.4573 C, and improves near-wall bias from -0.5626 to -0.3113 C; annual check is running. |
| 178 | 2026-09-26 | experiment | The annual 50-65N mixed-layer mask nearly preserved global A2 (1.130 C) but still failed the NA gate (0.9867 C); a narrower 55-65N annual probe is running. |
| 179 | 2026-09-26 | experiment | The 55-65N mixed-layer mask passes the pre-registered annual gate: global A2 1.127 C, NA RMSE 0.8723 C, near-wall bias -0.6038 C. Promote to reproducibility before any promotion. |
| 180 | 2026-09-26 | reproduction | The 55-65N mixed-layer/ice annual repeat reproduces to numerical precision and passes the same-resolution control gates. Promote to validated diagnostic, not a pure production default. |
| 181 | 2026-09-26 | summary | Added a human-facing summary of the validated 55-65N mixed-layer/ice diagnostic candidate. |

| 182 | 2026-09-26 | feature | Replaced the fixed-latitude MLD choice with an optional 2D stratification-derived MLD using the WOA 0.03 kg/m3 density threshold, clipped to 10--100m. |
| 183 | 2026-09-26 | experiment | The stratification-MLD 30d probe improved global A2 to 0.7557 C, NA RMSE to 0.4196 C, and near-wall bias to -0.2858 C versus the 55--65N 20m diagnostic. |
| 184 | 2026-09-26 | experiment | The stratification-MLD annual check failed: global A2 rose to 1.2793 C and NA RMSE to 1.0505 C despite a better near-wall bias. It is rejected as a default. |
| 185 | 2026-09-26 | feature | Added a minimal stateful dynamic-ice closure with ice thickness, latent growth/melt, conductivity insulation, brine salt flux, checkpoint/output diagnostics, and unit tests. |
| 186 | 2026-09-26 | experiment | The dynamic-ice 30d probe is numerically stable and identical to the stratification-MLD probe because no explicit ice nucleates in the first 30d; annual diagnostic is running. |

| 187 | 2026-09-26 | external | Built MOM6 at commit f49a00096 with FMS 2023.03, GSW, and CVMix in an external local directory; the ocean-only executable passed a 32x32 one-hour smoke run. |

| 188 | 2026-09-26 | audit | The first annual dynamic-ice run was stable but formed no ice because the -1.8C air floor removed the cold forcing needed for nucleation; dynamic-ice probes now omit that proxy floor. |
| 189 | 2026-09-26 | bugfix | Corrected brine salt flux: the latent-ice salt change is applied directly, not multiplied by dt. The first monthly probe blew up; after the fix it is stable and forms 18 explicit-ice cells. |
| 190 | 2026-09-26 | audit | Fixed cell_area to use R^2 times angular increments. The old sea-ice extents were too small by the Earth-radius factor; climate SST metrics were unaffected. |
| 191 | 2026-09-26 | experiment | The fixed monthly dynamic-ice 30d probe is stable, forms 18 cells, max thickness 1.757m, extent 3.33e10 m2, but worsens global/NA metrics versus the annual-air diagnostic. |

| 192 | 2026-09-26 | feature | Added --global-sst-restore-days so ocean_solver can run the same prescribed SST-restoring boundary condition as a future MOM6 comparison slice. |
| 193 | 2026-09-26 | experiment | A 30d global 30d-restoring probe is stable (global A2 0.8253 C) but is not better than the 55--65N mixed-layer diagnostic; it is a benchmark-enabling option, not a tuning win. |
| 194 | 2026-09-26 | external | Fixed the shared MOM6 grid to 720x260 0.5-degree global ALE, read WOA T/S, and completed a stable 1-day dynamic smoke. |
| 195 | 2026-09-26 | feature | Exported the shared 2023 monthly NCEP wind stress to an A-grid MOM6 forcing file with an unlimited time dimension. |
| 196 | 2026-09-26 | feature | Added --no-meridional-heat-flux and a wind-only 0.5-degree comparison script. |
| 197 | 2026-09-26 | experiment | The 30d 0.5-degree ocean_solver wind-only control passed: global A2 0.8658 C, raw bias/RMSE -0.0580/0.5660 C, NA RMSE 0.6751 C, near-wall bias -0.1442 C. |
| 198 | 2026-09-26 | external | Launched the matched 30d 4-rank MOM6 wind-only run; it is not scored until its NetCDF output is converted to the shared grid and reference. |
| 199 | 2026-09-26 | external | Completed the matched 30d MOM6 wind-only run with 4 MPI ranks; it was stable and used 49.6 min wall time. |
| 200 | 2026-09-26 | comparison | The first matched 0.5-degree 30d wind-only comparison is available: ocean_solver global A2 0.8658 C vs MOM6 1.0961 C; NA RMSE 0.6751 vs 0.6814 C. This remains a wind-only dynamic control, not a climate benchmark. |


| 201 | 2026-09-26 | tooling | Added the model-neutral external scorer, Stage-R protocol v2, and generic MOM6 wind-only benchmark. The scorer now has coordinate, wet-mask, steady-window, and mismatch tests; the full suite passes. |


| 202 | 2026-09-26 | external | Configured the Stage-R MOM6 slice with zero bulk fluxes plus WOA surface SST/SSS restoring; FLUXCONST=0.16666667 m/day matches the ocean_solver 30d direct restore over a 5m top cell. A 1-day smoke passed. |
| 203 | 2026-09-26 | experiment | The matched 30d prescribed-restore controls both passed: ocean_solver global/NA A2 0.8645/0.6661 C and MOM6 1.1148/0.5794 C. Wall times were 3.5 and 47.6 min for 4 MPI ranks. |
| 204 | 2026-09-26 | comparison | Stage-R 30d promotes only the protocol, not a climate claim: ocean_solver improves global A2/bias, MOM6 improves NA RMSE, and near-wall RMSE is nearly tied. The next gate is a 365d reproducible Stage-R check before Stage F. |


| 205 | 2026-09-26 | ice | Added a reproducible Stage-I closed-loop manifest for the 30d monthly dynamic-ice probe: explicit growth volume 2.41e10 m3, latent growth heat 7.38e18 J, brine salt change -8.86e16 kg, effective MLD 31.5 m, and surface heat residual -1.70e22 J. The annual no-floor control remains a no-ice negative control. |


| 206 | 2026-09-26 | experiment | The 365d prescribed-restore ocean_solver control passed stability but degraded to global A2 2.245 C and NA RMSE 1.858 C over the final 90d; heat/salt drift stayed small. This confirms the 30d Stage-R slice is not predictive of an annual climate score and should not be promoted. |


| 207 | 2026-09-26 | ice | Added the first annual minimal closed-loop diagnostic: monthly-air dynamic ice is stable and forms 77 cells (max 4.77 m, extent 1.32e11 m2), with finite growth/melt and brine salt flux. It improves near-wall bias to -0.256 C but fails the climate gate at global A2 1.362 C and NA RMSE 1.190 C. |


| 208 | 2026-09-26 | gate | The annual dynamic-ice closed loop passes stability and improves near-wall bias, but fails the same-resolution 0.5-degree ice-floor climate gate on global A2 and NA RMSE. It remains a diagnostic, not a promoted default. |


| 209 | 2026-09-26 | attribution | A 365d monthly-air no-ice control has global A2 1.378 C and NA RMSE 1.185 C, nearly matching the dynamic-ice run (1.362/1.190 C). The annual degradation therefore comes mainly from monthly-air forcing, not the ice closure itself. |


| 210 | 2026-09-26 | tooling | Added MOM6 monthly 2m air and lambda*(air-WOA SST) sensible-heat export tools. They are ready for a first Stage-F 30d slice, with the limitation that the current sensible proxy uses a prescribed rather than instantaneous SST in the heat-flux denominator. |


| 211 | 2026-09-26 | experiment | The first Stage-F 30d dynamic-bulk probe with monthly 2m air and no ice is stable but has global A2 1.692 C and NA RMSE 2.088 C. The same forcing with the Stage-I dynamic-ice closure improves to 0.956/0.874 C, showing that the ice/mixed-layer closure is a major climate lever rather than a diagnostic add-on. |


| 212 | 2026-09-26 | audit | Rescored the 30d monthly dynamic-ice benchmark on the standard final-10d window. It gives global A2/NA RMSE 1.231/1.316 C, not the earlier full-window 0.956/0.874 C. Ice still improves the Stage-F no-ice control, but the improvement is smaller than initially inferred. |

| 213 | 2026-09-26 | synthesis | Added the industrial-model gap matrix and recorded Stage-F 30d internal metrics; MOM6 Stage-F and 365d Stage-R remain running. |

| 214 | 2026-09-26 | experiment | The exact Stage-F prescribed sensible proxy ran 30d but failed drift: max|T| 51.4 C, global A2 5.63 C, NA RMSE 2.98 C. It is therefore not a valid climate comparison until the same prescribed flux is checked in MOM6. |


| 215 | 2026-09-26 | external | Configured a MOM6 exact Stage-F dynamic bulk slice by restoring toward live monthly 2m air with FLUXCONST_T=1.701 m/day (80 W/m2/K at rho=1035, Cp=3925) and FLUXCONST_S=0. The 1d local-filesystem smoke passed in 210 s; the 30d run is launched. |
| 216 | 2026-09-26 | audit | The earlier MOM6 Stage-F prescribed-proxy run reached day 30 with a stable logged mean temperature, but its final spatial/restart NetCDF write failed because the C: filesystem was full. It therefore has no valid spatial benchmark and is recorded only as a stability warning. |

| 217 | 2026-09-26 | external | Stopped the C-drive 365d Stage-R run at about day 118 before its guaranteed NetCDF failure and migrated the matched configuration to `/root/external_models/mom6_slice_050/p0_restore_ale_365d_local`. The local run is restarted from day 0 with 4 MPI ranks. |

| 218 | 2026-09-26 | experiment | Launched the 365d ocean_solver Stage-F dynamic-bulk no-ice control (monthly air, lambda 80 W/m2/K, 4 500m floor, FCT/TVD). This provides the internal annual counterpart for the exact MOM6 bulk comparison. |

| 218 | 2026-09-26 | external | Prepared and launched a local 365d MOM6 exact dynamic-bulk run (`p0_stage_f_dynamic_365d`) with the same live-air bulk closure and daily diagnostics as the 30d probe. |

| 219 | 2026-09-26 | audit | The first annual Stage-F no-ice run was killed when WSL stopped. It was relaunched after the restart; the 30d exact dynamic-bulk probe is the only other active MOM6 job. |

| 220 | 2026-09-27 | comparison | The first matched 30d exact dynamic-bulk controls both passed: ocean_solver global/NA A2 1.692/2.088 C and MOM6 1.782/2.207 C. This is now a direct Stage-F comparison, not the failed prescribed proxy. |

| 221 | 2026-09-27 | experiment | The 365d ocean_solver Stage-F dynamic-bulk no-ice control completed: global A2 1.297 C, NA RMSE 0.799 C, near-wall RMSE 0.788 C, heat drift -0.962 percent. It improves NA/near-wall over the ice-floor candidate but is not promoted because global A2 and heat drift are worse. |

| 222 | 2026-09-27 | benchmark | Added standardized 3D profile scoring and ran a first 30d Stage-F comparison. MOM6 global 3D RMSE is 1.129 C; ocean_solver 3D RMSE is 2.842 C, mostly from deep-layer bias. Surface skill still favors ocean_solver, but the 3D gap is now explicit. |

| 223 | 2026-09-27 | comparison | Added the matched 30d Stage-F 3D diagnostic: MOM6 global 3D RMSE 1.129 C vs ocean_solver 2.842 C. Ocean_solver is better at the surface but has a severe deep-ocean warm bias, especially at 4000m (+6.35 C). |

| 224 | 2026-09-27 | performance | Added a diagnostic performance table: ocean_solver completes the matched 30d Stage-F slice in 3.5 min, while MOM6 takes 74.5 min on 4 MPI ranks. This supports a speed observation only, not model superiority. |

| 225 | 2026-09-27 | experiment | The 365d Stage-F MLD20 stratification run passed stability but degraded global A2 to 1.406 C and NA RMSE to 1.302 C. It is not promoted; the annual no-ice control also remains unpromoted because of worse global skill and heat drift. |

| 226 | 2026-09-27 | audit | Corrected the 3D scorer to apply the shared bathymetric vertical mask. The earlier +6.35 C 4000m warm bias was an artifact from scoring below-seafloor ghost layers. Corrected 30d global 3D RMSE is ocean_solver 0.861 C vs MOM6 1.036 C; MOM6 is modestly better in 40-60N. |

| 227 | 2026-09-27 | benchmark | Launched a 365d Stage-F ocean_solver rerun with 3D snapshots to provide the annual counterpart for the MOM6 exact dynamic-bulk 3D comparison. |

| 228 | 2026-09-27 | ops | WSL became inaccessible and was restarted. The two MOM6 365d runs and the ocean_solver annual 3D run were relaunched from day 0; prior partial runs are not usable for scoring. |

| 229 | 2026-09-27 | ops | After the WSL restart, both MOM6 365d relaunch attempts exited during startup; the ocean_solver annual 3D control was relaunched and remains active. |

| 230 | 2026-09-27 | ops | The annual ocean_solver 3D control died after ~90d during snapshot I/O. It was relaunched with 30-day 3D snapshots to reduce I/O/memory pressure. |

| 231 | 2026-09-27 | experiment | Completed the 365d ocean_solver Stage-F 3D snapshot control. Final-90d global 3D RMSE is 1.547 C, North Atlantic 3D RMSE is 1.276 C, and surface A2 is 1.204 C; stability PASS. |
| 233 | 2026-09-27 | ops | Isolated the first MOM6 365d startup failure by launching an annual run with the validated 30d configuration plus 365-day segment length. It passes startup and continues as the v2 annual dynamic-bulk comparison. |
| 232 | 2026-09-27 | ops | Removed failed MOM6 365d test directories and stale pip/uv caches; this restored enough host disk space for the annual MOM6 run. |
| 233 | 2026-09-27 | benchmark | Launched the annual MOM6 Stage-F dynamic-bulk v7 run from the validated 30d configuration with 365-day segment length and daily diagnostics. |
| 234 | 2026-09-27 | analysis | Added the Stage-F annual 3D layer decomposition. The annual skill loss is concentrated in the upper 500 m, pointing to mixed-layer/ventilation error rather than deep-ocean or surface forcing alone. |
| 235 | 2026-09-27 | experiment | A 30d Stage-F stratification-MLD probe (density-threshold MLD, dynamic ice) improved global A2 from 1.692 to 1.173 C, North Atlantic RMSE from 2.088 to 1.228 C, and near-wall RMSE from 1.958 to 0.941 C versus the Stage-F no-ice control. |
| 236 | 2026-09-27 | experiment | Launched the 365d Stage-F stratification-MLD probe to test whether the 30d upper-ocean improvement survives the annual gate. |
| 237 | 2026-09-27 | experiment | The 365d Stage-F stratification-MLD probe passed stability but did not beat the annual no-ice control: global A2 1.340 C and North Atlantic RMSE 1.230 C versus 1.297/0.799 C. Near-wall bias improved to -0.095 C. Do not promote. |
| 238 | 2026-09-27 | ops | Confirmed that the earlier MAXCPU=-1 annual failure was a disk-space issue: a 1d MAXCPU=-1 smoke run completed cleanly. Re-launched the annual MOM6 Stage-F run as v8 with no CPU-time limit to avoid stopping near the 8h mark. |
| 239 | 2026-09-27 | experiment | A 30d Stage-F northern-only stratification-MLD probe (20--60N, density-threshold MLD, dynamic ice) gave global A2 1.322 C, North Atlantic RMSE 0.827 C, and near-wall RMSE 0.533 C versus the Stage-F no-ice control 1.692/2.088/1.958 C. |
| 240 | 2026-09-27 | experiment | Launched the 365d Stage-F northern-only stratification-MLD probe to test whether the 30d improvement survives the annual gate. |
| 241 | 2026-09-27 | experiment | The 365d northern-only stratification-MLD run without dynamic ice passed stability but worsened North Atlantic RMSE to 1.605 C versus the annual no-ice control 0.799 C. Global A2 1.281 C. Do not promote. |
| 242 | 2026-09-27 | experiment | The 365d northern-only stratification-MLD run with dynamic ice also failed the annual gate, producing 5787 explicit-ice cells and a 11.03 m maximum thickness. It is retained only as a negative diagnostic. |
| 243 | 2026-09-27 | experiment | The 365d Stage-F dynamic-ice/no-MLD ablation passed stability and improved global A2 to 1.213 C versus the no-ice control 1.297 C, but North Atlantic RMSE worsened to 0.983 C and near-wall RMSE to 1.095 C. It also produced 5918 explicit-ice cells and 11.03 m maximum thickness. Not promoted. |
| 244 | 2026-09-27 | audit | Corrected the prior 40--60N probe label: the run used a fixed 100m mixed layer in the band, not the stratification-derived MLD mode. It improved NA/near-wall to 0.848/0.533 C but global A2 remained 1.562 C; the mislabeled annual launch was stopped before scoring.

| 245 | 2026-09-27 | experiment | The corrected 30d Stage-F 40--60N stratification-MLD probe passed stability but scored global A2 1.600 C, NA RMSE 1.246 C, and near-wall RMSE 0.969 C, worse than the fixed 100m 40--60N probe. The next annual gate is the fixed-depth variant. |

| 246 | 2026-09-27 | experiment | Launched the 365d Stage-F fixed 100m 40--60N no-ice probe because it beat both the Stage-F no-ice control and the stratification-derived 40--60N probe on the 30d regional gate. |

| 247 | 2026-09-27 | experiment | The 365d Stage-F fixed 100m 40--60N no-ice probe passed stability but failed the annual gate: global A2 1.343 C, NA RMSE 1.451 C, near-wall RMSE 1.209 C versus the no-ice control 1.297/0.799/0.788 C. Do not promote. |

| 248 | 2026-09-27 | experiment | Launched a 30d Stage-F dynamic-ice + fixed 100m 40--60N mixed-layer diagnostic to test whether the regional fixed-depth gain can be combined with the Stage-I ice closure before any annual run. |

| 249 | 2026-09-27 | experiment | The 30d Stage-F dynamic-ice + fixed 100m 40--60N probe passed stability but scored essentially the same as the no-ice fixed-depth probe (global A2 1.562 C, NA RMSE 0.848 C, near-wall RMSE 0.534 C). The fixed-depth annual gate had already failed, so do not annualize the combined probe. |

| 250 | 2026-09-27 | benchmark | Added standardized latitude-band x depth 3D metrics to the solver and external 3D scorers. The annual Stage-F no-ice control confirms the dominant cold bias is in 40--20S and 20--40N between 15 and 50m, not only near the 40--60N wall. |

| 251 | 2026-09-27 | diagnostic | Added a standardized MLD bias/RMSE metric to the 3D scorer. The annual Stage-F no-ice control has model MLD 170.8m versus reference 31.5m (bias +139.3m, RMSE 655.7m), so the current solver entrains far deeper than the WOA initial state. |

| 252 | 2026-09-27 | experiment | A 30d Stage-F fine-upper-z vertical-grid probe (0,-5,-10,-20,-35,-50,-75,-100,-150,-200,-300,-500,-1000,-2000m) passed stability but worsened the climate gate to global A2 1.699 C, NA RMSE 2.110 C, near-wall 1.970 C. Do not promote; the default 14-level grid remains better on the current protocol. |

| 253 | 2026-09-27 | experiment | A 30d Stage-F low-mixing probe (kappa_v=1e-7, kappa_conv=0.005) was stable but essentially unchanged versus the Stage-F no-ice control (global A2 1.695 vs 1.692 C, NA RMSE 2.086 vs 2.088 C, near-wall 1.955 vs 1.958 C). A scalar mixing reduction is therefore not the main MLD fix. |

| 254 | 2026-09-27 | experiment | Completed the 365d Stage-F dynamic-ice/no-MLD 3D rerun in 28.9 min; it is stable but annual 3D RMSE is essentially unchanged versus the no-ice control (1.543 vs 1.547 C) and MLD worsens to 288.7 m mean. Do not promote. |
| 255 | 2026-09-27 | comparison | The direct industrial target remains the MOM6 annual v8 3D score; the ocean_solver annual control and standardized latitude-depth/MLD metrics are now ready, while the MOM6 v8 run continues. |

| 256 | 2026-09-27 | benchmark | Added standardized latitude-band MLD diagnostics to the 3D scorer and protocol. The annual no-ice control has 40--60N MLD 247.1m and 60--40S MLD 224.6m; the dynamic-ice annual run improves 40--60N to 153.8m but worsens 60--40S to 314.9m, so a band/ice-state dependent closure is required. |
| 257 | 2026-09-27 | ops | Lost the v8 annual MOM6 run to WSL shutdown. Re-launched the exact dynamic-bulk comparison as v10 with daily temperature and salinity; a 1-day salt smoke test passed, and the persistent WSL session is holding the run. |
| 258 | 2026-09-27 | experiment | Added a latitude-band gate for dynamic ice and ran a 30d 40--65N probe. It is stable, but unlike the full dynamic-ice diagnostic it does not improve global/NA metrics; the band table records that a future closure needs separate northern and southern ventilation behavior. |
| 258b | 2026-09-27 | fix | Corrected the band gate so an off-mask cell keeps normal surface exchange; the first probe had accidentally zeroed air-sea flux outside the ice band. |
| 259 | 2026-09-27 | analysis | Rescored the 30d full dynamic-ice diagnostic on the protocol final-10d window; it is equivalent to the 40--65N-only probe and no-ice control. The earlier 0.707C result was a whole-30d average and is not comparable. |
| 259b | 2026-09-27 | ops | WSL is currently unavailable because the host D: volume is full; the annual MOM6 v10 run must be relaunched after freeing disk space. |
| 260 | 2026-09-27 | ops | Validated a 10-day temp+salt diag_table on a 1-day MOM6 smoke and relaunched the annual Stage-F comparison as v11 to fit the available disk while keeping the final-90d 3D/MLD gate. |
| 261 | 2026-09-27 | benchmark | The corrected 30d Stage-F band-ice/fixed-MLD probe passed all pre-registered gates: global A2 1.562 C, NA RMSE 0.848 C, global 3D RMSE 0.777 C, and MLD bias +16.3 m versus the no-ice control 1.692/2.088/0.861/+36.7 m. It was annualized for the required check. |
| 262 | 2026-09-27 | experiment | The matching 365d final-90d check rejected the band-ice/fixed-MLD candidate: global A2 4.759 C, NA RMSE 8.434 C, global 3D RMSE 4.042 C, and salt drift -0.1497%. The annual no-ice Stage-F control remains the internal baseline. |
| 263 | 2026-09-27 | fix | Tightened the benchmark gate for signed near-wall bias so a large positive overshoot cannot pass merely by crossing zero. Added tests for both the rejected overshoot and a signed improvement toward zero. |
| 264 | 2026-09-27 | comparison | No further fixed-depth closure probes are queued. The active gate is the MOM6 annual v11 final-90d 3D temperature/MLD comparison; NEMO remains a possible second industrial target after that score. |
| 265 | 2026-09-27 | audit | Found that the first band-ice/fixed-MLD annual check used a different bathymetry (`90.8%` ocean) than the 30d probe and annual no-ice control (`67.9%`). Its poor metrics are therefore invalid and must not be used to reject the candidate. |
| 266 | 2026-09-27 | experiment | Launched a corrected band-ice/fixed-MLD annual v2 run with the same 67.9% ocean bathymetry as the annual no-ice control. The final-90d climate/3D/MLD gate is pending. |
| 267 | 2026-09-27 | experiment | The corrected band-ice/fixed-MLD annual v2 run completed in 34.5 min and passed stability, heat/salt bounds, and signed near-wall bias, but failed the annual climate gate: global A2 1.246 C, NA RMSE 1.495 C, global 3D RMSE 1.559 C. It improves MLD bias to +121.2 m and 40--60N MLD bias to +17.2 m, so it remains a diagnostic only. |
| 268 | 2026-09-27 | fix | The external 3D scorer now uses the actual time coordinate for the steady window instead of assuming uniform record indices. Added a test for non-uniform time coordinates so the MOM6 annual scorer cannot select the wrong records. |
| 269 | 2026-09-27 | diagnostic | Per-day scoring of the valid band-ice/fixed-MLD annual run shows the candidate wins through about day 180 but loses after day 300 in North Atlantic and near-wall 3D temperature. The failure is therefore seasonal, not a uniformly bad constant-depth choice. |
| 270 | 2026-09-27 | diagnostic | Candidate-minus-control 3D difference in the final-90d window is concentrated in the 40--60N upper ocean: +1.35C at the surface, +1.37C at 5m, and +0.65C at 100m. This supports a seasonal upper-ocean heat-capacity/ventilation target rather than another global scalar. |
