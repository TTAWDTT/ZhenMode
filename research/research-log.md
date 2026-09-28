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

| 213 | 2026-09-28 | research | Registered the full industrial-alignment acceptance matrix using MOM6, MITgcm, NEMO and OMIP primary sources; retained century, independent skill, GPU/distributed and adjoint requirements. |
| 214 | 2026-09-28 | protocol | Precommitted default scalar diffusion budget/accuracy gates in 5a0b6b0 and the nonzero-biharmonic real-topography stress matrix in ef62d56. |
| 215 | 2026-09-28 | fix | Default scalar Laplacian and biharmonic now use conservative wet-face exchanges; 18 initial failures became passes, with 36 direct regressions and 271 full-suite tests passing. Masked budget ratios improve from order 1e-6 to order 1e-18 without a mean correction. |
| 216 | 2026-09-28 | validation | Four real-ETOPO synthetic-forcing 1-day runs and four nonzero-biharmonic 7-day runs completed; kernel hashes checked. This is short numerical evidence, not whole-model budget, century or climate qualification. |

| 217 | 2026-09-28 | protocol | Precommitted actual-stage budgets in 32d09bd after checking MOM6, MITgcm and ECCO extensive/native-budget guidance; stage identity and independent source closure are separate gates. |
| 218 | 2026-09-28 | tooling | Added read-only actual-core stage/source instrumentation and runtime-forcing aware per-step accumulation. 27 new regressions cover analytic sources, state equality, artificial leakage detection and subquantum float32 input; latest full suite is 298 passed. |
| 219 | 2026-09-28 | evidence | Real-ETOPO float64 one-day baseline and ice runs are stable but expose approximately -2.08e21 J unexplained fixed-node enthalpy in the nonlinear stage. Decomposition closes; conservation is explicitly NOT qualified. Next investigate process attribution and moving-volume compatibility rather than compensate the mean. |
| 220 | 2026-09-28 | verification | Audited output matches the frozen pre-instrumentation d9ffce9 kernel exactly in six fields for four nonuniform masked float32/float64 ice/no-ice cases. A newly installed wheel imports stage_budgets and starts the CLI under isolated Python outside the repo. |
| 221 | 2026-09-28 | protocol | Precommitted actual nonlinear process/top transport and linearized eta*C attribution in 77a4215 after checking MOM6 coupling and MITgcm moving-volume tracer equations. Original source budget remains unchanged. |
| 222 | 2026-09-28 | tooling | Added actual substep-mean top transport and four nonlinear process tables; 16 new regressions and 314 full-suite tests pass. Frozen fb322c6 states match exactly in four masked float32/float64 ice/no-ice samples; MMS and lint pass. |
| 223 | 2026-09-28 | evidence | Two real-ETOPO synthetic-forcing one-day audits identify actual top heat transport as the approximately -2.08e21 J proxy budget gap; adding linearized surface inventory still leaves approximately +2e21 J. Do not declare conservation or climate qualification. All runtime source hashes checked. |
| 224 | 2026-09-28 | audit | Code inspection also finds convection scaling both kappa and dt by conv_nsub. The one-day attribution uses conv_nsub=1 and cannot diagnose this separate physical-coefficient issue; independently research/reproduce before changing the solution. |
| 225 | 2026-09-28 | protocol | Registered native wet-face column constraint/adjoint and actual projection convergence gates in 11b5609, using MOM6 coupling, MITgcm grid/free-surface and JAX CG primary sources. |
| 226 | 2026-09-28 | fix | Seven initial failures expose surface-mask column divergence and inappropriate gradient/zero-tolerance CG. Native 3D face constraint/adjoint, positive matrix sign and dtype stopping now pass 16 direct gates; full suite is 330 passed, MMS/lint pass. |
| 227 | 2026-09-28 | evidence | Frozen ab56075 masked fixtures leave 5.9-7.2 percent native residual at cap 150; corrected fixtures leave order 1e-10. Old cap 1000 can amplify energy or become nonfinite; corrected solves stop at tolerance. |
| 228 | 2026-09-28 | evidence | Two real-ETOPO one-day cases remain stable but leave 1.46 percent cumulative native projection residual and about -2.03e21 J fixed-node enthalpy gap. Only 2.5-2.6 percent heat-gap reduction; do not qualify conservation, production convergence, century or climate. Iteration environment provenance also needs explicit recording. |
| 229 | 2026-09-28 | precision | Supplementary float32 real-ETOPO baseline/ice one-day runs are stable but eta displacement residual is about -3.5e8/-3.3e8 m3 and nonlinear accounting heat residual about -1.88e18 J. Float64 diagnostic sums do not fix state-update precision; preserve actual units and investigate rather than call the normalized drift zero. |
| 230 | 2026-09-28 | protocol | Registered real-grid projection convergence/provenance gates in 0bdc47d after JAX CG and PETSc CG/Jacobi research; caps are not convergence evidence and implicit derivatives require converged solves. |
| 231 | 2026-09-28 | implementation | 02a385b adds immutable effective solve settings/provenance, exact native Jacobi and worst-step accumulation. 36 direct tests, 350 full-suite tests, lint and MMS pass; no physical matrix, source, mean or tracer repair is hidden in the preconditioner. |
| 232 | 2026-09-28 | evidence | 96 fixed-right-hand-side solves on 12 frozen cce8f87 actual predictors: 150 caps fail both precisions, Jacobi/300 fails two float64 samples, 600/1200 pass sampled gates. Jacobi/600 takes about 31 percent less CPU solve time than none/600 at equal sampled accuracy; not GPU/whole-model throughput. |
| 233 | 2026-09-28 | evidence | Four actual one-day integrations are stable with verified source hashes. Float64 worst-step projection residual is about 1.09e-12, but heat-gap reduction is only 0.028 percent. Float32 cumulative residual passes while worst-step 5.429e-5/5.411e-5 fails the unchanged 5e-5 gate. Conservation, century and climate remain unqualified. |
| 234 | 2026-09-28 | protocol | Registered precision failure decomposition in e72a3ed. First materialized capture changes temperature by about one ULP and is rejected; 2e74ba7 retains this failure and requires separate reference advancement, quantized-metric versus arithmetic comparison and no claim of bitwise batch replay. MOM6/MITgcm research maps unmatched vertical weights, wet faces and missing barotropic time-average transport before any new physical change. |
| 235 | 2026-09-28 | evidence | Separate unmodified audit single-step trajectories have 18/17 float32 failures, with worst snapshots at steps 13/9. On the same captured predictors, a consistent float64 solve quantized back to float32 reduces native residual from about 5.5e-5 to 3.0e-7/2.1e-7. Transport arithmetic and metric quantization are much smaller; implicates the current solver/correction path, not an unavoidable state-precision floor. Capture differences and non-bitwise batch scope remain explicit; no production mixed-precision change or physical-conservation claim. |
| 236 | 2026-09-28 | protocol | Registered actual true-residual/refinement gates in 3804066 after Netlib/JAX primary research and local JAX recurrence inspection. Keep 1e-9/5e-5 and compare 14 frozen predictors with bounded original-RHS absolute stopping floors; do not presume the cause before measuring. |
| 237 | 2026-09-28 | evidence | 86 solves show recursive float32 residual about 3.6e-6 while recomputed b-Apsi is about 5.0e-5, with public/mirrored solves matching exactly. Native velocity correction passes 8/8 float32 samples; no-floor corrections amplify residual to 0.281/0.113 and remain failed controls. |
| 238 | 2026-09-29 | fix | 2042ce1 adds bounded native-transport corrections and immutable CLI/provenance. Independent smooth-pressure float32 regression fails old code at 1.129e-4; 45 projection tests including active-branch local JVP/VJP now pass. Full suite 359 passed, 23 old warnings; lint/MMS pass. No physical/source/mean or whole-state float64 substitution. |
| 239 | 2026-09-29 | evidence | Six full 144-step real-ETOPO/synthetic-forcing trajectories pass stability and unchanged per-step projection gates. Worst float64/2deg about 1.09e-12; float32/2deg 4.318e-6/4.322e-6; float32/1deg 4.150e-6/4.141e-6. All runtime source and bathymetry hashes checked. Heat proxy gap about 2e21 J persists; no physical conservation, century or climate qualification. |
| 240 | 2026-09-29 | protocol | c3c47dd registers independent control-volume and actual barotropic time-average witnesses after MOM6/MITgcm research; 82e9ce0 implements frozen-kernel diagnostics. First erroneous r_bot factory keyword fails before computation, is corrected to PhysicsConfig and its log retained. |
| 241 | 2026-09-29 | evidence | Full-wet 50m column has 67.5m node-proxy thickness; sheared native versus barotropic column divergence differs by 48.78 percent, staircase by 65.74 percent. Actual old-substep mean closes surface update to 1.48e-14; endpoint substitution leaves 0.00595. Real node-proxy column depth errors span about -901/+925m at 5/95 percentiles. Missing physical identities are confirmed, not fixed. |
| 242 | 2026-09-29 | scope | Corrected previous 2deg text shape 180x65x14 to actual 180x66x14; original JSON/NPZ were correct. Retained old protocol discrepancy rather than rewrite history. Center cutoff +/-65 gives full-cell edges +/-66; 1deg has edges +/-65, so the pair is not a same-domain convergence experiment. Complete industrial roadmap remains active. |
| 243 | 2026-09-29 | protocol | Rechecked MITgcm finite-volume/C-grid/moving-surface and MOM6 mean-Q primary sources; precommitted physical geometry/extensive shared-flux migration gates in 7f9a52c. Donor reference is not the final climate method or a second qualified mainline. |
| 244 | 2026-09-29 | implementation | 1d04acf adds partial cells, intersecting faces, V/N and shared Q, explicit sources, material top/bottom and actual C-grid substep-mean coupling. Thirty direct regressions and initial 389-test suite pass; legacy driver is not migrated. |
| 245 | 2026-09-29 | failure | Original real-ETOPO reference completes four float64 100-step groups but rejects four float32 groups at first-step surface ratios 4.247e-6/4.396e-6. Passing donor/wave CFL cannot qualify a failed surface identity; rejected steps are not zero-error budget PASS. |
| 246 | 2026-09-29 | research | Frozen diagnosis exactly reproduces old residuals; final 64-bit metric alone still fails 2e-6 due to incoming V/A-h representation. Goldberg/JAX research precedes precision protocol 6f5b89e. Data-free 180x66 regression fails old code at 3.576e-6. |
| 247 | 2026-09-29 | fix | cb38ba7 diagnoses pressure surface from stored V with explicit 64-bit division/subtraction returning original dtype. No mean/volume repair, whole-state promotion or relaxed gate; X64 is required. Thirty-three direct regressions pass. |
| 248 | 2026-09-29 | evidence | Eight real-ETOPO linear-wave/two-tracer groups complete 100 steps at dt60. Surface/content gates pass, but nonuniform float32 extrema violate unchanged 2e-6 bounds by up to 2.33e-5 salt units: retain 6/8 and overall FAIL. Production conservation/century/climate remain unqualified. |
| 249 | 2026-09-29 | validation | Final 392 tests pass with 23 old warnings; configured/research lint, diff-check and production MMS ALL PASS (4.30). Independent wheel target installs/imports both new core modules under isolated Python. Runtime source hashes match cb38ba7; no production cutover or industrial PASS. |
| 250 | 2026-09-29 | protocol | Registered bounded shared transport/precision gates in 8803e73; read primary multidimensional FCT/SSP equations before implementation. Method/precision choice committed in71fc1c9 before accuracy results. |
| 251 | 2026-09-29 | evidence | Twelve frozen prescribed-Q controls distinguish arithmetic from stored inventories. Native32 patterned excursion9.91e-5; arithmetic64/store32 and promoting only V or N still fail. Both64 and reconstructed pair32 pass this fixture with equal73728 inventory bytes, twice native32. No barotropic replay or GPU-cost attribution. |
| 252 | 2026-09-29 | implementation | 2b9d632 adds centered metric reconstruction, six-face extensive FCT, checked SSPRK2 and explicit V/N64 with momentum32 permitted. Seventeen new gates include independent scalar-loop oracle, sources/partial cells/dry/CFL and local JVP/VJP. Cosine ratios3.37/3.30 meet registered3.2, but observed orders1.75/1.72 do not prove asymptotic2 or coupled dynamics order. |
| 253 | 2026-09-29 | validation | Eight same real-ETOPO100-step dt60/BT4 groups qualify explicit64 inventory with pure64/momentum32; worst content budget5.03e-14, bound excursion3.27e-13. Hashed snapshots independently recomputed; water residuals -4.23 to+3.26m3, not giant-inventory normalized alone. Native same-dtype donor still6/8; original frozen surface failures reproduce. |
| 254 | 2026-09-29 | validation | Full409 tests/23 old warnings, configured/research lint, MMS4.30 and isolated installed FCT execution pass. Verifier positive and five distinct negative metadata controls checked after correcting Windows path normalization. Production migration, complete physics/global geometry, century/climate/forecast, GPU/distributed and whole-adjoint remain unqualified; next actual3D momentum/common-depth pressure and cutover. |
| 255 | 2026-09-29 | protocol | 047aed7 registers common-depth partial-cell pressure, weighted energy/true-residual rotation and actual layer/fast-Q coupling after MOM6/MITgcm/Engwirda primary research. Latest user qualification explanation did not change implementation: preceding goal turn no progress; take the available repair rather than claim a wait. |
| 256 | 2026-09-29 | evidence | Four installed-JAX CG controls falsify atol-only repair: primal atol/primal start dot error0.28587; primal atol/zero start1.0; zero atol/primal start1.32e-4; zero/zero2.52e-15. Selection6a67a99 retains all failures and registers both solver changes plus fixed real-reference inputs. |
| 257 | 2026-09-29 | implementation | 41ccecd adds physical interfaces/common-depth represented-profile pressure, active3D layer velocities, force counted once and shared actual fast-Q/FCT. Zero CG start/atol corrects local implicit adjoint without relaxing energy/residual gates. Manufactured direct amplitude correction and misplaced test-block failures retained;31 direct/81 adjacent tests pass. |
| 258 | 2026-09-29 | evidence | Eight clean41ccecd real-ETOPO180x66x14 density-feedback linear momentum references complete100x60s/BT4. Worst content budget8.75e-18, surface discrepancy5.41e-16m, bounds2.34e-13; water changes -0.01294 to+0.000458m3. Not full nonlinear physics, production or climate. |
| 259 | 2026-09-29 | verification | Independent geometry/initial-density/inventory snapshot verifier and five negative controls pass. First area-formula cancellation error1.41e-14 retained; equivalent radians(diff) formulation fixes verifier without loosening1e-14. Source/snapshot hashes match; recorded stage gates are not independent step replay. |
| 260 | 2026-09-29 | validation | Full440 tests/23 old warnings, lint, production MMS4.30 and isolated installed active momentum execution pass; generated untracked build removed safely. Green suite does not prove physical consistency on uncovered partial faces. |
| 261 | 2026-09-29 | protocol | Re-read actual MITgcm flux/PV/shear/metric and MOM6 thickness-Coriolis equations. 0481332 preregisters independent thin1m/neighbor30m uniform-v physical force equality at unchanged1e-8 before nonlinear migration; no inference from skew energy alone. |
| 262 | 2026-09-29 | failure | Partial-face analytic acceleration0.002m/s2 becomes0.006477225575 (3.2386 times), while rotation energy change8.45e-16 and true residual4.72e-16 pass. sqrt(W) cross interpolation incorrectly imports neighbor wet-volume square-root ratios. Retain FAIL and require physical overlap/dual-volume repair before nonlinear/production cutover. Full industrial goal remains active and unmet. |
| 263 | 2026-09-29 | protocol | b55a130 registers exact spherical wet half-cell rectangles and u/v quadrant-overlap rotation after mature MITgcm/MOM6 research. Two independent east/north thin-face regressions FAIL before core edit; no force clipping, damping, masked shallows or shifted gates. |
| 264 | 2026-09-29 | implementation | f26d686 integrates actual common physical volume with exact transpose and wet dual mass. Thin-face force relative2.22e-16;46 direct/455 full tests pass. Physical energy quadrature explicitly changes; original sqrt(W) failure replayed at frozenf6bb886. |
| 265 | 2026-09-29 | verification | Same eight real-grid overlap references pass; independent inventory/geometry/hash verifier and five negative controls pass before subsequent source changes.6000s per group is not nonlinear/production/climate qualification. |
| 266 | 2026-09-29 | failure | 62d9b8d preregisters physical pressure/continuity work. Actual regular/irregular relative0.001344/0.003076 FAIL; offline shared-contact/mass controls below4.1e-17. Physical dual mass is not face_area*point_distance. Original power in W, hashes and FAIL retained. |
| 267 | 2026-09-29 | protocol | Previous goal response explained century proof and inspected authoritative evidence but made no implementation change; revalidate as no implementation progress. fcd9761 registers paired hydrostatic/public-fast mass-adjoint repair and unchanged1e-12 work,1e-6 FD,1e-12 adjoint,3.5 spatial refinement gates BEFORE edits. |
| 268 | 2026-09-29 | implementation | Seven new independent force/work cases FAIL before repair.9a2f219 factors physical horizontal measures once and pairs BOTH pressure operators/CFL with shared-Q divergence; retains scalar center distances.106 adjacent direct cases and final11 force/AD cases pass; no added damping or state repair. |
| 269 | 2026-09-29 | measurement | SAME actual pressure-work cases now3.016e-17/4.087e-17. Independent regular smooth spherical cell-mean MMS east ratios3.9867/3.9964, north3.8781/3.9352 pass frozen3.5. Does not prove arbitrary-cut-cell or full coupled temporal/spatial order. |
| 270 | 2026-09-29 | verification | Eight SAME real-ETOPO100x60s references pass at CLEAN9a2f219. Inventory maximum relative4.953e-18, surface5.412e-16m, bounds2.345e-13; source/snapshot hashes and five verifier negative controls independently checked. Original reports retained and not qualified by newer method verifier. |
| 271 | 2026-09-29 | validation |466 full regressions/23 existing warnings in440.06s, lint,60 local links and research YAML pass. Legacy production MMS4.30 and isolated installed physical-mass active momentum/shared-inventory step pass. Own generated untracked build removed only after resolved-path/tracked/reparse checks. Component success does not repair un-migrated production budgets. |
| 272 | 2026-09-29 | reflection | Read Herbin et al. actual staggered pressure/dual-mass/kinetic balance; regular MAC averages assume different supports than our spherical moving wet intersections. Next implement verified nonlinear dual continuity and momentum, buoyancy exchange, EOS/real physics and actual driver/restart migration. Preserve complete industrial roadmap; no century/climate/forecast/GPU/full-adjoint completion claim. |
| 273 | 2026-09-29 | protocol |32a2548 registers actual shared-Q return and ordinary-MAC moving-dual falsification before diagnosis/API edits, after Herbin/MITgcm actual r-star research. Preserve independent physical support and unchanged gates. |
| 274 | 2026-09-29 | implementation | Four missing-flux API cases FAIL before edit;0eb56ed returns SAME immutable Q already passed to V/N, not endpoint-rebuilt flux. Four new/61 adjacent cases pass; arithmetic/state unchanged. |
| 275 | 2026-09-29 | failure | Actual clean0eb56ed isolated rain yields zero common-wet dual mass change, while ordinary MAC predicts1.45e12/4.14e12m3, relative1 FAIL. Uniform-rain and scalar/kinematic controls pass. Failed mapping hypothesis is NOT an implemented nonlinear operator; support-exchange product is NOT a physical source fix. |
| 276 | 2026-09-29 | verification | Same eight real100x60s references pass at clean0eb56ed; independent final inventories/geometry/last local V-Q, column Q, boundaries and hashes pass. Nine verifier negatives reject. Separate scalar-loop dual geometry/NPZ audit and eight1e8m3 exchange corruptions confirm diagnosis; not every-step independent replay. |
| 277 | 2026-09-29 | validation |470 whole-suite tests/23 old warnings in441.80s, legacy MMS4.30 and isolated installed active shared-Q step pass. Specific live full-suite handle completed exit0; no duplicate restart. Previous implementation goal made progress; user century-proof discussion adds no new physical qualification. Next physically consistent moving support/flux selection and actual nonlinear/production work; complete industrial objective active and unmet. |
| 278 | 2026-09-29 | research | Re-read actual MITgcm flux/metric/r-star, Quiros Rodriguez stationary cut-cell second-kind capacities and Arrufat shifted-VOF reconstruction. c6aae95/1b129ef pre-register full wet half-prism trial, actual-Q local commutation, column mobility and rejection of plain RT0 contact traces BEFORE probe. Published Cartesian/static results do not qualify spherical moving ocean physics. |
| 279 | 2026-09-29 | implementation |5929a58 adds read-only candidate probe, not core geometry migration. Initial missing exterior wall-zero entries cause shape exception;88d59ca fixes it and retains failure log.3768060 masks old uniform mode at actual open layers and adds old/new mobility comparison; original unmasked output retained, not authoritative. |
| 280 | 2026-09-29 | evidence | ALL four rain and eight real-Q snapshots pass independent half-prism stock/oracle, local commutation and actual mass/source increments plus offline column pairing.25 direct negatives reject. Separate sparse matrix audit over12 grids has defect at most1.11e-16 and12 negatives reject; initial unrestricted-wall assertion retained and boundary entries fixed in independent general matrix, not by gate relaxation. |
| 281 | 2026-09-29 | failure | Plain RT0 spreading of actual nonzero Q across full primary face would leak into blocked bottom wedge; hypothetical unsmoothed maximum6.60e5m3/s, NOT actual core leakage. New-support constructed-profile old uniform cross/KE up to0.788 and symmetric capacity difference0.311 forbid mass-only migration. Kinematic PASS remains physical reconstruction UNQUALIFIED; no actual nonlinear solver, long integration or climate/GPU/adjoint qualification. |
| 282 | 2026-09-29 | protocol |3a91aed preregisters actual shared wet-contact traces and clipped vertical primitive after cut-cell capacities, Herbin and MITgcm research; original masses/forces/time path unchanged. Missing API fails collection, not13 numerical operators. |
| 283 | 2026-09-29 | implementation |d7e24b9 returns/checks reconstruction in actual linear step; initial Python all on traced bool fails10 tests and is replaced with JAX reduction.13 new/74 adjacent cases pass without gate relaxation; wall wedges stay zero and top/bottom/divergence match actual Q. |
| 284 | 2026-09-29 | provenance |Initial old-core capture records worktree hashes at process END after edits and is REJECTED as qualified baseline. Re-capture executes/hash-checks five immutable3a91aed git blobs; four source/velocity-precision fixtures have exact V/N/u/v in all16 comparisons. This is not all-real-trajectory bitwise equality. |
| 285 | 2026-09-29 | evidence |SAME eight real100x60s/BT4 groups pass at cleand7e24b9; maximum normalized bottom primitive error6.3472e-16. Host last-inventory/actual-Q and separate wet-contact/primitive audits pass all8 and reject9+5 corruptions. Disturbed real trajectories have tiny JIT roundoff differences; stages are not independently replayed at every step. |
| 286 | 2026-09-29 | validation |483 full tests/23 old warnings in472.40s, legacy MMS4.30 and isolated installed active Q64/inventory64/momentum32/wet-wall step pass. Frozen mapped traces are not physical point velocities, nonlinear momentum or production cutover; full century/climate/forecast/GPU/distributed/adjoint objective remains active and unachieved. |
| 287 | 2026-09-29 | research |e377d9d qualifies pending wet-trace evidence; user century-proof discussion was planning, not new physical progress. Re-read Piola normal mapping, RT moments, MITgcm spherical velocity/free surface, cut-cell capacities and Herbin dual mass. a3cad34 preregisters project-derived latitude-arc weights, north interior bubble and ACTUAL paired dual-Q integration before new tests/core edits. |
| 288 | 2026-09-29 | failure |Two old-code physical constant-U witnesses FAIL; first sampled relative error0.20133448. This rejects physical-U promotion of prior constant-area trace, not its restricted mapped-Q contract. Omitted north correction and old area split/new center bubble controls also reject. |
| 289 | 2026-09-29 | implementation |4113bcc returns/checks metric trace plus full half-prism mass/dual Q in actual linear step; current force mass/source/state unchanged.15 new and13 prior direct cases,89 final-source adjacent cases pass. Physical half-sphere and surface quadrature, JVP/FD, local source32/64 increments and invalid-input gates pass. No cosine floor or fake geometry source. |
| 290 | 2026-09-29 | evidence |SAME eight ETOPO100x60s references pass at clean4113bcc; maximum normalized dual commutation4.43229e-16. Independent last primary/metric/dual snapshots pass and reject9+5+5 corruptions. Four small fixtures keep all16 state fields exact, MMS4.30 and installed active metric/dual step pass. Full suite live, not yet qualified; nonlinear/wall/kinetic/time/production and complete industrial goals remain unachieved. |
| 291 | 2026-09-29 | validation |Specific full-suite handle completes exit0:498 passed,23 existing warnings,514.28s. Final-source89 adjacent tests also pass; all started validation processes are terminal, no observation-timeout restart. This goal turn is PROGRESS: actual metric/dual integration and qualified evidence, not industrial completion. Next physical kinetic/velocity/wall support and paired force/rotation/fast/time nonlinear migration; non-diagonal mass would require re-derived projection, not automatic diagonal C reuse. |
