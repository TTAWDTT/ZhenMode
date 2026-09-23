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

