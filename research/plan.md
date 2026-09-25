# Current Research Plan

Updated: 2026-09-22

Locked diagnostic baseline: `candidate_65n_07_gm0`
Runner: `scripts/run_candidate_baseline.sh`

## Ordered next steps

1. **Reproduce the candidate baseline.** [COMPLETE 2026-09-22]
   Run one 365d integration with `candidate_65n_07_gm0`.
   Confirm global A2 RMSE is near the previous `1.336 C` and the North
   Atlantic regional RMSE is near `1.043 C`.

2. **Attribute the GM0 benefit.** [COMPLETE 2026-09-22]
   Compare GM0 against GM500 at 0.7 degree using horizontal velocity,
   heat transport, and local heat-tendency diagnostics.
   Determine whether GM0 helps through physically improved circulation or
   whether it removes an over-strong closure.

3. **Diagnose North Atlantic boundary-current structure.** [COMPLETE 2026-09-22]
   Focus on `40..60N`, especially the advective cooling and near-wall sector.
   Inspect SST, SSH, currents, meridional heat transport, and vertical
   redistribution at 0.7 degree.

4. **Diagnose near-wall gradients and local advection.** [COMPLETE 2026-09-22]
   Compare SST gradients, velocity-gradient alignment, and the saved solver
   advection term between GM0 and GM500 over the final 90 days.
   Result: GM500 produces stronger near-wall advective cooling without changing
   boundary-current speed. The worst cold cells cluster in local wall sectors
   and often have weak mean currents.

5. **Attribute vertical heat redistribution in the worst wall cells.** [COMPLETE 2026-09-22]
   For the 100 coldest near-wall cells, decompose surface and subsurface heat
   tendencies. GM500 has much stronger local advective cooling; GM0 leaves the
   coldest cells with positive net surface warming. Vertical export is small,
   and the cells are a mixed coastal/near-wall population.

6. **Separate coastal geometry from high-latitude ventilation.** [COMPLETE 2026-09-22]
   Land-adjacent cells remain the larger error source (`1.539 C` RMSE in GM0
   versus `1.065 C` for interior). The `59--60N` interior cluster still exists
   but is secondary, so the next step is coastal masking/ventilation diagnosis.

7. **Diagnose coastal masking/ventilation.** [COMPLETE 2026-09-22]
   Land-adjacent and transitional cells dominate the near-wall error. The
   `0..3` and `4..7`-cell bands explain about 85% of near-wall SSE, while the
   `>=8`-cell group is much better. This is not only a shallow-water problem.

8. **Test coastal mask/ventilation lever.** [COMPLETE 2026-09-22]
   `--min-depth 500` was stable for 365d and improved global A2 by 4.1%. It
   also improved the near-wall coastal/transitional bands, but the broader
   North Atlantic sector was slightly worse. Promote it cautiously as a
   diagnostic candidate, not a production default.

9. **Validate the new coastal-mask candidate.** [COMPLETE 2026-09-22]
   The 365d repeat reproduced the same global A2 RMSE (`1.282 C`). Keep
   `min-depth 500` as the preferred diagnostic candidate, not the production
   default.

10. **Decide the next coastal lever.** [COMPLETE 2026-09-22]
   The 300 m floor was stable but did not dominate min-depth 500. Stop scanning
   the mask floor. Keep `min-depth 500` as the preferred diagnostic candidate.

11. **Re-open the bulk-lambda upper sweep.** [COMPLETE 2026-09-22]
   Lambda 120/160/240 are stable and improve the North Atlantic / near-wall
   cold bias, but global A2 worsens slowly. Keep lambda80 as the global-favored
   candidate.

12. **Reproduce the regional-bias candidate.** [COMPLETE 2026-09-22]
   The lambda 160 repeat reproduces global A2 `1.304 C`, North Atlantic
   `1.009 C`, and near-wall `1.010 C`. Keep lambda160 + min-depth 500 as the
   preferred regional-bias diagnostic candidate, not a production default.

13. **Diagnose near-land ventilation/current structure.** [COMPLETE 2026-09-22]
   The lambda160 benefit is mostly stronger surface bulk restoring rather than a
   boundary-current restructuring. The remaining immediate-land band is still
   cold, and upper-ocean vertical export is not the dominant term.

14. **Test the lambda120 + smooth80 compromise.** [COMPLETE 2026-09-22]
   Lambda120 + smooth80 gives global A2 `1.281 C`, NA `1.044 C`, and near-wall
   `1.063 C`. It is a useful compromise but does not beat lambda160 + smooth80
   on regional skill.

15. **Reproduce lambda160 + smooth80.** [COMPLETE 2026-09-22]
   The repeat reproduces global A2 `1.294 C`, NA `0.997 C`, and near-wall
   `1.019 C`. Promote lambda160 + smooth80 as the all-around diagnostic
   candidate; keep lambda80 + smooth80 as the pure global-A2 candidate.

16. **Isolate the immediate-land cold band.** [COMPLETE 2026-09-22]
   The 0--3-cell band is a first-order global error source: if it were perfect,
   global A2 would drop from 1.294 to 0.947 C. Min-depth 750/1000 improve global
   A2 but degrade NA/near-wall, so the mask floor is not the main fix.

17. **Test a structural coastal/vertical lever.** [COMPLETE 2026-09-22]
   The 19-level probe blew up. An 18-level probe passed but did not improve the
   0--3-cell bias. Smooth160 improved global/NA slightly but worsened near-wall;
   keep smooth80.

18. **Design a coastal-ventilation/boundary experiment.** [PROBED 2026-09-22]
   A local SSH relaxation test in 300--320E/55--60N was essentially a no-op for
   the coastal band. It should not be promoted to a 365d experiment.

19. **Test a non-SSH coastal lever.** [COMPLETE 2026-09-23]
   A coastal T restoring diagnostic improved global A2 to 1.253 C, NA to
   0.978 C, and near-wall to 0.979 C with tau=3d. This confirms the 0--3-cell
   band is a first-order boundary-condition error, but it is not yet a physical
   closure.
   `0.5`, `0.6`, and `0.65` degrees currently blow up. Do not push finer
   resolution until the 0.7-degree stability margin and boundary-current
   diagnostics are understood.




20. **Test a tapered coastal constraint.** [REJECTED 2026-09-23]
   Cosine tapering at tau=0.5d is worse than the hard band (30d A2 0.935 vs 0.919 C), and an equal-strength tau=0.25d probe remains worse (0.924 vs 0.919 C). Reject tapering and keep the hard-band diagnostic.

21. **Run a coastal heat-budget diagnostic.** [ACTIVE 2026-09-23]
   Decompose near-wall SST evolution into advection, diffusion, surface flux, and residual bands. Use the result to identify the missing physical closure instead of adding another local scalar forcing.


22. **Run a coastal heat-budget diagnostic.** [COMPLETE 2026-09-23]
   The hard coastal restore supplies +0.875 K/d in the 0--3-cell near-wall band and +0.573 K/d in the 4--7-cell band, while diffusion and bulk terms are near zero. It also suppresses local convection by 1.163 K/d in the 0--3-cell band. Treat it as a boundary-value/mixed-layer control, not a missing scalar diffusion or bulk flux.


23. **Stabilize and test finer resolution.** [ACTIVE 2026-09-23]
   With dt=1800s and nu_h=2e6, 0.65/0.60/0.55/0.50-degree 10d and 30d probes pass. The 365d 0.65/0.60 runs improve all three climate metrics over 0.70. Promote 0.60 as the finest validated diagnostic resolution; wait for the 0.50 365d run before further promotion.


24. **Validate the 0.50-degree diagnostic rung.** [COMPLETE 2026-09-23]
   The 0.50-degree 365d run passes with dt=1800s and nu_h=2e6. It improves global A2 to 1.204 C, NA 40--60N to 0.914 C, and near-wall 55--60N to 0.976 C. Promote it as the finest validated all-around diagnostic resolution, not the locked production-like baseline.


25. **Combine 0.50 degree with coastal restore.** [COMPLETE 2026-09-23]
   The 365d combined run improves global A2 to 1.078 C, NA to 0.818 C, and near-wall to 0.749 C. Record it as the diagnostic upper bound; resolution does not make the coastal constraint redundant.


26. **Sweep gentler restore strengths at 0.50 degree.** [COMPLETE 2026-09-23]
   The 30d ladder is no-restore 0.893 C, tau=3d 0.866 C, tau=1d 0.843 C, and tau=0.5d 0.831 C. The tau=1d 365d run gives 1.100/0.831/0.776 C. Keep tau=1d as the gentler compromise and tau=0.5d as the diagnostic upper bound.


27. **Lock the 0.50-degree candidate.** [COMPLETE 2026-09-23]
   The 0.50-degree lambda80 candidate reproduces global A2 1.171 C, NA 1.025 C, and the same raw biases. Promote it to the current production-like baseline and keep the 0.70-degree candidate as fallback.


28. **Test lambda120 at 0.50 degree.** [COMPLETE 2026-09-23]
   The 30d lambda120 probe gives A2 0.866 C, worse than lambda80 0.836 C. Keep lambda80 in the locked 0.50-degree candidate.



29. **Validate the 0.45-degree all-around rung.** [COMPLETE 2026-09-24]
   Run one repeat of the 365d 0.45-degree lambda80 candidate with the same
   seeds/flags. If it matches the first run, promote it to the current
   production-like baseline and make 0.50 the fallback.

30. **Stop the resolution ladder.** [COMPLETE 2026-09-24]
   0.40/0.35 add little global skill, cost substantially more, and worsen the
   near-wall raw bias. Do not continue blind refinement.

31. **Design a boundary-current/lateral heat-transport closure.** [ACTIVE]
   Use the coastal heat-budget and velocity/gradient diagnostics to test an
   explicit near-boundary lateral transport or boundary-layer closure, rather
   than another local diffusion/restore scalar.

32. **Guard the candidate boundary.** [ACTIVE]
   Once 0.45 is locked, use it as the fixed reference for new closure A/B tests.
   Do not combine unvalidated physics changes in the first diagnostic run.


33. **0.45-degree repeat result.** [COMPLETE 2026-09-24]
   The repeat reproduces global/NA A2 `1.1488/1.0105 C` and near-wall bias
   `-0.9175 C`. Promote `candidate_65n_045_gm0` to the production-like
   baseline; keep the 0.50-degree candidate as fallback.


34. **Run the first GM transport A/B ladder.** [PROBED 2026-09-24]
   Use the 0.45-degree candidate as the fixed control and run 30d GM500 and
   GM1000 probes. Promote only an all-around improvement to a 365d check.


35. **Validate GM500 at 0.45 degree for 365d.** [REJECTED 2026-09-24]
   The 30d A/B promotes GM500: global `0.9190 -> 0.9132 C`, NA A2
   `0.8494 -> 0.8485 C`, and near-wall bias `-0.7425 -> -0.7403 C`.
   GM1000 is rejected because it makes the regional/near-wall band colder.


36. **Reject GM500 as a production closure.** [COMPLETE 2026-09-24]
   Its 30d gain reverses by 365d: global A2 `1.1488 -> 1.1718 C`, NA A2
   `1.0105 -> 1.1086 C`, and near-wall bias `-0.9175 -> -0.9430 C`.
   Keep GM0 and look for a boundary closure that does not overcool the band.


37. **Reject the GM/Redi branch.** [COMPLETE 2026-09-24]
   Redi500 duplicates GM500 because both share the same skew-flux operator in
   this prototype. After the GM500 annual failure, do not spend a separate
   365d run on Redi.

38. **Test a freezing-point sea-ice proxy.** [PASSED 2026-09-24]
   The 365d GM0 candidate has 3030 cells colder than `-1.8 C`. A 30d
   `--ice-air-floor` probe improves global A2 by 1.63% and removes all
   sub-freezing cells. Run the 365d stability/climate check next.


39. **Validate the ice floor annually.** [COMPLETE 2026-09-24]
   The 365d check improves global A2 from `1.1488` to `1.1126 C`, slightly
   improves NA/near-wall metrics, and removes all sub-freezing cells.
   Run one reproducibility repeat before promotion.


40. **Lock the 0.45-degree ice-floor candidate.** [COMPLETE 2026-09-24]
   The repeat reproduces global/NA A2 `1.1126/1.0096 C` and near-wall bias
   `-0.9121 C`, with zero sub-freezing cells. Make this the production-like
   baseline and keep the no-proxy GM0 run as fallback.

41. **Re-baseline the next error diagnosis.** [COMPLETE 2026-09-24]
   Use `candidate_65n_045_icefloor` as the fixed reference. Re-score the
   remaining error bands before choosing the next physical lever.


42. **Re-check the coastal forcing target.** [PROBED 2026-09-24]
   In the 0..3-cell band the NCEP target is `0.933 C` colder than WOA, versus
   a model raw bias of `-1.229 C`. A wet-cell-only air-target smoother gives a
   tiny global gain but worsens regional/near-wall metrics; reject it.


43. **Next targeted coastal test.** [ACTIVE 2026-09-24]
   The unresolved boundary-current/eddy heat transport remains the main
   candidate. Test only a gentle constraint or parameterization against the
   new ice-floor baseline; do not combine it with marine-air smoothing.


44. **Promote tau=30d coastal proxy to 365d.** [PASSED 2026-09-24]
   On the ice-floor baseline, the 30d `tau=30d, cells<=7` run improves
   global/NA A2 to `0.8962/0.8438 C` and near-wall bias to `-0.7318 C`.
   Validate for 365d before deciding whether it can be a documented
   diagnostic parameterization.


45. **Bracket the coastal-proxy strength.** [ACTIVE 2026-09-24]
   The 365d tau=30d proxy improves global/NA A2 to `1.0914/0.9973 C` and
   near-wall bias to `-0.8842 C`. Run the tau=10d 30d rung next, then only
   promote the best all-around strength after reproducibility.


46. **Validate tau=10d coastal proxy.** [PASSED 2026-09-24]
   The 30d tau=10d proxy beats tau=30d: global/NA A2 `0.8827/0.8344 C`
   and near-wall bias `-0.7117 C`. Run its 365d check next.


47. **Repeat tau=10d before validation.** [COMPLETE 2026-09-24]
   The 365d tau=10d proxy improves global/NA A2 to `1.0597/0.9777 C` and
   near-wall bias to `-0.8345 C`. Repeat it; then classify it as the best
   diagnostic parameterization, not a pure dynamical production default.


48. **Classify tau=10d coastal proxy.** [COMPLETE 2026-09-24]
   The repeat reproduces global/NA A2 `1.0598/0.9777 C` and near-wall bias
   `-0.8344 C`. Record it as the validated diagnostic parameterization; keep
   the ice-floor candidate as the conservative production-like baseline.


49. **Bracket tau=3d.** [PROBED 2026-09-24]
   Run a 30d tau=3d probe only to see whether the diagnostic curve is near
   saturation. Do not promote a stronger restore unless the 365d check is
   clearly better and stable.


50. **Validate tau=3d as the strong-proxy rung.** [COMPLETE 2026-09-24]
   The 30d ladder continues improving through tau=3d (`0.8519/0.8139 C`,
   near-wall bias `-0.6519 C`). Run tau=3d for 365d, but keep tau=1d out of
   promotion because it is a tighter data constraint.


51. **Set the coastal-proxy diagnostic ladder.** [COMPLETE 2026-09-24]
   tau=10d is the gentler validated compromise (`1.0598/0.9777 C`), while
   tau=3d is the strongest validated diagnostic rung (`1.0033/0.9392 C`).
   Stop here; tau=1d is a tighter data constraint and should not be promoted.


52. **Separate diagnostics from production defaults.** [COMPLETE 2026-09-24]
   Keep the ice-floor candidate as the conservative production-like baseline.
   Record tau=3d/10d as diagnostic parameterizations. Next work should either
   physicalize the boundary transport or move to another error source.


53. **Stop and consolidate.** [COMPLETE 2026-09-24]
   The incremental closure, forcing, and structural-flag probes are bracketed
   or rejected. Record the stop node in
   `research/to_human/2026-09-24_stop_node_summary.md`. The next branch should
   be a physical boundary-current closure, full mixed-layer/ice treatment, or a
   new bulk-flux formulation.


54. **Create the standardized benchmark protocol.** [COMPLETE 2026-09-26]
    Add `docs/benchmark_protocol_zh.md` and a reusable metric module that
    scores the locked candidate from a saved run without rerunning analysis
    scripts. The current 365d ice-floor repeat passes with raw global RMSE
    `0.9136 C` and zero sub-freezing cells.

55. **Add the minimal thermodynamic mixed-layer/ice closure.** [PASSED 2026-09-26]
    Add `src/mixed_layer_ice.py` with a heat-capacity mixed layer, freezing
    point, ice growth/melt, and a brine-rejection salt-flux sign convention.
    Add unit tests before solver coupling.

56. **Couple the closure into the solver.** [PROBED 2026-09-26]
    Add an opt-in `--mixed-layer-depth` solver flag that spreads surface heat
    flux over a well-mixed slab. Unit tests confirm the parameter reaches
    `FDPhysParams`. Next: add sea-ice salt flux and run the 30d/365d benchmark
    comparison.

57. **Try to exceed industrial models on a defined slice.** [ACTIVE]
    Do not try to beat mature models globally.  Choose a narrow benchmark slice,
    fix forcing/grid/reference, and compare only on reproducible metrics such
    as SST RMSE, coastal bias, mixed-layer diagnostics, and GPU wall time.
