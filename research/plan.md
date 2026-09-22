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

17. **Test a structural coastal/vertical lever.** [NEXT]
   Stop the mask-floor scan. Run a narrowly scoped 30d probe with finer vertical
   resolution, then only continue to 365d if the coastal 0--3-cell bias improves.

18. **Only then revisit finer resolution.** [PROBED 2026-09-22: 0.6 still fails]
   `0.5`, `0.6`, and `0.65` degrees currently blow up. Do not push finer
   resolution until the 0.7-degree stability margin and boundary-current
   diagnostics are understood.



