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
