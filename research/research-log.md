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
