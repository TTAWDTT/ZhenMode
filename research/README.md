# Research code and retained evidence

Production starts at `ocean-solver` / `src/run_long_integration_global.py`, whose
installed module dependencies are declared in `pyproject.toml`. The production
CLI does not select these research candidates automatically.

| Location | Entry and dependency | Role |
| --- | --- | --- |
| `src/` | Production CLI -> `jax_solver_global.make_solver_global` -> FD step, forcing, diagnostics and restart | Installed production modules and explicitly selected alternative APIs |
| `research/experiments/` | Candidate modules and their `run_*` / `*_audit` / `verify_*` entry points | Separate methods, controlled comparisons, retained failures and protocols |
| `research/experiments/material_top_band/` | `dynamic_evidence` -> `dynamic`; `pressure_static_evidence` -> inventory pressure / real geometry | Synthetic moving-stock and static inventory-pressure controls |
| `scripts/controlled_window/` | `prepare_source` -> verified historical Git objects; source instrumentation -> workers / supervisors -> scalar postprocessors | Frozen-input reproduction tools; source identities are in `docs/controlled_window/` |
| `scripts/quality_speed/` | `gate` -> contract / artifact manifest; synchronized lifecycle timer | Registered acceptance tools; synthetic controls do not establish industrial qualification |
| `tests/` | `scripts/run_tests.py` -> pytest; Windows bounded runner for selected CPU checks | Unit, CLI, packaging and numerical regression controls |
| `docs/` | Decision records, protocols, hashes and measured receipts | Scientific claims and their limits; keep failed gates and historical evidence |
| `archive/regional/` | See its README | Retained regional implementation and historical attribution |

Start with the [project entry map](../README.md),
[r-star experiment index](experiments/material_rstar_coordinates/README.md),
[restart replay contract](experiments/material_restart_replay/README.md) and
[controlled-window reproduction guide](../docs/controlled_window/README.md).

## 2026-10-02: remove completed document-editing helpers

The nine former root `append_log*`, `update_state`, `update_hypotheses` and
`write_*` scripts were fixed-text editors from the 2026-09-21 survey. They were
not runtime or reproduction dependencies and could overwrite later document
updates or duplicate log entries. Their resulting findings, state, log,
literature notes, FCT protocol and progress report remain tracked. The scripts
remain available in Git history; future edits should update the actual documents
with reviewable diffs instead of replaying those one-off writers.

Frozen source preparers, candidate implementations, original input manifests,
patches, failures and third-party reference material are retained. A file being
old or absent from the production import graph does not make it disposable.
