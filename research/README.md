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

## 2026-10-02: share dashboard channel draining and verify cleanup

`dashboard/channel_io.py` now supplies the identical raw-byte `drain()` used by
`dashboard/cluster.py` and `scripts/publish_dashboard.py`. Both existing public
aliases remain available. The shared function has the same AST as both original
functions at base `4fcf3082a54a12206b6626920c7925da2f14c700`: receive size 65536,
initial timeout 4 seconds, 1-second idle deadline after a receive and 0.15-second
polling. The two callers retain their distinct ASCII conversion policies.

The 15 offline tests use a fake clock and channel, synthetic credential contents
and a denied SSH client. They cover raw bytes, timeout extension, empty and
nonpositive timeouts, errors, signatures, both aliases and ASCII behavior.
They do not read real credentials or open a connection.

The same selected regression set passed before and after cleanup: 198 passed,
1 deselected. The additional shared-alias assertion failed before extraction
and passes afterward. Full-repository Ruff, four existing reproduction/gate
CLI help commands, the helper import and local wheel construction passed.
The wheel contains all 39 declared production modules with identical source
bytes and preserves the `ocean-solver` entry point. Local JAX is unavailable;
the complete suite and FD manufactured-solution check remain CI requirements.

Activate the project-local virtual environment first. Reproduce the selected
regression set with:

```powershell
New-Item -ItemType Directory -Force logs/code_cleanup | Out-Null
python scripts/run_bounded_research_tests.py tests/test_dashboard_channel_io.py tests/test_mechanical_source_projection.py tests/test_n_probe.py tests/test_quality_speed_contract.py tests/test_packaging.py tests/test_material_top_band_dynamic.py tests/test_material_real_geometry.py tests/test_inventory_pressure.py -q -k 'not shared_implementation'
python scripts/run_bounded_research_tests.py tests/test_dashboard_channel_io.py -q
python scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.dynamic_evidence --output logs/code_cleanup/dynamic.json
```

The existing synthetic dynamic regression's complete JSON is byte-identical
before and after cleanup, SHA-256
`c4459dcac5412ab1bb4b456ea1ca12417be8ed9d221c39bce3d04951339017bc`.
Its dynamic source hash is unchanged:
`25568711d98ea867bbac8267dc8f1fcb3d1f226308f6e39d38c5b278e183349b`.
This comparison checks cleanup behavior; the report still declares industrial
quality and speed qualification false.

| Bounded check | Before wall / peak RSS | After wall / peak RSS |
| --- | --- | --- |
| Selected regression set | 79.547 s / 114,503,680 bytes | 84.769 s / 115,466,240 bytes |
| Existing synthetic dynamic report | 66.303 s / 37,654,528 bytes | 61.485 s / 37,359,616 bytes |

Each run used one CPU, a 180-second wall limit and a 4-GiB process-tree memory
limit, with exit 0 and no bound stop. Timing differences are verification
receipts, not a speed claim. Installed production sources, numerical research
modules, frozen source preparers/manifests, quality gates and existing evidence
documents are unchanged.
