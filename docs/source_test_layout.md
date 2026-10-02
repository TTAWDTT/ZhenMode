# Source and test layout contract

The canonical implementation lives in `src/ocean_solver`; `src/compat` contains
40 installed historical import/script bridges. The default production method,
step order, dtype defaults, physical operators and candidate refusal gates are
unchanged by this migration. Baseline: `5bb8879aa0894af441c483f34a57451f9e9df681`.

## Responsibilities and dependencies

The production chain is `runtime.entry -> runtime.application`, input/context/
forcing services, `fd.factory -> fd.integration`, then runtime records, recovery
and output. FD operators depend on canonical configuration, geometry and
physics modules. Canonical modules never import bare compatibility facades or
the FD/audit legacy export modules. Current data loaders live in `data`; shared
configuration lives in `configuration`; candidates remain opt-in under
`candidates/{fv,material}`. Validation/scoring and MOM6 export tools have their
own `validation/benchmarks` and `interop/mom6` owners.

`docs/source_test_layout.json` is the full inventory: every source move, all 100
original test moves, extracted support definitions, and complete Python source
and test file trees. The move declaration is for navigation and snapshot path
resolution, never a substitute for hashing actual implementation bytes.

Tests are grouped by `fd`, `runtime`, `data`, `validation`, `candidates/{fv,material}`,
`research/{rstar,contracts}`, and `infrastructure`. Original function names,
parameter IDs and assertions remain. The 32 original cross-test import sites
use explicit support modules. Research-shared weak-time diagnostics also moved
out of their test owner. NumPy quadrature/dense reference calculations remain
independent; no oracle is replaced with a call to the implementation it checks.

## Compatibility, inputs and packaging

All 40 bare imports resolve to their canonical module objects, including private
attributes on moved modules. Historical definition module names remain valid
for pickle; FD state/parameter classes keep identity, field order, defaults and
JAX pytree behavior. The existing private FD monkeypatch boundary remains the
actual consuming operator module, as documented in `docs/code_modularization.md`.
The two documented root launchers remain executable as direct files. The
`ocean-solver` console command points to `ocean_solver.runtime.entry:main`.

Tests use an installed distribution; the pytest `pythonpath=['src']` shortcut is
removed. Package discovery installs 96 canonical and 40 bare bridge Python
files. Root `data/` stays ignored while source/test directories named `data`
are tracked and linted. Configuration and WOA/wind/air cache defaults resolve
to the same checkout roots as before; environment overrides retain precedence.
Wheel defaults retain their previous distribution-relative root behavior.

Source identity conservatively includes all 96 canonical files: production
requires 121 source files and the material/research solver registry 103. No
minimal import closure is substituted. Missing files reject, byte tampering
changes hashes, and restart still requires the actual current source envelope.
Historical strict checkpoints require their historical source commit.

Current source snapshots resolve old logical archive labels through the layout
map and also include real canonical files, all compatibility bridges, support
helpers/package initializers and the declaration. Their hashes are real current
bytes. Current producer and verifier source checks use that same actual-file
mapping. Historical `git show` selectors and frozen protocol labels are retained;
an old receipt is never relabeled as verified against this new layout.

## Redundancy removed and deliberately retained

One duplicate 23-line `_synth_grid` body from the FCT and monotone advection test
suites is removed. Both bodies had identical ASTs and were only called by those
two owners; there was no CLI, dynamic registration or serialization API. One
unchanged builder now lives in `tests.support.fd.advection`. Both complete
limiter suites passed after this extraction. There are zero whole-file deletions
beyond relocations and zero numerical implementation removals. Removing root
`src`/`tests` path insertion and imports of test owners removes obsolete layout
coupling; this is reported separately from deletion of duplicate code.

Public compatibility bridges, legacy re-exports, compiled aliases, independent
oracles and frozen failures are retained. Static absence of a call does not
prove a public/private exported hook is dead. The historical coastal-budget,
horizontal-viscosity, resolution and salinity-restoring analysis scripts are
also retained: they belong to old result/protocol contexts, and changing their
legacy loaders or weighting would widen this layout task. No new reducer is
used to reinterpret their historical scores.

## Reproduce the regression checks

Use Python 3.12+ and install `.[dev]` in a project-local venv. Then run:

```bash
python -m ruff check .
python -m pytest tests/infrastructure tests/data -q
python -m pytest tests/fd/test_fct_advection.py tests/fd/test_monotone_advection.py -q
python scripts/capture_test_collection.py capture --output candidate_collection.json
python scripts/capture_test_collection.py compare reference_collection.json candidate_collection.json --layout docs/source_test_layout.json
python -m pip wheel . --no-deps --wheel-dir dist
```

Capture `reference_collection.json` on the baseline after the isolated POSIX
`resource` import portability fix. Collection comparison preserves the full
original function/parameter suffix and rejects missing, changed, colliding or
unmapped nodes. Run installed validation from a directory outside the checkout,
with the wheel installed in a separate local venv and `PYTHONPATH` unset:

```bash
python /path/to/checkout/scripts/validate_installed_modularization.py --source-root /path/to/checkout/src
ocean-solver --help
python -m ocean_solver.runtime.entry --help
```

The validator inserts no checkout paths. It checks all 136 installed Python
files against checkout bytes, all 40 alias identities, actual import origins,
121 source hashes, pickle/pytree and CLI help without loading external inputs.

For synthetic full-state comparison, use `scripts/capture_modularization_state.py`
on immutable baseline sources and candidate sources with the same dtype and
scan setting; use `scripts/compare_modularization_captures.py` to compare payloads.
The candidate bridge directory is `src/compat`. Exercise `--dtype float32` and
`float64`, with default scan and `--no-scan`. Each capture has 275 records.
`scripts/capture_driver_modularization.py` additionally exercises eight accepted
synthetic steps, ordinary and JIT wind, and two real interruptions/restarts;
its 344 retained records compare with only declared source/output location
normalization. Source hashes and verified envelope checksums are retained in
sidecars, not disguised as baseline hashes.

On Windows, `scripts/run_bounded_research_tests.py` bounds each invocation and
its process tree to one CPU, 180 seconds and 4 GiB. Use native limits elsewhere.
These tests use small synthetic inputs; they establish regression equivalence,
not complete-step temporal order, climate/forecast skill, industrial superiority
or acceleration. The prior scientific limits and negative controls still apply.

## Validation record — 2026-10-02

Before adding the separate MMS exit-status control, collection has 1805 nodes:
all 1791 original nodes mapped exactly, plus seven architecture cases, four
collection-mapping controls and three current-archive contracts. No original
test case or parameter identifier is lost. All 140 declared move destinations
exist and are tracked rather than ignored.

All four dtype/scan captures compare 275/275 records byte-for-byte; the JIT driver
comparison is 344/344 with no differences. Earlier successful captures are not
repeated after metadata-only changes. A real external-directory wheel install
checks 136 files, 40 aliases and 121 hashes. Seventeen compatibility tests and
17 new-contract/limiter tests passed. Data/provenance regression produced 145
passes and 98 input-dependent skips; its one old relative CLI directory failed,
was corrected to the repository fixture, and the affected test then passed.
The failure is retained in the local receipt instead of being omitted.

455 pre-existing docs/research/archive files with evidence/document suffixes
were byte-compared against the baseline, with zero differences. Their ordered
filename/SHA-256 ledger digest is
`383a66a9e68341d6b82e8c3e73f0fda85115b2c6f8d0d31607e392dd84af4f5c`.
The license is unchanged. Public evidence contains relative names and synthetic
receipts only; external inputs, local absolute paths and secrets are excluded.
