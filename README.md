# ZhenMode

ZhenMode is a JAX ocean solver for the hydrostatic primitive equations on a
longitude-periodic, latitude-truncated finite-difference grid. The default
production entry uses the existing FD method. FV/C-grid, material-volume and
r-star work remain explicit candidate or research APIs.

## Install and inspect the CLI

Use Python 3.12 or newer in a project-local virtual environment. From the
repository root:

```bash
python -m venv .venv
# Linux/macOS/WSL:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ocean-solver --help
```

The console command and `python -m ocean_solver.runtime.entry` use the same
application. The documented direct-file entry
`python src/run_long_integration_global.py` is retained. Production runs require
external bathymetry, initial tracer and selected forcing inputs; `--help` does
not load them. Dependency and optional development requirements are declared in
[pyproject.toml](pyproject.toml). For CPU/CUDA environment details, see
[GPU runtime and validation](docs/gpu_runtime_zh.md).

## Where the implementation lives

```text
ocean_solver.runtime.entry
  -> application -> inputs + context + forcing
  -> fd.factory -> fd.integration -> processes + operators
  -> runtime.records + recovery + output
```

| Location | Responsibility |
| --- | --- |
| `src/ocean_solver/configuration.py` | Physical/grid configuration and shared input defaults |
| `src/ocean_solver/fd/` | State/parameters, operators, physical processes, fast mode, full step and factory |
| `src/ocean_solver/runtime/` | CLI, input/forcing preparation, accepted-step execution, records, output and recovery |
| `src/ocean_solver/geometry/`, `data/`, `physics/` | Grid geometry/loading, climatology/reanalysis/input validation, ice closure |
| `src/ocean_solver/audit/`, `diagnostics/`, `provenance/` | Budgets/monitoring, saved diagnostics, actual source identities and strict restart |
| `src/ocean_solver/validation/`, `interop/` | Manufactured solutions, benchmark/scoring tools and MOM6 forcing exports |
| `src/ocean_solver/candidates/` | Opt-in FV/C-grid and material-volume implementations |
| `src/compat/` | Installed historical bare-import/script bridges to the same canonical module objects |
| `tests/` | Tests grouped by actual subsystem or scientific contract; shared builders/oracles in `tests/support/` |
| `research/experiments/` | Independent candidate kernels, NumPy oracles, protocols and retained failures |
| `docs/`, `archive/` | Documentation/evidence and retired historical material |
| `scripts/`, `dashboard/`, `configs/` | Utilities, run dashboard and grid descriptions |

Canonical package modules import canonical implementations. Old imports such
as `config`, `jax_solver_global`, `material_top` and
`run_long_integration_global` remain available after installation. Legacy type
pickle names and the shared class objects are preserved. Private FD fault
injection must patch the actual consuming module; see the
[layout and compatibility contract](docs/source_test_layout.md) and
[earlier production modularization](docs/code_modularization.md).

## Develop and run tests

Install the development extra first; pytest imports the installed package.
It no longer relies on a `pythonpath = ["src"]` setting.

```bash
python -m ruff check .
python -m pytest tests/ -q
# A subsystem can be run by itself:
python -m pytest tests/runtime/ -q
python -m pytest tests/data/ -q
python -m pytest tests/candidates/material/ -q
python src/jax_solver_global.py
```

Tests are organized under `fd`, `runtime`, `data`, `validation`,
`candidates/{fv,material}`, `research/{rstar,contracts}`, and `infrastructure`.
Shared fixture construction and independent numerical reference calculations
live in `tests/support`; test modules do not import other test modules.
The [machine-readable migration map](docs/source_test_layout.json) preserves
original test function and parameter identifiers.

Data-dependent tests may skip without local inputs. CI creates a synthetic,
test-only bathymetry stand-in before running the full suite:

```bash
python scripts/make_synthetic_bathymetry.py
python -m pytest tests/ -q
```

Synthetic tests, byte comparisons and manufactured spatial solutions are
regression evidence. They do not establish forecast skill, complete-step
temporal order or industrial qualification.

## Inputs and restart

Bathymetry, WOA climatology and atmospheric reanalysis are not bundled. Input
defaults and existing environment overrides retain their resolution rules;
missing required relief fails explicitly. Moving a module does not move the
checkout's `data/` or reanalysis cache root. For an installed wheel, defaults
retain the previous distribution-relative behavior; configure input locations
explicitly for production use.

`--checkpoint-days` writes full-state checkpoints and must align with
`--snap-days`. `--restart-from` verifies effective grid/parameters/forcing,
actual source bytes, dtype/backend, counters and retained output identity.
Old checkpoints tied to old source hashes require their historical commit.
Neither old state-only files nor reconstructed source hashes are strict
restarts. See [restart and budget limits](docs/debug_validation_zh.md).

## Scientific boundary

The default domain has periodic longitude and closed latitude walls; it is not
full polar coverage. Default horizontal resolution is one degree and CLI
latitude limit is +/-60 degrees. Mixing, seasonal forcing, ice and projection
options have their own support and validation limits. Candidate paths retain
explicit refusals and negative controls; changing directory structure does not
close their physical, stability, gradient or order gates.

There is no established century reliability, independent climate/forecast
accuracy or superiority to an industrial model. This repository does not claim
an unverified acceleration. Read the [repair status](docs/legacy_core_repair_status_zh.md),
[delivery plan](docs/production_delivery_plan_zh.md),
[acceptance requirements](docs/century_climate_acceptance_zh.md), and the relevant
[research index](research/README.md) for evidence and remaining limitations.
Historical receipts and independent oracles remain bound to their original
protocols and commits; test counts do not override failed scientific gates.
