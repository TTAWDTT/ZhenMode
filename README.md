# Ocean Solver

A JAX ocean model solving the hydrostatic primitive equations on a
longitude-global, latitude-truncated finite-difference grid.

## Mainline: read these files first

```text
run_long_integration_global.main
  -> config + grid + WOA / atmospheric forcing
  -> jax_solver_global.make_solver_global
  -> jax_solver_global._step_impl
  -> diagnostics + snapshots + restart_contract
```

| Responsibility | Source |
| --- | --- |
| CLI, initialization, forcing schedule and output | `src/run_long_integration_global.py` |
| State, operators, split-explicit barotropic mode and complete step | `src/jax_solver_global.py` |
| Physical parameters and grid / bathymetry | `src/config.py`, `src/grid.py` |
| Wind, air and initial tracer data | `src/forcing.py`, `src/wind_reanalysis.py`, `src/air_reanalysis.py`, `src/woa_data.py` |
| Saved diagnostics, scoring and versioned restart | `src/diagnostics.py`, `src/benchmark_metrics.py`, `src/restart_contract.py` |
| Regression suite and numerical decision log | `tests/`, [D1–D46](docs/decisions.md) |

The CLI does **not** import `material_top.py`, the FV/C-grid alternatives or
the r-star experiments. They are explicitly selected research APIs, not
replacement production defaults.

## What is qualified, and what is not?

| Path | Current evidence | Boundary |
| --- | --- | --- |
| Production FD CLI | Existing solver, forcing, diagnostics and versioned restart | Numerical stop gates are not closed physical budgets or climate / forecast skill |
| `src/material_top.py` opt-in nodal candidate | Corrected 2-degree, float64, no-cap, fixed-January 600/300 s trials reach 30 days with moving-stock audits | 1-degree trials retain capacity and subsequent negative-top failures; hard-convection gradient and other gates remain open |
| Completed-bed r-star weak operators | Registered pressure work/rest and specified point-order controls; bounded content/source controls and full 1-degree prescribed tracer stages on CPU/CUDA | New local state semantics; not a solved ocean step, factory or restart migration |
| `weak_time.py` candidate | Smooth Fourier time order approximately 2 | Moving-limiter signed/pulse orders approximately 0.77/1.16 and 0.62/0.86 fail the 1.9 gate on both backends |
| FV/C-grid alternative | Separate retained research controls and failures | Not the current original-core repair path or a production dependency |

The [repair status](docs/legacy_core_repair_status_zh.md) is the detailed evidence
record; sections 36–37 cover the latest weak-content/capacity/time work.
The [delivery execution plan](docs/production_delivery_plan_zh.md) defines the
current order, objectives, dependencies and acceptance gates. The earlier
[repair plan](docs/legacy_core_repair_plan_zh.md) remains a historical reference.
The [S0/S1 execution record](docs/production_delivery_execution_20260930_zh.md)
records the input/first-rejection repairs, retained failures and verified scope.
The [r-star index](research/experiments/material_rstar_coordinates/README.md)
maps their code and audit commands. Test counts do not override failed gates.
No current evidence establishes century reliability, independent climate /
forecast accuracy or superiority to an industrial model; see the
[acceptance requirements](docs/century_climate_acceptance_zh.md) and
[industrial roadmap](docs/industrial_alignment_roadmap_zh.md).

## Capabilities

- Default horizontal resolution: 1 degree; default latitude limit: +/-60 degrees,
  not full polar coverage. Real ETOPO2022 supplies bathymetry.
- Conservative spherical FD operators, split-explicit barotropic subcycling
  (`--mode-split`), JAX JIT and scan loops. Local RK stages do not establish
  complete-step second-order accuracy.
- Tracers, momentum and free surface with wind / bulk heat forcing and optional
  seasonal cycles; WOA2023 supplies initial temperature/salinity.
- Configurable mixing, GM/Redi and polar-edge controls. Mixed-layer heat and
  dynamic ice remain opt-in prototypes; unsupported candidate processes are
  rejected rather than silently enabled.
- CPU and supported Linux/WSL CUDA execution. GPU availability is not a speed
  benchmark, and distributed / multi-GPU qualification is not established.

## Environment and commands

Python 3.12+, NumPy, SciPy, netCDF4 and JAX are required.
From the repository root:

```bash
pip install -e ".[dev]"
ocean-solver --help
python scripts/run_tests.py
```

`ocean-solver` and `python src/run_long_integration_global.py` use the same
entry point. Example CLI syntax, **not a qualified configuration**:

```bash
ocean-solver --days 1 --dt 60 --dtype float64 --use-scan \
  --tag smoke --out-dir results --log-dir logs
```

For native Windows CPU / Linux or WSL CUDA details, see
[GPU runtime and validation](docs/gpu_runtime_zh.md) and `environment-gpu.yml`.
Data-dependent tests may skip. Test commands are not physical integrations.

## Data, resolution and restart

Bathymetry and climatology / reanalysis files are not bundled. Set
`OCEAN_SOLVER_BATHYMETRY` or put the relief file in `data/`; missing relief
fails explicitly. A synthetic **test-only**, non-ETOPO stand-in is available:

```bash
python scripts/make_synthetic_bathymetry.py
python -m pytest tests/ -q
```

Default resolution accepts integer multiples of the 0.1-degree source grid;
`--resolution-remap area` permits arbitrary positive spacing. The driver scales
default `dt_bt/nu_h/nu_bi` with resolution, but vertical, advection and
geometry-dependent limits still require validation.
See [resolution evidence](docs/resolution_cfl_limits.md).
Projection parameters and actual residuals must be verified; see the
[projection review](research/experiments/projection_residual_control/review.md).

`--checkpoint-days` writes versioned full-state restarts and must align with
`--snap-days`. `--restart-from` verifies grid, parameters, forcing, source,
dtype/backend and retained output identity. Old state-only files are not verified
restarts. Use frozen original sources for historical replay, not current hashes.
See [restart and budget limits](docs/debug_validation_zh.md).

## Directory boundaries and reading order

| Directory | Role |
| --- | --- |
| `src/` | Installed flat modules: production core, support and explicitly opt-in candidates |
| `tests/` | Regression and independent numerical oracles; intentionally retained negative controls |
| `docs/` | Current documentation index, decisions, plans and evidence |
| `research/experiments/` | Isolated candidates / audits; use each experiment's README |
| `research/plan.md` | Historical research log, **not** the current execution plan |
| `archive/regional/`, `docs/archive/` | Retired solver and historical documents; not production imports |
| `scripts/`, `dashboard/`, `configs/` | Utilities, run dashboard and remap grid descriptions |
| `results/`, `data/`, `scratch/` | Mostly ignored evidence, inputs and local artifacts; do not delete failed witnesses as redundant code |

Read this entry map, then [decisions](docs/decisions.md),
[repair status](docs/legacy_core_repair_status_zh.md) and the relevant
[experiment index](research/experiments/material_rstar_coordinates/README.md).
[All documentation](docs/README.md) is indexed separately.
The historical 65N/0.7-degree command is in `scripts/run_candidate_baseline.sh`;
it remains a diagnostic recipe, not a default or qualification.
