# Ocean Solver

A global finite-difference ocean model solving the hydrostatic primitive
equations on a lat-lon grid, written in JAX.

**This is the whole project.** One solver, one driver, one test suite:

| | |
| --- | --- |
| Solver core | `src/jax_solver_global.py` |
| Driver (CLI) | `src/run_long_integration_global.py` |
| Tests | `tests/` |
| Design rationale | `docs/decisions.md` (D1-D46) + `docs/README.md` |

Everything else is support: data loaders (`src/forcing.py`,
`src/wind_reanalysis.py`, `src/woa_data.py`), grid/bathymetry
(`src/grid.py`), configuration (`src/config.py`), scoring and fetching
utilities, plus two clearly-marked side directories —
`archive/regional/` (a retired regional spectral solver) and `docs/archive/`
(historical work logs). Neither is imported by the main line. FV/C-grid experiment
modules in `src/` are research references, not dependencies of the production driver.

## Production and Migration Status

The production CLI retains the original FD defaults. The active repair path
stays inside that same core: opt-in static nodal geometry -> actual barotropic
mean transport -> accepted tracer stages -> gated process-time repairs.
Candidates are selected explicitly in the API/smoke runner, not silently promoted
to production. See the [original-core repair plan](docs/legacy_core_repair_plan_zh.md)
and [registered results, compatibility and remaining failures](docs/legacy_core_repair_status_zh.md).

Independent local geometry, transport, rotation, convection and nonlinear time
checks do not qualify the whole model. Full-step gravity and wind/linear-drag
coupling still have first-order errors; complete moving heat/salt inventories,
versioned production restart and long-window climate/forecast validation remain open.
Real WOA/NCEP fixed-month one-day runs are numerical smoke, not forecast skill.
The [GPU environment and backend validation](docs/gpu_runtime_zh.md) likewise
do not establish industrial-model superiority or century reliability.

FV/C-grid work is a separate research-only alternative, not the current
production migration target. Its previous failed gates are retained in the
[nonlinear research evidence](research/experiments/cgrid_hydrostatic_momentum/nonlinear_dual_review.md)
and [paired/physical-frame history](research/experiments/cgrid_hydrostatic_momentum/paired_dynamics_review.md).
Those results must not be attributed to the original FD method or treated as
proof that replacing it is necessary.

## Features

- **Domain**: longitude-global lat-lon, truncated at default latitude
  +/-60 deg (`--lat-max`), 1 deg default resolution; not full polar coverage
- **Numerics**: conservative finite-difference horizontal operators
  (divergence/gradient are exact adjoints, spherical `cos(lat)` mass
  weighting); the barotropic mode is sub-cycled under a split-explicit
  scheme (`--mode-split`)
- **Time stepping**: split process updates and a JIT-compiled, `lax.scan`-based
  inner loop; local RK stages do not imply second-order accuracy of the full step
- **Advection**: flux-form tracer advection on the rigid-lid surface term;
  `--project-adv-vel` projects the stage-2 velocity column-divergence-free
  but does not close the whole-step heat/volume budget; `--monotone-adv` switches horizontal
  tracer fluxes to donor-cell (default is centered); `--fct-adv` enables an
  experimental TVD/MUSCL flux-limited horizontal transport (limited, but not
  yet a full Zalesak 3D FCT limiter)
- **Vertical mixing**: GM/Redi skew-flux (`--kappa-gm`, `--kappa-redi`),
  eddy viscosity (`--nu-h`), biharmonic (`--nu-bi`), convective adjustment
  (`--kappa-conv`), polar-edge Rayleigh sponge (`--sponge-days`)
- **Forcing**: NCEP/NCAR R1 reanalysis wind (`src/wind_reanalysis.py`),
  bulk air-sea heat flux, seasonal wind cycle (`--seasonal-wind`),
  WOA2023 initial fields (`src/woa_data.py`)
- **Grid**: real ETOPO2022 bathymetry with smoothing; `--min-depth`
  excludes shallower ocean cells

Mixed-layer heat deposition and dynamic ice are opt-in prototypes. Recent
budget, boundary, checkpoint, packaging and scoring fixes are documented in
[`docs/debug_validation_zh.md`](docs/debug_validation_zh.md), including the
validation limits. Numerical stability is not climate or forecast skill.

Shared SST scoring now uses wet-area weights and coordinate-based angular
windows (`area_weighted_angular_box_v2`), not equal-cell/index-window scores.
Comparison rejects different metric versions, domains or reference fields;
archived scores are not silently upgraded. Initial-reference and endpoint
content-change diagnostics are NOT independent climate or closed-budget
evidence. Current regression results are recorded in the repair status above;
test counts are not climate qualification.
See the [scoring protocol](docs/benchmark_protocol_zh.md) and
[century/climate acceptance requirements](docs/century_climate_acceptance_zh.md).

When enabling `--project-adv-vel`, set and verify `--projection-niter`,
`--projection-rtol`, `--projection-preconditioner` and
`--projection-max-refinements`. Effective settings are saved in run provenance.
CG's recursive residual can underestimate the actual residual; bounded native
transport correction now retains an original-RHS absolute stopping floor.
Legacy none/150 remains unqualified. See the
[actual residual review](research/experiments/projection_residual_control/review.md)
for measured gates and failures, not a universal configuration recommendation.
This is not whole-model conservation, all-grid convergence or century
qualification (D35-D36).

## Requirements

- Python 3.12+
- NumPy, SciPy, netCDF4
- JAX (GPU strongly recommended for production runs)

For Linux/WSL CUDA setup, see [the GPU runtime guide](docs/gpu_runtime_zh.md)
and `environment-gpu.yml`; native Windows JAX is CPU-only.

## Quick Start

```bash
pip install -e ".[dev]"          # installs the `ocean-solver` console script

python scripts/run_tests.py      # pytest suite (data-dependent tests skip)

# Example legacy CLI command, NOT one-year qualification or a general safe dt:
# mode split at dt=3600 -- 24 barotropic subcycles of
# 150 s. Without --mode-split the explicit free surface caps dt at 60 s.
ocean-solver --days 365 --dt 3600 \
  --mode-split --use-scan --seasonal-wind --wind-year 2023 \
  --resolution 1.0 --dtype float32 \
  --tag g365d --out-dir results --log-dir logs
```

`ocean-solver` and `python src/run_long_integration_global.py` are the same
entry point.

Bathymetry and WOA/Wind data are **not** in the repository. Point the solver
at the ETOPO2022 relief file with the `OCEAN_SOLVER_BATHYMETRY` environment
variable, or drop it in `data/`. Without it, grid construction fails with a
message naming that variable, and the data-dependent tests skip.

To exercise the grid / bathymetry / remap tests without the real file,
generate a clearly-labelled synthetic stand-in (it is **not** ETOPO and is
only for the test suite):

```bash
python scripts/make_synthetic_bathymetry.py   # -> data/ETOPO_..._surface.nc.npz
python -m pytest tests/ -q
```

## Resolution

By default `--resolution` accepts integer multiples of the 0.1 deg ETOPO
source grid. Add `--resolution-remap area` to build a conservative
spherical-area remapped grid at arbitrary positive spacings (e.g. 0.37 deg).
The time-step-sensitive physics parameters (`dt_bt`, `nu_h`, `nu_bi`) are
auto-scaled by the power of `dx` their CFL demands (`dx^1`, `dx^2`, `dx^4`),
but this does not guarantee whole-model stability on a finer grid: vertical,
advective, tracer and geometry-dependent bounds still need validation.
At `--resolution 1.0` (or
with no `--resolution`) the scaling is a no-op, preserving the 1 deg
defaults bit-for-bit.

Measured behavior across the resolution ladder, including the
`--project-adv-vel` closure for the flux-form surface leak, is in
[`docs/resolution_cfl_limits.md`](docs/resolution_cfl_limits.md).

## Project Structure

```
ocean-solver/
+-- src/                     # Core source (flat modules, installed as top-level py-modules)
|   +-- config.py            # GlobalGridConfig, PhysicsConfig, Config/DEFAULT_CONFIG
|   +-- grid.py              # global grid + ETOPO bathymetry
|   +-- jax_solver_global.py # global FD solver core (JAX JIT)  <- the main line
|   +-- run_long_integration_global.py  # long-run driver (CLI entry point)
|   +-- forcing.py           # wind stress / heat flux generators
|   +-- wind_reanalysis.py   # NCEP/NCAR R1 reanalysis wind loader
|   +-- woa_data.py          # WOA2023 climatological initial fields
|   +-- bench_climatology_global.py     # climatology scoring
|   +-- fetch_sla_monthly.py / fetch_ssh_abs_monthly.py  # altimetry fetchers
|   +-- erddap_fetch.py      # shared retry/backoff download helper for the above
+-- tests/                   # pytest suite
+-- docs/                    # Current mainline documentation (see docs/README.md)
+-- archive/
|   +-- regional/            # RETIRED regional spectral solver (not runnable, see its README)
+-- scripts/                 # run_tests.py, status_board.py, publish_dashboard.py,
|                            # make_synthetic_bathymetry.py
+-- dashboard/               # web run dashboard
+-- configs/                 # CDO target grid description
+-- results/                 # run outputs + figure scripts (gitignored, a few report figures are tracked)
+-- data/                    # bathymetry / climatology (gitignored, create it yourself)
+-- scratch/                 # ignored local artifacts, probes, downloads, slides
```

## Documentation

Index: [`docs/README.md`](docs/README.md). The mainline set:

- [`docs/decisions.md`](docs/decisions.md) — **why the solver looks like this** (D1-D38)
- [`docs/solver_technical_report_zh.md`](docs/solver_technical_report_zh.md) — solver technical report
- [`docs/resolution_cfl_limits.md`](docs/resolution_cfl_limits.md) — resolution limits and the CFL fix
- [`docs/deep-heat-poisoning-root-cause.md`](docs/deep-heat-poisoning-root-cause.md) — the column heat-leak root cause
- [`docs/hydrostatic_primitive_equations.md`](docs/hydrostatic_primitive_equations.md) — the equations
- [`docs/repositioning_memo_zh.md`](docs/repositioning_memo_zh.md) — project positioning

## Legacy: regional spectral solver

The original regional pseudospectral solver (FFT horizontal operators, IMEX
Strang splitting) is retired and lives in
[`archive/regional/`](archive/regional/README.md). It is **not runnable
as-is** — the shared config classes it depended on (`GridConfig`,
`TimeConfig`, `make_grid`) have been deleted. The current global FD solver
supersedes it entirely.

## Candidate diagnostic baseline

The current diagnostic baseline is the reproduced 65N, 0.7-degree candidate:

```bash
bash scripts/run_candidate_baseline.sh
```

Locked configuration: `lat-max=65`, 0.7-degree resolution, annual NCEP R1 2m
air forcing, `lambda_bulk=80`, `kappa_v=1e-6`, `kappa_conv=0.01`, localized
convective adjustment, `kappa_gm=0`, FCT/TVD transport, and projected stage-2
advective velocity. This is a diagnostic baseline, not yet a changed default.
Override days with `DAYS=30 bash scripts/run_candidate_baseline.sh`.
