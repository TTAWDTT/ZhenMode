# Ocean Solver

A global finite-difference ocean model solving the hydrostatic primitive
equations on a lat–lon grid, written in JAX.

## Features

- **Domain**: global, lat ±60°, adjustable horizontal resolution (`--resolution`)
- **Numerics**: conservative finite-difference horizontal operators; the
  barotropic mode is sub-cycled under a split-explicit scheme (`--mode-split`)
- **Time stepping**: RK2 with a JIT-compiled, `lax.scan`-based inner loop
- **Advection**: flux-form tracer advection on the rigid-lid surface term;
  `--project-adv-vel` projects the stage-2 velocity column-divergence-free
  to close the column heat budget; `--monotone-adv` switches horizontal
  tracer fluxes to donor-cell (the default remains centered for legacy runs)
- **Vertical mixing**: GM/Redi skew-flux (κ_gm), eddy viscosity (`nu_h`),
  biharmonic (`nu_bi`), convective adjustment (`kappa_conv`)
- **Forcing**: NCEP/NCAR R1 reanalysis wind (`wind_reanalysis.py`), bulk
  heat flux, seasonal wind cycle, WOA2023 initial fields (`woa_data.py`)
- **Grid**: real ETOPO2022 bathymetry with smoothing and a `min_depth` floor

## Project Structure

```
ocean_solver/
+-- src/                     # Core source code (production layout)
|   +-- config.py            # GridConfig, GlobalGridConfig, PhysicsConfig, TimeConfig
|   +-- grid.py              # global grid + ETOPO bathymetry
|   +-- jax_solver_global.py # global FD solver core (JAX JIT)
|   +-- run_long_integration_global.py  # long-run driver (CLI entry point)
|   +-- forcing.py           # wind stress / heat flux generators
|   +-- wind_reanalysis.py   # NCEP/NCAR R1 reanalysis wind loader
|   +-- woa_data.py          # WOA2023 climatological initial fields
|   +-- bench_climatology_global.py     # climatology scoring
|   +-- fetch_sla_monthly.py / fetch_ssh_abs_monthly.py  # altimetry fetchers
|   +-- archive_regional/    # retired regional spectral solver (see its README)
+-- scripts/                 # Entry point scripts
|   +-- run_tests.py         # pytest suite runner
|   +-- status_board.py      # regenerates status_zh.md from the RUNS registry
|   +-- publish_dashboard.py # publishes the dashboard
+-- tests/                   # pytest suite
+-- dashboard/               # web run dashboard
+-- configs/                 # CDO target grid description
+-- docs/                    # Documentation
+-- data/                    # Data files (gitignored)
```

## Requirements

- Python 3.12+
- NumPy, SciPy, netCDF4
- JAX (GPU strongly recommended for production runs)

## Quick Start

```bash
pip install -e .

# Run the test suite
python scripts/run_tests.py

# Global smoke / production run
python src/run_long_integration_global.py --days 365 --dt 60 \
  --mode-split --use-scan --seasonal-wind --wind-year 2023 \
  --resolution 1.0 --dtype float32 \
  --tag g365d --out-dir results --log-dir logs
```

## Resolution

By default `--resolution` accepts integer multiples of the 0.1° ETOPO source
grid, preserving historical block-averaged grids. Add
`--resolution-remap area` to build a conservative spherical-area remapped grid
at arbitrary positive spacings (for example 0.37° or 0.85°). The
time-step-sensitive physics parameters (`dt_bt`, `nu_h`, `nu_bi`) are
auto-scaled by the appropriate power of `dx` (`dx^1`, `dx^2`, `dx^4`), so a
smaller grid stays stable without hand-tuning. At `--resolution 1.0` the
scaling is a no-op.

Measured behavior across the resolution ladder, including the `--project-adv-vel`
closure for the flux-form surface leak, is in
[`docs/resolution_cfl_limits.md`](docs/resolution_cfl_limits.md).

## Legacy: regional spectral solver

The original regional pseudospectral solver (FFT horizontal operators, IMEX
Strang splitting) has been retired and moved to
[`src/archive_regional/`](src/archive_regional/README.md). The current global
FD solver supersedes it. Background on the repositioning is in
[`docs/repositioning_memo_zh.md`](docs/repositioning_memo_zh.md).

## Documentation

- [`docs/solver_technical_report_zh.md`](docs/solver_technical_report_zh.md) — solver technical report
- [`docs/resolution_cfl_limits.md`](docs/resolution_cfl_limits.md) — resolution limits and the CFL fix
- [`docs/deep-heat-poisoning-root-cause.md`](docs/deep-heat-poisoning-root-cause.md) — the column heat-leak root cause
- [`docs/hydrostatic_primitive_equations.md`](docs/hydrostatic_primitive_equations.md) — the equations
