# Ocean Solver

A spectral-method ocean model solving hydrostatic primitive equations on a regional domain.

## Features

- **Equations**: Hydrostatic primitive equations (momentum, continuity, tracer transport)
- **Numerics**: Pseudospectral method (FFT-based) for horizontal operators, finite-difference for vertical
- **Time stepping**: IMEX Strang splitting — linear terms (diffusion + Coriolis) via matrix exponential, nonlinear terms explicit
- **JAX acceleration**: JIT-compiled solver, ~12x faster than numpy/scipy backend
- **Grid**: 128x128 horizontal (0.1 deg resolution, ~9 km), 14 vertical z-levels (upper-ocean focused)
- **Domain**: NW Pacific subtropical region (lon 143.6E-156.3E, lat 28.6N-41.3N)
- **Bathymetry**: ETOPO2022, remapped via CDO

### Physics parameterizations

- **Equation of state**: Linear (default) or UNESCO nonlinear EOS (`eos_type='unesco'`)
- **Subgrid closure**: Smagorinsky nonlinear viscosity (`smag_cs`, default 0 = disabled)
- **Bottom friction**: Linear (default) or quadratic drag (`bottom_friction='quadratic'`, `cd=0.0025`)
- **Surface forcing**: 2D spatially-varying wind stress (idealized Stommel gyre, or real NOAA PSL NCEP/NCAR R1 10m reanalysis wind) and heat flux (meridional gradient)
- **Vertical mixing**: Constant eddy viscosity/diffusivity (`nu_v`, `kappa_v`)
- **Month-scale stability**: non-periodic forcing is tapered to zero at the periodic spectral y-seam, plus convective adjustment (`kappa_conv`) and optional Haney SST restoring (`tau_restore_days`) for stable multi-week forced runs

## Project Structure

```
ocean_solver/
+-- src/               # Core source code
|   +-- config.py      # GridConfig, PhysicsConfig, TimeConfig
|   +-- grid.py        # OceanGrid + ETOPO bathymetry reading
|   +-- state.py       # ModelState dataclass + initialization
|   +-- spectral_ops.py # FFT-based operators (derivatives, Poisson solver)
|   +-- jax_solver.py  # JAX JIT solver (primary, ~12x faster)
|   +-- integrator.py  # Numpy IMEX Strang splitting integrator
|   +-- momentum.py    # Momentum equation RHS (numpy)
|   +-- tracers.py     # Temperature & salinity transport (numpy)
|   +-- pressure.py    # Pressure gradient & hydrostatic balance
|   +-- eos.py         # Linear + UNESCO equation of state
|   +-- forcing.py     # 2D wind stress and heat flux field generators
|   +-- wind_reanalysis.py  # real NOAA PSL NCEP/NCAR R1 10m wind stress loader
|   +-- woa_data.py    # WOA2023 climatological initial fields
|   +-- verify_real_run.py          # short forced stability check
|   +-- verify_real_run_long.py     # multi-day/month stability + pass criterion
|   +-- verify_real_wind.py         # real-wind smoke test
|   +-- test_wave_speed.py    # Wave speed validation (c = sqrt(gH))
|   +-- test_forcing.py       # Ekman transport + Sverdrup response tests
|   +-- compare_jax_numpy.py  # JAX vs numpy agreement check
+-- scripts/           # Entry point scripts
|   +-- run_tests.py          # Unified test runner
|   +-- run_wind_test.py      # Wind forcing experiment
|   +-- run_1day.py           # Smoke test
|   +-- diagnose_stability.py # Stability diagnostics
+-- tests/             # Unit tests (pytest)
+-- configs/           # Configuration files
|   +-- target_grid.txt # CDO target grid description
+-- docs/              # Documentation
+-- data/              # Data files (gitignored)
```

## Requirements

- Python 3.12+
- NumPy, SciPy, netCDF4
- JAX (CPU backend)
- CDO 2.5+ (via WSL on Windows, for grid remapping)

## Quick Start

```bash
# Install (editable)
pip install -e .

# Run all tests
python scripts/run_tests.py

# Run individual tests
python src/test_wave_speed.py      # Surface gravity wave speed
python src/test_forcing.py         # Ekman transport + Sverdrup response
python src/compare_jax_numpy.py    # JAX vs numpy agreement

# JAX solver benchmark
python src/jax_solver.py
```

## Month-Scale Stable Forced Runs

Sustained wind + heat forcing exposes a spectral boundary artifact: the domain is
periodic in both x and y (FFT derivatives), but realistic forcing is **non-periodic
in y**, creating a step discontinuity at the meridional seam that pumps spurious
grid-scale energy at the boundary. Three combined fixes keep a forced run stable
over a month (`verify_real_run_long.py`):

1. **Forcing taper** — non-periodic wind/heat fields are tapered to zero over the
   `forcing_taper_cells` edge cells (`_taper_y` / `taper_2d_y` in `forcing.py`), so
   they are continuous across the periodic seam.
2. **Convective adjustment** (`kappa_conv`, on by default) — statically unstable
   columns are mixed vertically.
3. **Haney SST restoring** (`--restore-days`, working value 5 d) — relaxes the surface
   temperature toward the initial WOA field to anchor a stable forced equilibrium.

```bash
# Real NOAA PSL NCEP/NCAR R1 10m wind + heat, 30 days, SST restoring 5 d
python src/verify_real_run_long.py --days 30 --restore-days 5 --real-wind

# Idealized Stommel gyre wind + heat, 30 days, SST restoring 5 d
python src/verify_real_run_long.py --days 30 --restore-days 5
```

**Pass criterion (forced-equilibrium bar).** A continuously heated forced ocean
legitimately equilibrates a few °C above the initial SST, so "no drift vs initial
SST" is not the right bar. Instead the run must show **convergence, not accumulation**:

- no NaN / Inf in u, v, T;
- max|u| stays bounded (< 10 m/s);
- `monotonic_drift` false — the max-T over the final quarter does not climb > 2 °C
  above the first half;
- `amplitude_bounded` true — final max-T stays within a 12 °C absolute ceiling above
  the initial max (catches only a true runaway hotspot).

Both the idealized-gyre run (step a) and the real-wind run (step b) pass this bar,
reaching a stable ~30-day forced equilibrium.

## Configuration

All physics parameters are in `src/config.py` as frozen dataclasses:

```python
from config import DEFAULT_CONFIG
import dataclasses

# Enable all features
physics = dataclasses.replace(
    DEFAULT_CONFIG.physics,
    eos_type='unesco',
    smag_cs=0.1,
    bottom_friction='quadratic',
    cd=0.0025,
)
```
