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
- **Surface forcing**: 2D spatially-varying wind stress (Stommel gyre) and heat flux (meridional gradient)
- **Vertical mixing**: Constant eddy viscosity/diffusivity (`nu_v`, `kappa_v`)

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
