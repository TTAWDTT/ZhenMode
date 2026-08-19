# Ocean Solver

A spectral-method ocean model solving hydrostatic primitive equations on a regional domain.

## Features

- **Equations**: Hydrostatic primitive equations (momentum, continuity, tracer transport)
- **Numerics**: Pseudospectral method (FFT-based) for horizontal operators, finite-difference for vertical
- **Time stepping**: IMEX Strang splitting — linear terms (diffusion + Coriolis) via matrix exponential, nonlinear terms explicit
- **Grid**: 128×128 horizontal (0.1° resolution, ~9 km), 14 vertical z-levels (upper-ocean focused)
- **Domain**: NW Pacific subtropical region (lon 143.6°E–156.3°E, lat 28.6°N–41.3°N)
- **Bathymetry**: ETOPO2022, remapped via CDO
- **Performance**: ~542K points/s/core on single CPU (NumPy), ~4× faster than Veros/NumPy

## Project Structure

```
ocean_solver/
├── src/               # Core source code
│   ├── config.py      # GridConfig, PhysicsConfig, TimeConfig
│   ├── grid.py        # OceanGrid + ETOPO bathymetry reading
│   ├── state.py       # ModelState dataclass + initialization
│   ├── spectral_ops.py # FFT-based operators (derivatives, Poisson solver, etc.)
│   ├── integrator.py  # IMEX Strang splitting time integrator
│   ├── momentum.py    # Momentum equation RHS
│   ├── tracers.py     # Temperature & salinity transport
│   ├── pressure.py    # Pressure gradient & hydrostatic balance
│   └── eos.py         # Linearized equation of state
├── scripts/           # Entry point scripts
│   ├── run_wind_test.py       # Wind forcing experiment
│   ├── run_1day.py            # Smoke test
│   └── diagnose_stability.py  # Stability diagnostics
├── tests/             # Unit tests
├── configs/           # Configuration files
│   └── target_grid.txt # CDO target grid description
├── docs/              # Documentation
└── data/              # Data files (gitignored)
```

## Requirements

- Python 3.12+
- NumPy
- netCDF4
- CDO 2.5+ (via WSL on Windows, for grid remapping)

## Usage

```bash
# Run wind forcing test
python scripts/run_wind_test.py

# Run from project root
python -m src.grid        # Print grid info
python -m src.config      # Print configuration
```
