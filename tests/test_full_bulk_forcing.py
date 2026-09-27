import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import all_wet_grid

from config import PhysicsConfig
from full_bulk_forcing import (
    build_full_bulk_fields,
    net_surface_heat_flux,
    saturation_specific_humidity_kg_kg,
    validate_full_bulk_forcing_grid,
)
from jax_solver_global import JaxStateG, _compute_tracer_tendency, make_solver_global


def _solver(fields):
    grid = all_wet_grid(nx=12, ny=8, nz=4)
    physics = replace(
        PhysicsConfig(), nu_h=0.0, nu_bi=0.0, nu_v=0.0,
        kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
        kappa_gm=0.0, kappa_redi=0.0,
    )
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing,
        T_atm=np.full((grid.nx, grid.ny), 20.0), lambda_bulk=0.0,
        full_bulk=True,
        downward_shortwave=fields["downward_shortwave_w_m2"],
        downward_longwave=fields["downward_longwave_w_m2"],
        specific_humidity_air=fields["specific_humidity_air_kg_kg"],
        wind_speed=fields["wind_speed_m_s"],
        precipitation_rate=fields["precipitation_rate_kg_m2_s"],
        transfer_coefficient=1.3e-3,
        T_init=np.full((grid.nx, grid.ny, grid.nz), 20.0),
        S_init=np.full((grid.nx, grid.ny, grid.nz), 35.0),
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype="float64", return_params=True)
    return grid, physics, params


def test_full_bulk_fields_and_saturation_humidity_contract():
    shape = (12, 3, 4)
    fields = build_full_bulk_fields(
        np.full(shape, 20.0), np.full(shape, 0.01),
        np.full(shape, 300.0), np.full(shape, 400.0),
        np.full(shape, 1e-5), np.full(shape, 8.0),
    )
    assert fields["sensible_transfer_w_m2_k"].shape == (12, 3, 4)
    assert fields["latent_transfer_w_m2_kgkg"].shape == (12, 3, 4)
    q_sat = saturation_specific_humidity_kg_kg(20.0)
    assert q_sat == pytest.approx(0.01447, abs=2e-4)


def test_full_bulk_solver_warms_and_salts_surface():
    shape = (12, 8)
    fields = {
        "downward_shortwave_w_m2": np.full(shape, 400.0),
        "downward_longwave_w_m2": np.full(shape, 300.0),
        "specific_humidity_air_kg_kg": np.full(shape, 0.005),
        "wind_speed_m_s": np.full(shape, 5.0),
        "precipitation_rate_kg_m2_s": np.zeros(shape),
    }
    grid, _, params = _solver(fields)
    state = JaxStateG(
        jnp.zeros((grid.nx, grid.ny, grid.nz)),
        jnp.zeros((grid.nx, grid.ny, grid.nz)),
        jnp.full((grid.nx, grid.ny, grid.nz), 20.0),
        jnp.full((grid.nx, grid.ny, grid.nz), 35.0),
        jnp.zeros((grid.nx, grid.ny)),
        jnp.zeros((grid.nx, grid.ny)))
    dTdt, dSdt = _compute_tracer_tendency(state, params)
    assert float(jnp.max(dTdt[:, :, 0])) > 0.0
    assert float(jnp.max(np.abs(dTdt[:, :, 1:]))) == 0.0
    assert float(jnp.min(dSdt[:, :, 0])) > 0.0

def test_full_bulk_dynamic_step_updates_surface_forcing():
    shape = (12, 8)
    fields = {
        "downward_shortwave_w_m2": np.full(shape, 400.0),
        "downward_longwave_w_m2": np.full(shape, 300.0),
        "specific_humidity_air_kg_kg": np.full(shape, 0.005),
        "wind_speed_m_s": np.full(shape, 5.0),
        "precipitation_rate_kg_m2_s": np.zeros(shape),
    }
    grid = all_wet_grid(nx=12, ny=8, nz=4)
    physics = replace(
        PhysicsConfig(), nu_h=0.0, nu_bi=0.0, nu_v=0.0,
        kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
        kappa_gm=0.0, kappa_redi=0.0,
    )
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, init_fn, _, _, _, step_dyn = make_solver_global(
        grid, physics, 60.0, forcing=forcing,
        T_atm=np.full((grid.nx, grid.ny), 20.0), lambda_bulk=0.0,
        full_bulk=True,
        downward_shortwave=np.full((grid.nx, grid.ny), 400.0),
        downward_longwave=np.full((grid.nx, grid.ny), 300.0),
        specific_humidity_air=np.full((grid.nx, grid.ny), 0.005),
        wind_speed=np.full((grid.nx, grid.ny), 5.0),
        precipitation_rate=np.zeros((grid.nx, grid.ny)),
        transfer_coefficient=1.3e-3,
        T_init=np.full((grid.nx, grid.ny, grid.nz), 20.0),
        S_init=np.full((grid.nx, grid.ny, grid.nz), 35.0),
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype="float64", return_params=True,
        dynamic_forcing=True)
    state = init_fn(T_init=jnp.full((grid.nx, grid.ny, grid.nz), 20.0),
                    S_init=jnp.full((grid.nx, grid.ny, grid.nz), 35.0))
    full_bulk_fields = {
        "downward_shortwave_2d": jnp.full((grid.nx, grid.ny), 400.0),
        "downward_longwave_2d": jnp.full((grid.nx, grid.ny), 300.0),
        "specific_humidity_air_2d": jnp.full((grid.nx, grid.ny), 0.005),
        "sensible_transfer_2d": jnp.zeros((grid.nx, grid.ny)),
        "latent_transfer_2d": jnp.zeros((grid.nx, grid.ny)),
        "precipitation_rate_2d": jnp.zeros((grid.nx, grid.ny)),
    }
    warm = step_dyn(
        state, jnp.zeros((grid.nx, grid.ny)), jnp.zeros((grid.nx, grid.ny)),
        jnp.zeros((grid.nx, grid.ny)),
        T_atm_3d=jnp.full((grid.nx, grid.ny, 1), 20.0),
        full_bulk_fields=full_bulk_fields)
    assert float(jnp.max(warm.T[:, :, 0])) > 20.0

def test_full_bulk_forcing_grid_contract_matches_solver_grid():
    grid = all_wet_grid(nx=12, ny=8, nz=4)
    forcing = {
        "lat": grid.lat,
        "lon": grid.lon,
        "wet_mask": np.ones((grid.nx, grid.ny), dtype=bool),
    }
    validate_full_bulk_forcing_grid(forcing, grid)

    forcing["wet_mask"] = np.zeros_like(forcing["wet_mask"])
    with pytest.raises(ValueError, match="wet mask differs"):
        validate_full_bulk_forcing_grid(forcing, grid)
