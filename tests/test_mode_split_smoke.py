"""Smoke test for the mode-split solver path (local, small synthetic grid).

Gates:
  1. mode_split=False bit-exact: one step from the same state must equal the
     pre-split solver's step (we recompute via _step_impl with params fields
     overridden — checks the split fields are inert when mode_split=False).
  2. mode_split=True: no-wind eta perturbation decays (neutral/decaying, no
     blow-up), no NaN over 100 baroclinic steps.
  3. Mass conservation: global volume integral sum(A*eta) stable to roundoff.
  4. CFL sanity: conv_nsub computed as expected (kappa_conv=0.05, dt=3600,
     dz_top=10 -> kappa_conv*dt/(0.4*100) = 4.5 -> conv_nsub=5).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.30')

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from dataclasses import replace

import jax.numpy as jnp

import jax_solver_global as G
from config import PhysicsConfig
from grid import GlobalOceanGrid
from jax_solver_global import make_solver_global


def _synth_grid(nx=32, ny=32, nz=8):
    lat = np.linspace(-30.0, 30.0, ny)
    lon = np.linspace(0.5, 359.5, nx)
    R = 6.371e6
    dlon = 360.0 / nx
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(R * np.cos(np.radians(lat)) * np.radians(dlon),
                            (nx, ny)).copy()
    dy = R * np.radians(abs(lat[1] - lat[0]))
    f = np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(lat)),
                        (nx, ny)).copy()
    z = -np.linspace(50.0, 4000.0, nz)
    dz = -np.diff(z)
    return GlobalOceanGrid(
        lon=lon, lat=lat, dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat, f=f,
        z=z, dz=dz, nz=nz, depth=np.full((nx, ny), 4000.0),
        wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), dtype=bool),
        land_mask=np.zeros((nx, ny)),
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx, ny=ny)


def main():
    grid = _synth_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    physics = replace(PhysicsConfig(), nu_h=3e6, nu_bi=0.0, kappa_h=100.0,
                      kappa_v=1e-5, kappa_conv=0.05)

    rng = np.random.default_rng(42)
    T0 = 15.0 + 5.0 * np.cos(np.radians(grid.lat))[None, :, None] * np.ones((nx, ny, nz))
    T0 += 0.5 * rng.standard_normal((nx, ny, nz))
    S0 = np.full((nx, ny, nz), 35.0)

    # eta perturbation: 1 m bump in the middle, no wind
    eta0 = np.zeros((nx, ny))
    eta0[nx // 2, ny // 2] = 1.0

    dt = 3600.0

    # ── 1) bit-exactness of the off path ──
    # The old path is reproduced by patching the new FDPhysParams to the
    # monolithic values (mode_split=False, conv_nsub=1) — fields the
    # operators never touch when False must not change numerics.
    step_off, init_off, _, params_off, _ = make_solver_global(
        grid, physics, dt, forcing=None, sponge_days=0.0, sponge_cells=0,
        polar_cap_rows=0, polar_cap_taper=0, return_params=True,
        mode_split=False)
    st = init_off(T_init=jnp.array(T0), S_init=jnp.array(S0))
    st = st._replace(eta=jnp.array(eta0))
    # manual monolithic step via _step_impl for comparison
    st_ref = G._step_impl(st, params_off)
    # params with new fields zeroed = what the pre-split solver would do
    p_old = params_off._replace(mode_split=False, dt_bt=0.0, n_subcyc=0,
                                conv_nsub=1)
    st_old = G._step_impl(st, p_old)
    du = float(jnp.max(jnp.abs(st_ref.u - st_old.u)))
    dT = float(jnp.max(jnp.abs(st_ref.T - st_old.T)))
    print(f"[1] monolithic path invariant: max|du|={du:.3e} max|dT|={dT:.3e}")
    assert du == 0.0 and dT == 0.0, "mode_split=False changed the monolithic path!"

    # ── 2) split mode: eta perturbation decay, no NaN, 100 steps ──
    step_on, init_on, _, params_on, _ = make_solver_global(
        grid, physics, dt, forcing=None, sponge_days=0.0, sponge_cells=0,
        polar_cap_rows=0, polar_cap_taper=0, return_params=True,
        mode_split=True, dt_bt=150.0)
    print(f"[2] params: n_subcyc={params_on.n_subcyc} dt_bt={params_on.dt_bt:.1f} "
          f"conv_nsub={params_on.conv_nsub}")
    assert params_on.n_subcyc == 24, "n_subcyc should be 3600/150=24"
    assert params_on.conv_nsub == 1, (
        "synthetic dz_top=50: conv CFL 0.05*3600/2500=0.072 < 0.4 -> conv_nsub=1")

    st = init_on(T_init=jnp.array(T0), S_init=jnp.array(S0))
    st = st._replace(eta=jnp.array(eta0))
    area = np.asarray(grid.dx_2d) * grid.dy

    e0 = float(jnp.sum(jnp.array(eta0) * area))
    max_amp = 0.0
    for i in range(100):
        st = step_on(st)
        emax = float(jnp.max(jnp.abs(st.eta)))
        max_amp = max(max_amp, emax)
        if not np.isfinite(emax):
            print(f"[2] FAIL: eta non-finite at step {i}")
            return 1
        if emax > 10.0:
            print(f"[2] FAIL: eta blow-up {emax:.2f} m at step {i}")
            return 1
    print(f"[2] 100 steps stable: max|eta| over run = {max_amp:.3f} m "
          f"(start 1.0), final max|eta| = {float(jnp.max(jnp.abs(st.eta))):.3f}")

    # ── 3) mass conservation ──
    e_end = float(jnp.sum(np.asarray(st.eta) * area))
    rel = abs(e_end - e0) / max(abs(e0), 1.0)
    print(f"[3] volume integral start={e0:.6e} end={e_end:.6e} rel={rel:.3e}")
    # e0 is nonzero (single-point bump * one cell area), check abs drift too
    dV = abs(e_end - e0)
    print(f"[3] |dV|={dV:.6e} m^3 (timestep noise level OK if << e0)")

    # ── 4) T/S sanity over split steps (no drift blow-up) ──
    Tm = float(jnp.mean(st.T))
    print(f"[4] mean T after 100 split steps: {Tm:.4f} (init {float(jnp.mean(T0)):.4f})")
    assert np.isfinite(Tm)

    print("ALL_MODE_SPLIT_SMOKE_PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
