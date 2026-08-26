"""Probe: what drives the polar bottom blow-up? Decompose the first-step tendency.

max|u| grows 0 -> 1.65 -> 3.6 -> 5.3 m/s at lat~82 iz=13 (bottom), independent
of nu_h. No wind forcing. So u is driven by the pressure gradient force (PGF).
Initial eta=0, so PGF = baroclinic (density) PGF only.

This script computes each tendency term separately at the initial state and
reports where the largest du/dt is, to find what's injecting energy at the pole.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_global_grid, GlobalGridConfig
from woa_data import get_initial_fields
import jax_solver_global as G
from jax_solver_global import (
    JaxStateG, _compute_pressure_gradient, _compute_bt_rho_pgf,
    _compute_vertical_velocity, _advection_flux_form, _coriolis_rotation_2d,
    _d_dx, _d_dy, _laplacian_h, _d2_dz2,
)

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"


def main():
    print("Building real global grid...")
    grid = make_global_grid(GlobalGridConfig(), BATHY)
    T_init, S_init = get_initial_fields(grid)
    phys = DEFAULT_CONFIG.physics
    step, init_state, diag = G.make_solver_global(grid, phys, 60.0, forcing=None,
                                                   eos_type='linear')
    state = init_state(T_init, S_init)
    p = G.make_solver_global.__wrapped__ if hasattr(G.make_solver_global, '__wrapped__') else None

    # Rebuild the FDPhysParams the solver uses (make_solver_global builds it
    # internally; replicate via the public builder path by calling make_solver
    # and pulling params from a closure is not possible. Instead reconstruct
    # the params struct the same way make_solver_global does.)
    base = G.make_fd_params(grid)
    from jax_solver_global import FDPhysParams
    nx, ny, nz = base.nx, base.ny, base.nz
    H_sw = float(jnp.sum(jnp.array(grid.dz)))
    dz_norm = (jnp.array(grid.dz).reshape(1, 1, -1) / H_sw)
    params = FDPhysParams(
        dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
        inv_dx=base.inv_dx, inv_dy=base.inv_dy,
        inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
        f=base.f, wet_mask=base.wet_mask, wet_mask_3d=base.wet_mask_3d,
        dz_denom_interior=base.dz_denom_interior,
        dz_bnd_top=base.dz_bnd_top, dz_bnd_bot=base.dz_bnd_bot,
        d2z_hm=base.d2z_hm, d2z_hp=base.d2z_hp, d2z_denom=base.d2z_denom,
        d2z_h0_top=base.d2z_h0_top, d2z_h0_bot=base.d2z_h0_bot,
        dz_3d=base.dz_3d, dz_surface=base.dz_surface,
        surface_mask=base.surface_mask, bottom_mask=base.bottom_mask,
        nx=nx, ny=ny, nz=nz,
        nu_h=phys.nu_h, nu_v=phys.nu_v, kappa_h=phys.kappa_h, kappa_v=phys.kappa_v,
        kappa_conv=phys.kappa_conv, nu_bi=phys.nu_bi, kappa_bi=phys.kappa_bi,
        T_ref=phys.T_ref, S_ref=phys.S_ref, eos_type='linear',
        r_bot=phys.r_bot, cd=phys.cd, bottom_friction=phys.bottom_friction,
        tau_x_2d=jnp.zeros((nx, ny)), tau_y_2d=jnp.zeros((nx, ny)),
        Q_heat_2d=jnp.zeros((nx, ny)),
        H_sw=H_sw, dz_norm=dz_norm, dt=60.0,
        T_atm_3d=jnp.zeros((nx, ny, 1)), lambda_bulk=0.0,
    )

    # Full momentum tendency decomposition at init state.
    w = _compute_vertical_velocity(state, params)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, params)   # 0 (u=v=0)
    f_3d = params.f[:, :, None]
    cor_u = f_3d * state.v      # 0 (v=0)
    cor_v = -f_3d * state.u     # 0 (u=0)
    pgf_x, pgf_y = _compute_pressure_gradient(state, params)
    diff_u = params.nu_h * _laplacian_h(state.u, params)   # 0 (u=0)
    bot_u = -params.r_bot * state.u * params.bottom_mask   # 0

    def loc(a, name):
        a = np.array(a)
        am = np.abs(a)
        idx = np.unravel_index(np.argmax(am), am.shape)
        print(f"  {name:18s} max|.|={am.max():.3e}  at ix,iy,iz={idx} "
              f"(lat={grid.lat[idx[1]]:.1f}, iz={idx[2]})")
        return idx

    print("\n--- tendency decomposition at init (u=v=0, eta=0) ---")
    loc(adv_u, "adv_u")
    loc(cor_u, "cor_u")
    loc(pgf_x, "pgf_x")
    loc(diff_u, "diff_u")
    loc(bot_u, "bot_u")
    print("\n--- vertical profile of pgf_x at the hot column ---")
    # Find the global max pgf location, print its column vertical profile.
    pgx = np.array(pgf_x)
    am = np.abs(pgx)
    ix, iy, iz = np.unravel_index(np.argmax(am), am.shape)
    print(f"  hottest pgf_x at ix={ix} lon={grid.lon[ix]:.1f}, lat={grid.lat[iy]:.1f}")
    print(f"  T column: {np.array(state.T)[ix, iy, :]}")
    print(f"  pgf_x column: {pgx[ix, iy, :]}")
    print(f"  |pgf_x|*dt (60s) -> u increment: {am[ix,iy,iz]*60:.3f} m/s at iz={iz}")

    # Barotropic density PGF (used in free surface)
    bt_x, bt_y = _compute_bt_rho_pgf(state, params)
    print(f"\n  barotropic rho PGF: max|bt_x|={np.abs(np.array(bt_x)).max():.3e}, "
          f"max|bt_y|={np.abs(np.array(bt_y)).max():.3e}")


if __name__ == "__main__":
    main()
