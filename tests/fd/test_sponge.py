"""Regression tests for the polar-edge lateral sponge.

The sponge is a Rayleigh damping applied in the linear half-step. Two
invariants matter and both used to be violated:

1. **Land / below-seafloor ghost keeps its pre-step value.** The diffusion
   mask restores the sentinel ``T_ref``/``S_ref`` on dry cells; the sponge
   must not drag it back off. It did, because the relaxation target was the
   raw unmasked ``T_init`` (whose land entries are arbitrary WOA garbage).
   Regression: ``test_sponge_preserves_land_and_ghost_sentinel``.

2. **No initial field must not mean "relax to zero".** With ``T_init=None``
   ``init_state`` fills ``T_ref``, so the target has to be ``T_ref`` too.
   Regression: ``test_sponge_without_init_field_keeps_reference_state``.

And the sponge must stay a strict no-op when it is switched off
(``sponge_days=0``), which is the production default.

Uses a small synthetic GlobalOceanGrid (no real data / bathymetry).
Run:  python -m pytest tests/test_sponge.py -v
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from dataclasses import replace

import jax.numpy as jnp

import zhenmode.model.solver.timestepping.step as G_processes
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.geometry.grid import GlobalOceanGrid

T_REF = PhysicsConfig().T_ref      # 15.0
S_REF = PhysicsConfig().S_ref      # 35.0

LAND_T = 999.0                     # deliberately absurd, so any leak is loud
LAND_S = 777.0

DT_HALF = 1800.0


def _synth_grid(nx=24, ny=20, nz=6):
    """Global grid with a land block and a shallow patch (ghost layers)."""
    lat = np.linspace(-60.0, 60.0, ny)
    lon = np.linspace(0.5, 359.5, nx)
    R = 6.371e6
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(R * cos_lat * np.radians(360.0 / nx),
                            (nx, ny)).copy()
    dy = R * np.radians(abs(lat[1] - lat[0]))
    f = np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(lat)),
                        (nx, ny)).copy()
    z = -np.linspace(50.0, 4000.0, nz)
    dz = -np.diff(z)

    wet = np.ones((nx, ny))
    wet[5:9, :] = 0.0                  # land block
    wet[0, 0] = 0.0                    # single land cell
    depth = np.where(wet > 0.0, 4000.0, 0.0)
    depth[12:16, 4:8] = 1500.0         # shallow patch -> ghost layers below

    wet3 = np.zeros((nx, ny, nz))
    for k in range(nz):
        wet3[:, :, k] = (np.abs(z[k]) <= depth).astype(float) * wet

    return GlobalOceanGrid(
        lon=lon, lat=lat, dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat, f=f,
        z=z, dz=dz, nz=nz, depth=depth, wet_mask=wet,
        ocean_mask=wet > 0.0, land_mask=1.0 - wet, wet_mask_3d=wet3,
        nx=nx, ny=ny)


def _init_fields(grid):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    T = np.full((nx, ny, nz), 10.0)
    T[:, :, 0] = 25.0
    T[:, :, -1] = 2.0
    T[grid.wet_mask == 0.0] = LAND_T
    S = np.full((nx, ny, nz), S_REF)
    S[grid.wet_mask == 0.0] = LAND_S
    return T, S


def _build(grid, T0, S0, sponge_days, sponge_cells):
    """Solver with every dissipation switch off, so only the sponge acts."""
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_v=0.0, kappa_h=0.0,
                      kappa_v=0.0, nu_bi=0.0, kappa_bi=0.0, kappa_conv=0.0)
    nx, ny = grid.nx, grid.ny
    zeros = np.zeros((nx, ny))
    _, init_fn, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=(zeros, zeros, zeros),
        T_atm=None, lambda_bulk=0.0, sponge_days=sponge_days,
        sponge_cells=sponge_cells, T_init=T0, S_init=S0,
        polar_cap_rows=0, polar_cap_taper=0, mode_split=True,
        return_params=True)
    state = init_fn(T_init=None if T0 is None else jnp.array(T0),
                    S_init=None if S0 is None else jnp.array(S0))
    return state, params


def _half_steps(state, params, n=5):
    for _ in range(n):
        state = G_processes._linear_half_step(state, params, DT_HALF)
    return state


# ── tests ──────────────────────────────────────────────────────────

def test_sponge_off_is_a_noop():
    """sponge_days=0 (production default) must leave the state untouched."""
    g = _synth_grid()
    T0, S0 = _init_fields(g)
    state, params = _build(g, T0, S0, sponge_days=0.0, sponge_cells=0)
    out = _half_steps(state, params, n=5)
    for name in ("T", "S", "u", "v"):
        a = np.asarray(getattr(state, name))
        b = np.asarray(getattr(out, name))
        assert np.array_equal(a, b), f"{name} changed with the sponge off"
    print("  [PASS] sponge_days=0 is an exact no-op")


def test_sponge_preserves_land_and_ghost_sentinel():
    """Land and below-seafloor ghost must stay at T_ref / S_ref."""
    g = _synth_grid()
    T0, S0 = _init_fields(g)
    state, params = _build(g, T0, S0, sponge_days=5.0, sponge_cells=4)
    out = _half_steps(state, params, n=5)

    dry = g.wet_mask == 0.0
    ghost = (g.wet_mask[:, :, None] > 0.0) & (g.wet_mask_3d == 0.0)
    assert dry.any() and ghost.any(), "test grid lost its dry/ghost cells"
    # The sponge must actually be live in the polar band for this to bite.
    assert float(np.abs(np.asarray(params.sponge_rate)[dry]).max()) > 0.0

    T = np.asarray(out.T)
    S = np.asarray(out.S)
    assert np.allclose(T[dry], T_REF, rtol=0, atol=0),         f"land T leaked off the sentinel: {np.unique(T[dry])}"
    assert np.allclose(S[dry], S_REF, rtol=0, atol=0),         f"land S leaked off the sentinel: {np.unique(S[dry])}"
    assert np.allclose(T[ghost], T_REF, rtol=0, atol=0),         f"ghost T leaked off the sentinel: {np.unique(T[ghost])}"
    assert np.allclose(S[ghost], S_REF, rtol=0, atol=0),         f"ghost S leaked off the sentinel: {np.unique(S[ghost])}"
    print(f"  [PASS] {int(dry.sum())} dry + {int(ghost.sum())} ghost cells "
          f"held at the sentinel with the sponge on")


def test_sponge_relaxes_wet_band_and_leaves_interior_alone():
    """Wet cells in the band are pulled toward the target; interior is not."""
    g = _synth_grid()
    T0, S0 = _init_fields(g)
    state, params = _build(g, T0, S0, sponge_days=5.0, sponge_cells=4)

    rate = np.asarray(params.sponge_rate)[:, :, 0]
    band = (rate > 0.0) & (g.wet_mask > 0.0)
    interior = (rate == 0.0) & (g.wet_mask > 0.0)
    assert band.any() and interior.any()

    # Perturb the wet band away from the target (T_clim == T_init).
    T_pert = np.array(state.T)
    T_pert[band] += 5.0
    st = state._replace(T=jnp.array(T_pert))
    out = _half_steps(st, params, n=5)

    d0 = np.abs(np.asarray(st.T) - np.asarray(state.T))[band]
    d1 = np.abs(np.asarray(out.T) - np.asarray(state.T))[band]
    assert np.all(d1 < d0), "the sponge did not pull the band back"
    assert np.allclose(np.asarray(out.T)[interior],
                       np.asarray(state.T)[interior], rtol=0, atol=0),         "the sponge perturbed wet cells outside the band"
    print(f"  [PASS] band pulled back (max {d0.max():.3e} -> {d1.max():.3e}), "
          f"interior bit-identical")


def test_sponge_without_init_field_keeps_reference_state():
    """No initial field: the target is T_ref/S_ref, never zero."""
    g = _synth_grid()
    state, params = _build(g, None, None, sponge_days=5.0, sponge_cells=4)
    out = _half_steps(state, params, n=5)
    for name, ref in (("T", T_REF), ("S", S_REF)):
        f = np.asarray(getattr(out, name))
        assert np.allclose(f, ref, rtol=0, atol=0),             f"{name} drifted off {ref}: range [{f.min():.4g}, {f.max():.4g}]"
    print("  [PASS] sponge with no T_init/S_init stays at T_ref/S_ref")


if __name__ == "__main__":
    test_sponge_off_is_a_noop()
    test_sponge_preserves_land_and_ghost_sentinel()
    test_sponge_relaxes_wet_band_and_leaves_interior_alone()
    test_sponge_without_init_field_keeps_reference_state()
    print("ALL PASS")
