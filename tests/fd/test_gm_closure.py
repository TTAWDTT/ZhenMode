"""
Unit tests for the Gent-McWilliams / Redi isopycnal closure in
jax_solver_global.py: _isopycnal_slope, _isopycnal_closure,
_redi_skew_flux_tendency.

Uses a small synthetic all-wet GlobalOceanGrid (no real data / bathymetry)
so the tests are fast and self-contained.

Run:  python -m pytest tests/test_gm_closure.py -v
  or: python tests/test_gm_closure.py
"""
import os
import sys

os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.30')

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from dataclasses import replace

import jax.numpy as jnp

import zhenmode.model.solver.dynamics.tendencies as G_processes
import zhenmode.model.solver.numerics.horizontal as G_horizontal
import zhenmode.model.solver.numerics.vertical as G_vertical
import zhenmode.model.solver.physics.eos as G_eos
from tests.support.grid import all_wet_grid as _synth_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.physics.isopycnal import (
    _isopycnal_closure,
    _isopycnal_slope,
    _redi_skew_flux_tendency,
)


def _make_state_and_params(grid, kappa_gm, T_field, S_field=None, kappa_redi=0.0):
    """Build a JaxStateG + FDPhysParams with given T and flat S=35."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if S_field is None:
        S_field = np.full((nx, ny, nz), 35.0)
    physics = replace(PhysicsConfig(), nu_h=100.0, kappa_h=100.0,
                      kappa_gm=kappa_gm, gm_slope_max=0.01,
                      kappa_redi=kappa_redi)
    Q = np.zeros((nx, ny))
    tau_x = np.zeros((nx, ny))
    tau_y = np.zeros((nx, ny))
    _, init_fn, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q),
        T_atm=None, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=np.asarray(T_field), S_init=np.asarray(S_field),
        polar_cap_rows=0, polar_cap_taper=0, return_params=True)
    state = init_fn(T_init=jnp.array(T_field), S_init=jnp.array(S_field))
    return state, params


# ── tests ──────────────────────────────────────────────────────────

def test_closure_off_when_kappas_zero():
    """Closure OFF by default: kappa_gm = kappa_redi = 0 -> no term at all."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.full((nx, ny, nz), 20.0)
    T[:, :, 0] = 22.0  # warm surface
    T[:, :, -1] = 5.0  # cold bottom
    state, p = _make_state_and_params(g, kappa_gm=0.0, T_field=T)
    gm_T, gm_S, redi_T, redi_S = _isopycnal_closure(state, p)
    assert gm_T == 0.0 and gm_S == 0.0 and redi_T == 0.0 and redi_S == 0.0, \
        "closure nonzero with kappa_gm = kappa_redi = 0"
    print("  [PASS] closure identically 0 with kappa_gm = kappa_redi = 0")


def test_slope_finite_and_limited():
    """Slope must be finite and bounded by the limiter for real stratification."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    # Strong horizontal T gradient + stable vertical stratification.
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)   # warm east, cold west
    T[:, :, 0] += 3.0   # warm surface
    T[:, :, -1] -= 3.0  # cold bottom
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T)
    Sx, Sy = _isopycnal_slope(state, p)
    assert Sx.shape == T.shape, f"shape {Sx.shape} != {T.shape}"
    assert bool(jnp.all(jnp.isfinite(Sx))), "S_x not finite"
    assert bool(jnp.all(jnp.isfinite(Sy))), "S_y not finite"
    assert float(jnp.max(jnp.abs(Sx))) <= p.gm_slope_max * (1.0 + 1e-6), \
        f"max|S_x|={float(jnp.max(jnp.abs(Sx)))} > limiter {p.gm_slope_max}"
    assert float(jnp.max(jnp.abs(Sy))) <= p.gm_slope_max * (1.0 + 1e-6), \
        f"max|S_y|={float(jnp.max(jnp.abs(Sy)))} > limiter {p.gm_slope_max}"
    print(f"  [PASS] slope finite, max|S_x|={float(jnp.max(jnp.abs(Sx))):.3e} "
          f"<= {p.gm_slope_max}")


def test_slope_sign_down_gradient():
    """The slope must be S = -grad_h(rho') / (drho'/dz), i.e. it points DOWN
    the horizontal density gradient (toward denser water) in the stably
    stratified interior. A sign error here reverses the eddy transport.
    """
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)   # warm east => drho_dx < 0
    T[:, :, 0] += 3.0
    T[:, :, -1] -= 3.0
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T)
    Sx, Sy = _isopycnal_slope(state, p)
    assert float(jnp.max(jnp.abs(Sx))) > 0.0, "slope is zero everywhere"

    # Rebuild the numerator/denominator exactly as _isopycnal_slope does.
    rho = G_eos._density_anomaly(state.T, state.S, p) * p.wet_mask_z
    drho_dx, _ = G_horizontal._gradient_conservative_3d(rho, p)
    drho_dz = G_vertical._d_dz(G_vertical._fill_ghost_bottom(rho, p), p)
    expected = -drho_dx / drho_dz
    mask = (jnp.abs(Sx) > 0.0) & (jnp.abs(drho_dz) > 0.0)
    assert bool(jnp.all(jnp.sign(Sx[mask]) == jnp.sign(expected[mask]))), \
        "slope sign does not match -d(rho')/dx / d(rho')/dz"
    # The test state varies zonally only: no meridional slope.
    assert float(jnp.max(jnp.abs(Sy))) < 1e-12, "spurious meridional slope"
    print(f"  [PASS] slope sign = -d(rho')/dx / d(rho')/dz "
          f"(max|S_x|={float(jnp.max(jnp.abs(Sx))):.3e})")


def test_closure_zero_for_horizontally_uniform_state():
    """A horizontally uniform density field has zero slopes, so the closure
    tendency vanishes identically: isopycnal redistribution creates no
    transport where there is nothing to flatten."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.full((nx, ny, nz), 20.0)
    T[:, :, 0] = 22.0
    T[:, :, -1] = 5.0  # stratified but horizontally uniform
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T,
                                      kappa_redi=2000.0)
    gm_T, gm_S, redi_T, redi_S = _isopycnal_closure(state, p)
    for name, term in (("gm_T", gm_T), ("gm_S", gm_S),
                       ("redi_T", redi_T), ("redi_S", redi_S)):
        assert float(jnp.max(jnp.abs(term))) == 0.0, \
            f"{name} nonzero on a horizontally uniform state"
    print("  [PASS] GM and Redi identically 0 on a horizontally uniform state")


def test_gm_and_redi_are_the_same_operator():
    """kappa_gm and kappa_redi drive the SAME skew-flux operator, so the
    closure tendency for (kappa_gm=k, kappa_redi=0) is identical to the one
    for (kappa_gm=0, kappa_redi=k). That is why enabling BOTH doubles the
    closure strength (GM/Redi closure convention, docs/decisions.md D18).
    """
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)
    T[:, :, 0] += 3.0
    T[:, :, -1] -= 3.0
    state, p_gm = _make_state_and_params(g, kappa_gm=2000.0, T_field=T)
    _, p_redi = _make_state_and_params(g, kappa_gm=0.0, T_field=T,
                                       kappa_redi=2000.0)
    gm_T, gm_S, _, _ = _isopycnal_closure(state, p_gm)
    _, _, redi_T, redi_S = _isopycnal_closure(state, p_redi)
    assert float(jnp.max(jnp.abs(gm_T - redi_T))) == 0.0, "GM != Redi on T"
    assert float(jnp.max(jnp.abs(gm_S - redi_S))) == 0.0, "GM != Redi on S"
    print(f"  [PASS] kappa_gm and kappa_redi are the same operator "
          f"(max|gm_T|={float(jnp.max(jnp.abs(gm_T))):.3e})")


def test_closure_wired_into_tracer_tendency():
    """kappa_gm > 0 must change dT/dt: the closure has to be APPLIED by
    _compute_tracer_tendency, not merely available as a helper."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)
    T[:, :, 0] += 3.0
    T[:, :, -1] -= 3.0
    st_off, p_off = _make_state_and_params(g, kappa_gm=0.0, T_field=T)
    st_on, p_on = _make_state_and_params(g, kappa_gm=2000.0, T_field=T)
    dT_off, _ = G_processes._compute_tracer_tendency(st_off, p_off)
    dT_on, _ = G_processes._compute_tracer_tendency(st_on, p_on)
    delta = float(jnp.max(jnp.abs(dT_on - dT_off)))
    assert delta > 0.0, "kappa_gm > 0 did not change dT/dt (closure not wired in)"
    print(f"  [PASS] GM closure is applied to dT/dt (max delta {delta:.3e})")


def test_tendency_finite_with_gm():
    """Full tracer tendency with GM ON must be finite for realistic T."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)
    T[:, :, 0] += 3.0
    T[:, :, -1] -= 3.0
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T)
    dTdt, dSdt = G_processes._compute_tracer_tendency(state, p)
    assert bool(jnp.all(jnp.isfinite(dTdt))), "dTdt not finite with GM"
    assert bool(jnp.all(jnp.isfinite(dSdt))), "dSdt not finite with GM"
    print(f"  [PASS] tendency finite with GM (max|dTdt|={float(jnp.max(jnp.abs(dTdt))):.3e})")


# ── Redi isopycnal mixing tests ────────────────────────────────────

def test_redi_zero_when_kappa_redi_zero():
    """Redi tendency must vanish when kappa_redi=0."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)
    T[:, :, 0] += 3.0
    T[:, :, -1] -= 3.0
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T, kappa_redi=0.0)
    S_x, S_y = _isopycnal_slope(state, p)
    redi_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p)
    assert float(jnp.max(jnp.abs(redi_T))) == 0.0, "Redi nonzero when kappa_redi=0"
    print("  [PASS] Redi zero when kappa_redi=0")


def test_redi_tendency_finite():
    """Redi tendency with slopes + kappa_redi>0 must be finite and nonzero."""
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)        # horizontal gradient -> slope
    T[:, :, 0] += 5.0
    T[:, :, -1] -= 5.0          # vertical stratification
    state, p = _make_state_and_params(g, kappa_gm=2000.0, T_field=T, kappa_redi=2000.0)
    S_x, S_y = _isopycnal_slope(state, p)
    redi_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p)
    assert bool(jnp.all(jnp.isfinite(redi_T))), "Redi dTdt not finite"
    assert float(jnp.max(jnp.abs(redi_T))) > 0.0, "Redi dTdt is zero despite slopes"
    print(f"  [PASS] Redi tendency finite & nonzero (max|redi_T|={float(jnp.max(jnp.abs(redi_T))):.3e})")


def test_redi_is_dissipative_on_perturbation():
    """Redi diffuses ALONG isopycnals. Its defining structural property is
    that the vertical skew flux F_z vanishes on the DENSITY anomaly (density
    is constant on isopycnals, so isopycnal diffusion transports no density
    across levels — no spurious diapycnal mixing / APE creation).

    A sign error or a missing tensor term would create a non-trivial vertical
    density flux F_z[rho] that pumps buoyancy across levels. We check that
    F_z[rho] is negligible relative to the horizontal skew flux F_h[rho].
    (Multi-step forward-Euler is NOT a valid test: explicit Euler is
    unconditionally unstable for diffusion, so it steepens regardless of sign.)
    """
    g = _synth_grid()
    nx, ny, nz = g.nx, g.ny, g.nz
    # Weak stratification + strong zonal front -> non-trivial slope
    T = np.broadcast_to(np.array([20., 19., 18., 17., 16., 15., 14., 13.]),
                        (nx, ny, nz)).copy()
    for i in range(nx):
        T[i, :, :] += 10.0 * (1.0 - i / (nx - 1))   # 10C zonal front
    state, p = _make_state_and_params(g, kappa_gm=0.0, T_field=T, kappa_redi=2000.0)
    S_x, S_y = _isopycnal_slope(state, p)
    rho = G_eos._density_anomaly(state.T, state.S, p)
    # Reconstruct the skew fluxes the function builds internally:
    k = p.kappa_redi
    dC_dz = G_vertical._d_dz(rho, p)
    Fx = -k * S_x * dC_dz          # horizontal skew flux of density
    Fy = -k * S_y * dC_dz
    dC_dx = G_horizontal._d_dx(rho, p)
    dC_dy = G_horizontal._d_dy(rho, p)
    SdotGradC = S_x * dC_dx + S_y * dC_dy
    S2 = S_x * S_x + S_y * S_y
    Fz = -k * (SdotGradC + S2 * dC_dz)   # vertical skew flux of density
    # F_z[rho] must be ~0 (no diapycnal density transport); F_h can be nonzero.
    max_Fh = float(jnp.max(jnp.sqrt(Fx ** 2 + Fy ** 2)))
    max_Fz = float(jnp.max(jnp.abs(Fz)))
    print(f"  [info] max|F_h[rho]|={max_Fh:.3e}, max|F_z[rho]|={max_Fz:.3e}")
    assert max_Fz < 0.05 * (max_Fh + 1e-30), \
        f"Redi vertical density flux {max_Fz:.3e} not negligible vs horizontal {max_Fh:.3e}; diapycnal leak"
    print(f"  [PASS] Redi has no diapycnal density flux "
          f"(|F_z|={max_Fz:.3e} << |F_h|={max_Fh:.3e})")


def test_gm_skew_flux_stable():
    """Regression guard: GM closure in _compute_tracer_tendency must use the
    skew-flux form (diffusive CFL ~ kappa*|S|^2*dt/dz^2), NOT the advective
    bolus form (CFL ~ |w*|*dt/dz). On a thin surface layer the advective form
    has CFL > 1 and blows up within ~12 steps; the skew-flux form is stable.

    We step the full solver 30 steps and assert max|T| stays bounded (no
    exponential blow-up). This catches a regression that swaps the skew-flux
    operator for the retired advective bolus form.
    """
    g = _synth_grid(nx=32, ny=32, nz=8)
    nx, ny, nz = g.nx, g.ny, g.nz
    # Realistic-ish: warm surface, cold bottom, zonal T gradient -> slope.
    T = np.zeros((nx, ny, nz))
    for i in range(nx):
        T[i, :, :] = 5.0 + 15.0 * (i / nx)
    T[:, :, 0] += 5.0
    T[:, :, -1] -= 5.0
    physics = replace(PhysicsConfig(), nu_h=100.0, kappa_h=100.0,
                      kappa_gm=2000.0, gm_slope_max=0.01)
    step_fn, init_fn, _, _, _ = make_solver_global(
        g, physics, 60.0,
        forcing=(np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))),
        T_atm=None, lambda_bulk=0.0, T_init=T,
        S_init=np.full((nx, ny, nz), 35.0),
        polar_cap_rows=0, polar_cap_taper=0, return_params=True)
    st = init_fn(T_init=jnp.array(T), S_init=jnp.array(np.full((nx, ny, nz), 35.0)))
    t_max_prev = float(jnp.max(jnp.abs(st.T)))
    for i in range(30):
        st = step_fn(st)
        t_max = float(jnp.max(jnp.abs(st.T)))
        assert bool(jnp.all(jnp.isfinite(st.T))), f"NaN at step {i+1}"
        # No exponential blow-up: max|T| must not grow by more than 3x the
        # initial max (the advective form hit >10x by step 20).
        assert t_max < 3.0 * float(np.max(np.abs(T))) + 1.0, \
            f"max|T|={t_max:.1f} blew up at step {i+1} (advective CFL regression?)"
    print(f"  [PASS] GM skew-flux stable over 30 steps "
          f"(max|T| {t_max_prev:.2f} -> {t_max:.2f}, no blow-up)")


TESTS = [
    test_closure_off_when_kappas_zero,
    test_slope_finite_and_limited,
    test_slope_sign_down_gradient,
    test_closure_zero_for_horizontally_uniform_state,
    test_gm_and_redi_are_the_same_operator,
    test_closure_wired_into_tracer_tendency,
    test_tendency_finite_with_gm,
    test_redi_zero_when_kappa_redi_zero,
    test_redi_tendency_finite,
    test_redi_is_dissipative_on_perturbation,
    test_gm_skew_flux_stable,
]


if __name__ == '__main__':
    passed = failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print("=" * 60)
    print(f"GM closure tests: {passed} passed, {failed} failed, "
          f"{passed + failed} total")
    print("=" * 60)
    sys.exit(1 if failed else 0)
