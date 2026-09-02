"""Step-by-step diagnostic: find which field blows up first and how fast."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, JaxState, _step_impl, _linear_half_step, _explicit_full_step, _compute_nonlinear_residual

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
physics = DEFAULT_CONFIG.physics
dt = 300.0

step_fn, init_state, _ = make_solver(grid, physics, dt, forcing=None)
params = step_fn.__wrapped__(init_state()) if hasattr(step_fn, '__wrapped__') else None

# Rebuild params manually
from jax_solver import _compute_params
params = _compute_params(grid, physics, dt, forcing=None, eos_type='linear')

state = init_state()
key = jax.random.PRNGKey(42)
key1, key2, key3 = jax.random.split(key, 3)
state = JaxState(
    u=state.u + jax.random.normal(key1, state.u.shape) * 0.001,
    v=state.v + jax.random.normal(key2, state.v.shape) * 0.001,
    T=state.T + jax.random.normal(key3, state.T.shape) * 0.001,
    S=state.S,
    eta=state.eta,
)

print("Step  step  max|u|      max|v|      max|T-Tref|  max|eta|    max|dudt|    max|dTdt|")
print("-" * 95)

for i in range(1, 56):
    # Compute residual before stepping
    dudt, dvdt, dTdt, dSdt = _compute_nonlinear_residual(state, params)
    
    max_u = float(jnp.max(jnp.abs(state.u)))
    max_v = float(jnp.max(jnp.abs(state.v)))
    max_T = float(jnp.max(jnp.abs(state.T - physics.T_ref)))
    max_eta = float(jnp.max(jnp.abs(state.eta)))
    max_dudt = float(jnp.max(jnp.abs(dudt)))
    max_dTdt = float(jnp.max(jnp.abs(dTdt)))
    
    has_nan = bool(jnp.any(jnp.isnan(state.u)))
    
    if has_nan:
        print(f"  {i:3d}   NaN detected!")
        break
    
    if i <= 10 or i % 5 == 0 or max_dudt > 1.0:
        print(f"  {i:3d}   {max_u:.4e}  {max_v:.4e}  {max_T:.4e}  {max_eta:.4e}  {max_dudt:.4e}  {max_dTdt:.4e}")
    
    if max_u > 1e6 or max_dudt > 1e6:
        print(f"  -> BLOWUP at step {i}")
        break
    
    state = step_fn(state)

# Also test: is the linear half-step stable on its own?
print("\n=== Linear half-step only (no nonlinear) ===")
state2 = init_state()
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(99), 3)
state2 = JaxState(
    u=state2.u + jax.random.normal(key1, state2.u.shape) * 0.001,
    v=state2.v + jax.random.normal(key2, state2.v.shape) * 0.001,
    T=state2.T + jax.random.normal(key3, state2.T.shape) * 0.001,
    S=state2.S, eta=state2.eta,
)
dt_half = dt / 2.0
for i in range(1, 101):
    state2 = _linear_half_step(state2, params, dt_half)
    state2 = _linear_half_step(state2, params, dt_half)
    if i % 20 == 0:
        max_u = float(jnp.max(jnp.abs(state2.u)))
        max_T = float(jnp.max(jnp.abs(state2.T - physics.T_ref)))
        has_nan = bool(jnp.any(jnp.isnan(state2.u)))
        print(f"  Step {i:3d}: max|u|={max_u:.4e}  max|T-Tref|={max_T:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan:
            break

# Test: nonlinear step only (no linear)
print("\n=== Nonlinear step only (no linear) ===")
state3 = init_state()
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(77), 3)
state3 = JaxState(
    u=state3.u + jax.random.normal(key1, state3.u.shape) * 0.001,
    v=state3.v + jax.random.normal(key2, state3.v.shape) * 0.001,
    T=state3.T + jax.random.normal(key3, state3.T.shape) * 0.001,
    S=state3.S, eta=state3.eta,
)
for i in range(1, 56):
    state3 = _explicit_full_step(state3, params, dt)
    if i % 5 == 0 or i <= 5:
        max_u = float(jnp.max(jnp.abs(state3.u)))
        max_T = float(jnp.max(jnp.abs(state3.T - physics.T_ref)))
        has_nan = bool(jnp.any(jnp.isnan(state3.u)))
        print(f"  Step {i:3d}: max|u|={max_u:.4e}  max|T-Tref|={max_T:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or max_u > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break
