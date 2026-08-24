"""
JAX (CPU) vs PyTorch (GPU) — numerical alignment bench, extended.

Three comparisons:
  A. Baked-in forcing path: jax step(state) vs torch step(state) on GPU,
     same WOA-seeded init, 8 steps. (extends bench_jax_torch.py to GPU torch)
  B. Dynamic forcing path: jax step(state, JaxForcing) vs torch
     step(state, TorchForcing) on GPU — the cross-backend check that the
     dynamic-forcing port mirrors jax field-for-field.
  C. Dynamic-vs-static per-backend: each backend's dynamic path ≡ its own
     static path (jax ~4e-14, torch exactly 0). Sanity that the dynamic
     path is a faithful plumbing change, not a physics change.

Run:  .venv-gpu-torch/Scripts/python.exe src/bench_jax_torch_gpu.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import torch
from torch_solver import (make_solver as torch_make_solver, make_forcing as
                          torch_make_forcing, TorchState, _compute_params,
                          _step_impl as torch_step_impl)
from forcing import wind_stress_gyre, heat_flux_meridional
from config import DEFAULT_CONFIG
from grid import make_grid
from woa_data import get_initial_fields


def _to_device(ntuple, device):
    vals = [v.to(device) if isinstance(v, torch.Tensor) else v for v in ntuple]
    return type(ntuple)(*vals)


def _cmp(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    rel = diff / denom
    print(f"  {name:>5}: max_abs={diff.max():.6e}  "
          f"mean_rel={rel.mean():.3e}  max_rel={rel.max():.3e}")
    return float(diff.max())


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0
    N = 8
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)
    T_init, S_init = get_initial_fields(grid)

    cuda_ok = torch.cuda.is_available()
    dev = torch.device('cuda') if cuda_ok else torch.device('cpu')
    print(f"grid {nx}x{ny}x{nz}  dt={dt}s  steps={N}")
    print(f"jax CPU  | torch {torch.__version__} on {dev} "
          f"(cuda={cuda_ok})")
    print()

    # ── JAX solver (CPU) ──
    import jax_solver as js
    j_step, j_init, _ = js.make_solver(grid, physics, dt, forcing=forcing)
    j_forcing = js.make_forcing(grid, jnp.asarray(tau_x), jnp.asarray(tau_y),
                                jnp.asarray(Q_heat))

    # ── Torch solver (GPU) ──
    t_params = _compute_params(grid, physics, dt, forcing=forcing)
    if cuda_ok:
        t_params = _to_device(t_params, dev)
    t_init_state = torch_make_solver(grid, physics, dt, forcing=forcing)[1]
    t_forcing = torch_make_forcing(grid, tau_x, tau_y, Q_heat)
    if cuda_ok:
        t_forcing = _to_device(t_forcing, dev)

    # ═══════════════════════════════════════════════════════════════════
    # A. Baked-in forcing path: jax step(state) vs torch step(state) GPU
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("A. Baked-in forcing: jax CPU step(state) vs torch GPU step(state)")
    print("=" * 60)
    jst = j_init(T_init=T_init, S_init=S_init)
    tst = t_init_state(T_init=T_init, S_init=S_init)
    if cuda_ok:
        tst = _to_device(tst, dev)

    jst = j_step(jst)  # compile warmup
    jax.block_until_ready(jst.u)
    tst = torch_step_impl(tst, t_params)
    if cuda_ok:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N - 1):
        jst = j_step(jst)
    jax.block_until_ready(jst.u)
    t_jax = (time.perf_counter() - t0) / (N - 1) * 1000
    t0 = time.perf_counter()
    for _ in range(N - 1):
        tst = torch_step_impl(tst, t_params)
    if cuda_ok:
        torch.cuda.synchronize()
    t_pt = (time.perf_counter() - t0) / (N - 1) * 1000

    print(f"Results after {N} steps (baked-in):")
    _cmp('u', jst.u, tst.u.cpu())
    _cmp('v', jst.v, tst.v.cpu())
    _cmp('T', jst.T, tst.T.cpu())
    _cmp('S', jst.S, tst.S.cpu())
    _cmp('eta', jst.eta, tst.eta.cpu())
    print(f"  timing: jax CPU {t_jax:.2f} ms/step | torch GPU {t_pt:.2f} ms/step")

    # ═══════════════════════════════════════════════════════════════════
    # B. Dynamic forcing path: jax step(state, jf) vs torch step(state, tf) GPU
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("B. Dynamic forcing: jax step(state, jf) vs torch step(state, tf) GPU")
    print("=" * 60)
    jst2 = j_init(T_init=T_init, S_init=S_init)
    tst2 = t_init_state(T_init=T_init, S_init=S_init)
    if cuda_ok:
        tst2 = _to_device(tst2, dev)

    jst2 = j_step(jst2, j_forcing)
    jax.block_until_ready(jst2.u)
    tst2 = torch_step_impl(tst2, t_params, t_forcing)
    if cuda_ok:
        torch.cuda.synchronize()
    for _ in range(N - 1):
        jst2 = j_step(jst2, j_forcing)
    jax.block_until_ready(jst2.u)
    for _ in range(N - 1):
        tst2 = torch_step_impl(tst2, t_params, t_forcing)
    if cuda_ok:
        torch.cuda.synchronize()

    print(f"Results after {N} steps (dynamic forcing):")
    _cmp('u', jst2.u, tst2.u.cpu())
    _cmp('v', jst2.v, tst2.v.cpu())
    _cmp('T', jst2.T, tst2.T.cpu())
    _cmp('S', jst2.S, tst2.S.cpu())
    _cmp('eta', jst2.eta, tst2.eta.cpu())

    # ═══════════════════════════════════════════════════════════════════
    # C. Per-backend dynamic ≡ static (sanity)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("C. Per-backend: dynamic path ≡ static path (same forcing)")
    print("=" * 60)
    # jax
    js_a = j_init(T_init=T_init, S_init=S_init)
    js_b = j_init(T_init=T_init, S_init=S_init)
    for _ in range(3):
        js_a = j_step(js_a)               # static
        js_b = j_step(js_b, j_forcing)    # dynamic, same fields
    jx = 0.0
    for name in ('u', 'v', 'T', 'S', 'eta'):
        d = float(np.abs(np.asarray(getattr(js_a, name)) -
                         np.asarray(getattr(js_b, name))).max())
        jx = max(jx, d)
    print(f"  jax  dynamic-vs-static max_diff (3 steps) = {jx:.6e}")
    # torch
    ts_a = t_init_state(T_init=T_init, S_init=S_init)
    ts_b = t_init_state(T_init=T_init, S_init=S_init)
    if cuda_ok:
        ts_a = _to_device(ts_a, dev); ts_b = _to_device(ts_b, dev)
    for _ in range(3):
        ts_a = torch_step_impl(ts_a, t_params)              # static
        ts_b = torch_step_impl(ts_b, t_params, t_forcing)   # dynamic, same fields
    tx = 0.0
    for name in ('u', 'v', 'T', 'S', 'eta'):
        d = float(np.abs(np.asarray(getattr(ts_a, name).cpu()) -
                         np.asarray(getattr(ts_b, name).cpu())).max())
        tx = max(tx, d)
    print(f"  torch dynamic-vs-static max_diff (3 steps) = {tx:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
