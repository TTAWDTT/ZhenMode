"""
Verify the PyTorch backend on GPU + the dynamic-forcing port.

Two checks, mirroring the jax_solver dynamic-forcing verification
(commit 9113d46):

  1. GPU run: build the solver, move params + state to CUDA, run a short
     integration, confirm no NaN/Inf and that fields actually changed
     (i.e. the GPU step did real physics, not a no-op).

  2. Dynamic-vs-static forcing equivalence: with the SAME wind/heat baked
     into params, step(state) (baked-in path) must agree with
     step(state, make_forcing(...)) (dynamic path) to round-off — the
     torch analogue of jax's 4e-14 equivalence. Then a SECOND check:
     swapping to a DIFFERENT forcing via the dynamic path must change the
     state (proving the swap actually takes effect, not silently ignored).

Run:  .venv-gpu-torch/Scripts/python.exe src/verify_torch_gpu_dynamic_forcing.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import DEFAULT_CONFIG
from grid import make_grid
from forcing import wind_stress_gyre, heat_flux_meridional
from torch_solver import (
    make_solver, make_forcing, TorchState, TorchForcing, SolverParams,
)


def _to_device(ntuple, device):
    """Move every torch.Tensor leaf of a namedtuple to ``device``."""
    vals = []
    for v in ntuple:
        if isinstance(v, torch.Tensor):
            vals.append(v.to(device))
        else:
            vals.append(v)
    return type(ntuple)(*vals)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    cuda_ok = torch.cuda.is_available()
    print("=" * 64)
    print("CHECK 0 — environment")
    print("=" * 64)
    print(f"torch {torch.__version__}  cuda build {torch.version.cuda}")
    print(f"cuda.is_available = {cuda_ok}")
    if cuda_ok:
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"(capability {torch.cuda.get_device_capability(0)})")
    print(f"grid: {nx}x{ny}x{nz}  dt={dt}s")

    # ───────────────────────────────────────────────────────────────────
    # CHECK 1 — GPU short integration
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("CHECK 1 — GPU short integration (10 steps, float64)")
    print("=" * 64)
    step_fn, init_state, diag_fn = make_solver(grid, physics, dt,
                                               forcing=forcing, compile=False)

    state = init_state()
    # perturb so the step has signal to move
    g = torch.Generator().manual_seed(42)
    state = TorchState(
        u=state.u + torch.randn(state.u.shape, generator=g) * 0.01,
        v=state.v + torch.randn(state.v.shape, generator=g) * 0.01,
        T=state.T + torch.randn(state.T.shape, generator=g) * 0.01,
        S=state.S,
        eta=state.eta,
    )

    if cuda_ok:
        device = torch.device('cuda')
        # move params too: step closes over `params`; rebuild a device-resident
        # solver by re-running _compute_params is not needed — we can drive
        # _step_impl directly with device-moved params. But the public step_fn
        # captured CPU params. Simplest: re-make solver, then move the closed
        # params is impossible from outside. Instead move state to GPU and
        # note params stay CPU → cross-device ops would fail. So build a
        # device-resident step by moving params through a custom closure.
        from torch_solver import _compute_params, _step_impl, _init_state
        params = _compute_params(grid, physics, dt, forcing=forcing)
        params = _to_device(params, device)
        state = _to_device(state, device)

        def gpu_step(s, forcing=None):
            return _step_impl(s, params, forcing)

        step_fn = gpu_step
        print(f"running on {device}")
    else:
        print("CUDA not available — running CHECK 1 on CPU (degraded)")

    t0 = time.perf_counter()
    state = step_fn(state)  # first step (includes any lazy init)
    if cuda_ok:
        torch.cuda.synchronize()
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(9):
        state = step_fn(state)
    if cuda_ok:
        torch.cuda.synchronize()
    t_9 = (time.perf_counter() - t0) / 9 * 1000

    u = np.asarray(state.u.cpu())
    T = np.asarray(state.T.cpu())
    eta = np.asarray(state.eta.cpu())
    finite = np.isfinite(u).all() and np.isfinite(T).all() and np.isfinite(eta).all()
    print(f"first step: {t_first:.3f}s   steady-state: {t_9:.2f} ms/step")
    print(f"u    range=[{u.min():.4e}, {u.max():.4e}]  max|u|={np.abs(u).max():.4e}")
    print(f"T    range=[{T.min():.4f}, {T.max():.4f}]")
    print(f"eta  range=[{eta.min():.4e}, {eta.max():.4e}]")
    print(f"all finite: {finite}")
    moved = (np.abs(u).max() > 0) and (np.abs(eta).max() > 0)
    print(f"CHECK 1 (GPU runs + fields moved): "
          f"{'PASS' if (finite and moved) else 'FAIL'}")

    # ───────────────────────────────────────────────────────────────────
    # CHECK 2 — dynamic vs static forcing equivalence
    # ───────────────────────────────────────────────────────────────────
    # Same wind/heat baked into params AND built as a TorchForcing from the
    # same fields → step(state) and step(state, jf) must agree to round-off.
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("CHECK 2 — dynamic vs static forcing equivalence (3 steps)")
    print("=" * 64)
    dev = torch.device('cuda') if cuda_ok else torch.device('cpu')

    # fresh solver, baked-in forcing
    from torch_solver import _compute_params, _step_impl
    params = _compute_params(grid, physics, dt, forcing=forcing)
    if cuda_ok:
        params = _to_device(params, dev)

    # identical randomized initial state for both paths
    g = torch.Generator().manual_seed(7)
    base = init_state()
    base = TorchState(
        u=base.u + torch.randn(base.u.shape, generator=g) * 0.05,
        v=base.v + torch.randn(base.v.shape, generator=g) * 0.05,
        T=base.T + torch.randn(base.T.shape, generator=g) * 0.05,
        S=base.S,
        eta=base.eta + torch.randn(base.eta.shape, generator=g) * 0.01,
    )
    if cuda_ok:
        base = _to_device(base, dev)

    s_static = base
    s_dyn = TorchState(*(t.clone() for t in base))

    jf = make_forcing(grid, tau_x, tau_y, Q_heat)
    if cuda_ok:
        jf = _to_device(jf, dev)

    for _ in range(3):
        s_static = _step_impl(s_static, params)              # baked-in
        s_dyn = _step_impl(s_dyn, params, jf)                # dynamic, same fields

    max_diff = 0.0
    for name in ('u', 'v', 'T', 'S', 'eta'):
        a = np.asarray(getattr(s_static, name).cpu())
        b = np.asarray(getattr(s_dyn, name).cpu())
        d = float(np.abs(a - b).max())
        max_diff = max(max_diff, d)
        print(f"  {name:>4}: max_abs(static - dynamic) = {d:.6e}")
    print(f"CHECK 2 (same forcing, static vs dynamic): "
          f"{'PASS' if max_diff < 1e-10 else 'FAIL'}  (max_diff={max_diff:.2e})")

    # ───────────────────────────────────────────────────────────────────
    # CHECK 3 — dynamic forcing actually swaps (different forcing changes state)
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("CHECK 3 — dynamic forcing swap takes effect (diff forcing ≠ same)")
    print("=" * 64)
    # A clearly different forcing: doubled wind stress + zero heat.
    tau_x2 = tau_x * 2.0 + 0.05
    tau_y2 = tau_y * 2.0 - 0.03
    Q_zero = np.zeros_like(Q_heat)
    jf2 = make_forcing(grid, tau_x2, tau_y2, Q_zero)
    if cuda_ok:
        jf2 = _to_device(jf2, dev)

    s_a = TorchState(*(t.clone() for t in base))   # dynamic, original forcing
    s_b = TorchState(*(t.clone() for t in base))   # dynamic, different forcing
    for _ in range(3):
        s_a = _step_impl(s_a, params, jf)
        s_b = _step_impl(s_b, params, jf2)

    diff_ab = 0.0
    for name in ('u', 'v', 'T', 'S', 'eta'):
        a = np.asarray(getattr(s_a, name).cpu())
        b = np.asarray(getattr(s_b, name).cpu())
        diff_ab = max(diff_ab, float(np.abs(a - b).max()))
    print(f"  max_abs(orig-forcing - diff-forcing) after 3 steps = {diff_ab:.6e}")
    print(f"CHECK 3 (forcing swap changes the trajectory): "
          f"{'PASS' if diff_ab > 1e-6 else 'FAIL'}  (diff={diff_ab:.2e})")

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  GPU available + runs : {cuda_ok and finite and moved}")
    print(f"  dynamic ≡ static     : {max_diff < 1e-10}  (max_diff={max_diff:.2e})")
    print(f"  forcing swap effective: {diff_ab > 1e-6}  (diff={diff_ab:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
