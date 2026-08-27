# Plan: Semi-implicit (Crank-Nicolson) free-surface step for the global FD solver

## Root cause (empirically proven this session)

The global FD solver (`src/jax_solver_global.py`) blows up (NaN ~step 1000-1400,
max|u| 27@1000 → 1361@1200) under real WOA stratification + advection. Decisive
diagnostics this session established the mechanism:

1. **Density PGF is the sole energy source** (zero `_density_anomaly` → ocean at rest).
2. **No-advection (residual PGF only)**: FD grows baroclinic KE **unbounded**
   (36x over 1400 steps, still climbing). The depth-varying baroclinic PGF in the
   explicit RK2 residual injects energy linearly.
3. **Regional spectral solver** (`jax_solver.py`, same PGF structure, same
   stratification, same 1400 steps, advection ON): **STABLE** (max|u| 0.5-2.8, E flat).
   → The FD explosion is a **discretization defect**, not shared physics.
4. **Regional spectral, no-advection**: KE_bc grows initially (8x by step 400, the
   physical PE→KE adjustment from the unbalanced rest init) then **settles to a
   bounded oscillatory equilibrium**. FD no-adv never equilibrates (grows unbounded).
   → Both start from the same imbalance; spectral **reaches equilibrium**, FD does not.

**Mechanism**: the FD linear half-step's **explicit forward-backward (Sielecki)
free-surface step has phase error**. The spectral solver's **exact matrix-exponential**
free-surface step has correct phase, so the baroclinic adjustment (driven by the
residual PGF) reaches a bounded oscillatory equilibrium. The FB phase error prevents
equilibration → the persistent density PGF does net work every step → unbounded
baroclinic KE growth → advection cascades it to small scale → exponential blowup.

The phase error enters via the **barotropic projection**: the SW step evolves
(eta, ubt, vbt); the baroclinic velocity (from the residual PGF) is depth-averaged
into ubt, the SW step updates ubt/eta, and the barotropic delta is projected back
to 3D (`u_new = u + (ubt_new - ubt_old)`). With FB phase error, this projection
injects energy into the baroclinic mode. The exact matrix-exp projection is
energy-neutral.

## The fix

Replace the explicit FB free-surface step (`_free_surface_step_fd`) with a
**semi-implicit Crank-Nicolson SW step** — the FD analogue of the spectral
matrix exponential. The CN scheme has correct phase (2nd-order, symplectic-like)
and is unconditionally stable, so the baroclinic adjustment equilibrates like the
spectral solver.

### Discretization (semi-implicit CN shallow-water step)

Continuity + momentum, trapezoidal (CN) in time, with the barotropic PGF + wind
forcing F = F_rho + wind, implicit linear bottom drag:

```
(eta^{n+1} - eta^n)/dt = -H_sw/2 * (div(U^{n+1}) + div(U^n))          ... (1)
U^{n+1} = (U^n + dt/2*(-g*grad(eta^n + eta^{n+1}) + F^n + F^{n+1})) / (1 + r_bt*dt/2)   ... (2)
```
where U=(ubt,vbt). F is treated explicitly (F^{n+1}=F^n; it depends on T/S which
don't change within the linear half-step). Substituting (2) into (1) gives the
**Helmholtz equation for eta^{n+1}**:

```
[ I - (dt/2)^2 * g*H_sw * L ] eta^{n+1} = rhs
```
where L = div(grad(·)) is the **self-adjoint conservative Laplacian** (already
implemented: `_gradient_conservative` is the exact adjoint of
`_divergence_conservative` under the area-weighted inner product). The operator
`A = I - (dt/2)^2 * g*H_sw * L` is SPD on the wet domain (−L is PSD), so **CG
converges** (already verified: `_diag_helmholtz_test.py` shows CG converges, mass-
conserving, resid ~1e-10 on the real masked global grid).

After solving for eta^{n+1}, recover U^{n+1} from (2) (explicit given eta^{n+1}),
apply sponge decay + polar cap (same as current FB step), project barotropic delta
back to 3D.

### Implementation steps (all in `src/jax_solver_global.py`)

1. **Add `_free_surface_step_cn(eta, u, v, p, F_rho_x, F_rho_y, dt_half)`** — a new
   function alongside `_free_surface_step_fd`. Builds the Helmholtz operator
   `A(eta) = wet_mask * (eta - (dt_half^2)*g*H_sw*lap_consistent(eta))`, the RHS
   from the old state + forcing, solves via `jax.scipy.sparse.linalg.cg` (maxiter
   ~100, tol ~1e-8), then recovers ubt/vbt. Preserves the existing polar-cap
   filter, sponge decay, conservative mass, and barotropic-delta projection (lift
   the `_cap` closure + projection logic from `_free_surface_step_fd`).

2. **Wire a selector** in `_linear_half_step`: use `_free_surface_step_cn` instead
   of `_free_surface_step_fd`. Gate behind a new `semi_implicit_fs: bool` flag on
   `PhysicsConfig` (default **False** to preserve the validated spectral-baseline
   behavior unchanged) OR a `make_solver_global` kwarg. Set True only in the global
   FD path. (PhysicsConfig defaults untouched per the integrity mandate — use
   `dataclasses.replace` in the driver to flip it.)

3. **Validate with diagnostics first** (no source change to the production path
   until the gate passes):
   - Re-run `_diag_noadv_decisive` with the CN step: expect baroclinic KE to
     **settle to bounded oscillation** (like spectral no-adv), not unbounded growth.
   - Re-run the full-step (advection ON): expect **no NaN through 1400+ steps**.

4. **If the CN step stabilizes**: run the **pure-free-wave decay gate**
   (sum_eta ≤ 1e-10 relative AND max|eta|, max|u| decay over 800+ steps), then the
   **10d production probe** (polar cap ON + real wind + bulk=40), confirm both
   watchdogs (T + eta) clean, `pytest` (55 tests) green.

5. **If CN does NOT stabilize**: the phase-error hypothesis is wrong; fall back to
   the deeper architectural fix — a per-column implicit treatment of the 3D
   baroclinic PGF itself (tridiagonal vertical Helmholtz), the true FD analogue
   of integrating the full 3D system exactly.

## Risk / guardrails

- **No change to `jax_solver.py`** (spectral baseline, READ ONLY).
- **PhysicsConfig defaults unchanged** — flip `semi_implicit_fs` via
  `dataclasses.replace` in the global driver only.
- New code isolated in `jax_solver_global.py`; the CN step is a new function, the
  FB step is retained (selectable), so the change is reversible.
- CG inside the JIT step: verify it compiles and is fast enough (the Helmholtz
  diag already showed CG converges in few iterations on this grid). If CG is too
  slow or unstable under jit, fall back to lon-FFT + lat-tridiagonal (the grid is
  lon-periodic with per-row-uniform dx).
- Push HELD until 10d clean on BOTH watchdogs. Commits to main locally, no origin
  push. git user TTAWDTT.

## What this does NOT change

- The validated spectral `jax_solver.py` is untouched.
- Pre-registered R1/R4 pass/fail criteria are not moved.
- A1/A2 thresholds are not relaxed; honest FAIL is fine.
