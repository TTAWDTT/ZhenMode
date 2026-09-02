# GM Sub-Grid Baroclinic Closure for the FD Global Solver

## Context (verified this session)

The FD global solver's energy injection is a **physical resolution limitation**, not a numerical defect (verified by matched-control re-run):
- Spectral solver, dx=9108m (fine), nu_h=100, no-adv, real WOA: **bounded** (max|u| 1.3→19 peak→7.6 decay).
- Spectral solver, dx=111km (coarse), same settings: **injects** (max|u| 17→82).
- FD + full exact linear step (exact diffusion + exact free surface), dx=111km: **still injects** (max|u| 9→86).

The baroclinic Rossby radius (~30-50km) is resolvable at dx=9108m (3-5 pts) but sub-grid at dx=111km (0.3 pts). The depth-varying baroclinic PGF (in the RK2 residual, identical structure in both solvers) drives unresolved internal-gravity-wave modes that inject energy. The user-approved "exact/implicit FD linear step" is disproved.

**Probes this session confirmed** the closure must be GM-style bolus transport, NOT enhanced diffusion:
- kappa_gm=5000 (50× background): max|u| 9→88 (no arrest, ~same as baseline 9→86).
- kappa_gm=50000 (500× background): max|u| 9→67 (slowed only, not arrested).
Horizontal Laplacian diffusion decays the large-scale density gradient too slowly to outpace the PGF injection. GM bolus transport actively releases baroclinic available potential energy (APE) — the mechanistically correct target.

## Design: GM Bolus Transport Closure

The Gent-McWilliams closure represents unresolved eddies as an advective bolus transport that flattens isopycnal slopes, releasing APE. Implemented as an additional tracer advection by the bolus velocity `(u*, v*)`, proportional to the isopycnal slope.

### Physics

1. **Isopycnal slope** (linear EOS, `rho' = RHO_0·(-ALPHA_T·T' + BETA_S·S')`):
   ```
   S_x = ∂z_iso/∂x = -(∂rho'/∂x) / (∂rho'/∂z)
   S_y = ∂z_iso/∂y = -(∂rho'/∂y) / (∂rho'/∂z)
   ```
   where `∂rho'/∂z` is the vertical density gradient (stratification, >0 when stable). Slope-tapering/slope-limiting applied to avoid singularity at weak stratification.

2. **Bolus (eddy-induced) velocity** (down-gradient along isopycnal slope):
   ```
   u* = -κ_GM · S_x
   v* = -κ_GM · S_y
   ```
   `κ_GM` is the GM diffusivity (tunable, ~500-5000 m²/s typical for 1°; will calibrate).

3. **Tracer transport by bolus** (added to the existing scalar advection in `_compute_tracer_tendency`):
   ```
   dT/dt += -(u*·∂T/∂x + v*·∂T/∂y)
   dS/dt += -(u*·∂S/∂x + v*·∂S/∂y)
   ```
   The bolus transports tracers down the isopycnal slope, flattening density gradients and releasing APE — directly countering the baroclinic PGF injection.

4. **Vertical bolus closure**: `w*` diagnosed from `(u*, v*)` continuity (same pattern as `_compute_vertical_velocity`) so the bolus is non-divergent in the interior. This keeps the bolus transport mass-conservative.

5. **NOT applied to momentum** (GM is a tracer-only closure in its standard form; the momentum effect is indirect via the flattened density → reduced PGF).

### Implementation (in `src/jax_solver_global.py`)

New functions:
- `_isopycnal_slope(state, p)` → returns `S_x, S_y` (nx, ny, nz) using `_d_dx`/`_d_dy`/`_d_dz` on `rho_prime`. Apply slope limiter (tanh-clip large slopes) and mask by `wet_mask_z`. Guard `∂rho'/∂z` with a floor to avoid division blow-up in weakly-stratified/convective columns.
- `_gm_bolus_velocity(state, p)` → returns `u_star, v_star, w_star` (nx, ny, nz). `u*,v* = -κ_GM·S`; `w*` from continuity of `(u*,v*)`.
- `_gm_tracer_transport(T, u_star, v_star, w_star, p)` → bolus advection tendency `-(u*·∂T/∂x + v*·∂T/∂y + w*·∂T/∂z)`, dealiased, masked.

Hook into `_compute_tracer_tendency`: add `_gm_tracer_transport(T, u*, v*, w*, p)` and `_gm_tracer_transport(S, ...)` to `dTdt, dSdt`. Because `_compute_tracer_residual` subtracts only `kappa_h·lap`, the GM term (a distinct advection) survives into the residual correctly — no double-count.

New `PhysicsConfig` fields (via `dataclasses.replace`, production defaults preserved):
- `kappa_gm: float = 0.0` (m²/s, GM eddy diffusivity; 0 = disabled, the default).
- `gm_slope_max: float = 0.01` (dimensionless slope limiter, standard ~0.01).

### Calibration & gate

1. **Coarse-grid gate** (the discriminating test): run the coarse 128×128 dx=111km no-adv test with GM ON. Success = injection arrested or reduced to sub-linear growth (max|u| bounded or slow-linear, NOT 9→86 exponential). Sweep `kappa_gm` over {500, 1000, 2000, 5000}.
2. **Fine-grid non-regression**: run the fine regional test with GM ON — must remain bounded (GM should be near-inactive where slopes are resolvable; verify it doesn't degrade the fine-grid result).
3. **Full-step gate** (the real test): full `_step_impl` with advection ON + real wind + bulk flux, 10-day probe. Gate = 10 days clean (no NaN, max|T| bounded, eta watchdog clean) — the standing push-release gate.
4. **pytest**: 55 existing tests must stay green; add a unit test for `_isopycnal_slope` (known stratification → expected slope) and bolus non-divergence.

### What this does NOT do (integrity guardrails)
- Does NOT relax A1/A2 thresholds or cook comparisons.
- Does NOT modify the validated spectral `jax_solver.py` (it has no GM because it runs fine-resolution where baroclinic modes are resolvable).
- Production `PhysicsConfig` defaults unchanged (`kappa_gm=0`); closure activated only via `replace(...)` in the global driver.
- Push remains HELD until the 10-day clean gate (both T + eta monitors) is met.
- Honest FAIL is acceptable: if GM bolus does NOT arrest the coarse/full-step injection, report it and reassess — do not tune kappa_gm to pathology.

### Risk / fallback
The main risk is the bolus velocity becoming singular or noisy where `∂rho'/∂z → 0` (weakly stratified columns). Mitigated by the stratification floor + slope limiter + dealias. If GM bolus proves insufficient even when correctly implemented, the fallback is the Redi isopycnal-mixing tensor (along-isopycnal rather than horizontal diffusion) — a larger implementation, held in reserve.

## Steps
1. Add `kappa_gm`, `gm_slope_max` to `PhysicsConfig` (defaults 0 / 0.01).
2. Implement `_isopycnal_slope`, `_gm_bolus_velocity`, `_gm_tracer_transport` in `jax_solver_global.py`.
3. Wire GM transport into `_compute_tracer_tendency` (gated by `p.kappa_gm > 0`).
4. Add unit tests (slope sign/magnitude, bolus non-divergence).
5. Coarse-grid gate sweep (kappa_gm {500,1k,2k,5k}) — verify injection arrested.
6. Fine-grid non-regression — verify no degradation.
7. Full-step 10-day probe with real wind + bulk — the push gate.
8. Commit (unpushed); report results honestly.
