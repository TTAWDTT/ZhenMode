# Solver controls and fixed real-reference inputs

Registered 2026-09-29 after the first direct failures and before the repaired
solver or real-grid momentum references. The preceding goal turn answered the
user's qualification question but did not change authoritative implementation;
classify it as no progress and take the available solver repair, not a wait.

## Rejected solver hypothesis and retained controls

The original direct suite has 3 failures and 15 passes, retained in
`results/industrial_alignment/cgrid_momentum_direct.log`. Two failures are an
incorrect manufactured-input strength assertion: analytic velocities agree,
but density amplitude0.1 yields only3.324e-7m/s, below the chosen1e-6 activity
witness. Increase ONLY this direct fixture amplitude to1.0; keep the analytic
accuracy and activity gates unchanged. The third failure is a genuine implicit
rotation VJP error, despite forward energy/residual and finite differences passing.

The four controls in `cgrid_rotation_adjoint_controls.json` give:

| captured absolute tolerance | captured initial guess | relative dot error |
| --- | --- | --- |
| 1e-13 times primal RHS norm | primal scaled velocity | 0.2858719181 |
| 1e-13 times primal RHS norm | zero | 1.0 |
| zero | primal scaled velocity | 1.320895808e-4 |
| zero | zero | 2.520578260e-15 |

The initial hypothesis that removing the absolute tolerance alone would repair
both starts was falsified. Retain the original report/log, including that failed
expectation. Select BOTH zero initial guess and zero solver absolute tolerance;
keep relative tolerance1e-13, maxiter100 and independent primal residual/energy
gates unchanged. The explicit primal arithmetic residual floor remains an
external acceptance check, not a captured stopping scale for the adjoint.

[JAX CG documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html)
requires convergence of both implicit solves. Inspection of installed JAX0.11.2
`jax/_src/scipy/sparse/linalg.py`, `_isolve`, confirms the SAME partial solver
with captured x0/tol/atol is passed for solve and transpose_solve. A large primal
initial guess and primal-scaled absolute tolerance are inappropriate for the
small reverse RHS of weighted physical velocities. This is a tested correction
in our calling configuration, not a blanket claim that JAX CG is incorrect.
Recheck all four controls, with only zero/zero expected to pass this fixture;
also check scaled cotangents and the active coupled predictor. This does not
qualify long trajectories or the whole-model adjoint.

## Eight real-grid cases, before running them

Use the existing real ETOPO180x66x14 geometries and interfaces through8000m:
smoothed80/minimum500 and unsmoothed0/minimum10, each with layer velocity64/32,
each with rest/nonzero density disturbance. Geometry/pressure/fast mode and
V/N inventories are explicitly64. Each case advances100 macrosteps of60s, BT4.
No fallback timestep, masking, precision policy or threshold is permitted.

Represented global density-anomaly reference coefficients in positive-down
metres are [1.0,1e-3,1e-7]. Inventory tracer0 is its exact physical cell mean
(kg/m3), plus0.1*sin(longitude) in disturbed cases; tracer1 is constant35.
Initially eta/u/v are zero. Derive density each step from N0/V, rather than
keeping a prescribed density while claiming active feedback. This synthetic
density-only experiment has no thermodynamic EOS or observed climate meaning.

Retain protocol gates: absolute surface discrepancy <=1e-12+1e-12*eta scale,
content budgets and constant concentration <=1e-12, initial global tracer bounds
within1e-12, valid pressure/rotation/fast/FCT/matched-Q stages every step.
Rest pressure <=1e-12m/s2 throughout; report rest velocities without inventing
a post-result bound. Disturbed pressure and layer shear must be nonzero, with
an activity witness maximum speed >1e-9m/s over the complete100-step run.
Report rotation energy/residual, physical V changes and closed boundaries;
archive first failed step and last accepted state rather than label rejection
as a zero-error budget PASS. Store source/input hashes and initial/final NPZ.

The full nonlinear production cutover, physical forcing/mixing/ice, independent
climate/forecast, century, polar topology, GPU/distributed and full adjoint
requirements remain unchanged and unmet.
