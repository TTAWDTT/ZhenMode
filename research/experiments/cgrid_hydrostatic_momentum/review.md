# Review: linear momentum progress, but partial-face physical rotation FAIL

2026-09-29. Protocol047aed7, solver/input selection6a67a99, implementation41ccecd,
independent verifier/additional partial protocol0481332. **Overall physical
momentum qualification is NOT achieved.** Full industrial objective stays active.

## Implemented and independently checked

Physical geometry now retains global vertical and latitude interfaces.
`cgrid_momentum` integrates neighboring hydrostatic pressure on common wet
intervals, subtracts an explicit represented density profile for reconstruction
and restores its physical surface load. The active linear step evolves actual
layer velocities: rotation/pressure mean force enters fast momentum once,
deviations drive shear, and actual fast mean Q drives existing V/N FCT.
Density in the real references is recomputed each step from tracer0=N0/V.
Geometry, inventories, pressure and fast arithmetic remain64;32 layer velocity
storage is explicit. Pressure/physical faces are frozen within each macrostep.
There is no complete EOS, nonlinear momentum, real surface physics/mixing/ice or
production driver cutover, and no full temporal-order qualification.

The original rotation forward solve had a28.59percent local VJP dot error.
Four same-matrix controls falsify an absolute-tolerance-only explanation:
primal-scaled atol/zero start fails completely; zero atol/primal start still
errs1.32e-4. Only BOTH zero atol and zero initial guess pass, at2.52e-15.
Installed JAX0.11.2 uses the same captured CG solve for the implicit transpose.
Repair retains tol1e-13/maxiter100, independently checked primal residual floor
and weighted energy gate. Scaled cotangents, FD/JVP/VJP for pressure/rotation
and the active coupled predictor pass; not a whole-model or long adjoint claim.

Two first direct failures were a manufactured amplitude mismatch, not incorrect
analytic force:0.1kg/m3 produced3.324e-7m/s under a1e-6 activity threshold. The
precommitted selection increases this fixture's amplitude to1.0, without moving
the activity or error gates. A subsequent misplaced test-block failure is kept
in `cgrid_momentum_direct_fixed.log`; correcting its scope gives31 direct tests,
81 including prior physical/FCT tests, and440 full-suite tests with23 old warnings.
Configured/research lint and legacy MMS ALL PASS, convergence ratio4.30. The
independently installed module executes an active layer-pressure step under
isolated Python; generated build copies were verified untracked and removed.

## Eight registered short real-grid references

`results/industrial_alignment/cgrid_momentum_reference.json` and eight hashed
NPZ files originate from clean41ccecd. Same ETOPO180x66x14, interfaces through8000m,
smoothed80/minimum500 and unsmoothed0/minimum10, velocity64/32, represented
quadratic rest/nonzero density perturbation. All eight complete100x60s with BT4.
This is **6000 simulated seconds per group**, not a production climate run.

- Worst content budget relative error8.7507e-18, computed from summed cellwise
  changes rather than subtracting two giant global inventories.
- Signed water inventory changes -0.01294 through+0.000458m3; maximum equivalent
  mean surface change3.812e-17m. Do not report these as literally zero.
- Maximum surface identity discrepancy5.412e-16m; maximum constant-tracer
  error2.312e-13 and global bound excursion2.345e-13, below unchanged1e-12.
- Maximum pure64 rotation energy change4.151e-16, stored32 change2.084e-9;
  maximum true relative solve residual1.178e-16.
- Rest maximum pressure acceleration1.692e-19m/s2; velocities below8.891e-16m/s.
  Disturbed maximum speeds0.01863/0.03198m/s with nonzero vertical shear, so the
  pressure/momentum path is not a no-op.
- Maximum donor outflow fraction2.070e-4 and gravity bound0.002767; no timestep
  fallback, clipping, bathymetry change or after-result gate relaxation.

Independent `verify_reference.py` recomputes spherical area, partial thickness,
initial density, final contents/constants/bounds/deep geometry and water changes
from snapshots, and checks all100 recorded per-case stage histories and source
hashes. Five negative controls (duration/dtype/stage/hash/duplicate) are rejected.
It is NOT an independent replay of every pressure/rotation stage. An initial
area verifier computed diff(radians(longitude)) rather than radians(diff(longitude)),
losing cancellation accuracy by1.413e-14 relative. Original failure log retained;
rearranging the equivalent formula preserves its1e-14 gate. Separate evaluation
in a different operation order gave4.938e-15 discrepancy; no core/input change.
Old surface/precision/donor failures are not reclassified or overwritten.

## Additional physical counterexample: decisive FAIL

After re-reading MOM6 thickness/PV and MITgcm actual momentum equations, register
`partial_rotation_protocol.md` before running `probe_partial_rotation.py`.
The current sqrt(W)-scaled four-face interpolation is skew and energy preserving,
but that alone does not make it the physical fixed-z Coriolis force.

The21m column has a1m bottom face adjacent to30m bottom faces. On its common wet
interval20:21m all contributing north velocities are2m/s and f=0.001/s, so the
analytic east acceleration is0.002m/s2. The candidate gives0.006477225575m/s2:
**3.238612788 times expected**, relative error2.238612788 against the unchanged
1e-8 consistency gate. Yet the implicit60s rotation has energy change8.448e-16,
true solve residual4.722e-16 and returns valid=True. Raw report/log retained as
`results/industrial_alignment/cgrid_partial_rotation.json/.log`, exit1.

Cause: using a fixed four-face coefficient on sqrt(W)*velocity makes the physical
acceleration depend on sqrt(neighbor wet volume/local wet volume), rather than
integrating the neighbor velocity on the common physical wet interval. In this
witness the amplification is0.5*(1+sqrt(30)). The dense matrix test correctly
checks the chosen algebra but cannot prove this algebra is the physical force.
The uniform constant-depth test and near-rest real references miss this defect.

Do NOT call the eight short references whole physical momentum PASS, transplant
this rotation into production, or merely replace f with f+vorticity to claim
nonlinear advection. Next derive/verify physical cross-face overlap and momentum
dual-volume consistency, including moving tops, before adding matched nonlinear
momentum and switching actual driver/initialization/checkpoint/physical sources.
No force clamp, stronger damping, discarded shallow cells or relaxed gate is an
acceptable repair. Then repeat existing direct/real gates at the repaired commit.

Century reliability, independent climate and forecast, complete polar geometry,
GPU/distributed performance, whole-model adjoint/learning and production
engineering remain required. This turn makes concrete implementation/evidence
progress but does not satisfy or shrink the active industrial objective.
