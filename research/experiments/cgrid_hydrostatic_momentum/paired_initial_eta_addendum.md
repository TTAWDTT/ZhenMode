# Actual starting eta is part of independent near-rest work evidence

2026-09-29. Register BEFORE changing the kernel/evidence schema. First actual
eight-input experiment at clean6fb6c88 completes the first100-step rest case
with all runtime gates PASS; independent NumPy last-step energy audit FAILS
(`paired_dynamics_host_partial.log`). Report is NOT qualified by runtime PASS.

The host reconstructs eta0 from previous V/A-H. At a near-rest state its
last-bit reassociation differs from compiled eta actually supplied to the
solver. PE is quadratic in a value near zero, so a relative energy-only
comparison cannot distinguish this geometry-arithmetic uncertainty. The
ordinary disturbed canonical host checks64/32 pass (momentum relative
1.94e-15 and energy relative3.78e-16), but do not close the near-rest gap.

Record the ACTUAL starting eta returned from the active solver and compare
it independently to previous V/A-H at a64eps*(abs(V/A)+abs(H)) geometry
arithmetic floor. ONLY this coordinate/representation check has that floor.
Use the recorded starting eta for independent work; do NOT infer eta0 from
Q or the endpoint to force continuity to pass. Preserve existing pressure,
energy, volume, source, field and solve gates. No solver/state/force/Q repair.

Also independently check the area-weighted substep-mean eta: with zero source,
its global mean must equal the initial mean. A uniform eta-mean corruption is
in the nullspace of B^T, so a pressure-gradient-only audit cannot reject it.
Use the existing1e-12 m +1e-12*eta-scale gate, not a new energy tolerance.
New source hashes and report basename are required; retain initial partial
trajectory/failure. Deliberately stop known-unqualified reference work before
re-running; never duplicate a still-live process or restart on a mere timeout.
