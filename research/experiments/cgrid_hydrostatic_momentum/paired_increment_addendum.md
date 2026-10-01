# Increment solve after a held-force witness

2026-09-29. Before changing the actual solve. The direct suite's first13
tests pass; expanded suite has three test-fixture NumPy/.at mistakes (fixed
without operator changes), then32 pass and one new density-force witness FAILS.
Retain logs `paired_dynamics_expanded.log` and `paired_dynamics_direct.log`.

With g=0, C=0, initially resting momentum but nonzero eta, solving for the
FULL endpoint lets the uncoupled large existing surface state dominate GMRES
relative stopping. The independently recomputed raw momentum-force defect
reaches0.02723581 (per-rho0 force integrated over time); endpoint velocities
and whole frozen energy gates alone do not detect this source-relative error.
This is NOT evidence of a physical budget failure in the old production core.

Use the algebraically equivalent block INCREMENT equation

    (H-h*L/2)*delta = h*(L*x0 + source)
    x1 = x0 + delta.

Compute its RHS directly, not by subtracting two large endpoint expressions.
GMRES solves the increment with tol1e-14. Independently check BOTH increment
and full endpoint residuals using the existing1e-12 relative +64eps norm
floor; report their maximum relative residual. No state/mean-Q/source repair,
no force-test tolerance relaxation and no original gate removal. This controls
small forced changes without pretending it eliminates endpoint addition
roundoff or qualifies moving/nonlinear dynamics.
