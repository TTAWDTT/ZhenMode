# Actual surface-storage precision failure, before any dynamics/gate edit

2026-09-29. Runtime source d6dcf63, clean launch, unchanged end hashes. Specific
real handle97559 is terminal exit1; eight requested10x60s groups complete with
four disturbed PASS and four near-rest FAIL at step1. Preserve original
`results/industrial_alignment/nonlinear_dual_real_10.json/log` and all snapshots.
Do NOT turn four PASS into the registered eight-group qualification.

## Independent evidence, not a new physical source

On smooth rest both momentum residual1.7445e-16 and continuity gates pass,
while raw energy residual=-4.23614388437e-22 perrho0 exceeds unchanged
5.04374325582e-30 energy tolerance. Solved surface displacement is at most
6.35218934194e-18m; storing its volume increment on the large existing64-bit
top stock loses it, and actual diagnosed eta1 remains0. Raw rest similarly
fails (-4.60880779897e-22, tolerance8.57640214852e-30).

Independent NumPy computes the IDENTIFIED storage transformation, not the
unexplained budget residual:

    eta_solved = 2*eta_bar - eta0
    surface_storage_work = .5*g*sum(A*(eta_stored-eta_solved)
                                      *(eta_stored+eta_solved)).

For smooth/raw rest, subtracting this independent transformation leaves
8.045e-33/-1.881e-34, inside original gates. This is evidence for where the
finite representation loses PE, NOT permission to invent energy, relabel an
unexplained remainder, loosen tolerances, or claim conserved stored state.
Saved independent witness: `results/industrial_alignment/nonlinear_surface_storage_probe.json`.

The four disturbed last steps pass NumPy-only actual weighted-time/Q/dual
advection/pressure/curvature/impulse/reaction/V/N/work/cast audits. Worst true
momentum relative7.65344e-16, energy change/work relative4.57217e-16; eleven
deliberate corruptions reject. Their10-step histories are runtime checks,
NOT independent replay of every step or thermodynamic qualification.

## Primary research and next decision

[MITgcm nonlinear surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)
requires the SAME thickness/continuity form in volume and tracer transport;
[exactConserv option](https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html)
recomputes divergence after the pressure solve. Neither is a theorem that
unresolved perturbations survive volume storage. [JAX numerics FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
warns compilation may reassociate arithmetic; therefore actual compiled eta0
and eta_bar must remain recorded and independently geometry-checked.
MOM6's documented sub-roundoff thickness safeguards are NOT a transferable
license to clip our physical source or rescale kinetic energy.

Before a repair, compare two explicit engineering choices:

1. Compensated extensive volume/free-surface representation, used consistently
   through geometry, Q, momentum time identity, FCT, diagnostics and restart.
   It must preserve SAME local Q and the registered physical stored-state
   budget, not just carry an unused second eta or use unmatched topology.
2. Explicit measured storage work, separate from physics and velocity casts,
   with an independently derived arithmetic bound and cumulative numerical
   budget. This can diagnose finite representation, but must NOT be called
   exact stored-state energy conservation or a century certificate. It may
   not replace compensating state if representation drift exceeds requirements.

Choose based on independent precision/long-accumulation witnesses, not which
is easier to pass. Register selected state/time/serialization and bounds
BEFORE core edits. Add tiny-force/tiny-source cases, original rain witness,
deliberate false storage-work and altered eta/Q controls, then repeat ALL eight
NEW trajectories with new hashes. Original FAIL stays visible. Full
buoyancy/EOS/real physics/production/global, century and sensitivity, independent
climate/forecast, GPU/distributed and whole-adjoint/learning scope remains active.

Preregister a concrete representation comparison before running it: initial
volume3.5e11m3, signed increment +/-spacing(V)/32, N100000/1000000 steps; exact
target from Decimal.from_float at60 digits. Compare actual JIT plain addition,
compensated addition and compensated addition with an explicit JAX optimization
barrier. Plain addition must lose at least99percent of registered total source;
barrier-compensated high/low representation must match the exact binary target
on this fixture. Measure ordinary compensated behavior too; no presumed XLA
success. Record source hashes and retain all outputs. This arithmetic witness
does not yet qualify geometry, dynamics, FCT, serialization or century physics.
