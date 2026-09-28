# Actual paired dynamics: qualification in progress

2026-09-29. Full industrial goal remains active and unachieved. Protocolf136f6f
precedes implementation; increment addendumab125d9 precedes small-force
stopping correction; core6fb6c88 actually advances momentum AND inventories
using the same midpoint Q. Not a read-only mass probe.

## Actual equations and method change

Choose physical wet-contact horizontal-field L2 kinetic mass, mixed entries
and true contact overlaps. Integrate planetary f at latitude quadrature nodes
and assemble its skew force in the SAME basis. Column B uses actual face
areas, surface pressure uses B^T; re-derive simultaneous midpoint rather than
reuse diagonal mobility or the old layer mean/deviation transport correction.
Recover hydrostatic FORCE by multiplying existing acceleration by its
declared common-wet mass before solving with NEW M. Actual mean Q drives
closed Qz, bounded V/N, wet/dual transport and physical velocity. Inventories,
solve/source/Q are64; momentum storage32/64. Frozen, cast and moving-geometry
kinetic work are separately exposed, not claimed as closed nonlinear energy.

## Evidence and retained failures

- Missing API first fails collection, not numerical operator gates.
- First13 direct checks pass; expanded tests find three NumPy/.at fixture
  errors (test-only fix). Held-density-force witness then FAILS: full-endpoint
  GMRES stopping is dominated by existing eta for a small momentum forcing;
  raw integrated force defect0.02723581. Preserve all logs and gates.
- Solve increments directly, checking BOTH true increment/endpoint residuals;
  no state/source/Q repair.33 direct tests pass in37.01s: independent physical
  integrals/dense midpoint state/Q/work, signed sources, actual density force,
 32/64 storage, temporal refinement, local tangent/FD and rejection guards.
- Disturbed NumPy canonical audits64/32 pass (momentum1.94e-15, energy3.78e-16).
  Installed active step outside src passes; legacy MMS stays ALL PASS (4.30).
- First real rest case100x60s passes runtime gates, but independent near-rest
  host work audit FAILS. Recomputed eta0 from V reassociates the last bits;
  relative near-zero PE changes. Initial partial report stays UNQUALIFIED.
  Reference PID33232 intentionally stopped before edits, not on a timeout.
- Addendumf6551fc registers actual starting eta and independent geometry floor
  ONLY for representation, plus global mean-eta to reject the B^T nullspace
  corruption. No energy, force, source or continuity gate relaxation.

Logs: `results/industrial_alignment/paired_dynamics_*` and
`cgrid_paired_dynamics_reference*`. Full-suite handle42215 completes:573 pass,
23 old warnings,673.50s; launch/end core/test hashes match (C27D2F6A/1BAE3D3C).
This is PRE-eta source, not final-source qualification. Eta-recording core
782cd63 then passes80 adjacent tests in59.16s. Final full/eight-case evidence
is still pending; no finished process is falsely described as live.

Final-source one-step smoke at clean1802e36 uses ALL eight immutable real-grid
inputs with ACTUAL new dynamics. Runtime and independent host audits pass;
9 momentum/Q/eta-geometry/eta-nullspace/force/V/N/work/metric corruptions reject.
Maximum independent momentum relative3.1085101e-15 and energy5.0355504e-16.
This is eight60s steps, NOT eight6000s histories or climate qualification.
Final isolated installed782cd63 active step also passes outside src; initial
eta is64 and same-Q/inventory64 with momentum32. New full/100-step processes
are pending; original unqualified partial evidence is retained.

Final full-suite handle91137 completes exit0:574 passed,23 old warnings,
863.58s. Launch/end hashes match ADAC8C8C(core)/4418BBA4(test). Installed final
step, ruff/diff/YAML and60 changed-document local links pass. Of the NEW
100-step cases, the first64 rest and disturbed histories/independent last-step
audits pass; others remain pending, not assumed passed. Near-rest host work
relative is now1.2204098e-17 without relaxing its gate; geometry eta difference
7.46e-16m is independently checked, not repaired. Disturbed moving-M work
at its last step is-111.4006 perrho0, still NOT claimed as nonlinear closure.

Preregistered source witness9259164 is reproduced by actual current kernel.
Launch/end-pinned report `paired_source_momentum_pinned.json` records uniform
rain1e-6m/s for600s, assumed zero incoming horizontal momentum, f/g/force0.
Frozen work residual iszero, BUT actual axial momentum rises1.2e-5 relative;
top speed stays0.1 instead of isolated-source-conserving0.0999914293m/s.
Independent kinetic gain1.40904723e8 perrho0 agrees with reported moving-M
energy within its explicit cancellation floor. This is a NEW physical source
contract FAIL, not a retrospective failure of declared frozen equations.
An incoming momentum/source and mixing-work model is required. Do not fix
this by arbitrary Mdot damping or post-hoc global rescaling. First late-only
hash witness is not authoritative; pinned f52737a script confirms unchanged
launch/end sources and explicit uniform endpoint speed before analytical L.

Actual nonlinear/moving-energy research now notes that cubic/piecewise
Hamiltonian time coupling cannot inherit frozen quadratic midpoint's energy
identity. See `research/literature/nonlinear_paired_energy_20260929.md`;
do not add arbitrary Mdot or post-hoc KE repair instead of actual advection.

## Required next work

Record starting eta, rerun final-source tests/installed step and NEW eight-case
real evidence with independent last-step mass/C/B, momentum/Q/work/cast/moving
mass and inventory corruptions. Old FV reference is a historical baseline.
Then ACTUAL nonlinear momentum/moving kinetic-buoyancy exchange and ACTUAL
initialization/restart/driver migration; finish EOS/real sources/mixing/ice.
Global topology, century/sensitivity, independent climate/forecast, fair
GPU/distributed, whole adjoint/learning and engineering remain required.
Zero wall-normal flow is NOT qualified wall reaction or industrial completion.
