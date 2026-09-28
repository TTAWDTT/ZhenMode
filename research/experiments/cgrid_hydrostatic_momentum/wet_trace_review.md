# Actual wet-contact reconstruction: frozen spatial traces qualified

Protocol3a91aed preceded regressions/core edits; implementationd7e24b9.
Actual linear_momentum_surface_step now reconstructs the SAME shared64 Q
already passed to V/N, returns its wet support and tests its validity. This
is not an unused diagnostic or a completed nonlinear/production migration.

## Construction and retained limits

W/E/S/N traces use actual shared wet intervals, not a full-height extension
through step-wall or moving-top wedges. Exact clipped vertical primitives
match interface Q and constant mapped cell divergence. These outputs are
face-integrated mapped fluxes, NOT physical point velocities. Their longitude
face trace is constant in normalized sin(latitude) area, not latitude arc.
The next physical reconstruction must address that distinction together with
dual fluxes and moving momentum support; changing transverse flux alone can
break mass commutation. Current D40/D41 masses and force arithmetic remain.

The missing API initially fails collection, not13 numerical operator tests.
The first implementation also fails10 tests under JIT because Python all
tries to coerce traced booleans. Replacing it with a JAX stack/reduction fixes
that implementation defect without changing any acceptance threshold. Both
initial logs remain.13 direct regressions subsequently pass;74 adjacent
tests include actual32/64 steps, pressure, wall traces and shared Q.

## Frozen-state provenance and exact comparison

The first baseline capture imported old code but collected worktree source
hashes at process END, after edits. Its wet_trace_frozen_inputs.json and
associated original snapshots are REJECTED as provenance-qualified evidence.
Do not silently treat the late manifest as the code loaded by that process.

The authoritative capture executes five immutable git blobs from3a91aed
(config, finite_volume, bounded_transport, barotropic_transport, cgrid_momentum)
and hashes those actual blobs. No source checkout/worktree change was needed.
wet_trace_frozen_verified_inputs.json and the four verified snapshots record
that baseline. For an irregular5x4 partial/land geometry, velocity64/32 and
explicit source on/off, V/N/u/v are bitwise equal in all16 field comparisons.
This proves only those four fixtures. Additional materialized JIT outputs
produce tiny roundoff differences in some disturbed real references; no claim
of bitwise equality for all800 real steps or all configurations is made.

## Clean implementation qualification

At cleand7e24b9, eight SAME real ETOPO cases,100 steps each at60s/BT4,
pass previous inventory/pressure/rotation/fast/transport gates plus wet trace
validity and bottom closure at every accepted step. Maximum recorded normalized
bottom error6.3472e-16; worst content budget relative4.95283e-18. Volume change
ranges from-0.00537109375 to+0.01318359375m3. These short linear references
are not century, climate, real-forcing or nonlinear time-coupling evidence.

Independent host geometry/inventory/local-Q audit passes all eight snapshots
and rejects nine existing metadata/flux corruptions. Separate host wet-interval,
side integral, net divergence and point primitive audit passes all eight and
rejects five wet-specific corruptions. Maximum absolute bottom and side
integral errors are6.11e-10/9.31e-10m3/s under the preregistered participating
flux gate. These audits independently reconstruct LAST snapshots and inspect
recorded stage flags, not every intermediate operator of each trajectory.

Full suite:483 passed,23 existing warnings,472.40s. Legacy MMS ALL PASS with
ratio4.30. Isolated wheel installation imports the installed wet module and
executes actual inventory64/Q64/momentum32 coupling, local V and wall checks.
Configured/research lint passed. No damping, geometry source or force repair.

## Reproduction and next actual work

- python research/experiments/cgrid_hydrostatic_momentum/run_reference.py
- python research/experiments/cgrid_hydrostatic_momentum/verify_reference.py --report results/industrial_alignment/cgrid_wet_trace_reference.json --negative-controls
- python research/experiments/cgrid_hydrostatic_momentum/verify_wet_trace_reference.py --negative-controls
- python -m pytest tests/test_wet_flux_reconstruction.py tests/test_cgrid_momentum.py tests/test_cgrid_pressure_work.py tests/test_momentum_shared_flux.py -q
- python -m pytest tests/ -q

Artifacts under results/industrial_alignment: cgrid_wet_trace_reference.json
and eight hashed NPZ; wet_trace_before.log; wet_trace_traced_boolean_failure.log;
wet_trace_adjacent.log; wet_trace_full_tests.log; wet_trace_inventory_verification.log;
wet_trace_independent_verification.log; wet_trace_state_comparison.json/log;
wet_trace_frozen_verified_inputs.json and verified NPZ; wet_trace_legacy_mms.log;
wet_trace_install.log. Original unqualified late-manifest capture is retained.

Next resolve metric physical trace/velocity and matched dual Q, full half-prism
mass, wall reactions, pressure/rotation/fast capacity and moving time coupling
TOGETHER; then implement ACTUAL nonlinear momentum and kinetic/buoyancy work.
Complete EOS/real forcing/mixing/ice and production initialization/restart/driver
cutover. Global topology, century/sensitivities, independent climate/forecast,
fair GPU/distributed, whole adjoint/learning and engineering remain unqualified.
