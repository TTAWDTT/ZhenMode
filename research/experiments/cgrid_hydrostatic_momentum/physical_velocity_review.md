# Physical mean-Q velocity: qualification record, not production dynamics

2026-09-29. Protocol9a7fa6b preceded witnesses and implementationc08754d.
Coordinate addendumbe5e2f3 preceded corrected evidence recording6bd7f50.
This goal turn is PROGRESS: an actual linear-step physical frame, direct
boundary/continuity tests, full regressions and independent kinetic selection.
The complete industrial objective remains active; production is still legacy FD.

## Why the new view is needed

The existing mapped qz/A is a MATERIAL-coordinate transport view, not absolute
Eulerian downward velocity. Two surface witnesses first FAIL under that
interpretation; the partial fixture discrepancy reaches1.87441905e-6m/s.
This is not evidence the previous production driver used that interpretation.

`src/physical_velocity.py` uses SAME actual mean Q, starting geometry and top
volume source as the actual linear step. Physical east/north components use
the spherical flux mapping with custom radius inferred from actual metrics.
The absolute vertical lift is top netQ/A*(1-alpha); grid motion subtracts
the actual source/A; relative boundary velocity includes source/A. Lower
fixed interfaces keep shared Qz/A, seafloor/blocked normal flow stayszero.
Rain/evaporation are signed boundary inflow, not fictitious bulk compression.
No Q, source, inventory or velocity state update was changed by this view.

The constructor validates X64 geometry/Q/source, metric identities including
common spherical radius/periodic longitude, finite positive metrics, actual
lower continuity and top-only wet sources. Actual linear-step output includes
the reconstruction and rejects invalid frames. This is no unused helper.
Invalid point coordinates are not clipped and poles are not cosine-floored.

## Direct and regression evidence

-42 direct tests cover all four actual contact normals, shared interfaces,
  blocked wedges and walls, custom radius, partial/land/moving geometry,
  source/no-source/signed source-only rest, independent spherical JVP/FD,
  actual grid-motion/local V increments for32/64 momentum, malformed inputs
  and false point validity. Source and physical geometry/Q remain64.
-131 final-source adjacent tests pass in172.68s.
-Full suite540 passed,23 existing warnings,409.92s. Core/test SHA256 captured
  at launch and rechecked after completion; subsequent edits only change
  research evidence recording/verifiers, not these production/core test files.
-Four immutable3a91aed baseline fixtures retain all16 V/N/u/v fields exactly;
  the previous late-captured baseline remains REJECTED. No bitwise assertion
  for all real trajectories, step sizes, platforms or nonlinear physics.
-Legacy MMS ALL PASS with4.30 convergence ratio. Isolated installed active
  inventory/Q64,momentum32, actual physical surface frame and local V step PASS.
  The new module is in the wheel; imports resolve outside the repository.

Logs are under `results/industrial_alignment/physical_velocity_*`.

## Failures retained rather than changing the gate

The first direct test draft accidentally applied the free-surface inflow
condition to ALL fixed internal interfaces, and reused eager top coordinates
for a separately compiled step. Correct the test scope/use the actual returned
top, not the field or gate. Its10 failures are retained in
`physical_velocity_direct_expanded.log`; final42 cases pass.

Luna's preliminary kinetic probe incorrectly mixed east and north components
in the inner product and mislabeled diag(Gram) as old face masses. Parent
review fixes the vector dot product, actual face proxies, dimensions and units.
An independent depth-partition/physical-component integral then FAILS the
still-missing1/cos(phi) in the north bubble: relative errors0.0005063 and
0.0003738. Correct the basis, not tolerance. Symmetric weighted-basis product
assembly avoids separate roundoff-dependent entries; canonical physical-field
and Gram actions agree within7.79e-16. The full-wet zonal checkerboard field
has KE about0.3333633 of its same-coefficient lumped proxy. Different field/FV
quadratures are NOT retrospective failures of an old discrete KE norm.
All preliminary outputs are retained as `luna_unqualified_*` and
`physical_kinetic_canonical_*`; they are not authoritative gate6 successes.

Original eight linear histories at cleanc08754d PASS, but the independent
endpoint point audit FAILS: recomputed host top/bottom can land on the other
side of a discontinuous support after JIT reassociation. Stored top/height
alone still fail at bottom. The original report is retained UNQUALIFIED for
physical point evidence. Add actual compiled sample depth arrays; independently
validate those coordinates against V/area geometry at64eps stored-depth floor.
This floor applies ONLY to coordinate provenance; field/divergence/source
thresholds and actual support intervals are unchanged. No coordinate repair.

## Current real-reference and host qualification

The corrected SAME eight100x60s groups at clean6bd7f50 complete PASS under the
NEW `cgrid_physical_velocity_coordinate_reference.json` basename. Original
reports remain retained/unqualified for point evidence. Specific reference,
independent inventory/wet/dual/physical and kinetic process handles all finish
exit0. Host audits pass all eight last snapshots and reject9 inventory/Q,
10 wet/dual and15 physical metric/frame/source/coordinate/stage corruptions.
No existing gate is dropped. Maximum recorded lower bulk continuity iszero;
dual commutation4.432293e-16, bottom closure6.347202e-16. Global V delta remains
[-0.00537109375,+0.01318359375]m3; it is not rounded away or claimed as zero.

`physical_kinetic_actual_eight.json` records script/report/snapshot hashes and
independently constructs PSD symmetric physical Gram actions for all eight
actual mean-Q fields;48/96 quadrature passes the original1e-12 relative gate.
For dynamically disturbed groups, common-wet FV proxy exceeds field KE by
about0.898--2.952percent; full half-prism proxy by3.012--9.751percent, with SAME
actual Q-derived normal coefficients. Rest-group energies are only roundoff
and their ratios are NOT interpreted as physical errors. These data reject
equating all three metrics, not the use of a declared FV discrete norm.
Field-L2 adoption would need matching coupled mass/force/fast/time equations.

Authoritative logs: `physical_velocity_inventory_verification.log`,
`physical_velocity_wet_dual_verification.log`,
`physical_velocity_independent_verification.log`,
`physical_kinetic_actual_eight.log`. The first coordinate and kinetic FAIL
artifacts are retained, not retroactively relabeled PASS.

## Next actual production migration, with full objective unchanged

This field is actual macrostep MEAN Q on frozen starting geometry, NOT exact
endpoint velocity or a moving-time ALE/GCL/nonlinear/kinetic-power proof.
Sources in the eight real reference groups arezero; signed source tests are
small fixtures. Host audits check last snapshots and recorded stage flags,
not independent replay of every intermediate actual step.

Choose the physical velocity/kinetic norm explicitly. If the reconstructed
field L2 norm is used, its mixed terms require a coupled mass action; deriving
pressure, Coriolis, wall reaction and fast projection must be simultaneous.
Do not silently put kinematic half-prism mass into old force equations or
reuse diagonal C=sum(S^2/M) for a coupled mass. Then implement ACTUAL nonlinear
momentum/buoyancy work and moving source/time coupling in the actual step,
followed by actual EOS/sources/mixing/ice, initialization/restart/driver cutover.
Global topology, true century/sensitivities, independent climate/forecast,
fair GPU/distributed, whole adjoint/learning and engineering remain required.

Primary research and gate definitions are in
[`protocol`](physical_velocity_protocol.md) and
[`coordinate addendum`](physical_velocity_coordinate_addendum.md).
