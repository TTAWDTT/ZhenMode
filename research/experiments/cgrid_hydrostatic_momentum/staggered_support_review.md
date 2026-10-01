# Half-prism selection: kinematics passes, physical reconstruction NOT qualified

Preregistered c6aae95, with pre-diagnosis contact-trace clarification1b129ef.
Read stationary cut-cell capacities and mass-momentum-consistent VOF in
addition to MITgcm/MOM6/Herbin; applicability limits are in
[`staggered support protocol`](staggered_support_protocol.md).
The current industrial goal remains complete in scope and incomplete in fact.

## What was tested

No core state, geometry, force, Q or production driver changed. Offline trial
5929a58 uses ALL four actual-source snapshots from clean0eb56ed plus ALL eight
real ETOPO last-step snapshots with actual shared Q. Each input snapshot and
report hash is checked; the original real-Q verifier also passes. Proposed
momentum support is the union of complete wet HALF primary prisms, including
stepped-bottom wedges and wall dual half prisms. It is NOT the previous D40
common-wet rectangle support or the same velocity/energy representation.

The first probe omitted exterior zero flux entries for wall duals and failed
with a shape exception BEFORE measurements.88d59ca fixes those entries. The
first old-uniform-mode comparison also extended a uniform mode into closed
layers;3768060 masks the comparison exactly at open contact layers. Retain
both earlier artifacts, but use the clean3768060 report for authoritative
comparison. Neither probe error changes any production/linear-core arithmetic.

## Actual-input kinematic and algebraic results

All12 groups pass the registered offline gates, without shifting tolerances:
- Scalar-index exact-sphere wet-half-prism integration agrees with proposed
  masses. Including wall/closed prisms gives total dual stock equal to primary
  stock in each direction; no hidden removal of bottom wedges.
- Shared-Q dual divergence commutes with the half-prism mass map. Maximum
  absolute commutation residual is6.69e-10m3/s at large real-grid fluxes;
  acceptance uses participating flux, not total ocean stock.
- Actual local mass increments agree with mapped source minus dual divergence.
  Maximum fraction of the registered transported-amount/stored64-volume
  tolerance is0.0293. Existing min-height/ordinary-MAC rain FAIL is retained,
  not rewritten as a PASS under a different mass definition.
-24 dual-flux corruptions of1e8m3/s and one flat-latitude-weight counterexample
  reject. A separate sparse latitude divergence/mapping construction checks
  all12 input grids: matrix defect at most1.11e-16;12 more matrix corruptions
  reject. Its general matrix includes external wall entries; actual recorded
  wall fluxes are zero. The earlier unrestricted-wall matrix assertion failure
  is retained, not explained away by increasing tolerance.

Contact area S and mass M imply column mobility C=sum(S^2/M), minimum-energy
mode e=S/M, u_fast=e*Q/C and u_slow=u-u_fast. Offline mass-weighted orthogonality,
kinetic decomposition, layer/column force and conjugate pressure work pass
the registered1e-12 gates in all groups. This is necessary ALGEBRA, not an
implemented pressure/fast/nonlinear time-coupling or physical-accuracy proof.

For the preregistered constructed velocity profiles, the properly wet-masked
old uniform-velocity decomposition on the NEW proposed mass has cross-energy
as large as0.78774 of total KE. Old common-wet and proposed column capacities
have maximum symmetric relative difference0.31066. These are candidate-
geometry comparisons, NOT measured errors in actual trajectories and NOT a
claim the existing D40/D41 linear scheme violates its own energy definition.
One cannot swap mass alone and retain its old uniform correction/fast matrix.

## A physically invalid shortcut is rejected

Plain mapped RT0 averages constrain only integrated Q. If a nonzero primary
face Q is spread across a full wet primary face height, portions cross the
blocked bottom wedge instead of the shared contact. Using BOTH sides' actual
heights, the rejected extrapolation has maximum hypothetical wall flux about
1.94e5m3/s on smoothed disturbed cases and6.60e5m3/s on unsmoothed disturbed
cases. This is NOT wall leakage of actual recorded core Q: actual Q is still
on closed/shared faces and unchanged. Rest cases have only tiny roundoff Q;
rain-only controls have zero Q. Mass closure cannot validate that trace.

Authoritative result is KINEMATIC_PASS_PHYSICAL_RECONSTRUCTION_UNQUALIFIED,
not nonlinear-momentum/industrial PASS. Need trace-compatible enriched
reconstruction on actual contact intervals and geometric boundary capacities;
wet/closed velocity support and wall reactions must be defined. A local
vertical correction could enforce interior divergence once restricted traces
are specified, but has NOT been implemented or qualified. Moving-surface
time coupling, spherical metric/vertical momentum, physical pressure/Coriolis
accuracy, kinetic/buoyancy exchange and positive/CFL gates must be established
together before actual nonlinear implementation/cutover. Do not inject
arbitrary sources, clip force, broaden the wet face or use stronger damping.

## Reproduction and next mainline work

At clean3768060:
`python research/experiments/cgrid_hydrostatic_momentum/probe_staggered_support.py --out results/industrial_alignment/staggered_support_masked_uniform.json`

Artifacts in results/industrial_alignment:
- staggered_support_masked_uniform.json/log, with all12 input hashes and sources;
- staggered_support_matrix_audit.log;
- retained staggered_support_selection.json/log (old unmasked comparison);
- staggered_support_selection_shape_failure.log;
- staggered_support_unrestricted_wall_matrix_failure.log.

Configured and explicit research lint pass. The latest470-test/MMS/installation
qualification remains that of unchanged0eb56ed core; this offline diagnostic
does not add production tests or advance a new integration. Next actual work
must resolve the wet-contact/velocity reconstruction, then implement paired
nonlinear momentum/fast physics and migrate the ACTUAL initial/restart/driver
with EOS/real forcing/mixing/ice. No endless unused reference alternative;
global topology, century/sensitivities, independent climate/forecast, fair
GPU/distributed, whole adjoint/learning and engineering remain required.
