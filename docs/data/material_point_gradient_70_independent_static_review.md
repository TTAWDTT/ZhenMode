# Independent static review — material-point gradient v1

Reviewer Lady (`/root/scalar_parity_review`), 2026-10-13. Scope: uncommitted source/API/guard/control review in `D:/CodexWorktrees/thermal-full-scan/ocean-solver`, base HEAD `ece37c7b7ad383314e4dcfd982641d9ca541e1c3`. No src edits, new numerical diagnostic, native run, or repeated tests were performed by this reviewer. This continues the acceptance-gate and independence-bookkeeping review; it is a static development receipt, not independent numerical qualification.

## Verdict

**STATIC-SCOPE-ACCEPT for the implementation; readiness OPEN pending control repairs and new full CI.** No blocking defect found in the five production source changes. Two actual local test failures are identified below and must remain failure evidence, then be repaired in the test harness with the original criteria before treating the local controls as complete. Do not promote the source to a passed scientific/native/ranking result. Candidate momentum remains `dual_face_v1`; the failed ece point-momentum option is not selected. Best configuration source remains3c8.

The scalar operator change is confined to `_material_point_scalar_rate`'s two direct gradients: horizontal face concentration column moment, and vertical point desired RHS. Its helper owner still uses actual integrated `(fx*dz_node, fy*dz_node)` tracer faces. The change preserves full `−T*mass_residual/h`, baseline, mass owner, conservative projection, and joint scalar bounds.

## API, defaults, and dispatch

`make_solver_global` appends `material_point_gradient_scheme='minmod_v1'` after the existing arguments; prior positional meanings remain intact. The FDPhysParams field is appended with the matching default, preserving previous field indices/default alignment. `_material_point_scalar_rate` selects fourth only for the exact explicit string `fourth_uniform_v1`; omission and old param-shaped objects resolve to original physical_tracer_gradient. Public factory rejects unknown strings. Both temperature and salt scalar owners use this routine naturally; no momentum, continuity, or global vertical-gradient substitution is introduced.

The development contract appends a default gradient argument, serializes fourth only when nondefault in both ocean_options and development, and validates by reconstructing the same contract. Tampered options/declaration fail digest comparison. Frozen contract construction and its thresholds/profiles are unchanged. The existing worker `wave_zhenmode.integrate` passes ocean_options through `make_solver_global(**...)`, so the serialized selector reaches the actual native owner; no hidden adapter drops it. The default contract remains free of an extra serialized default option, and the declared old digest is checked by the local test.

A new namedtuple field changes parameter fingerprints even for default numerical behavior. This is expected: new installed-source identity and restart metadata must be generated, and old default numerical replay cannot be described as checkpoint metadata compatibility. The optional ece momentum implementation is present in the base source, but this gradient patch does not select it; contracts for the new experiment must leave momentum at dual_face_v1.

## Guard and primitive correctness

Factory fourth requires material_point_v4 and literal float64. Material-point already requires integrated_reference_v1, nodal_capacity, centered_fourth pressure/continuity, centered_second horizontal tracer, and no monotone_adv/fct_adv/project_adv_vel. The inherited integrated owner rejects partial columns, nonuniform spacing, dry masks, missing physical endpoints, split mode, non-closed meridional boundary, non-symmetric external mode, and fewer than6 nodes. The new check is downstream of those existing guards; it does not relax them. Default minmod remains legal in all prior supported owners.

The fourth primitive implements the exact reviewed five-point formulas in downward physical depth: endpoint weights (−25,48,−36,16,−3)/(12h) and (−3,−10,18,−6,1)/(12h), interior (1,−8,0,8,−1)/(12h), reflected negative right stencils. Spacing uses the first interface only, which is appropriate inside the factory's uniform envelope. Local concentration differences preserve constant zero. There is no slope epsilon clipping, state-dependent slope damping, grid-dependent parameter fit, or altered threshold. Standalone primitive assumes >=5 uniform wet nodes; the production supported owner remains>=6 and must stay so.

## Baseline, mass residual, and conservative structure

The original physical_tracer_gradient body is unchanged. Its original use in conservative endpoint surface allowance is unchanged, as are the MUSCL reconstruction, integrated_reference_scalar_rate, point_vertical_antiflux, and joint_scalar_blend bodies. The new operator is an ordinary local function selection; it does not monkeypatch their globals. Baseline params still only replace the vertical_tracer_scheme with muscl_band_v3. All mass_faces, Fz, shifted_numerator, mass_numerator, and reference handling remain unchanged.

The desired RHS retains complete mass residual, so a reversed/inconsistent Fz continues to produce a nonzero constant-tracer tendency instead of being clipped. Anti-flux surface/bed/stagnant-face pins remain under original point projection; rate stays low+alpha*delta with a single domain blend. Conservative budget closure relies on unchanged shared face divergence and pinned anti-flux endpoints. Actual new controls check this at the real scalar owner, not merely the derivative primitive. Static preservation of that algebra is accepted; arbitrary-state monotonicity/full-time stability is not inferred.

## Actual controls and repair requirements

Observed `controls-gradient-attempt01.log` reports **168 passed,2 failed,1 warning in96.70s**. No reviewer test rerun occurred. The two FAILs are test-harness defects, not measured PDE discrepancies:

1. `test_unlimited_target_preserves_bilinear_amplitude_parity`: under @jax.jit, `np.broadcast_to(amplitude*...)` attempts conversion of a JitTracer before any parity comparison. Use JAX broadcast or construct concrete shape profiles outside JIT, preserving the original manufactured fields and `atol=1e−20 C/s`. Both v and q amplitude-dependent broadcasts require review. The original target/parity test did not execute its claimed numerical criterion in attempt01.
2. `test_tests_use_support_instead_of_importing_other_test_files`: importing fields from tests.fd.test_material_point_transport violates the existing repository source-layout rule. Move a genuinely shared fixture to tests.support and update callers, or construct the fixture locally. Do not weaken/exempt the repository rule.

The new polynomial controls specify independently written analytic derivatives0..4 and true constant exact zero. Unsupported selector/owner/dtype checks cover new guards; existing integrated owner tests exercise its uniform/fully-wet envelope. The real owner constant, reversed-mass negative, Kelvin gauge, inventory, actual-Fz isolation, and low_rate selector equality controls use production integrated faces correctly. Tolerances1e−14 C/m, constant1e−17 C/s, gauge/inventory1e−13, and odd target1e−20 C/s are the declared unchanged criteria. These controls are development and regression evidence, already exposed to author/debugging; they are not heldout independent physics controls. Polynomial degree tests plus actual-owner controls protect distinct failure classes rather than restating the implementation alone.

Ruff/full CI completion has not been certified by this reviewer. Successful local control repair should be recorded as a separate attempt; full new CI is still required against the final frozen source. Fresh source identities and fresh unused active non-affine profiles/native gates remain distinct from local software tests. Neither the derivative controls nor the odd desired target alone proves improvement of every final-rate error.

## Evidence correction and lineage

The original controls-declaration.json cited scalar-point-gradient-parity-3c8/result.json, which is the rejected attempt01 with missing horizontal layer integration. This was identified to root during review. `controls-evidence-erratum01.json` now explicitly retains that declaration and rejects promotion of its original evidence, then designates attempt02-owner-units/result.json and its independent review as active evidence. That append-only correction resolves the reference issue; package/report provenance must retain this corrected link.

Attempt02 mechanism uses shared prescribed continuum fields/parameters and frozen installed3c8 source. It supports a local opt-in hypothesis, with odd response reduced to floating cancellation scale and fine-grid even error slightly worse; it does not establish a new native ranking. Those observations helped choose fourth and are now development contacts, so they cannot later be reused as untouched qualification points. Full qualification must remain OPEN until its actual declared criteria pass without drifting thresholds.

## Audited source snapshot

These file byte identities were obtained directly from the uncommitted checkout. Any subsequent source change requires an appended review delta. Test repair changes are permitted but must be recorded separately.

|file|SHA256|
|---|---|
|src/zhenmode/model/solver/dynamics/transport.py|d033239ea9139f05fc186d75605516f3c4d9e138f516dd9d34fa95095ebb3501|
|src/zhenmode/model/solver/numerics/vertical.py|6d5693b4a6dde1842be9e216e4674a42d53794b048189701620c02242e348dab|
|src/zhenmode/model/solver/factory.py|257343e3130fda018da8b77c45ae882cbe3197453378a24166e7a2a57e864ccd|
|src/zhenmode/model/solver/state.py|5b56be6af8df33327c1f0daab76728b1c081209ee1589e0b8df2f8ef5f2212d5|
|src/zhenmode/benchmarks/internal_wave.py|e9315d131847231fe9994abf8ad5dd7417584ff9842fa6130d0ddc63feb8dd1f|
|tests/fd/test_material_point_gradient.py (attempt01)|b6ce168710a1b836871abeee4c2311b830aecb2a1412077e1c33bd642f03cdc2|

`git diff --check` was clean at inspection. This is a content-bound static review, not a claim that the frozen133-file envelope, build, source manifests, full CI, or native pilot already exist for the new candidate.

## Append-only readiness review after harness repair — 2026-10-13

**READY-TO-FREEZE/BUILD within the accepted static scope.** The test harness repairs remove both identified blockers, and the actual affected-subset receipt reports48 PASS,1 warning,38.14s; supervisor exit0, wall39.4981s, one CPU,4GiB sampled group RSS budget, peak1114157056 bytes, no stop reason. No reviewer computation/test/native rerun was used. The all-library Ruff PASS is root-reported; this reviewer did not rerun Ruff. `git diff --check` remains clean. Full new frozen-source CI, source-envelope packaging, and qualification are still pending, so scientific/native/rank acceptance remains OPEN.

The repaired parity target uses jnp.broadcast_to and jnp trigonometric operations for amplitude-dependent v and q, preventing the former NumPy conversion of a JIT tracer. Grid/profile/amplitudes, exact algebraic parity meaning, integrated fx*h/fy*h owner, and1e−20 C/s criterion are unchanged. The original transport fixture function body is exactly408 characters and matches the moved support body character-for-character after the function rename. Both original and new scalar test modules now import the support fixture; the original source-layout rule is preserved. No production algorithm change was made to get these two tests through.

All five production-source file hashes below are identical to the first static audit. The170 declared control cases are the original157 plus13 new cases (5 polynomial,5 unsupported,1 contract,1 actual owner/baseline,1 parity). Record the actual runs distinctly: attempt01 was168 PASS/2 FAIL; attempt02 was48 PASS on the affected gradient/material-point/infrastructure subset, including both repaired failures. The union covers all170 control cases with at least one pass, but there was **no single170 PASS run**, and attempt01 must remain a failed run. This union is local regression evidence over unchanged production source, not full new CI or heldout numerical qualification.

The documentation's appended optional-gradient section correctly limits the method to opt-in material-point desired/moments, preserves baseline/minmod/default/guards, rejects attempt01 units, retains attempt02 cancellation-scale odd evidence and fine-grid even-error worsening, keeps dual_face_v1 momentum, and states full-source CI/independent controls/native pilot pending. It does not falsely claim native improvement. Reviewed documentation byte hash: c4dd040d3f78ec32adbfcd5f3f431bc5a61663415aa5567f2227674a053d9757.

### Test-hash label erratum

The first audit's table labelled b6ce168710a1b836871abeee4c2311b830aecb2a1412077e1c33bd642f03cdc2 as the attempt01 test. That label was incorrect: source reading and the later hash inventory occurred while root was repairing the tests. This hash belongs to the repaired current test file. The actual preserved failed-test-gradient-attempt01.py.txt hash is8862a5273beda4b8d8a91ce3b3457639a7b6f89c45a7ae77cf267ffda59e4244. The original audit text/table remains intact, with this append-only correction. Production-source hash binding and the recorded168 PASS/2 FAIL remain unchanged.

|final reviewed file|SHA256|
|---|---|
|src/zhenmode/model/solver/dynamics/transport.py|d033239ea9139f05fc186d75605516f3c4d9e138f516dd9d34fa95095ebb3501|
|src/zhenmode/model/solver/numerics/vertical.py|6d5693b4a6dde1842be9e216e4674a42d53794b048189701620c02242e348dab|
|src/zhenmode/model/solver/factory.py|257343e3130fda018da8b77c45ae882cbe3197453378a24166e7a2a57e864ccd|
|src/zhenmode/model/solver/state.py|5b56be6af8df33327c1f0daab76728b1c081209ee1589e0b8df2f8ef5f2212d5|
|src/zhenmode/benchmarks/internal_wave.py|e9315d131847231fe9994abf8ad5dd7417584ff9842fa6130d0ddc63feb8dd1f|
|tests/fd/test_material_point_gradient.py repaired|b6ce168710a1b836871abeee4c2311b830aecb2a1412077e1c33bd642f03cdc2|
|tests/fd/test_material_point_transport.py support import|2eae6ea65184a92e640266a327b9e6496e5bb05fddf7bb0b27c06c709a13cc04|
|tests/support/uniform_tracer_channel.py moved fixture|7bd1f38b58e54b017c7f58b65e9619f9f9f802f6af6f1d76f6d8e18d29f0bd1d|

Root may now freeze/build a new candidate within this source scope. Before selecting any further unseen source/math,1D, or ACTIVE2D controls, the reviewer must receive the new frozen source identity and binding package envelope; no new data/profile has been selected or evaluated here. Those future controls require their own predeclared plants, corruptions, resources/timing and contact/lineage record. Passing them must never reuse attempt02 observations as untouched holdout evidence or relax the original thresholds. New qualification candidate keeps momentum dual_face_v1, not the failed point-momentum combination.
