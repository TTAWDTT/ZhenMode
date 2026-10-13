# Saved native pilot audit — 70d4085

Verdict: **SCOPE-PASS with recorded provenance limitations**, for saved indices 0 and 1 only. This verdict covers file identity, contract linkage, native/output consistency, the original diagnostic geometry formula and an independently implemented readback of the fixed development metrics. It does not close an untouched physical qualification gate.

The audit declaration precedes raw-array access. `audit_saved.py` imports no project modules and calls neither a solver nor project scoring functions. It uses the declared linear analytic reference, a separately implemented pointwise evaluator and snapshot-wise dot products plus explicit time trapezoids. This shares the physical equations, original contract, NumPy double precision and already contacted development arrays with production. The candidate and its reported metrics existed before this audit, so this is a post-production arithmetic audit, not a blind or held-out reference. No fit, threshold change, native integration, source edit, commit or PR occurred.

Controls passed before real arrays were read: NPZ-serialized known trapezoidal RMS sqrt(1.5), exact zero RMS, constant inventory, identical-array positive twin; deliberately perturbed array, changed contract, altered bytes and incorrect capacities were rejected by the respective comparators. These controls test the audit machinery, not the physical solver. Attempt 1 then stopped before any native arrays because WSL Git could not interpret the Windows worktree pointer. Its failure is retained. Using the main repository's Git directory repaired that lookup; the successful finite audit took 1.724845 s, one CPU affinity and 71,921,664 bytes peak RSS, under the predeclared 180 s / 4 GiB cap.

All **133 Python files** agree across exact Git blobs at revision `70d408583aa99db63e49d958d89dcaccad273f0e`, the preserved Git archive, build inputs, wheel and installed package. Git blob SHA-1 was independently recomputed from archive bytes. The complete package hash registries in wheel identity, declaration, each row's run receipt and native producer match. Wheel SHA-256 is `7dbf38429a48355d2d10fe775c3bfd705d78497f19c9b832e4acddd6a5a22da2`. Input, contract, configuration, executable and saved output hashes link consistently. This identifies installed bytes; it does not retroactively prove each historical Python module's loaded path.

Both rows contain **41 complete instantaneous snapshots**, 0–60,000 s in 1,500 s increments, for 4,800 declared steps of 12.5 s. T/S/eta/u/v in native and adapted output are element-for-element identical after flattening. Every adapted numerical array is finite float64. Native capacities stored as `dz_node` have shape (1,1,16), endpoint half-widths 3.3333333333333286 m and maximum width 6.666666666666671 m. The capacities are exactly identical between old/default and candidate. Diagnostic `h` is exactly the original formula: broadcast this fixed `dz_node`, then add eta to the top cell. `h` stays positive, sums to H+eta, and adapted u/v volume diagnostics equal `h*area`. These diagnostic volumes should not be conflated with dynamically moving solver capacities: the native worker builds the fixed grid and saves `params.dz_node`.

Candidate `fourth_uniform_v1`, `material_point_v4`, `cubic_uniform_v1` and `integrated_reference_v1` are explicit and agree in the frozen contract, native actual_controls and output resolved_options. The worker passes `**c['ocean_options']` to the factory. `momentum_vertical_velocity_scheme` is absent from this dictionary and actual_controls; the frozen factory's default is `dual_face_v1`. Thus dual-face momentum is an effective default inferred from the exact source/worker call, not a separately saved runtime parameter. Neither native-run.json nor the output stores historical loaded Python module paths. These are explicit lineage limitations; the frozen source was not changed to conceal them.

Default row 0 agrees **exactly on all 17 nonmetadata adapted arrays** with the saved band-stage baseline. Candidate row 1 passes all unchanged original error and inventory thresholds; independently recomputed values agree with stored production metrics within the declared numerical comparator tolerance.

| Metric | Candidate independent readback | Frozen 3c8 independent readback |
|---|---:|---:|
| eta relative RMS | 0.0005387828306036288 | 0.0005388509039568597 |
| u relative RMS | 0.00003192140293356053 | 0.00003198853242504118 |
| v relative RMS | 0.000004082506786198244 | 0.000004440723369795227 |
| T RMS, degC | 7.088858339175953e-9 | 7.334242326890638e-9 |
| S RMS, psu | 0 | 0 |
| volume/T inventory/S inventory relative maxima | 2.220446049250313e-16 each | 2.220446049250313e-16 each |
| original composite max ratio | 0.005387828306036287 | 0.005388509039568596 |

The composite is strictly lower than saved 3c8 by **0.0126331%**. Candidate T also satisfies the separate pilot ceiling 8.530755561576719e-7 degC. This is a small descriptive improvement on a contacted development case.

Full-trajectory candidate-minus-3c8 maxima are T 4.974666012458329e-9 degC, S exactly 0, eta 2.3138961845587086e-11 m, u 1.885760712893366e-9 m/s, v 1.27318446868749e-9 m/s. All initial differences are zero; T/eta/u/v change during integration. Counts, plain-array RMS and final-snapshot differences are preserved in `row-01-result.json`; these comparisons are descriptive, without new physical acceptance thresholds.

Indices 2–9 had not been executed at the captured journal snapshot. Source/math/new-science and CI are separate gates. This audit does not mark CI 38306749152 PASS, cover future rows, certify global SOTA/full physics, or overwrite any earlier 1D/2D, 47/362, mean or arithmetic FAIL/OPEN evidence. Future saved rows require a new append-only audit entry for their unreviewed indices; indices 0 and 1 should not be rerun merely to accumulate more passing checks.

Evidence: `audit-declaration.json`, `controls-result.json`, `attempt01-failure.json`, `identity-result.json`, `row-00-result.json`, `row-01-result.json`, `audited-row-ledger.jsonl`, `audit-summary.json`, and the standalone `audit_saved.py`.
