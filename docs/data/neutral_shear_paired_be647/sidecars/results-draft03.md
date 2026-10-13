# Neutral shear, paired horizontal product — attempt02 result draft03

**DRAFT03 — three main cases complete; controls and final stage summary pending.** This version adds the completed fine case 2 terminal review to the already completed cases 0/1. Prior cases were not rechecked. Cases 3/4 and the original final summary remain outside this version. [Draft01](results-draft.md) and [draft02](results-draft02.md) are preserved unchanged; draft02 SHA256 is ed21d8f1eb30e9791a21e0abc1e47976e10e84a617bad80be526d8d90b61b9e4. This document issues no execution permission. Evidence remains contacted DEVELOPMENT.

Case 0 passes the unchanged scientific barriers and the strict comparison against the saved historical coarse T_RMS. The new T_RMS is `0.00004407068051203133`, below the old saved value `0.00037185579607669175`. The T and S RMS values decreased; u, v, eta and rho RMS increased slightly while remaining within their original barriers. No statement that every error decreased is supported.

The independent terminal review is [PILOT_TERMINAL_REVIEW01.json](review/PILOT_TERMINAL_REVIEW01.json), SHA256 `c0d5951bc49358540c1ac956f7cca66bf9570badaedcb76e2896eb1554a5af86`, with [readable scope and limits](review/PILOT_TERMINAL_REVIEW01.md). Its status is PASS_RECORD_BINDING_REVIEW: 473 metadata/hash checks, 146 distinct file identities and 35 actual loaded source paths. The present report transcribes existing receipts; the terminal review did not decode arrays, run a model, call an oracle or rerun a scientific audit.

## Frozen source and actual execution identity

| Item | Identity |
| --- | --- |
| Solver revision | `be647d70fc7394f3014dca8bc776e207d4f02ae3` |
| Selected wheel SHA256 | `44cf1162882fbd745be9b40221800858ac83de950590cca315c8ece544db583a` |
| Wheel/source manifest SHA256 | `c217495711be5216ea139219bc42088dd3f05285d851955f691625dd9a375848` |
| Selection SHA256 | `6af6737ad8eca03cd588dbd5551a0635525449cdd58eea1cf7a1d555764a842a` |
| Attempt02 declared plan SHA256 | `e8fa4dcd323f464777e441f7aa6b14fdf2b8a4e2150dc883065f66254688a165` |
| Definitions SHA256 | `f09f42e9970d48dfd33649f2f7dd58bd9871c707ab92d6ddb3e170b6b891a703` |
| Producer bundle SHA256 | `67ecb7e0b1a77cc8ea0480b4c11c8b9a311dc2f403fd678a4ddab0aaa6866188` |
| Observer bundle SHA256 | `e6a53b6c126e9b5cea0011376e86a14b722a04bce286fb36fde3951f5d879dda` |

The installed package is reused from `horizontal-paired-product-v1/installed`. The complete 133-file source maps captured before and after case 0 match the selected manifest. The case 0 worker independently recorded 35 loaded ZhenMode module paths, all in that selected installed tree, and their bytes match the manifest. The appendix reproduces those actual paths and SHA values from the terminal review record.

The recorded runtime used Python `/root/.venvs/zhenmode-managed-bench-20261005/bin/python`, JAX 0.11.2, NumPy 2.5.3, x64 enabled and device `cuda:0`. Actual selected options include `paired_product_fourth` tracer transport, `fourth_uniform_v1` material-point gradient, `dual_face_v1` momentum vertical velocity and `binomial_v1` tangent filter. Existing full CI for this same solver revision is reused: 1,799 pytest passes and the previously reviewed MMS/Ruff outcomes. No second CI run is claimed for this unchanged function or adapter-only successor.

## Problem and fixed observation rules

The locked configuration has H = 80 m, Lx = 24,000 m, Ly = 32,000 m, constant initial eta = 0.025 m, f = 0.00013 s^-1, dt = 12.5 s, duration = 60,000 s and output spacing = 1,500 s. Temperature and salinity reference values are 15 degrees C and 35 psu. The declared buoyancy profile is linear in z with a cosine meridional perturbation; geostrophic zonal velocity varies with y and z. A compensated tracer cosine is advected by that analytic velocity. Full expressions, coefficients, cases and gates remain in [declared-plan.json](declared-plan.json).

The native trajectory evolves its velocities (`freeze_adv_vel=false`). The analytic momentum acceleration for this profile is zero. Thus this is not evidence for arbitrary nonlinear momentum acceleration. Sources, mixing, drag, ice and polar forcing are disabled as declared.

The independent observer evaluates the original fixed physical-z point definitions. It uses actual area and actual nodal capacity for weighted errors and discrete inventories, including actual eta in the observer's top capacity. This does not alter the source's static native capacity. The upper-cap transport and background-stock discrepancy remain separately reported. Archived w/Fz values do not establish a new vertical-velocity accuracy qualification.

Original barriers are retained: T_RMS <= 0.008 degrees C; S_RMS <= 0.002105263157894737 psu; normalized u/v/eta/rho errors <= 0.1; volume drift <= 1e-10; T/S inventory drift <= 1e-8; finite fields, positive native h, T >= -1.8 degrees C and S >= 0. Initial/null tolerances remain 5e-11 and constant-source core RHS tolerance remains 1e-17. Three-grid requirements remain adjacent normalized T/S L1 improvement ratios >= 1.25 and finest normalized L1 <= 0.05. The finest L1 gates now pass in the original case 2 receipt; adjacent-grid summary gates remain pending and are not inferred from RMS improvement.

## Failed predecessor and explicit analytic reuse

The original `neutral-shear-paired-be647` coarse initialization failed with exit 1 before a constant-source archive, independent initializer PASS, timing or full native run. The constant controls used `jnp.full(shape, ph['Tref_C'])` and its salinity equivalent. JSON integers 15 and 35 led to int64 fixtures; the paired transport float64 guard correctly rejected them. Its parent wall time was 12.832388576 s and sampled peak RSS was 934154240 bytes. The failure disposition SHA is `87b289e623995e7c77aae0ca36b713ce283e03bead04c14c05acaa879fbd18da`. Saved physical initial arrays and parameters from before that failure are not an old initializer qualification.

Attempt02 uses a separate directory and new once-only identities. Its producer changes exactly those two calls to add `dtype=jnp.float64`. Both reference constants are exactly representable in float64; their values, science thresholds, six observer/core files and definitions are unchanged. The predecessor is preserved read-only. No failed launch or verdict was replaced with PASS.

The original 176 analytic checks had already completed once, before the failed initialization: actual exit 0, parent time 1.0028088669999988 s and sampled peak RSS 34467840 bytes. Their original verdict SHA is `cc80ec648fabcc1d65fb64b48b1bb97e232f848c652f3c1d30acf580d685dd15`. They were not rerun for attempt02.

Instead, an explicitly gated standard-library metadata bridge checks the origin records and current source, definitions, cores, package, physics, modes and gate identities; plan differences are limited to scope/declaration metadata. The [new binding](review/analytic01-binding.json), SHA `bb64108e4ecab249abed7acabc5ae793dae4dfa6022f352e99dc0bee25ad0a07`, says `receipt_kind=REUSED_FROM_PRIOR_ANALYTIC_CHECK`, `reuse=true` and `new_observer_calls=0`. It directly references the old verdict and its actual successful gate/launch/resource chain. Its own metadata-only execution took 2.007637579 s and sampled 20897792 bytes peak RSS. Those costs belong to the bridge, not to a fictitious new analytic execution. There is no new-stage copied analytic verdict posing as fresh execution.

Attempt02 case 0 then obtained its own physical initialization, constants, actual parameters, receipt and independent four-SHA binding. The binding SHA is `b6472e94c8df83f7199f21d06f75207a571909c4b387e1714d811873dcc381b8`; the initializer verdict SHA is `d642f1ae792e27c0efb15f9f9cbd3d2a77e3521c97f3570276912c8b621473e3`. The terminal review preserves all four actual file hashes rather than substituting a predecessor's initializer result.

## Completed coarse result

Case 0 uses 8 x 12 x 17 points. The worker reports one completed full native run, 4,800 steps and 41 outputs. Its run receipt and terminal record are byte-identical. Independent adapter and science receipts bind the raw native output, transformed audit archive, metadata, source context and original thresholds.

| Whole-trajectory metric from saved receipts | Old coarse receipt | Attempt02 case 0 |
| --- | ---: | ---: |
| T RMS, degrees C | 0.00037185579607669175 | 0.00004407068051203133 |
| S RMS, psu | 0.00009785685754199627 | 0.00001159755332239367 |
| u RMS, m/s | 6.267188586846027e-7 | 6.267190974384345e-7 |
| v RMS, m/s | 3.9936859710475823e-7 | 3.993686689417414e-7 |
| eta RMS, m | 1.610920454078542e-7 | 1.610920517871372e-7 |
| rho RMS, kg/m^3 | 3.203975302594049e-8 | 3.208585535915747e-8 |

The [science verdict](00-main/science-verdict01.json) is PASS, SHA `abec6f2aa16308cd012e1a64686241ae824670fe0bef5cad2a1db3d7d15027bb`. The [strict comparison](review/strict-coarse-comparison01.json) has status PASS_STRICT_SAVED_RECEIPT_COMPARISON, SHA `6fe28229f57788384eaa1ec2571aeefbb9661c2b586ab42e3529580a5ef693de`. It compares finite saved decimal T_RMS values by strict less-than, without slack, rounding or a substituted ratio. The historical verdict SHA is `147163432215ed57ab88037b1974f09c65dbf859a2f2d43ec3cb436c737da837`. Historical arrays were not reopened and the historical metric was not recomputed.

The science journal records 220 observations, 212 assertions and no failed assertions. Reported volume relative drift is 0; T inventory relative drift is `3.857457593415844e-12`; S inventory relative drift is `2.2699437768703233e-16`. Both constant-source RHS observations are 0. Reported minimum h is 2.5 m, minimum T is 13.816318021268343 degrees C and minimum S is 34.9790962402127 psu. All original field, density, inventory and safety barriers pass in that receipt.

The descriptive upper-cap transport discrepancy is `6.885974213414107e-8` m^2/s and its T background inventory proxy error is `3.750000000000001e-6` degrees C m. These are retained descriptive diagnostics, not new PASS thresholds or evidence of a generic moving-surface qualification.

## Timing and actual resources

| Case 0 phase | Actual parent wall time, s | Sampled peak host RSS, bytes | Outcome |
| --- | ---: | ---: | --- |
| Initialize | 16.241666230000007 | 1003831296 | Exit 0; own independent initial binding |
| Timing prefix | 28.470822485 | 1244246016 | Exit 0; 24 native calls only |
| Full native | 33.683090842 | 1244717056 | Exit 0; 4,800 steps, 41 outputs |
| Independent initializer observer | 1.2048087579999995 | 34525184 | PASS |
| Independent adapter | 1.805594947 | 40923136 | PASS binding |
| Independent science observer | 1.4038625919999994 | 40710144 | PASS |

The timing prefix used two reset 12-step windows of the same actual compiled stepper; it was not a second full run. Recorded compilation time was 15.379697265999999 s and the two windows were 0.16727922799999817 s and 0.09288325300000011 s. The saved planning forecast was 143.0737705274989 s. This forecast is distinct from the subsequently observed 33.683090842 s full-run parent time; no new timing fit was performed for this draft.

The full-run worker separately reports compilation of 15.318990342 s, integration and diagnostics of 5.496945495000002 s, and producer elapsed time of 28.522851455999998 s. These worker intervals are not interchangeable with parent wall time. Native limits were one CPU, 600 s and 4 GiB host memory; observer limits were one CPU, 180 s and 4 GiB. Host memory is sampled process-group RSS, not a hard cgroup cap. GPU 0 with allocator fraction 0.4 does not imply a whole-device memory cap.

## Completed case 1 extension

**PASS_RECORD_BINDING_REVIEW**, case 1 only. Reviewer: Lady / arithmetic_static_fresh. The machine record [CASE1_TERMINAL_REVIEW01.json](review/CASE1_TERMINAL_REVIEW01.json), SHA256 `427847be63658c62306b7f0662c95c32da7c2859a3c7833ccb878b8844af494e`, contains 1,220 metadata/hash checks and 57 file hashes. All checks passed. Array archives were hashed only as opaque file bytes; no NPZ was decoded, and no model, oracle, scientific core or observer was executed.

The chain checked here covers case 1's own initializer, four-file independent binding, two-window timing, immutable case-specific allowance, native phase gate, launch/worker markers, actual successful parent resource receipt, native and terminal byte identities, adapter, original science verdict and observer input/output bindings. The three worker phases each recorded complete before/after source maps matching all 133 entries of the sealed manifest. The full native worker's 35 actual module paths all point into the selected installed tree. This extension compares recorded source captures to the sealed manifest; it does not rehash the immutable installed source or rerun source/CI qualification.

Case 0 is inherited from the completed terminal review SHA `c0d5951bc49358540c1ac956f7cca66bf9570badaedcb76e2896eb1554a5af86` and was not rechecked. The analytic bridge and strict coarse comparison references are matched to that review's recorded SHA identities without reopening those records. The run gate's coarse science-binding prerequisite is retained as a previously root-qualified prerequisite, explicitly without a new check in this extension. This scope distinction is represented in `inherited_prerequisites_not_rechecked`; no whole case 0 chain is newly certified.

The case 1 science verdict is PASS, SHA `bb4d6ffcf6347a42486e798b9960702827c10fcc752edf1c5223adeef7b782d8`. Its recorded T_RMS is `2.906948527432644e-6` and S_RMS is `7.651418903982357e-7`. Recorded normalized T/S L1 values are `3.443243607504523e-5` and `3.443516315837632e-5`. The science journal contains 220 observations and 212 asserted checks, with no failed assertion. This is transcription of the original result, not a second numerical calculation. No historical case 1 strict-improvement threshold is introduced.

The native receipt reports 4,800 steps and 41 outputs, exit 0, actual parent wall time `51.128220907` s and sampled peak host RSS `1448624128` bytes. Native limits are one CPU, 600 s and 4 GiB sampled process-group RSS; observer limits are one CPU, 180 s and 4 GiB. Sampled RSS is not a hard cgroup limit. The run receipt SHA is `186b1618bd7bf3a710468149fa2ca7fbca6cf2c2abfb2659489ae02641a68bea`; terminal.json is byte-identical. The raw native SHA is `13d529b508e2625ede9ef3ee91a74400bc1d0e1bb12cd3a527f4612ef1b95b41`.

Original volume/T/S relative inventory observations are respectively `0`, `2.412962835030336e-13`, and `3.2914184764619687e-15`. Both constant-source RHS observations are zero. The descriptive upper-cap transport discrepancy is `6.930522393877726e-8` m^2/s; the T background inventory proxy remains `3.750000000000001e-6` degrees C m. These descriptions do not create new acceptance thresholds.

The preceding case 1 review retains its original evidence cutoff. The later case 2 completion is added separately below; zero-spice case 3, zero-shear-null case 4 and the final summary remain pending. No permission is created here. Contacted DEVELOPMENT, static-H/eta-cap and zero analytic nonlinear-acceleration limits remain unchanged.


## Completed fine case 2 and three-main receipt comparison

Case 2 (32 x 48 x 65) completed its one original 4,800-step/41-output native run, independent adapter and original science checks. Its science verdict is PASS, SHA `e28264427788a7bcfb5515e882cbe2941e36c29b27c3be7573b4b9bdf6deacdb`. The records-only [CASE2_TERMINAL_REVIEW01.json](review/CASE2_TERMINAL_REVIEW01.json), SHA `39d48111fba8ce36b247e9ae66aab6fa477aeb61a0539eff340b6dad07182756`, preserves 1,223 metadata/hash checks, 60 file identities and 35 actual worker paths. Its [scope statement](review/CASE2_TERMINAL_REVIEW01.md) separates recorded source captures from a fresh source audit. Cases 0/1 were inherited without rechecking.

The fine journal contains 222 observations and 214 assertions, with no failed assertion, including both original finest normalized-L1 gates. Reported volume/T/S relative inventory drifts are `0`, `1.5046820399849035e-14` and `2.4969381545573554e-15`. Both constant-source RHS observations are 0. Minimum h is 0.625 m, minimum T is 13.810397186262488 degrees C and minimum S is 34.978947362075 psu. The descriptive upper-cap transport discrepancy is `6.941674363056637e-8` m^2/s; the background T inventory proxy error remains `3.750000000000001e-6` degrees C m.

The following table directly transcribes the three new main-case receipts and the three saved historical source70 JSON receipts. Historical arrays were not opened, and no old metric or displayed ratio was recalculated. The strict historical acceptance comparison remains the originally declared coarse T_RMS rule; medium/fine comparisons below are descriptive, not new acceptance gates.

| Grid | Metric | Saved source70 | Selected be647 attempt02 |
| --- | --- | ---: | ---: |
| 8 x 12 x 17 | T RMS, degrees C | 0.00037185579607669175 | 0.00004407068051203133 |
| 8 x 12 x 17 | S RMS, psu | 9.785685754199627e-5 | 1.159755332239367e-5 |
| 8 x 12 x 17 | u RMS, m/s | 6.267188586846027e-7 | 6.267190974384345e-7 |
| 8 x 12 x 17 | v RMS, m/s | 3.9936859710475823e-7 | 3.993686689417414e-7 |
| 8 x 12 x 17 | eta RMS, m | 1.610920454078542e-7 | 1.610920517871372e-7 |
| 8 x 12 x 17 | rho RMS, kg/m^3 | 3.203975302594049e-8 | 3.208585535915747e-8 |
| 16 x 24 x 33 | T RMS, degrees C | 9.50470031446601e-5 | 2.906948527432644e-6 |
| 16 x 24 x 33 | S RMS, psu | 2.501285789334958e-5 | 7.651418903982357e-7 |
| 16 x 24 x 33 | u RMS, m/s | 3.937124966194141e-8 | 3.940144359945673e-8 |
| 16 x 24 x 33 | v RMS, m/s | 2.510846444238586e-8 | 2.5144035723863297e-8 |
| 16 x 24 x 33 | eta RMS, m | 1.0133159533964772e-8 | 1.0134820447526767e-8 |
| 16 x 24 x 33 | rho RMS, kg/m^3 | 8.810050518488863e-9 | 9.997809298960032e-9 |
| 32 x 48 x 65 | T RMS, degrees C | 2.3893683930442902e-5 | 1.8639626226425045e-7 |
| 32 x 48 x 65 | S RMS, psu | 6.28811511182653e-6 | 4.851795175610486e-8 |
| 32 x 48 x 65 | u RMS, m/s | 2.9475166435966397e-9 | 2.959939457146428e-9 |
| 32 x 48 x 65 | v RMS, m/s | 3.006819882529949e-9 | 3.04283398177509e-9 |
| 32 x 48 x 65 | eta RMS, m | 6.718963122851496e-10 | 6.730816123352594e-10 |
| 32 x 48 x 65 | rho RMS, kg/m^3 | 5.724012971667923e-9 | 5.8147541560952455e-9 |

T/S errors decreased on these three saved comparisons, while u/v/eta/rho RMS are higher on all three grids. The original barriers pass; there is no claim that every error improved. The historical coarse/medium/fine JSON SHA values are respectively `147163432215ed57ab88037b1974f09c65dbf859a2f2d43ec3cb436c737da837`, `17949d807a332500e848e2040bb38bc906954a78cb32878f4966e8bb58ebe89d` and `1c5e1a10d2b2ee18f49e8a170fd1ca6b68448211fa7ee70300e750399fd30cbe`.

| Main case | T normalized L1, saved source70 | S normalized L1, saved source70 | T normalized L1, new | S normalized L1, new |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.004432841208314006 | 0.004432843590280222 | 0.0005232532100036479 | 0.0005232674350391263 |
| 1 | 0.0011315530270999273 | 0.0011315965978100839 | 3.443243607504523e-5 | 3.443516315837632e-5 |
| 2 | 0.0002850273731176967 | 0.00028505070362881165 | 2.241470620123557e-6 | 2.1978454536730308e-6 |

## Actual resources across the three completed main cases

Every full native row below represents one completed original run of 4,800 steps and 41 outputs. These parent wall/RSS values come from actual resource receipts, not the planning forecasts.

| Case | Initialize wall, s | Timing-prefix wall, s | Full-native parent wall, s | Full-native peak sampled RSS, bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16.241666230000007 | 28.470822485 | 33.683090842 | 1244717056 |
| 1 | 18.444258705 | 38.096644565000005 | 51.128220907 | 1448624128 |
| 2 | 19.448346873 | 34.485600898 | 81.60238648 | 2062200832 |

Case 2's actual initializer observer took 1.4039101380000005 s with peak RSS 63660032 bytes; its adapter took 13.860633655 s with 565252096 bytes; its science observer took 5.421896773 s with 470335488 bytes. All report exit 0 and no stop reason. Case 2's worker separately reports compilation 20.914941177000003 s, integration/diagnostics 46.280012261 s and producer elapsed time 76.383623501 s. These intervals are not substituted for parent wall time.

Before the fine full run, the actual two-window timing prefix reported compilation 20.756259608000004 s, windows 0.34567720000000435 s and 0.08286946999999856 s, and peak sampled RSS 1374797824 bytes. The sealed [case2 measured plan](case2-measured-plan01.json), SHA `a2f44f1e02b791df01a74413358b6885c7fc83352dbc41bdbeb6573e51194f48`, records an initial-archive ZIP-member-size inspection only: all 17 member sizes total 8,040,208 bytes and its 41-output allowance is 329,648,528 bytes. The allowance includes extra acceleration and coordinate members beyond the full stored history.

The saved formula `1.25*(measuredlargestprefixRSS+3*41*ALL initialNPY member bytes+64MiB)` gives a recorded forecast of 3,038,565,340 bytes; the recorded time forecast is 259.1347213470026 s. These planning estimates were below the declared caps and are distinct from the actual fine native outcome above. This report does not repeat ZIP inspection or forecast arithmetic. Sampled process-group RSS, one-CPU affinity, native 600-second and observer 180-second limits remain the actual resource policy; it is not a hard cgroup or whole-GPU-memory certification.

## Physical and independence boundaries retained

The locked analytic potential is `Phi = g*eta + b0*(z-eta) + N2*(z^2-eta^2)/2`, with `b0 = Bamp*cos(pi*y/Ly)` and constant analytic eta. The source pressure diagnostic represents anomaly potential times rho0; it is not a newly interpreted total hydrostatic pressure. Fixed physical-z point observations, static native capacity, eta-cap transport/stock discrepancy and the zero analytic nonlinear momentum acceleration restrict the physical claim. Native velocities evolve; that fact does not turn this special analytic profile into a generic accelerating-flow test.

Producer and observer authors are distinct. Lady owns the outer observer bindings and this records-only extension, so this work is independent of the producer but is not a fresh independent audit of Lady's own wrapper implementation. Original scientific cores and thresholds are inherited unchanged. Root's separate fresh-context report/frame review remains required before final publication; no fresh frame verdict is invented here.

Three main cases now have original science PASS receipts. The original whole-stage summary has not been executed in this report's evidence cutoff: its wrapper awaits both native control PASS receipts before one final summary. Case 3 zero-spice flow and case 4 zero-shear null remain pending here, even if their workers advance after this draft is written. No current worker state or additional permission is inferred. There is no complete five-case PASS, holdout qualification, generic fourth-order proof or new ranking claim in draft03.


## Stage tables and remaining qualifications

| Main case | Grid | T normalized L1 | S normalized L1 | Science verdict in this draft |
| --- | --- | ---: | ---: | --- |
| 0 | 8 x 12 x 17 | 0.0005232532100036479 | 0.0005232674350391263 | PASS; terminal identity reviewed |
| 1 | 16 x 24 x 33 | 3.443243607504523e-5 | 3.443516315837632e-5 | PASS; terminal identity reviewed |
| 2 | 32 x 48 x 65 | 2.241470620123557e-6 | 2.1978454536730308e-6 | PASS; terminal identity reviewed |

| Stage-level requirement | Result in this draft | Future receipt slot |
| --- | --- | --- |
| Both adjacent-grid T L1 ratios >= 1.25 | PENDING | Summary receipt and bound case 0/1/2 verdicts |
| Both adjacent-grid S L1 ratios >= 1.25 | PENDING | Summary receipt and bound case 0/1/2 verdicts |
| Finest T/S normalized L1 <= 0.05 | Original case 2 gates PASS; final summary PENDING | Bound case 2 journal and future summary |
| Case 3 zero-spice geostrophic flow control | PENDING | Own initial, native, adapter and science receipts |
| Case 4 zero-shear null control | PENDING | Own initial, native, adapter and science receipts |
| All five original cases and complete stage qualification | PENDING | Final summary and independent terminal record |

Zero-spice remains a nonzero-flow balance control and is distinct from zero-shear null. No row is inferred from another row's PASS. Future report completion must cite the new immutable receipts and an append-only review, preserve this draft's scope, and keep every unsuccessful or refused attempt in lineage.

## Interpretation boundary

This coarse result supports a conditional improvement of the selected horizontal paired-product option on the declared contacted profile, under the original barriers. It does not establish all-grid convergence, full fourth-order accuracy, general three-dimensional consistency, arbitrary nonlinear-acceleration accuracy, arbitrary moving-surface behavior, or improved ranking. The optional scheme and its legacy default remain distinct; this report does not change solver defaults or source.

The static-capacity versus eta-cap discrepancy remains explicit. Existing source/CI qualification and the reused analytic controls support identity and software/analytic checks within their own scope. They do not replace the still-pending independent native controls and three-grid stage requirements. The stage remains open in draft03 despite completed main cases 0, 1 and 2; controls 3/4 and the original final summary have not been qualified by this report.

## Actual case 0 loaded source paths

The following entries are transcribed from the case 0 worker path observations preserved in the independent terminal review. These 35 actually loaded modules are a subset of the complete 133-file captured package. Each full path and hash is recorded below; no earlier-stage loaded-path list is substituted.

| Actual worker path | SHA256 |
| --- | --- |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/__init__.py` | `37d5510890297eb78a6974bdb3959c3b224f5e2c381939113af51d7a58e71188` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/__init__.py` | `239c1f8fd5b5a454930824715e99ecde75470b7c89274fbef5f7a1fba4619538` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/__init__.py` | `150aa43ee0dcf4c3217308ca7254bc7d196a7f2c70e5e9a0e5ace3e76cb940af` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/quality.py` | `3b5b05c518d2e927acb4dd0049c7f679ea36b5cfcb2f3f508120389da0e771a2` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/provenance/__init__.py` | `daf5d6bfcecc2b278842a3426572a906a824b2a1f520668da6bb8992e35bd64e` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/provenance/sources.py` | `b422a776558810077f543b8a3b3b99f9584ffafa0b292515a93ac1a96bb99bcc` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/sources.py` | `846b83f00b10a680b09b424878857e70cd54e99ca8a20aa73367da34f599e687` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/config.py` | `890d6b7666877a53d655f070030a5810e687f7f6201325d20b34003dcf7b39ea` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/__init__.py` | `229c94a88ab915568cdb3e04a06896ba1d78a817d0054ead4145186272b82d7e` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/__init__.py` | `b4e9370dd4ce8e3ed7d1dad3c2b35beec912a7d9b43f7f407c3c90e960b2079a` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/grid.py` | `54c5ebd94d6431e6301363e1e0ba6228212b311563a5edd3fde6c0760d9bebf3` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/__init__.py` | `92253b086944c0a62935772ba54bdbc89297fe9567822d267fd15b2cf2bb5611` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/__init__.py` | `929b966423d7801ab47df95b5b84043f63b7cfe905184555eff61e4bdfb426b1` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/backend.py` | `416c3ecf3ec992f66402af2e2b81db5c66f421686691592a01b6865036485197` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/column.py` | `ae9013fb0628e6cbdc8d2027bd899d7815e0864e76a649ed88e5dcb0968d405a` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/contacts.py` | `e91b99345dd7c9688045021e8a895d9b7193606879cb0b0927de6b3b76062e32` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/vertical.py` | `6d5693b4a6dde1842be9e216e4674a42d53794b048189701620c02242e348dab` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/channel.py` | `ab46eadafa84c0a0756b76f0451e0c636e36bcca5d76101cc6f6e62cc9151038` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/horizontal.py` | `430c3ed7b2222734c93b7e33d7003fe64b147e36a75c6b6d5584b5663a2e251b` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/__init__.py` | `a5ebfc7e1dfa65dc7dc52f47cd345a3e0afadf8d5236eb226ba1c22162e5e1ee` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/eos.py` | `50a2529714bef3967143ab28eccfc530cc92fe66d1125de70c4ba7147005a07a` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/pressure.py` | `53b065815da2905df325bb5deab1ecd11babb7eba54076235243158bad38d824` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/transport.py` | `23a93aac12a769afbe948cd812d1d7b68f4563a79b61b4f39034ae6e2e4f276a` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/projection.py` | `64313d7abca6fe4c4ea0f181f469272eb7d6c6c862c19375b2d1dfbedebfb968` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/isopycnal.py` | `8165becfd9a23844761362e36c20161260506e3cd3244a203ae44bde75de7298` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/state.py` | `851e96e868aed41a2f5882e9f1683f0f79056cd73ff35b281ba51cc0a8346d62` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/surface.py` | `0eade693636cd017d5772e2c0cee83c1a5050c49ad3a2936591356ef4cb163b8` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/vertical.py` | `e2305eed74d2c2caed01d70e4a734af4c5a7c06b14647d8bb0f3e118b5db8a3e` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/__init__.py` | `913ed45d926d13a9a89517aacd0095980dd2ff7c15e44bd1c0d850b94c09f6a6` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/subcycles.py` | `fada7101b16464e2e21ec0bac4bfff7d399cf4e9073a4af7035528203f74598b` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/tendencies.py` | `698a8d680089259d7f15a91166d8d89e2ac6d31800af3ca416883757c5f8ec89` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/fd_metrics.py` | `26a9c836450d9b0d0c5d932f0c910799c7036a68a6266f5c36138574ae79b478` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/barotropic.py` | `c4137232f53baaf4eec9ecac4406368f148a6ee1763f91f48957ca0ba9c0b0a6` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/step.py` | `d888979831d99467af6b4b39ed1008a13d8b616b52a93daa38be322b1964c4ae` |
| `/mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/factory.py` | `8123849af4ce529487be47687ad541f38946c9a32b692dc31e0bf6befaf4697b` |


## Actual case 2 loaded source paths

These are the fine worker's own observations, paired with its captured package hashes. Source installed bytes are inherited from the sealed qualification; this records-only extension did not rehash them. Case 0's original path table above is retained as historical evidence. Case 1's own table is in CASE1_TERMINAL_REVIEW01.json.

| Actual fine-worker path | Captured SHA256 |
| --- | --- |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/__init__.py | 37d5510890297eb78a6974bdb3959c3b224f5e2c381939113af51d7a58e71188 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/__init__.py | 239c1f8fd5b5a454930824715e99ecde75470b7c89274fbef5f7a1fba4619538 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/__init__.py | 150aa43ee0dcf4c3217308ca7254bc7d196a7f2c70e5e9a0e5ace3e76cb940af |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/quality.py | 3b5b05c518d2e927acb4dd0049c7f679ea36b5cfcb2f3f508120389da0e771a2 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/provenance/__init__.py | daf5d6bfcecc2b278842a3426572a906a824b2a1f520668da6bb8992e35bd64e |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/provenance/sources.py | b422a776558810077f543b8a3b3b99f9584ffafa0b292515a93ac1a96bb99bcc |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/inputs/sources.py | 846b83f00b10a680b09b424878857e70cd54e99ca8a20aa73367da34f599e687 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/config.py | 890d6b7666877a53d655f070030a5810e687f7f6201325d20b34003dcf7b39ea |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/__init__.py | 229c94a88ab915568cdb3e04a06896ba1d78a817d0054ead4145186272b82d7e |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/__init__.py | b4e9370dd4ce8e3ed7d1dad3c2b35beec912a7d9b43f7f407c3c90e960b2079a |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/grid.py | 54c5ebd94d6431e6301363e1e0ba6228212b311563a5edd3fde6c0760d9bebf3 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/__init__.py | 92253b086944c0a62935772ba54bdbc89297fe9567822d267fd15b2cf2bb5611 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/__init__.py | 929b966423d7801ab47df95b5b84043f63b7cfe905184555eff61e4bdfb426b1 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/backend.py | 416c3ecf3ec992f66402af2e2b81db5c66f421686691592a01b6865036485197 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/column.py | ae9013fb0628e6cbdc8d2027bd899d7815e0864e76a649ed88e5dcb0968d405a |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/contacts.py | e91b99345dd7c9688045021e8a895d9b7193606879cb0b0927de6b3b76062e32 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/vertical.py | 6d5693b4a6dde1842be9e216e4674a42d53794b048189701620c02242e348dab |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/channel.py | ab46eadafa84c0a0756b76f0451e0c636e36bcca5d76101cc6f6e62cc9151038 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/numerics/horizontal.py | 430c3ed7b2222734c93b7e33d7003fe64b147e36a75c6b6d5584b5663a2e251b |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/__init__.py | a5ebfc7e1dfa65dc7dc52f47cd345a3e0afadf8d5236eb226ba1c22162e5e1ee |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/eos.py | 50a2529714bef3967143ab28eccfc530cc92fe66d1125de70c4ba7147005a07a |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/pressure.py | 53b065815da2905df325bb5deab1ecd11babb7eba54076235243158bad38d824 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/transport.py | 23a93aac12a769afbe948cd812d1d7b68f4563a79b61b4f39034ae6e2e4f276a |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/projection.py | 64313d7abca6fe4c4ea0f181f469272eb7d6c6c862c19375b2d1dfbedebfb968 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/isopycnal.py | 8165becfd9a23844761362e36c20161260506e3cd3244a203ae44bde75de7298 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/state.py | 851e96e868aed41a2f5882e9f1683f0f79056cd73ff35b281ba51cc0a8346d62 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/surface.py | 0eade693636cd017d5772e2c0cee83c1a5050c49ad3a2936591356ef4cb163b8 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/physics/vertical.py | e2305eed74d2c2caed01d70e4a734af4c5a7c06b14647d8bb0f3e118b5db8a3e |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/__init__.py | 913ed45d926d13a9a89517aacd0095980dd2ff7c15e44bd1c0d850b94c09f6a6 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/subcycles.py | fada7101b16464e2e21ec0bac4bfff7d399cf4e9073a4af7035528203f74598b |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/tendencies.py | 698a8d680089259d7f15a91166d8d89e2ac6d31800af3ca416883757c5f8ec89 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/geometry/fd_metrics.py | 26a9c836450d9b0d0c5d932f0c910799c7036a68a6266f5c36138574ae79b478 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/dynamics/barotropic.py | c4137232f53baaf4eec9ecac4406368f148a6ee1763f91f48957ca0ba9c0b0a6 |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/timestepping/step.py | d888979831d99467af6b4b39ed1008a13d8b616b52a93daa38be322b1964c4ae |
| /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/horizontal-paired-product-v1/installed/zhenmode/model/solver/factory.py | 8123849af4ce529487be47687ad541f38946c9a32b692dc31e0bf6befaf4697b |
