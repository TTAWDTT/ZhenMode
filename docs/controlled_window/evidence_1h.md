# Fixed one-hour four-cell diagnostic, 2026-10-01

Completed once: 12 common rounds × 300 s = 3600 s, 48 accepted attempts. No extension/retry.
Host nagi; original repository/tree/data/Goal unchanged. Raw arrays remain local/private.
Historical source commit `212df951c351f82dba74fbc43db5e52b0ad34c47`; frozen protocol SHA256 `e238cbab8e1b99c85d3c3b8091e7e513dcc52dec58d5ff324188ec00f406adee`.
34 source files byte-match registered historical hashes; material_top.py has the recorded scalar-only instrumentation diff.
All 48 checkpoint hashes and 12 commit markers verified; trajectory agrees with authoritative markers.
All four first-step six-field maximum absolute differences against prior uninstrumented full-step artifacts are exactly zero; ice byte-equal and acceptance flags match.

## Registered contrasts and geography

I0 = archived initial state. I1 = frozen paired T/S layerwise donor sensitivity: 440 original-domain nodes; crop retains exactly 202, without reselection. It is a stitched sensitivity, not a scientifically trusted replacement.
B0 = original closed domain. B1 = remove old j0, new closed wall at old j1; rebuild only new wall projection diagonal. B1 is not an open-boundary experiment.
The fixed retained296 common observer area is 1.694260548556649e12 m². Deleted152 are absent from B1, never credited as improvement; constrained161 at old j1 are reported separately.
January2023 archived forcing held fixed; dt_fast25 ×12, actual_geometry_v2/joint_heun_v1, cap256, unchanged original gates.

## Common trajectory

Values in mm. Cell order I0B0, I1B0, I0B1, I1B1.

| Time s | Target four cells mm | Retained296 area means mm |
|---:|---|---|
| 300 | 1.445209, 3.116938, 1.393740, 3.025947 | -1.611584, -1.548044, -2.061869, -2.107184 |
| 600 | 1.365984, 5.646929, 1.350886, 5.586367 | -5.984068, -5.778171, -7.592367, -7.762412 |
| 900 | -0.700064, 5.724090, -0.514681, 5.984175 | -12.698099, -12.324556, -15.979859, -16.339392 |
| 1200 | -5.233592, 2.031969, -4.703334, 2.914026 | -21.375183, -20.857931, -26.675985, -27.267942 |
| 1500 | -12.619482, -6.014388, -11.729814, -4.382612 | -31.675416, -31.076612, -39.207557, -40.049101 |
| 1800 | -23.068854, -18.319654, -22.034178, -16.136775 | -43.300676, -42.706939, -53.183618, -54.267852 |
| 2100 | -36.563972, -34.284697, -35.900659, -32.163128 | -55.997478, -55.505287, -68.295432, -69.597473 |
| 2400 | -52.837793, -53.013409, -53.403125, -51.994561 | -69.555642, -69.256012, -84.309287, -85.793575 |
| 2700 | -71.393544, -73.488389, -74.388082, -74.998199 | -83.803063, -83.766895, -101.054591, -102.680981 |
| 3000 | -91.559073, -94.697242, -98.494274, -100.462728 | -98.600347, -98.867887, -118.410840, -120.138822 |
| 3300 | -112.565021, -115.721911, -125.200771, -127.668208 | -113.838777, -114.416251, -136.295962, -138.087478 |
| 3600 | -133.633460, -135.808872, -153.891948, -155.950907 | -129.441666, -130.305298, -154.656383, -156.477908 |

## Conditional effects and interaction

For Y: input at B0=I1B0−I0B0; input at B1=I1B1−I0B1; boundary at I0=I0B1−I0B0; boundary at I1=I1B1−I1B0. Main effects average the two conditional effects; interaction=I1B1−I0B1−I1B0+I0B0.

| Time s | Target: input B0/B1; boundary I0/I1; interaction mm | 296: input B0/B1; boundary I0/I1; interaction mm |
|---:|---|---|
| 300 | 1.671729, 1.632207, -0.051470, -0.090991, -0.039522 | 0.063540, -0.045316, -0.450285, -0.559141, -0.108856 |
| 600 | 4.280946, 4.235481, -0.015097, -0.060562, -0.045465 | 0.205897, -0.170046, -1.608299, -1.984241, -0.375943 |
| 900 | 6.424154, 6.498857, 0.185383, 0.260085, 0.074703 | 0.373543, -0.359533, -3.281760, -4.014836, -0.733076 |
| 1200 | 7.265561, 7.617360, 0.530257, 0.882057, 0.351800 | 0.517252, -0.591957, -5.300802, -6.410012, -1.109209 |
| 1500 | 6.605094, 7.347202, 0.889668, 1.631777, 0.742108 | 0.598804, -0.841544, -7.532140, -8.972489, -1.440348 |
| 1800 | 4.749200, 5.897403, 1.034676, 2.182879, 1.148203 | 0.593737, -1.084233, -9.882943, -11.560912, -1.677970 |
| 2100 | 2.279275, 3.737531, 0.663313, 2.121569, 1.458256 | 0.492190, -1.302041, -12.297954, -14.092186, -1.794232 |
| 2400 | -0.175615, 1.408564, -0.565332, 1.018848, 1.584180 | 0.299630, -1.484287, -14.753645, -16.537563, -1.783918 |
| 2700 | -2.094845, -0.610117, -2.994537, -1.509810, 1.484728 | 0.036168, -1.626390, -17.251528, -18.914086, -1.662558 |
| 3000 | -3.138169, -1.968454, -6.935201, -5.765486, 1.169715 | -0.267540, -1.727981, -19.810493, -21.270934, -1.460441 |
| 3300 | -3.156890, -2.467436, -12.635750, -11.946296, 0.689454 | -0.577474, -1.791516, -22.457185, -23.671227, -1.214042 |
| 3600 | -2.175413, -2.058959, -20.258489, -20.142035, 0.116454 | -0.863632, -1.821525, -25.214716, -26.172610, -0.957894 |

At 1h the target input contrast is negative in both domains, despite positive first-step contrast. On retained296, B1 lowers the area mean by 25.214716/26.172610 mm at I0/I1. The input main effect is −1.342578 mm and boundary main −25.693663 mm; interaction −0.957894 mm.
Geography matters: the newly constrained161 mean is less negative in B1, while the retained296 and common south rows are more negative. This is redistribution/sensitivity, not a demonstrated improvement.

## Original acceptance and numerical evidence

All 48 valid; no nonfinite fields; returned equals attempted in all six fields. No rejection occurred, so rollback was not triggered or tested anew.
Worst full-domain continuity residual 4.024558464266193e−16 m (gate1e−10); face mismatch2.842170943040401e−14 m²/s (gate1e−9); unpaired eta filter4.996003610813204e−16 m; local inventory roundoff ratio≤0.031533752 (gate1).
Minimum wet thickness≥1.871441446287196 m; |eta|≤0.628558553712804 m; |velocity|≤0.306572894410758 m/s; ice0. All schedules supported: first/second linear1, nonlinear2, momentum1, below cap256 and original CFL0.5.
Source arrays/inventory recorded per step in verified_window_summary.json. Only bulk_heat is nonzero among declared source channels. Accumulated heat source J in cell order: −6.063851031552315e19, −6.063342315445301e19, −6.078252580037741e19, −6.077996985446679e19; declared salt sources0. These budgets do not isolate eta causality.

## Stage observations and actual face records

The prior isolated25s negative target and full300s positive target are different numerical trajectories. In the full first step I0B0 eta starts0, predictor0, post-fast+0.0014452094860626834m; nontransport change−5.421e−19m. First L/N predictor modifies T/S, density-derived acceleration and velocity before the 12-substep fast map. This locates the observed sign in the full fast stage but does not identify a unique responsible operator; no stage ablation was run.
First I0B0 density acceleration before x/y=−1.14308409e−7/+2.49179606e−7 m/s², pre-fast=−5.06329736e−8/+3.71361924e−7; mean pre-fast u/v=−1.76655452e−4/−3.47404635e−5 m/s. Surface eta-gradient acceleration is zero in this first predictor.
The observed column_faces are actual returned step-mean transport arrays. B0 first internal face=oldj0/j1; B1 first internal face=oldj1/j2. Those are different geographic faces and their difference is not the same-face boundary contrast. Closed southern exterior is zero by operator construction; observed northern exterior transport max is zero.

| Time s | Actual first internal signed flux, cell order, m³/s | Target eta post-fast−initial, cell order, mm |
|---:|---|---|
| 300 | 18292595.758324, 19214722.264944, 19821802.877036, 19858604.707927 | 1.445209, 3.116938, 1.393740, 3.025947 |
| 600 | 47728608.119153, 49953251.552006, 51755646.103900, 51818884.365476 | -0.079226, 2.529991, -0.042854, 2.560420 |
| 900 | 70515502.798168, 73495785.467892, 76629447.033291, 76688409.452329 | -2.066048, 0.077160, -1.865568, 0.397808 |
| 1200 | 87325094.226740, 90577404.684700, 95236821.889133, 95271199.792155 | -4.533528, -3.692121, -4.188653, -3.070149 |
| 1500 | 99045697.617805, 102185074.252608, 108545337.262359, 108547390.220386 | -7.385890, -8.046357, -7.026479, -7.296637 |
| 1800 | 106722737.211668, 109480579.410543, 117641106.288967, 117614480.346439 | -10.449372, -12.305265, -10.304364, -11.754163 |
| 2100 | 111463007.757780, 113687257.231845, 123640074.321443, 123596524.243154 | -13.495118, -15.965043, -13.866482, -16.026354 |
| 2400 | 114329752.783635, 115971139.361931, 127590764.084573, 127546297.456787 | -16.273821, -18.728711, -17.502465, -19.831432 |
| 2700 | 116248970.112454, 117338477.958728, 130388675.699539, 130360017.418910 | -18.555751, -20.474981, -20.984957, -23.003639 |
| 3000 | 117941002.270149, 118563821.221842, 132716760.720824, 132718475.024054 | -20.165529, -21.208853, -24.106192, -25.464529 |
| 3300 | 119885386.241577, 120155587.925257, 135019583.454430, 135061985.141585 | -21.005948, -21.024669, -26.706497, -27.205480 |
| 3600 | 122320695.755483, 122359335.601130, 137511759.082046, 137599733.388721 | -21.068439, -20.086961, -28.691177, -28.282700 |

B0 first actual flux18,292,595.75832449 m³/s agrees with earlier continuity inference18,292,595.758324485; the new observation is actual transport, the old value remains an inference. Target nontransport eta changes across this window are ≤4.164e−17m. A continuity identity does not establish physical validity.

## Cost and bounded execution

One worker, CPU affinity mask1, exit0; total408.4274303s. Peak working set3,369,680,896bytes and commit3,345,633,280bytes, both below4GiB. No guard triggered.
Two cold kernel compilations115.511647/114.935447s, lowering1.472729/1.499045s; I1 reuses corresponding in-process executable. No persistent cache used.
| Cell | Cumulative wall s | Warm execution mean s | Warm rounds2–12 total wall s |
|---|---:|---:|---:|
| I0B0 | 160.630725 | 2.846972 | 38.285481 |
| I1B0 | 42.870658 | 2.826810 | 38.054649 |
| I0B1 | 160.001308 | 2.847950 | 38.196478 |
| I1B1 | 43.121952 | 2.863875 | 38.290154 |

No case approaches its900s budget. Execution and checkpoint/diagnostic costs are distinguished; prior65–67s single-step runs included compilation and were not warm-step cost.

## Inference limits and next review

This fixed1h result shows that first-step positive eta and delay in sign reversal do not imply sustained improvement. It supports reviewing wall-location redistribution and input sensitivity on common geometry; it neither repairs nor predicts the30h failure.
I1 has layerwise multi-profile columns and no proven physical initialization. B1 only moves a closed wall. Current raw WOA identities are recorded but historical raw hashes are missing. Original numerical checks passing do not qualify boundary physics or initialization.
No extra integration, scientific modification, Git push/merge, raw upload or Goal change was performed for this delivery.
Evidence: protocol.json/instrumentation.diff/worker.py/supervisor.py, per-round COMMITTED.json and private checkpoints, verified_window_summary.json, verified_artifact_hashes.json, resources.json and worker.log.
