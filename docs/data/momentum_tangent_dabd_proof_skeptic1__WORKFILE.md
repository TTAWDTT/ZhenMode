# Fresh hostile skeptic pass 1 WORKFILE

## Provenance/read order
Independent SKELETON written 2026-10-12T21:38:20 UTC before proof access, hash C2EC0F1A810D529F45656B1D1D980F7A0DE61E4D33EE3328309FB504868C90DC. Full spec/skill hashes are in SKELETON. Tools consulted after skeleton: full tool-stewardship and timing-discipline skills.

## Primary review (read 2026-10-12T21:38:39 UTC)
Primary WORKFILE SHA256 B74F91933A84852ABD88D377C8FAF5C27B1B984729D93E16665612EFADD4C8F6.
- Weighted Cauchy row estimate is correct: weights (1,2,1) total4, functional coefficients (1,-2,1) yield precisely 4(a²+2b²+c²). Aggregate interior weights <=4, so bound16 valid for every n>=4; boundary cannot exceed it.
- Direct contraction refinement uses ||R^T R x||²<=16||Rx||². This is an unstated intermediate, derivable from the already proved ||R||<=4; no circular target import. Exact controls pending.
- Nullspace recurrence, symmetry-based sum and affine preservation complete; physical y=(j+1/2)/n has same span as j for fixed n. O(n) missing-r zero convention correctly scatters all rows, including endpoints.
- Normal quadratic form endpoints +2 is consistent with -3 diagonal. Bound4 fills endpoint weights2+2, interior2+2. L is actually negative definite, but merely nonpositive is sufficient and not erroneous.
- Sine nonzero justification: first component sin(k*pi/(2n))>0 for k1..n. Distinct eigenvalues give complete basis including k=n; no omitted normalization requirement.
- Bn comparison follows its definition. Strict-odd label interpreted solely as pinned L/ghost convention, not implementation equivalence. No inference to generic tangential source.
- Low-wave series correct, only symbolic scope. No global PDE claim imported.
- Primary explicitly marks X0 receipts REUSED, not independent verification. Its overclaim witnesses are rational deductions awaiting my independent executed run.
Current provisional review: no algebra FIRE; exact independent controls and other arm review pending. This is not CLEAN yet.

## Variational review (read 2026-10-12T21:39:26 UTC)
PROOF SHA256 CB3DBB2252ACADFD938D79D1B71E4870EC211B25B192BEA0486B228591E84638.
- All-n domain and common w>0 scale are explicit. Real symmetric spectral lemma is proved by compact sphere stationary point plus invariant complement; no circularity found. Allowing max to be negative causes no issue.
- I.1 local square defect and I.2 aggregate count: n4 has only outer and next-outer indices; no double-count overlap, coefficients [4,12,12,4]. For n>=5 inner coordinates receive16. Identity correctly separates boundary deficit and summed local defect. Pending fresh script identity controls.
- I.3-I.5 use independent lemma, exact Gram kernel, recurrence, and symmetry. Scaled-y affine span explicitly handled. Transpose scatter costs two linear loops and is the pinned finite matrix for all n.
- II.1 bilinear difference+penalty and II.2 adjacent-sum form give both Loewner bounds with correct endpoints; scalar polynomial interval lifts to Fn. No sine dependence needed for contraction.
- II.4-II.8: correct ghost parity, nonzero first component, strict eigenvalue order, orthogonality and exceptional k=n norm. Telescoping division by sin(theta) used only k<n, so no Nyquist divide-by-zero. k=0 intentionally excluded, matching odd wall matrix. No missing eigenspace.
- II.9-II.11 and B polynomial correct on paper; series symbol scope explicit. Negative matrix entries establish lack of pointwise positivity, not lack of Loewner PSD.
- T1 is negativity witness; e0 is independent TV witness; normal T2 TV1 is legitimate unrestricted vector witness with physical native applicability expressly unclaimed.
- Weighted e0 values have exact rational expectations313/128 and2537/256; already anticipated in skeleton, not imported as independent results. My forthcoming code recomputes them from index-constructed matrices.
- Author script and code NOT read or reused. Author receipt logs are not my executed verification. n5/n8 author controls contribute no fresh skeptic evidence.
Current provisional: no FIRE; exact skeptic controls and spectral review pending.

## Spectral review (read 2026-10-12T21:39:58 UTC)
WORKFILE SHA256 FE8283810D2CB08D779526EDB62AAD16BF4DC2BB9B36DFE4B90E25A7C47CB1CC.
- S1 restriction P to0..n-3 and shift direction U correctly yield finite R with no extra zero-extension rows. Triangle bound4 valid; no Fourier/Parseval import, no boundary even/odd assumption.
- S3-S8 symmetric spectral map, exact kernel, scaled affine span and immutable-input scatter complete. Real arithmetic O(n) equality does not license bitwise equality.
- N1-N4 ghost endpoints, recurrence, distinct basis and Fn multiplier match independent skeleton; theta=pi alternating mode included, k0 excluded. Additional Fn O(n) tridiagonal composition is algebraically correct and will receive symbolic n4 control.
- N5-N7 include additional valid comparison0<=B<=Fn<=I, difference2s²(1-s²). This follows pointwise shared diagonalization, not source/native tangent equivalence.
- Failed v1 correctly retained and explicitly not counted as clean pass: T1 has input/outputTV9 exactly. Corrected e0 witness exact17/16. v2/v3 remain AUTHOR controls, contribute zero fresh skeptic passes.
- All target controls, model exclusions and no-design-selection are explicit. No hidden outside import beyond finite spectral lemma, elementary triangle/trig/monotonicity; those are self-contained all-n arguments.
No proof/code of any other running proof agent was accessed. No root static proof prose accessed. No production/native source accessed or changed.

## Capability and computation plan, frozen before launch
Read toolkit tools/README and full rankscreen GUIDE; rankscreen explicitly targets large multi-prime eliminations and is not general matrix algebra, so wrapping it would not produce the prescribed exact n4 controls. A one-shot fresh SymPy script is warranted for this task-specific matrix/trigonometric audit and is not a toolkit instrument; no installed sources changed. Installation quick-start and optional-backend acceptance consulted: no optional backend is used, and installation PASS is not scientific proof.
Only n4 finite matrices; exact rational and symbolic generic trig/local identities. CPU affinity one CPU; CPU and wall180s; address limit4GiB; monitor RSS every0.01s. Runtime unknown, measuring first bounded audit. Progress is completed named checks in controls.json; any unexpected false assertion fails closed and is retained. Source and all outputs stay in skeptic1. No scan, fit, data or native phase.

## Prelaunch receipt
UTC 2026-10-12T21:42:57.6586685Z; exact_audit.py SHA256 C98BF8FFCE63B57E252BD1E171E7C684ADC71113412879147180074D5E825D6F. Frozen denominator69 including denominator assertion; only n4 matrices. The scalar equality comparator receives deliberately unequal exact rationals, and PSD/contraction detectors receive both clean matrices and wrong1/4 copies. Shared deepest layers: pinned matrix definition and suite SymPy1.14 exact arithmetic; feared author indexing/witness/scope errors are attacked by independently assembled matrices, principal minors, symbolic free-vector identities and pre-proof expectations. This is a fresh skeptic pass, not an entirely disjoint CAS implementation or formal proof. There is no fitting or reference-guided parameter selection.

## Independent exact execution (first audit run)
Command: wsl -d Ubuntu-24.04 -- bash -lc 'source /root/.local/share/bootloops-suite/env.sh && cd /mnt/d/Github/ocean-solver/outputs/vertical-mean-round3-20261012/momentum-wall-filter-proof-70/skeptic1 && python exact_audit.py > stdout.txt 2> stderr.txt'
Script SHA256 c98bf8ffce63b57e252bd1e171e7c684adc71113412879147180074d5e825d6f. First run finished successfully:69/69, no unexpected failed check, stderr empty. Actual wall1.786877921s/CPU1.495117976s, sampled peak78,430,208bytes,136 RSS samples at0.01s, one CPU[0], hardCPU/wall180s/address4GiB. Tool shell duration4.7356137s includes startup/host overhead. No additional compute launched or needed.

Verbatim stdout (4 lines):
```
FRESH-SKEPTIC1 CONTROLS PASS n=4 checks=69
T1=[-1/8, 9/8, 33/8, 71/8]; T2=[5/8, 9/8, 9/8, 5/8]
PLANTS CAUGHT; Ft TV17/16>1; Fn TV1>0; T1 TV9=9
TIMING {"CPU_limit_seconds": 180, "RSS_interval_seconds": 0.01, "RSS_samples": 136, "address_limit_bytes": 4294967296, "cpu_affinity": [0], "cpu_seconds": 1.495117976, "exact_checks": 69, "finite_sizes": [4], "python_executable": "/root/.local/share/bootloops-suite/bootloops/.venv/bin/python", "sampled_peak_RSS_bytes": 78430208, "script_sha256": "c98bf8ffce63b57e252bd1e171e7c684adc71113412879147180074d5e825d6f", "status": "PASS", "sympy_version": "1.14.0", "wall_seconds": 1.786877921}
```
The script emits controls.json incrementally, receipt.json with birth/environment/resource record, and bugreports.json with exact sides/residuals for2 false copies and8 overclaims. No author check was credited as fresh evidence.

## Exhaustive claim/domain ruling
- Target I symmetry/Loewner/contraction: all-n zero-extension operator bound from skeleton and finite Cauchy route each close the gap; n4 all-principal-minor controls cover both order bounds and I-Ft^TFt. Stronger direct contraction refinement follows ||R^TRx||<=4||Rx|| and is not circular. Bounds do not identify finite R^TR with an infinitely wrapped convolution.
- Target I constant/physical-y affine/sum/kernel: recurrence on all n-2 consecutive triples yields exactly a+bj; y affine relation has nonzero slope1/n. n4 rank2, free-vector scatter, constant/affine/scaled affine/sum checks passed. O(n) mathematical equality only; native float order and performance are not tested or implied.
- Target II Green bounds: expansion at each endpoint yields diagonal3=1+2, upper form endpoint1=4-3, each interior sum coefficient2=4-2. Thus -4I<=L<=0 for every n. Polynomial mapping on self-contained finite symmetric diagonalization gives contraction independent of sine completeness. n4 both free-vector bilinear identities and all principal-minor controls passed.
- Target II basis: left ghost sin(-theta/2)=-sin(theta/2), right sum2sin(k*pi)cos(theta/2)=0. Interior addition identity applies also to ghost-substituted endpoints. k1..n first component>0, eigenvalues distinct; n nonzero eigenvectors of symmetric L form orthogonal complete basis. k0 absent, k=n Nyquist has norm n, other modes norm n/2 by finite telescope; division is only k<n. No sign/zero/normalization loophole found. n4 all modes/eigenmapping/Gram and generic trig/ghost/telescope controls passed.
- B comparison: exact definition(I+L/4)^2, polynomial and cos^4 multiplier passed. Additional spectral-arm0<=B<=Fn<=I is valid from2s²(1-s²)>=0 in same basis; n4 exact minor check passed. No unsupported statement about actual generic tangent edge padding is present.
- Symbol accuracy: series Fn departure theta^4/16+O(theta^6), B departure theta²/2-5theta^4/48+O(theta^6) independently executed. This yields symbolic orders only, not global PDE accuracy, eta mechanism, coupled step energy or production guarantees.
- Scope n>=4 real unrestricted vectors/equal fixed positive weights: proof quantifiers match the spec. Negative and TV witnesses are legitimate for unrestricted matrix claims even if not admissible physical native states. No arbitrary terrain or evolving volume imported. Failure for one nonuniform positive snapshot precludes an unrestricted all-positive dynamic-weight inference, while not proving a specified eta system unstable.
- Both positivity overclaims killed by n4 nonnegative e0 output entry -1/16. Ft TV17/16>1; Fn constant TV1>0 and maximum9/8>1. T1 TV9=9 verifies the retained spectral failed-v1 diagnosis and prevents its mistaken promotion. Fn constant result also disproves normal mean preservation: sum7/2 versus4, with no claim that normal mean must be preserved.
- False-copy plants independently caught by clean-twin PSD and contraction detector: Ft wrong norm8 versus4, quadratic -4; Fn wrong norm36 versus4. This is harness calibration, not a target FIRE.
- All imported mathematical facts are elementary and either derived in each proof or transparently used: finite symmetric diagonalization, triangle inequality/isometry, elementary trigonometric addition and strict sine monotonicity on(0,pi/2]. No target-equivalent MAY-ASSUME import found. If these elementary imports are disallowed, authors explicitly downgrade relevant spectral step to OPEN; no external source/native theorem was silently admitted.
- Alternative Ft-only with original B unchanged remains unselected. No implementation consumer has been reviewed in this pass. No production/native edit, new data, parameter sweep, optional/backend or toolkit source modification occurred.

## Verdict
CLEAN / NO-FIRE for pinned Targets I and II, fresh hostile skeptic pass1 only. This does NOT set VERIFIED-CLOSED. Exactly1 qualifying fresh clean pass exists from this arm; spectral v1 failure and author v2/v3 are excluded. Another new-context exhaustive clean pass must be consecutive, followed by nonauthor spec-only reconstruction and actual consumer interface trace before final workflow grade. This is a BootLoops working standard, not formal/refereed proof certification.

## Failed-v1 preservation inventory / closing provenance
Inspected filenames/sizes only, without reading author code: spectral/control_stderr.failed-v1.txt467bytes; controls.failed-v1.json2544bytes; exact_controls.failed-v1.py5726bytes. These retained files support the narrative's append-only failure history, while my T1 TV equality is independently executed. End-of-pass file hashes are recorded in hash_manifest.json. No supporting proof was changed by this skeptic.
