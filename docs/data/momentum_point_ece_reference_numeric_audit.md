# Independent finite numerical consumer audit

2026-10-13. **CONFIRMED finite binary64 reference-computation consistency.** Physical attribution to native errors remains **OPEN**. This audit assigns no new proof-closure grade and does not replay the exact proof arms or native integration.

The numerical comparator and 1e-7 relative RMS gate were declared in `declaration.json` and WORKFILE before calculation. Parent summary numbers and result.json scalars were already visible, and their contact is recorded. They supplied no fitted parameters or code choices. The root consumer source `reference.py` was opened only after the fresh independent script and its successful full numerical run existed.

## Independent route and measured agreement

`independent_numeric.py` derives Qt by integrating `numpy.polynomial.Legendre.basis(l)` from x=-1 and retaining degrees below n. The last column's degree-n coefficient is deliberately discarded, so this is the projected integral operator; it does not assume pointwise highest-degree endpoint closure. Qb=H e0e0^T-Qt. The script assembles the real canonical operator for U,V,B,E, balances it with sqrt(W), and appends three forcing-generator states c0,c,s:

c0dot=0, cdot=-nu*s, sdot=nu*c; initial (c0,c,s)=(1,1,0).

Their couplings are h0, hc, hs. Each saved time is evaluated directly with `scipy.linalg.expm` on the resulting real 52x52 matrix. This is distinct from the consumer's Hermitian eigendecomposition and three exponential-convolution terms; no consumer or project code is imported. No additional n grid was selected: the independent dynamics check uses n=16 and all 41 saved time samples only. Physical-node scalar reconstruction uses z=-linspace(0,H,16), x=-2z/H-1, theta2=B/(g alpha).

| Predeclared n16 comparison | Measured | Gate |
|---|---:|---:|
| Eta relative RMS over the 41 samples | 1.2767860426403096e-14 | <=1e-7 |
| Physical-node theta mode2 relative RMS | 6.930205540858173e-13 | <=1e-7 |
| Eta trapezoidal-time relative RMS | 1.2763933834538466e-14 | <=1e-7 |
| Theta mode2 trapezoidal-time relative RMS | 6.927613388479379e-13 | <=1e-7 |

Maximum absolute differences are 3.567216391634932e-23 m for E and 2.3920383104414312e-23 C for theta mode2. These are finite floating-point comparisons, not claims of mathematical exactness or arbitrary-precision digit certification. At the inherited seven-digit gate they agree by about 13.9 and 12.2 decimal relative digits respectively. There is no fit, threshold change, held-out scientific prediction claim, or raised-precision certificate here.

The augmented-system known-answer controls run before the prediction comparison. Constant oscillator maximum defect is 5.55e-17; resonant sine-forcing defect is 9.44e-16; all zero-a and nu=0 controls are <=1e-12. Reversing the rotation yields defect 1.4161468365471424>1e-3. A one-percent perturbation makes the actual relative comparator reject with 0.00990099009900991>1e-7; negating independent eta against the saved eta yields relative defect 2. These demonstrate that both the evolution and comparison harness can reject false copies.

Pilot: known-answer controls plus one same-dimension exponential at t=60000, without comparing root predictions. That exponential took 0.00045829000000008335 s; the recorded 41-evaluation projection is 0.018789890000003417 s. Full run followed only after `timing_plan.json` was saved.

All accepted compute used sourced BootLoops suite Python, NumPy 2.5.3, SciPy 1.16.3, CPU affinity one and BLAS threads one, 180 s wall/CPU fence and 4 GiB aggregate sampled-RSS cap. The supervisor samples its recursive process tree every 0.1 s.

| Run | True exit | Wall s | Peak aggregate sampled RSS bytes |
|---|---:|---:|---:|
| Timing pilot | 0 | 0.511483139 | 69,599,232 |
| Independent n16 full comparison | 0 | 0.409797261 | 72,077,312 |
| Raw readback 1, retained | 0 | 1.012774770 | 73,719,808 |
| Raw readback 2, including time-weighted n16 check | 0 | 0.910323880 | 73,564,160 |

The first readback output remains in `readback1/`. No numerical failure occurred. A PowerShell metadata conversion failed because result.json contains case-distinct k/K keys; it was retained as a setup error and corrected with native JSON parsing/-AsHashtable. No zero count from the failed conversion was accepted. An intermediate console Select-Object over hashtables displayed null properties, but the stored before/after JSON records contain the correct paths and all five `unchanged=true` identities.

## Raw saved-array metric replay

`raw_readback.py` loads NPZ arrays directly with allow_pickle=False and imports neither the project loader nor the project's first-order oracle or RMS function. It rebuilds first-order eta and T from the declared contract, computes native-minus-linear-reference errors, uses area for eta and actual saved h*area for T, and integrates the weighted spatial mean squares with trapezoidal time quadrature. The two actual primary contracts have identical physical parameters. The saved observations are ocean-solver physical nodes, not layer-mean scalar values.

The saved descriptive comparison uses n64 reference arrays, exactly as declared. Every re-emitted scalar is bitwise-equal as a Python float to result.json (maximum reported relative scalar difference 0.0):

| Quantity | 3c8 | ece |
|---|---:|---:|
| Canonical eta RMS m | 7.051645033463736e-10 | 7.051645033463736e-10 |
| Eta residual/original RMS | 0.05559509515431827 | 0.051360788113192515 |
| Canonical theta20+theta mode2 RMS C | 6.799621461113437e-9 | 6.799621461113437e-9 |
| T residual/original RMS | 0.382942825324212 | 0.3534436016803804 |

All original RMS and residual RMS values also reproduce in `raw_readback_result.json`. The raw output byte hashes match the result's published hashes: 3c8 `1fef45cf2409721d07fef49f74549a9739bb6b9ccbec4aa0408b11b100f93fab`; ece `97a7af537c47771f69cfe17573571045365b2cbe9f8b22c6219341899fa78088`.

Saved-array replay gives n32/n64 eta convergence 1.8758322128692746e-16 and theta convergence 5.522953216595545e-16. Reconstructing finest Bsurface from the stored nodal theta gives `(Bsurface+N2E)` relative RMS 4.3189190939452875e-16, passing 1e-7. Its last-bit difference from the producer's 2.8686e-16 arises because the saved theta has already undergone B basis reconstruction and division by g alpha before this checker multiplies back. Neither quantity needs threshold adjustment.

The NPZ does not retain n64 V/B coefficient histories. Therefore the n64 projected pressure and vertical-velocity boundary numbers cannot be independently replayed from the saved arrays alone. Their implementation expressions are source-audited and match the specification; this report does not promote that source check into an independently re-executed n64 boundary calculation. The fresh dynamic integration remains n16 only.

## Consumer source/interface audit

The inspected `reference.py` implements the permitted canonical object:

* Qt is truncated/projected Legendre integration; Qb includes the H e0e0^T constant. The skew gate does not imply exact finite-n endpoint closure.
* Second-order U,V,B,E start at zero. C1 and the independently source-licensed C2 correction enter only U0/V0, with h0[V0]=-Sv+Q, hc[V0]=Sv+Q, hs[U0]=Su. There is no direct Y-mode2 scalar self-advection source. Theta20 is evaluated separately; this does not claim that all second-order theta is Y uniform.
* First-order fields and the background are prescribed by the frozen linear reference. The input contract explicitly has zero external physical sources. No first-order/native re-fitting is performed.
* The Hermitian eigenphase a=-i lambda, exact coincident J limit, no eigenvalue clipping, unforced energy and imaginary-part gates match the specification. The n32/n64 convergence and finest projected boundary checks precede descriptive native comparisons, and failure exits without comparison.
* Physical scalar observations use fixed physical nodes. Descriptive full-T weights are the original saved h*area/time weights; they do not make the canonical evolution a finite-capacity native model.
* Fixed-H/linear-continuity dynamics exclude native filters, FD wall operators, time/process splitting, moving geometry, volume/kinematic eta*v terms. Those exclusions remain material. Agreement and small residual fractions support consistency of this canonical contribution, not exclusive causation, a new score, nonlinear truth, or global SOTA.

No new scalar nonlinear/alpha diagnosis was declared or computed. Alpha here is solely the existing canonical contract ratio N2/(g gamma).

## Actual source identity and CI readback

An independent filesystem traversal found **133 installed Python source files**. Their SHA256 dictionary equals all 133 entries in the wheel identity, result.source_before, and result.source_after. This is a count of that installed package, not of the entire repository or its tests. The root writes `source_after=before` only after an actual `package_source_hashes` equality assertion; the audit additionally checked final installed bytes independently. All five explicit before/after identity checks—consumer source, spec, result, reference NPZ and frozen contract—are unchanged.

The actual CI logs support distinct counts:

* ece commit `ece37c7b7ad383314e4dcfd982641d9ca541e1c3`, run 38293318389: `ci-ece-full-attempt02.log:841` reports **1737 passed, 6710 warnings in 1589.16s**. The same log contains that checkout SHA; its terminal record says completed/success.
* 3c8 commit `3c8e86bef2ee1fbfd50cca4343d6fc61694d9edc`, run 38286199660: `ci-3c8-full.log:672` reports **1721 passed, 6710 warnings in 1081.12s**. Its terminal record says completed/success.

These are recorded software CI outcomes, not scientific evidence for the canonical attribution. No CI was rerun by this auditor. Exact locators, raw lines and log hashes are in `raw_readback_result.json`.

## Verdict and lineage limits

CONFIRMED: independent n16 real augmented-matrix evolution at the saved times, comparison harness controls, saved descriptive metric replay, saved convergence/surface replay, inspected consumer interfaces and actual source/CI identities. Source-audited only: producer n64 pressure/velocity boundary outputs whose state histories were not saved. OPEN: exclusive native physical attribution and any new nonlinear mechanism diagnosis.

The deepest shared layers are the specified physics/contract, projected canonical operator, binary64 arithmetic and NumPy reconstruction. Core evolution algorithms and author/code are separate: SciPy real augmented exponential versus NumPy Hermitian spectral time convolution. No root code was reused. The saved samples are evaluation coordinates contacted by both routes; they are not a new independent scientific holdout. The acceptance gate is intentionally a finite seven-digit consistency audit, not the thirty-digit/raised-precision closure standard for a newly recognized mathematical constant.

