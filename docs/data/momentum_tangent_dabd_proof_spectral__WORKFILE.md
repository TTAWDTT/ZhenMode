# Spectral proving arm WORKFILE

Scope: TARGET_SPEC.md pinned, regime 1, n >= 4, fixed equal weights. Independent diversity arm: zero-extension shift convolution for tangential bound; midpoint sine basis for normal. No root/other arm/X0 derivation access.

Status: OPEN, skeleton created before supporting reading. X0 dispatch received as NO-FIRE; T1/T2 false copies caught, no derivation imported.

1. Preconditions and import ledger: pending protocols and toolkit source guide.
2. Predeclared computation plan: finite n=4 exact controls, T1/T2, both coefficient-1/4 plants; bounded sourced-suite SymPy CPU 1 / 180 s / 4 GiB with sampled RSS; no n scan.
3. Tangential zero-extension shift norm proof: pending.
4. Tangential invariants, kernel, evaluation correspondence: pending.
5. Normal midpoint sine basis, spectrum and contraction: pending.
6. Symbols and excluded overclaims: pending.
7. Exact script-of-record and receipts: pending.
8. Closure and remaining independent checks: pending. Self-grade ceiling CLOSED-UNVERIFIED; no final/native grade.

## Protocol reading receipt
Read prove-protocol and tool-stewardship in full; timing-discipline and planted-truth re-read separately after combined output truncation; independence-bookkeeping in full. No self-check upgrades workflow grade. Arithmetic controls are author checks only, not independent certification. Plants and clean twins retained. Runtime before measurement is unknown; first finite n4 control run is measurement, no extrapolated n scan authorized.

## 1. Capability selection and frozen finite-control plan

Consulted tools/README.md and full tools/rankscreen/GUIDE.md. rankscreen is a multi-prime screen for large exact-Q eliminations and explicitly is not a general sparse linear algebra library or final closure instrument. Its screen is not needed for the pinned n4 exact rational matrices. The problem-specific one-shot SymPy script below is justified by the small finite control and symbolic identity checks; no toolkit source is changed and no toolkit engine readiness is claimed. Installation guide consulted for capability entry. No optional/licensed backend used.

Known-answer startup: SymPy Rational(1,2)+Rational(1,3)=5/6, zero matrices, then exactly n4 only. Frozen expected controls are copied verbatim from TARGET_SPEC rather than generated from the implementation: T1 [-1/8,9/8,33/8,71/8]; T2 [5/8,9/8,9/8,5/8]; wrong Ft alternating output [0,2,-2,0], norm2 8 versus 4, quadratic form -4; wrong Fn alternating output -3x, norm2 36 versus 4. Clean twins must satisfy the matrix positivity/contraction detector. Comparator mismatch must fire.

Additional checks frozen before launch: symbolic-vector n4 O(n) loop versus dense matrix; n4 kernel dimension and affine identities; n4 normal midpoint sine residuals and orthogonality; symbolic recurrence, Fourier multiplier and filter-symbol identities, Taylor departure orders; total variation witnesses T1/T2. These are controls supporting the all-n proof, never an n scan or an all-n numerical certification. Per-check failure is an AssertionError and halts; no catch-and-continue. Progress unit: completed named exact checks in controls.json. First runtime unknown, measuring this bounded first run. Enforce Linux affinity one available CPU, RLIMIT_CPU=180 s, wall alarm=180 s, RLIMIT_AS=4 GiB; sample current RSS in a monitor thread and retain max/sample count in receipt.

Independence ledger: authored script depends only on pin and this arm's derivation; hand-written expected T1/T2/plants share TARGET_SPEC with every arm (declared deepest shared layer). This author's checks cannot certify this author's proof or independently validate the pin. No root/other arm code, proof, or X0 derivation read. External fresh skeptic and nonauthor reconstruction remain mandatory. Grade cannot exceed CLOSED-UNVERIFIED here.
## Failed finite control retained: v1 (not a clean pass)

The first run stopped at author-added Ft-not-TVD using T1. This was a mistaken choice of witness, not a target failure: T1 is monotone with endpoints -1/8 and 71/8, hence total variation 9 equals the input's 9. The exact assertion correctly fired and the failed script/output/control prefix are retained as *.failed-v1.*. No clean or independent pass counted. Shell command elapsed 2.4584155 s is a measured upper bound including interpreter and WSL/PowerShell overhead; no successful script receipt was emitted because the fail-closed assertion stopped it.

Predeclared correction before run v2: replace only the TV witness with e0=[1,0,0,0]. Hand-computed Ft e0=[15/16,1/8,-1/16,0] has TV=17/16 versus TV(e0)=1. Check its exact vector too. T1 still proves negative output. No parameter/family change, no n scan. Added one cheap matrix-vector check to the same n4 run; measured first attempted batch provides 2.459 s upper-bound precedent, planning still with hard CPU/wall180s/4GiB/oneCPU, actual duration will be recorded. This is an author-controlled revision, so the corrected witness is not independent verification. Denominator frozen at 35 checks.
## 3. Tangential proof: zero extension and shifts (all integers n >= 4)

Let E: R^n -> l2(Z) extend x by zero outside j=0,...,n-1. Let U be the integer shift (Uz)_j=z_(j+1); both U and U^2 preserve the l2 norm because they only reindex a convergent nonnegative sum. Let P restrict an integer-axis sequence to j=0,...,n-3; restriction cannot increase its l2 norm. These objects impose no reflection or periodic closure on the finite tangential vector.

Identity (S1): R = P (I-U)^2 E = P(I-2U+U^2)E. For each retained j, the expression is x_j-2x_(j+1)+x_(j+2), exactly the pinned R. The triangle inequality for the Hilbert norm, followed by the shift isometries, gives (S2): ||Rx|| <= ||(I-2U+U^2)Ex|| <= ||Ex||+2||UEx||+||U^2 Ex|| = 4||x||. The restriction is essential: the extra second differences of a zero extension at the external indices are discarded, not assigned to a finite boundary row. This is the shift/convolution route, not a finite boundary Green-form argument or a weighted Cauchy estimate. The script's convolution-Laurent-symbol also checks the algebraic multiplier identity (1-2z+z^2)(1-2/z+1/z^2)=(2-z-1/z)^2; no Fourier/Parseval import is necessary for this shift proof.

Set A=R^T R. For every x, (S3) x^T A x = ||Rx||^2 is nonnegative, and (S2) bounds it by 16||x||^2. Therefore A is real symmetric and 0 <= A <= 16I in Loewner order, and Ft=I-A/16 is real symmetric with 0 <= Ft <= I. By the finite symmetric-matrix eigenbasis lemma stated below, A has orthonormal eigenvectors with eigenvalues a in [0,16]; Ft has eigenvalues 1-a/16 in [0,1]. Expanding x in that basis gives (S4) ||Ft x||^2=sum_i(1-a_i/16)^2 c_i^2 <= sum_i c_i^2=||x||^2. The same argument applies over real x and any fixed common positive scalar quadrature weight.

Elementary linear-algebra import ledger: the proof uses the real symmetric eigenbasis fact, Euclidean norm triangle inequality, and the norm definition. This is ordinary finite linear algebra, not a specialized theorem imported from another proving arm. For completeness the eigenbasis fact follows by maximizing v^T A v on the unit sphere: compactness supplies a maximizer; its first derivative along every tangent w perpendicular to v is 2w^TAv=0, so Av is a scalar multiple of v. Symmetry makes v's orthogonal complement invariant. Repeat by induction on dimension. This also proves every spectral mapping claim for a polynomial p(A) by applying p to each eigenvector. If this elementary import is ruled inadmissible, downgrade this arm OPEN-AT S4 and retain S1-S3/invariants; no silent substitution of another arm's proof.

The all-n inequalities and finite eigenbasis argument are deductive proofs. Finite exact controls are tied to the matrix identities and each associated algebra below; they are not claimed to machine-prove a quantified norm inequality.

## 4. Tangential invariants, kernel, sum and O(n) identity

Identity (S5): R1=0 and Rj=0, since a second difference of constant or affine samples vanishes. Conversely Rx=0 says x_(j+2)-x_(j+1)=x_(j+1)-x_j for every retained j. Induction makes all first differences equal, hence x_j=a+bj. Thus null(R)=span(1,j), with linearly independent basis since n>=4. Scaled affine y_j=(j+1/2)/n is an invertible affine reparameterization of j, so the same statement holds for affine samples in y.

Identity (S6): Ax=0 if and only if Rx=0: forward follows from x^TAx=||Rx||^2, reverse directly from A=R^TR. Therefore Ft x=x if and only if x belongs to this exact two-dimensional span. Ft preserves every constant/affine sample. Symmetry and Ft1=1 give (S7) 1^T Ft x=(Ft1)^T x=1^T x; this preserves unweighted sum, and mean after dividing by n. Preservation does not rely on a universal tangential odd or Neumann boundary assumption.

For an O(n) application, compute r_j=x_j-2x_(j+1)+x_(j+2) for j=0,...,n-3 from the immutable input x. Start z=x and accumulate z_j-=r_j/16, z_(j+1)+=r_j/8, z_(j+2)-=r_j/16. Each row contributes -(r_j/16)(e_j-2e_(j+1)+e_(j+2)), hence (S8) z=x-R^T r/16=Ft x exactly over the reals. One second-difference pass and one transpose scatter take O(n) work and O(n) storage (storage can be reduced without mutating input prematurely). This does not promise bitwise identical floating-point evaluation order. Script check Ft-loop-is-matrix uses four free symbols and verifies the complete n4 linear identity, not one chosen vector.

## 5. Normal proof: explicit midpoint sine basis (all n >= 4)

For k=1,...,n set theta_k=k*pi/n and v_j^(k)=sin(theta_k*(j+1/2)). Extend these symbolic samples to j=-1 and j=n only to check the two ghosts. Identity (N1): v_-1=-v_0. Also v_n+v_(n-1)=sin(k*pi+theta_k/2)+sin(k*pi-theta_k/2)=2 sin(k*pi) cos(theta_k/2)=0, so v_n=-v_(n-1). Thus the pinned endpoint diagonal -3 precisely equals the second difference after the two odd ghosts are substituted; it is a separate normal boundary assumption.

The angle addition identity (N2) sin(u-theta)-2sin(u)+sin(u+theta)=-4sin(theta/2)^2 sin(u) holds for any u and theta. It gives the interior recurrence and, with (N1), both endpoint rows. Therefore (N3) L v^(k)=lambda_k v^(k), lambda_k=-4sin^2(k*pi/(2n)). The check general-sine-recurrence executes the free-angle identity, and left-odd-ghost/right-odd-ghost execute both symbolic ghost identities with integer k. The n4-sine-residual check separately tests the actual assembled endpoint rows.

Each v^(k) is nonzero: its first component sin(k*pi/(2n)) is strictly positive for 1<=k<=n. Since k*pi/(2n) increases in (0,pi/2], sin^2 is strictly increasing there. Thus all lambda_k are distinct. If k differs from ell, symmetry gives lambda_k <v^(k),v^(ell)> = <Lv^(k),v^(ell)> = <v^(k),Lv^(ell)> = lambda_ell <v^(k),v^(ell)>, hence their inner product is zero. The n nonzero orthogonal vectors are a basis of R^n. No unresolved completeness assumption remains; the n4-sine-orthogonal control tests the exceptional Nyquist normalization as well (diag(2,2,2,4)).

Polynomial mapping (N4): Fn v^(k)=(1-lambda_k^2/16)v^(k)=(1-sin^4(k*pi/(2n)))v^(k). These multipliers lie in [0,1), including 0 exactly at k=n. Expanding in the orthonormalized sine basis proves real symmetry, 0<=Fn<=I, and ||Fn x||<=||x||. Fn symmetry also follows directly because L^2 is symmetric. For k=n, v_j=sin(pi*(j+1/2))=(-1)^j, lambda_n=-4, and Fn v=0 exactly. The coefficient 1/16 was pinned before all arithmetic and was not selected by the controls.

An O(n) normal evaluation applies the pinned tridiagonal L twice, using first row -3x_0+x_1, interior x_(j-1)-2x_j+x_(j+1), last row x_(n-2)-3x_(n-1), then x-L(Lx)/16. This is exactly the same Fn, with no terrain or evolving weights inserted. Fn-loop-is-matrix checks this composition on free n4 symbols.

## 6. B comparison, symbols and excluded claims

For the pinned strict-odd normal B=(I+L/4)^2, (N5) B v^(k)=(1+lambda_k/4)^2 v^(k)=cos^4(theta_k/2) v^(k). This follows from 1-sin^2 a=cos^2 a. Writing s=sin(theta/2), the symbol difference is (N6) (1-s^4)-(1-s^2)^2=2s^2(1-s^2)>=0. Thus 0<=B<=Fn<=I in the same normal sine basis. This is a comparison between the two precisely defined normal matrices; no original tangential edge filter is identified with a strict-even reflection matrix.

As theta tends to zero, (N7) 1-fn(theta)=sin^4(theta/2)=theta^4/16+O(theta^6), and 1-b(theta)=1-cos^4(theta/2)=theta^2/2+O(theta^4). The free-angle exact Taylor check verifies both coefficients and orders. This is a symbol departure statement only; it is not a global PDE-order, whole-step stability, or eta-causality claim.

Both are not positivity/TVD certificates. Pinned T1 sends nonnegative [0,1,4,9] to [-1/8,9/8,33/8,71/8], giving a negative value. T1 is not a TV increase witness (retained v1 failure documents this). The endpoint pulse e0 has Ft e0=[15/16,1/8,-1/16,0] with discrete TV=17/16>TV(e0)=1, so Ft is not universally TVD. Fn has entry (0,2)=-1/16 at n4, so Fn e2 has a negative first component. Pinned T2 sends constant [1,1,1,1] to [5/8,9/8,9/8,5/8], so constant-normal preservation is false and its TV=1 exceeds zero input TV. The wrong-coefficient copies are separately caught: at n4 Ft_wrong alternating norm2=8>4 and quadratic=-4; Fn_wrong alternating norm2=36>4. Clean matrices pass the same PSD/contraction detector.

Equal fixed-weight Loewner and L2 arguments supply no dynamic eta-weighted contraction theorem, nonlinear coupled native energy theorem, pointwise positivity, or TVD guarantee. Odd normal closure remains a modeling assumption; generic tangential data do not justify a universal odd closure. The admissible tangential-only alternative Ft plus unchanged original B stays available, and this arm makes no design selection. No production/native files edited or consumed by the proof. Consumer source review and physical/software controls belong after algebra closure.
## Incremental algebra closure and measured additional plan

Sections S1-S8 and N1-N7 above are complete. Additional three same-n4/free-symbol arithmetic ties were identified before declaring closure: free-vector x^T R^TR x=(Rx)^T(Rx), free-symbol normal B difference=2s^2(1-s^2), and the normal n4 matrix Fn-B PSD. This adds no n or parameter sampling, fitting, precision change or backend. Prior successful v2 measured wall1.333565s/cpu1.153172s sampled peak72,052,736bytes (57 samples). Plan v3 uses the same machine/env/CPU/resource settings and these three cheap exact checks, with unchanged hard180s caps; runtime of v3 will be measured rather than assumed. Denominator now frozen38. Preserve v2 receipts and code. This is an author increment, not a fresh skeptic pass.
## 7. Script-of-record, identity ties and executed receipt

Script-of-record: exact_controls.py (current v3); stdout control_stdout.txt; complete exact named outcomes controls.json; environment/resource receipt receipt.json; stderr empty. v1 failed and v2 successful artifacts are retained separately and are not overwritten. Command executed via wsl Ubuntu-24.04, source /root/.local/share/bootloops-suite/env.sh, python <workspace>/spectral/exact_controls.py. SymPy1.14.0; exact rational/symbolic arithmetic only; n-values=[4].

Verbatim final stdout (four lines):

```text
AUTHOR-CONTROLS-PASS n=4 checks=38/38
T1=[-1/8, 9/8, 33/8, 71/8]; T2=[5/8, 9/8, 9/8, 5/8]
PLANTS CAUGHT: Ft norm2=8>4, quadratic=-4; Fn norm2=36>4
TIMING wall_seconds=1.352844 cpu_seconds=1.179166 sampled_peak_rss_bytes=71766016 samples=58
```

Arithmetic tie map: S1/S8 -> Ft-loop-is-matrix and convolution-Laurent-symbol; S2 -> shift isometry and norm triangle deductive argument plus clean-Ft-detector finite control (not an all-n computed certificate); S3 -> quadratic-Gram-identity and clean-Ft-detector; S4 -> finite spectral mapping from S3 and clean-Ft-detector; S5-S7 -> Ft-affine-and-sum and Ft-kernel-dimension, with recurrence induction providing the quantified kernel result; N1-N3 -> general-sine-recurrence, both ghost checks, n4-sine-residual; basis completeness -> nonzero/distinct-eigenvalue proof, checked finite orthogonality with n4-sine-orthogonal; N4 -> normal-F-symbol, n4-filter-spectral-map, Nyquist-killed and clean-Fn-detector; N5 -> B-symbol and n4-B-spectral-map; N6 -> B-difference-symbol and n4-normal-B-Loewner-comparison; N7 -> low-wave-departures. Fn O(n) -> Fn-loop-is-matrix. All T1/T2, nulls, clean twins, wrong coefficient plants, negative entries and TV witnesses have named exact controls.

No finite sampled identity or control replaces an all-n argument. In particular generic triangle inequality, compactness, sin monotonicity and recurrence induction are openly deductive sites for the fresh verifier to challenge; they are not described as executed all-n certificates. Real mathematical equalities apply to immutable-input real arithmetic; no bitwise software guarantee is claimed.

## 8. Initial closure ruling and outstanding mandatory stages

Self-grade: CLOSED-UNVERIFIED for both pinned targets I and II, all integers n>=4, under equal positive fixed weights and the stated two distinct boundary definitions. Route remained zero-extension/shift for tangent and explicit midpoint sine basis for normal. No root or other arm proof/code or X0 derivation was read. No production/native source or scientific data changed. No sweep, fit, optional or licensed backend was run.

Retained failed control v1 is an author witness-selection mistake, repaired by a separately predeclared endpoint-pulse TV witness. It is neither an independent skeptic failure nor a clean skeptic pass. Author controls v2/v3 are not the mandatory consecutive fresh-context skeptical passes. Root must arrange fresh skeptics and nonauthor spec-only reconstruction/interface ruling before any VERIFIED-CLOSED or final workflow grade; consumer implementation review and physical/software controls remain separate prerequisites for a native phase. This arm does not authorize that phase or select between combined and tangential-only design alternatives.

Grade downgrade path: if an all-n deductive site is refuted, OPEN-AT named S/N step; if the elementary symmetric eigenbasis import is ruled inadmissible, OPEN-AT S4 with S1-S3 and invariants intact. If finite control values fail replay, the author arithmetic record is DEAD at that check and closure reopens. The target pin is not silently modified.
