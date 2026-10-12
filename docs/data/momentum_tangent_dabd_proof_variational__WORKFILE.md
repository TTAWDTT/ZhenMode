# Variational proving arm — WORKFILE

State: OPEN. Regime 1. Read only pinned TARGET_SPEC.md and full prove-protocol skill. Parent reports X0 NOFIRE with T1/T2 and false copies caught; no X0 derivation imported. Forbidden cross-reading honored: no root proof or other proving-arm proof read.

Route lock: finite-dimensional variational / quadratic-form / weighted Cauchy plus discrete Green boundary identity. No infinite extension or Fourier proof.

## 1. Domain and assumptions
TODO: real n>=4, equal fixed weights, finite-matrix domain and imports.
## 2. Target I: Gram energy and weighted Cauchy
TODO: exact finite energy decomposition; Loewner bounds; contraction.
## 3. Target I: kernel, affine, sum and O(n) transpose
TODO: nullspace recursion; fixed-point space and preservation.
## 4. Target II: discrete Green and spectral bounds
TODO: endpoint-aware Green identity; complementary quadratic form; contraction.
## 5. Target II: finite sine eigenbasis and B comparison
TODO: endpoint recurrence, orthogonality/completeness, symbols and alternating kill.
## 6. Exclusions and pinned controls
TODO: positivity/TVD, T1/T2, coefficient plants, dynamic weights.
## 7. Exact certificate declaration and measured execution
TODO: CPU1, 180s, 4GiB sampledRSS; sourced suite SymPy only; n4 controls and predeclared n5/n8 algebra sanity only; scripts/logs retained.
## 8. Closure and provenance
TODO: author self-grade cannot exceed CLOSED-UNVERIFIED. Fresh skeptical and nonauthor/interface stages remain parent responsibilities.

### Increment 1 — finite-domain derivation and compute declaration

The real inner product is x^Tz; equal positive fixed quadrature w only multiplies it by w. All indices are finite, j=0,...,n-1, n>=4. No continuation or Fourier transform is used.

For each row (a,b,c), weighted Cauchy gives (a-2b+c)^2 <= 4(a^2+2b^2+c^2). Its exact defect is 2(a+b)^2+(a-c)^2+2(b+c)^2. Summing rows gives
16||x||^2-||Rx||^2 = 12(x_0^2+x_{n-1}^2)+4(x_1^2+x_{n-2}^2)+sum_{j=0}^{n-3}[2(x_j+x_{j+1})^2+(x_j-x_{j+2})^2+2(x_{j+1}+x_{j+2})^2].
This applies at n4 without overlapping extra endpoint terms because x_1 and x_{n-2} remain different.

The endpoint-aware finite Green identity is
-x^T L z = sum_{j=0}^{n-2}(x_{j+1}-x_j)(z_{j+1}-z_j)+2x_0z_0+2x_{n-1}z_{n-1}.
Its complementary identity is x^T(4I+L)z=sum_{j=0}^{n-2}(x_{j+1}+x_j)(z_{j+1}+z_j).
For z=x these give -4I<=L<=0. Endpoint diagonals -3 are essential.

Finite symmetric diagonalization is the only auxiliary linear algebra used. For completeness: a real symmetric matrix's Rayleigh quotient reaches a maximum on the compact unit sphere; differentiating along each tangent direction proves the maximizer is an eigenvector, and its orthogonal complement is invariant. Induction supplies an orthonormal eigenbasis. If this elementary lemma is rejected, contractions remain OPEN-AT the diagonalization step; no target itself is imported.

Compute declaration BEFORE code/execution: exact SymPy rational controls n4 only; optional n5/n8 predeclared finite symbolic algebra sanity, not eigenvalue scanning. Unknown runtime, measuring n4 first under CPU1/180s/4GiB sampledRSS. Progress unit: named passed assertions, accumulated in exact_controls.json and exact_controls.log. Sourced BootLoops suite environment; no licensed backend, no source-toolkit change, no native/project data. Toolkit index consulted: no general finite-filter quadratic certificate package exists; rankscreen is nearest exact-linear-algebra capability and its guide will be read before code. Direct problem-specific SymPy is justified because modular rank screens cannot certify the signed polynomial identities and pinned rational filter outputs.

### Increment 2 — preservation, finite sine route and excluded stronger claims

R1=0 and Rj=0. Conversely Rx=0 gives x_{j+2}-x_{j+1}=x_{j+1}-x_j, hence x_j=a+bj. Since R^TR is symmetric and x^TR^TRx=||Rx||^2, its nullspace equals ker(R). Thus F_t's eigenvalue-one space is exactly span(1,j). In scaled coordinates y_j=(j+1/2)/n the affine sample span is identical. Symmetry plus F_t1=1 yields 1^TF_tx=1^Tx. Two loops (form d_j=x_j-2x_{j+1}+x_{j+2}; add d_j,-2d_j,d_j to a length-n zero vector at j,j+1,j+2) give R^Td exactly and therefore O(n) evaluation.

For Target II, define theta=k*pi/n and s_j=sin((j+1/2)theta), k=1,...,n. Interior recurrence s_{j-1}+s_{j+1}=2cos(theta)s_j gives Ls=(2cos(theta)-2)s. Left finite ghost value s_{-1}=-s_0 and right finite ghost value s_n=-s_{n-1}, the latter because n*theta=k*pi. These are two scalar endpoint identities only, not infinite extension. Substitution into the recurrence gives exactly the -3 endpoint diagonal. Eigenvalues lambda_k=-4sin^2(theta/2) are strictly decreasing for k=1,...,n, so symmetric-matrix orthogonality makes the n nonzero sine vectors a basis. The squared norms are n/2 for k<n and n for k=n: telescope 2sin(theta)cos((2j+1)theta)=sin((2j+2)theta)-sin(2jtheta); sum and use sin(2n theta)=0. For k=n the entries are (-1)^j directly.

On this finite basis F_n has multiplier 1-sin^4(theta/2), whereas B_n=(I+L/4)^2 has multiplier cos^4(theta/2). F_n kills the alternating vector. Symbolic low-theta expansions will be checked but interpreted only as symbol departures.

Additional n4 exact controls predeclared BEFORE code: F_t e0 has a negative component and ordinary line-TV 17/16>TV(e0)=1; F_n e0 also has a negative component; F_n 1=T2 has TV1>TV(1)=0. These certify lack of pointwise positivity and TVD. Positive unequal W=diag(1,100,1,1) on e0 is a dynamic-weight exclusion control: F_t weighted squared norm 313/128>1 and F_n weighted squared norm 2537/256>1. Arbitrary variable positive weights cannot follow from equal fixed-weight contraction. No dynamic model is tested or asserted unstable.

### Increment 3 — measured exact controls and bounded follow-up plan

Executed command (WSL Ubuntu-24.04): source /root/.local/share/bootloops-suite/env.sh; cd this variational workspace; python exact_controls.py --sizes 4 --tag n4-pilot, stdout/stderr retained in exact_controls_n4-pilot.log. Exit 0. Every registered assertion passed. Verbatim log tail:
```
PASS Fn positivity and TVD exclusion
PASS unequal-weight exclusions
SUMMARY {"tag": "n4-pilot", "sizes": [4], "elapsed_seconds": 0.6600395210000016, "sampledRSS_peak_bytes": 68501504, "rss_sample_count": 56, "affinity": [0], "memory_limit_bytes": 4294967296, "cpu_seconds_limit": 180, "sympy_version": "1.14.0"}
```
The pilot measures the actual certificate in this sourced suite, including SymPy startup. Follow-up predeclared n5/n8 polynomial sanity: estimated 2.64s using conservative linear expression-count scaling from n4 (2*(8/4)*0.66004); this is conditional, not a measured duration. Same CPU1/180s/4GiB sampledRSS cap; no new input regime or eigenvalue scan. If rates collapse or asserts fail, retain that run and reopen the corresponding step.

### Increment 4 — completed all-n derivation and author closure

PROOF.md now completes original planned sections 1–8: finite spectral lemma and explicit downgrade path; local weighted-Cauchy SOS and finite boundary multiplicities; affine kernel, exact fixed space and mean; O(n) transpose accumulation; endpoint-aware bilinear Green/complement identities; complete finite sine basis with real normalization and both endpoint equations; B polynomial/symbol comparison and symbol-only low-wave orders; pinned rational controls, plants, positivity/TVD and varying-weight exclusions.

The n5/n8 sanity run exited zero and passed all assertions. Verbatim tail:
```
PASS n8 alternating kill
PASS n8 B polynomial
SUMMARY {"tag": "n5-n8-sanity", "sizes": [5, 8], "elapsed_seconds": 0.4398548700000049, "sampledRSS_peak_bytes": 68870144, "rss_sample_count": 36, "affinity": [0], "memory_limit_bytes": 4294967296, "cpu_seconds_limit": 180, "sympy_version": "1.14.0"}
```

Author closure state: Target I CLOSED-UNVERIFIED and Target II CLOSED-UNVERIFIED for every integer n>=4 in the pinned real equal fixed-weight domain. Exact finite certificates are controls, not proof by a finite scan. All-n closure rests on the displayed finite identities and recurrences. No root proof, other arm proof or implementation was accessed at any time. No numeric failures or deviations occurred in this arm. Initial TODO lines above are historical skeleton state; every item is now resolved in PROOF.md.

External stages remain: fresh skeptical exact passes, nonauthor spec-only reconstruction and consumer/interface trace. This arm does not claim VERIFIED-CLOSED, native stability, production changes, parameter fitting or a new scientific profile.
