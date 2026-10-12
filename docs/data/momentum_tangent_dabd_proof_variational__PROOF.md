# Independent variational proof — CLOSED-UNVERIFIED

Status is an author grade only. No fresh skeptic or nonauthor verifier is claimed.
The pinned spec, the full prove-protocol, tool-stewardship and timing-discipline
skills, the toolkit index, rankscreen guide and installation quick start were
read. No root proof, other proving-arm proof, implementation consumer or X0
derivation was read. Parent supplied only X0 NOFIRE and control-catch status.
No Fourier argument, infinite zero extension or infinite reflection is used.

## 1. Domain and elementary finite spectral lemma

Fix any integer n>=4 and real column vectors x,z in R^n. The inner product is
x^Tz and ||x||_2^2=x^Tx. A common fixed quadrature weight w>0 gives
||x||_w^2=w||x||_2^2, so each stated contraction transfers to that norm.
Indices never wrap. Loewner inequalities here mean inequalities of real
quadratic forms for every real vector.

We use finite real-symmetric orthogonal diagonalization, justified as follows.
For a symmetric real matrix A the continuous Rayleigh quotient has a maximum
on the compact unit sphere. At a maximizing unit vector v, differentiating
along every tangent vector h orthogonal to v gives h^TAv=0. Thus Av is a
multiple of v. Symmetry makes v's orthogonal complement invariant; induction
on dimension provides an orthonormal eigenbasis. Consequently, quadratic-form
bounds aI<=A<=bI place each eigenvalue in [a,b], and scalar polynomial bounds
on that interval transfer to A. This is an auxiliary theorem proved here,
not the filter claim assumed under a different name. If this elementary
lemma is inadmissible, the spectral/contraction conclusions are OPEN-AT this
step; the explicit finite quadratic identities remain proved.

## 2. Target I: finite Gram form and weighted Cauchy

Let R have row j equal to the second difference at j,j+1,j+2. For reals a,b,c,
weighted Cauchy gives

    (a-2b+c)^2 <= (1+2+1)(a^2+2b^2+c^2).

The exact nonnegative defect is

    4(a^2+2b^2+c^2)-(a-2b+c)^2
       =2(a+b)^2+(a-c)^2+2(b+c)^2.                    (I.1)

When row inequalities are summed, a coordinate has multiplicities 1,2,1
from the containing rows. The two outer coordinates therefore receive
coefficient 4, the next two receive 12 and every remaining coordinate 16.
At n=4 the two next-to-outer coordinates are distinct and there are no
remaining coordinates. Explicitly, for every n>=4,

    16||x||_2^2-||Rx||_2^2
       =12(x_0^2+x_{n-1}^2)+4(x_1^2+x_{n-2}^2)
        +sum_{j=0}^{n-3}[2(x_j+x_{j+1})^2
                         +(x_j-x_{j+2})^2
                         +2(x_{j+1}+x_{j+2})^2].      (I.2)

Every term on the right is nonnegative. Also x^TR^TRx=||Rx||_2^2>=0.
Hence 0<=R^TR<=16I. Its symmetry follows from transposition, so

    F_t=I-R^TR/16 is symmetric and 0<=F_t<=I.          (I.3)

In the orthonormal eigenbasis of R^TR, each eigenvalue gamma is in [0,16].
The corresponding F_t eigenvalue 1-gamma/16 is in [0,1]. Squaring each
coordinate in that basis proves ||F_tx||_2^2<=||x||_2^2 for every real x.
This includes real vectors with any signs and makes no boundary hypothesis
on the tangential samples.

## 3. Target I: affine fixed space, mean and evaluation

Both the constant vector 1 and the index vector (0,1,...,n-1)^T lie in ker R.
Conversely, Rx=0 says

    x_{j+2}-x_{j+1}=x_{j+1}-x_j, j=0,...,n-3.

Thus every first difference equals one constant b, and x_j=a+bj. Consequently
ker R=span(1,j). The identity x^TR^TRx=||Rx||_2^2 implies
ker(R^TR)=ker R. Therefore

    F_tx=x iff x belongs to span(1,j).                (I.4)

The scaled samples y_j=(j+1/2)/n generate the same affine space, since y is
an affine function of j with nonzero slope. All constant and affine samples
are preserved. Symmetry and F_t1=1 give

    1^TF_tx=(F_t1)^Tx=1^Tx.                           (I.5)

This proves unweighted sum and hence arithmetic mean preservation; with a
common fixed quadrature weight it also proves that weighted sum preservation.

For evaluation, compute d_j=x_j-2x_{j+1}+x_{j+2} for j=0,...,n-3. Start
q=0 in R^n. For each j add d_j to q_j, -2d_j to q_{j+1}, and d_j to
q_{j+2}. These are exactly the three entries of row j of R multiplied by d_j,
so after addition q=R^Td=R^TRx. Returning x-q/16 equals the pinned F_t
matrix. There are two loops of linear length and no periodic wrapping.

## 4. Target II: discrete Green identities with endpoint terms

L is the pinned finite symmetric tridiagonal matrix: offdiagonals 1,
interior diagonals -2, outer diagonals -3. Expanding the finite sums gives

    -x^TLz=sum_{j=0}^{n-2}(x_{j+1}-x_j)(z_{j+1}-z_j)
             +2x_0z_0+2x_{n-1}z_{n-1},              (II.1)

    x^T(4I+L)z=sum_{j=0}^{n-2}(x_{j+1}+x_j)
                                      (z_{j+1}+z_j). (II.2)

The endpoint coefficients in (II.1) are 1 from the adjacent difference
and 2 from the penalty, giving precisely 3. In (II.2) each endpoint
appears once, matching 4-3=1. These checks explain why replacing endpoints
by interior coefficients would invalidate the proof.

Set z=x. Identity (II.1) proves L<=0, while (II.2) proves -4I<=L. By
the finite symmetric lemma each eigenvalue lambda lies in [-4,0]. Thus
L^2 has eigenvalues lambda^2 in [0,16], and

    F_n=I-L^2/16 is symmetric, 0<=F_n<=I,
    and ||F_nx||_2<=||x||_2.                          (II.3)

This is the quadratic-form argument for the normal filter. It does not
rely on the subsequent explicit sine diagonalization.

## 5. Target II: finite sine recurrence, endpoint identities and real norms

Fix k in {1,...,n}, theta=k*pi/n, and s_j=sin((j+1/2)theta). The elementary
addition formula gives

    s_{j-1}+s_{j+1}=2cos(theta)s_j.                   (II.4)

For the left endpoint its single auxiliary value is s_{-1}=-s_0. For
the right endpoint n*theta=k*pi gives

    s_n=(-1)^k sin(theta/2),
    s_{n-1}=-(-1)^k sin(theta/2), so s_n=-s_{n-1}.    (II.5)

These are two scalar endpoint identities. No infinitely extended sequence
is introduced. Applying (II.4) at j=0 with the left identity gives
-3s_0+s_1=(2cos(theta)-2)s_0; applying it at j=n-1 gives the analogous
right endpoint equation. The interior recurrence and both endpoints imply

    Ls=lambda_ks, lambda_k=2cos(theta)-2
                         =-4sin^2(k*pi/(2n)).         (II.6)

The first component sin(k*pi/(2n)) is positive, so every s is nonzero.
The n eigenvalues are distinct: k*pi/(2n) strictly increases within
(0,pi/2], and sin is strictly increasing there. For symmetric L, subtracting
s_k^TLs_l=(Ls_k)^Ts_l shows that distinct eigenvalues give orthogonality.
Thus these n real vectors form an orthogonal basis of R^n.

Their exact real norms can also be obtained by finite telescoping. For k<n,
sin(theta)>0 and

    2sin(theta)cos((2j+1)theta)
       =sin((2j+2)theta)-sin(2jtheta).                (II.7)

Sum over j=0,...,n-1. The right side is sin(2n theta)-sin(0)=0. Hence
sum_j cos((2j+1)theta)=0, and sin^2 u=(1-cos(2u))/2 gives

    ||s_k||_2^2=n/2 for 1<=k<n;
    s_{n,j}=(-1)^j and ||s_n||_2^2=n.                (II.8)

So normalized real basis vectors are sqrt(2/n)s_k for k<n and
n^(-1/2)s_n. This finite normalization includes the exceptional Nyquist mode.

Applying the filter polynomial to (II.6) gives

    F_ns_k=[1-sin^4(k*pi/(2n))]s_k.                 (II.9)

For k=n the multiplier is zero, so the alternating vector is killed
exactly. In particular no zero-eigenvalue or constant-normal preservation
assertion is made for this odd wall matrix.

For the original strictly odd binomial matrix,

    B_n=(I+L/4)^2=I+L/2+L^2/16,
    B_ns_k=cos^4(k*pi/(2n))s_k.                      (II.10)

Here 1+lambda_k/4=1-sin^2(theta/2)=cos^2(theta/2), proving the comparison.
With theta tending to zero, the exact symbol departures expand as

    1-symbol(F_n)=theta^4/16+O(theta^6),
    1-symbol(B_n)=theta^2/2-5theta^4/48+O(theta^6).  (II.11)

These statements concern these filter symbols only. They imply neither
global PDE order nor an identification of a tangential edge filter with
strict even reflection.

## 6. Exact pinned controls and exclusions

At n=4, direct exact evaluation gives

    F_t[0,1,4,9]^T=[-1/8,9/8,33/8,71/8]^T (T1),
    F_n[1,1,1,1]^T=[5/8,9/8,9/8,5/8]^T (T2).

Define ordinary finite-line variation TV(x)=sum_{j=0}^{n-2}|x_{j+1}-x_j|.
On e_0=[1,0,0,0]^T,

    F_te_0=[15/16,1/8,-1/16,0]^T,
    F_ne_0=[3/8,5/16,-1/16,0]^T.

Both have negative components despite nonnegative input, so neither is a
pointwise positivity certificate. TV(e_0)=1 but TV(F_te_0)=17/16; also
TV(1)=0 but TV(F_n1)=1 from T2. Thus neither is a finite-line TVD
certificate. Loewner positivity is distinct from entrywise positivity.

The false-copy trap replacing 1/16 by 1/4 produces, on x=[1,-1,1,-1]^T,

    (I-R^TR/4)x=[0,2,-2,0]^T,
    squared output norm=8>4, x^T(I-R^TR/4)x=-4;
    (I-L^2/4)x=-3x, squared output norm=36>4.

Both plants were caught by executed exact assertions and are retained as
controls, not accepted filter variants.

Equal fixed weights do not imply arbitrary positive weights. Explicitly,
for W=diag(1,100,1,1) and e_0,

    e_0^TWe_0=1,
    (F_te_0)^TW(F_te_0)=313/128>1,
    (F_ne_0)^TW(F_ne_0)=2537/256>1.

This excludes an inference to all positive varying volume weights, which
already includes this positive snapshot. It does not assert instability
of a specified eta model. Dynamic eta-weighted contraction, coupled nonlinear
native stability, arbitrary terrain, and pressure/continuity effects remain
outside this finite algebra. Normal odd reflection is the separately imposed
L boundary assumption; generic tangential data do not justify universal odd
or Neumann reflection. Keeping original B_n and applying only F_t is an
admissible algebraic option; this proof selects no design.

## 7. Executed certificates and scope of evidence

`exact_controls.py` is the retained script of record. It runs only sourced
suite SymPy 1.14.0, one CPU affinity, CPU/time cap 180s and address-space cap
4GiB, with RSS sampled every 0.01s. No installed source was changed.
The known rational-answer gate runs before the substantive controls.

The exact local weighted-Cauchy identity (I.1), generic sine recurrence
(II.4), reduced endpoint parity (II.5), telescoping summand (II.7), filter
symbols and expansions (II.9)-(II.11) all have named script assertions.
The finite aggregate identities (I.2), (II.1)-(II.2), symmetry, affine
preservation, rank, transpose evaluation, alternating kill and B polynomial
were checked symbolically at the predeclared sizes n4, n5, n8. Every displayed
n4 control above was independently computed with exact rational arithmetic
in this arm. JSON receipts retain the actual values.

These checks are controls of the written algebra, not a finite-size inference
to all integers n. The coordinate multiplicity count, nullspace recursion,
bilinear expansion, scalar endpoint recurrence, strict eigenvalue ordering
and finite spectral lemma are the all-n proof. A fresh verifier must rebuild
them and reproduce the registered outputs with fresh code.

Verbatim retained n4 log tail:

```
PASS Fn positivity and TVD exclusion
PASS unequal-weight exclusions
SUMMARY {"tag": "n4-pilot", "sizes": [4], "elapsed_seconds": 0.6600395210000016, "sampledRSS_peak_bytes": 68501504, "rss_sample_count": 56, "affinity": [0], "memory_limit_bytes": 4294967296, "cpu_seconds_limit": 180, "sympy_version": "1.14.0"}
```

Verbatim retained n5/n8 log tail:

```
PASS n8 alternating kill
PASS n8 B polynomial
SUMMARY {"tag": "n5-n8-sanity", "sizes": [5, 8], "elapsed_seconds": 0.4398548700000049, "sampledRSS_peak_bytes": 68870144, "rss_sample_count": 36, "affinity": [0], "memory_limit_bytes": 4294967296, "cpu_seconds_limit": 180, "sympy_version": "1.14.0"}
```

Both runs exited zero. Author closure: Target I and II CLOSED-UNVERIFIED
for every pinned integer n>=4. Proof protocol final grade, two fresh skeptic
passes, nonauthor spec-only reconstruction and complete consumer interface
trace remain external stages. No native or production authorization follows.
