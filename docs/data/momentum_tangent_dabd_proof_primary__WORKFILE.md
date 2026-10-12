# Primary incremental workfile

X0 was completed NO-FIRE before this proving artifact. Pinned TARGET_SPEC only. No native/source changes. Domain n>=4, real vectors, fixed equal positive weights. T1/T2 and coefficient1/4 planted copies use X0 exact control receipts explicitly REUSED, not rerun by primary. Independent proof arms are running their own forced routes and this primary does not read them.

## Target I direct finite algebra

For R_j x=x_j-2x_(j+1)+x_(j+2), weighted Cauchy gives (R_j x)^2<=4(x_j²+2x_(j+1)²+x_(j+2)²). Summing gives ||Rx||²<=16||x||² because the summed coefficient of each x_i² is at most4. Consequently M=R^T R is real symmetric positive semidefinite and its eigenvalues lie in[0,16]. F_t=I-M/16 thus has eigenvalues in[0,1], proving 0<=F_t<=I and ||F_t x||²<=||x||². An equivalent direct estimate is ||F_t x||²=||x||²-||Rx||²/8+||R^TRx||²/256 <=||x||²-||Rx||²/16. No bound here uses dynamic geometry.

R1=0 and Rj=0 exactly; the second-difference recurrence Rx=0 gives x_i=a+bi, so kerR is exactly the affine span. F_t preserves constants/affine samples, its eigenvalue-one space is this span, and 1^T F_t x=(F_t1)^T x=1^T x. Matrix application needs O(n): compute r_j once, then transpose accumulation y_i=r_i-2r_(i-1)+r_(i-2) with missing r=0. This agrees entrywise with R^T R, with no ghost/symmetry assumption on generic tangent.

## Target II quadratic form and recurrence

L diagonal is -2 except -3 at physical half-grid endpoints, with two offdiagonals1. Direct expansion gives -x^T Lx=sum_(j0..n-2)(x_(j+1)-x_j)²+2x0²+2x_(n-1)². It is nonnegative and at most4||x||²: bound each difference² by2(x_j²+x_(j+1)²), then the two endpoint penalties complete their coefficient to4. L is symmetric, hence its spectrum lies in[-4,0]. F_n=I-L²/16 is therefore symmetric with spectrum[0,1] and L2 contraction. No pointwise positivity follows.

For theta=k*pi/n with integer k1..n, s_j=sin(theta*(j+1/2)). The interior recurrence s_(j-1)+s_(j+1)=2cos(theta)s_j, while s_-1=-s0 and s_n=-s_(n-1), yield Ls=-4sin²(theta/2)s at endpoints as well. These n eigenvectors have distinct eigenvalues because theta/2 spans(0,pi/2] and sin² is strictly increasing; real symmetric L makes them orthogonal and they are nonzero, furnishing a basis. Thus Fn's symbol is1-sin^4(theta/2), alternating k=n is killed. B_n=(I+L/4)² yieldscos^4(theta/2), exactly the strictly-odd binomial stencil by expanding L and its odd-ghost recurrence. These normal formulas do not describe the source's generic tangential edge padding. Series at theta0: 1-Fn=theta^4/16+O(theta^6), 1-Bn=theta²/2+O(theta^4); this is standalonefilter low-wave behavior, not globalPDE convergence.

## Fixed tests, traps and prohibited conclusions

For n4 R^TR=[[1,-2,1,0],[-2,5,-4,1],[1,-4,5,-2],[0,1,-2,1]]. Applied to x=[0,1,4,9], Ft returns[-1/8,9/8,33/8,71/8]; it is not positivity-preserving. L²=[[10,-5,1,0],[-5,6,-4,1],[1,-4,6,-5],[0,1,-5,10]]; Fn1=[5/8,9/8,9/8,5/8], so it need not preserve constantnormal data or pointwise bounds. Coefficient1/4 in Ft sends alternating x to[0,2,-2,0], ||result||²8>4 and x^TF_wrongx=-4; in Fn it sends alternating x to-3x and squarednorm36>4. X0 exact receipts independently caught these predetermined copies; primary repeats no solver/control execution.

Current derivation appears CLOSED-UNVERIFIED for TargetsI/II only, pending independent forced routes/skeptics/non-authorcross. No formalproof/refereed/native/wholemodel/TVD/positivity/dynamicvolume certificate. Tangential-only Ft with unchanged Bn remains an admitted alternative, unselected. Consumer must still separately establish implementations, resource performance, physicaloriginalgates and fullCI before any native change.


## Explicit overclaim witnesses before skeptic seal

For n4, Ft e0=[15/16,1/8,-1/16,0] has finite forward-difference TV17/16>TV(e0)=1; this establishes a concrete non-TVD witness (the quadratic T1 proves negativity, not TVincrease). Fn1=[5/8,9/8,9/8,5/8] has TV1>TV(1)=0. Fn e2=[-1/16,1/4,5/8,5/16] is not positivity-preserving. All are direct rational consequences of the displayed n4 matrices, no numerical runs were added. These witnesses are about unrestricted vector/matrix claims; no claim is made they are admissible nativewall solutions.
