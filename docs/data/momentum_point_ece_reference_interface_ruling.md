# Independent cross-verification and consumer interface ruling

Date: 2026-10-13. Role: nonauthor spec-only verifier. Grade: **CONFIRMED for the pinned algebra and stated canonical uses; OPEN for unexecuted numerical gates and physical attribution to native errors.** This record does not assign VERIFIED-CLOSED; root must retain its separate consecutive S2/S3 records.

The verifier read TARGET_SPEC.md first and immediately attempted the skeleton write; the first PowerShell write failed to parse and was corrected before any additional read. No producer, arm, skeptic, or root proof code/draft was read. No consumer implementation was read. Native arrays and native numerical error values were not read. Seven frozen source files, their hashes, the actual primary contract, and source-only provenance metadata were read for C2. Fresh code was written without reuse.

## General proof chain, rather than inference from finite tests

Put Y=ky, Z=m(z+H), T=omega*t. Direct differentiation of the prescribed fields gives

N(u1)=A²*f*omega*m²/k * cos(Y)sin(Y)sin(T)cos(T) * [cos²(Z)+sin²(Z)],

N(v1)=-A²*omega²*m²/k * cos(Y)sin(Y)sin²(T) * [cos²(Z)+sin²(Z)],

N(theta1)=-gamma*A²*omega*m * sin(Z)cos(Z)sin(T)cos(T) * [sin²(Y)+cos²(Y)].

The two Pythagorean identities and double-angle identities give exactly Target I for all the specified real coordinates and parameters. No dispersion relation, eigenmode qualification, free-surface identity, or second-order solution is used. `independent_exact.py` derives the left sides by differentiation, then machine-checks three symbolic residuals and y independence. Its expected expressions are comparison targets, not imported premises.

For Target II let d=-z be depth down from the reference top, x=2d/H-1, and define the continuous top integration operator T(p)(d)=integral_0^d p(s)ds. The Rodrigues form P_l(x)=1/(2^l*l!)*D^l[(x²-1)^l] gives P_l(±1)=(±1)^l. Direct differentiation of its polynomial coefficients yields

D(P_(l+1)-P_(l-1))=(2l+1)P_l, l>=1.

This is an elementary Legendre polynomial identity added as a derived step, not assumed from the pin. One can verify it coefficient by coefficient using

P_l(x)=2^(-l) sum_(j=0..floor(l/2)) (-1)^j*(2l-2j)!/[j!*(l-j)!*(l-2j)!] x^(l-2j).

The endpoint difference vanishes at x=-1. Consequently T(P_0)=H(P_0+P_1)/2 and T(P_l)=H(P_(l+1)-P_(l-1))/[2(2l+1)] for l>=1. Orthogonal projection on degrees 0..n-1 deletes only the out-of-space P_n contribution of column n-1. This gives the pinned Qt. The executed independent route uses Rodrigues polynomials and two definite polynomial integrals to compute each projected entry, without the pin recurrence as a generator.

For every n, the only diagonal entry of MQt+Qt^T M is its (0,0) value H. Every adjacent off-diagonal pair has sum

-H/[2(2l+1)(2l-1)] + H/[2(2l-1)(2l+1)] = 0.

All other pairs are zero. Removing the last degree removes both members of an out-of-space pair, so the conclusion holds for every finite n>=2; it is not extrapolation from n<=5. Since M e0=e0, Qb=H e0e0^T-Qt gives M Qb=Qt^T M. In the derivative of one half X^T W X, the f U/V terms cancel because M is symmetric, the V/B terms cancel by this adjoint identity, and Kg E V^T M e0 cancels -(g/H)E KH e0^T V. This proves W Aop+Aop^T W=0 for all parameters in Target II. The finite executions test transcription and boundary handling with separately written Fraction matrix products; they do not supply the quantifier proof.

At n=2, the column P1 has continuous top integral H(P2-P0)/6. Projection retains -H P0/6 and discards **H P2/6** (equivalently P2/6 for Qt/H). With H=2 the discarded term is P2/3. At either endpoint the continuous integral is zero, while its projected value is -H/6. Thus projected Qt is not a pointwise exact top integral of the highest retained polynomial. Projected Qb has the opposite endpoint discrepancy. No exact finite-n boundary closure follows from energy skew.

## Exact execution evidence

`independent_results.json` and `exact.stdout` contain all cases declared before execution in `declaration.json`. The final successful run used suite Python `/root/.local/share/bootloops-suite/bootloops/.venv/bin/python`, SymPy 1.14.0, one affinity CPU, 180 s wall/CPU limits and 4 GiB memory fence, with aggregate RSS sampled every 0.1 s. True exit=0, wall=2.623915938 s, peak aggregate sampled RSS=94,609,408 bytes.

T1 reproduces (27/112000, -27/80000, -3/5000). The additional predeclared phase cases include zero and negative f and phase endpoints. T2 reproduces Qt=[[1,-1/3],[1,0]], Qb=[[1,1/3],[-1,0]], M=diag(1,1/3), W_EE=7/8 and the zero adjoint residual. n=2,3,4,5 and all declared combinations of K=0 or 3/5 and f=0 or ±2/7 pass. The planted Qb=-Qt copy has residual entries (V0,B0) and (B0,V0) equal -6/5, so the harness catches the missing rank-one constant. A T1 comparison perturbed by 1 is also caught.

The first exact run is retained in `run1-failed`: the trigonometric-only normalization failed to identify an elementary rational cancellation in the general neighbor identity. Its printed expression has identical factored denominators; this was a checker limitation and not a certified nonzero mathematical witness. A full relaunch after adding rational simplification passed the same predeclared set. The error and failed code are retained in the scoreboard.

Verbatim final stdout excerpt:

```text
C3_wrong_phase_copy: CAUGHT
C3_Hermitian_plant: PASS
circularity_import_audit: PASS
overall: CONFIRMED
```

## Per-use interface ruling

| Consumer use | Hypotheses and trace | Ruling |
|---|---|---|
| Input mapping k=pi/Ly, m=q/H, omega=sqrt(f²+N2(k/m)²), K=2k, nu=2omega | Actual frozen primary contract has H=100, Ly=20000, gamma=.02, N2=3.924e-5, f=1e-4, A=.01, T15/S35. Its H,k,m,omega,g,N2 are positive; K>=0 and all coefficients real. The dispersion relation is a consumer definition, unnecessary for Target I but compatible with it. Frozen evaluation reference lines 34–45 gives that first-order definition. | CONFIRMED for coefficient definitions; OPEN for reading later time/native observation arrays, which has not occurred here. |
| C1 Su into U0 and Sv into V0 | Target I proves these forces are independent of depth and proportional to sin(2ky), so each occupies only vertical Legendre degree 0 for the declared sin(Ky) velocity coefficients. nu=2omega is a direct variable substitution. | CONFIRMED. |
| C1 theta self-advection has zero Y-mode2 | Target I's N(theta1) has no y dependence, hence no cos(2ky) component. This statement concerns the self-advection source; linear second-order vertical advection of T0 still drives theta2_mode2. | CONFIRMED at stated strength. A claim that all theta2 is Y uniform is outside the pin and would be REFUTED as a consumption claim. |
| C1 theta20 | Integrate the source from time zero: theta20=-gamma*A²*m/4*sin(2m(z+H))*sin²(omega*t). Its derivative equals the freshly derived N(theta1) and its initial value is zero. This is the mean self-advection contribution in the specified fixed-H canonical idealization. The Y-mean vertical velocity is zero under fixed-bed incompressibility and zero bed flow, consistent with the canonical model. | CONFIRMED for this canonical contribution. Not a native full temperature solution. |
| C1 T1 composed theta20 | At the T1 phases, theta20=-3/2000 and its derivative=-3/5000, equal to N(theta1). Both checks are executed exactly. | CONFIRMED. |
| C2 EOS/background/interface | C2 is outside Targets I/II and is independently licensed by frozen source/contract, with gamma nonzero and alpha=N2/(g gamma). See the source derivation below. The seven frozen blobs match actual 3c8 provenance metadata. | CONFIRMED for this continuous linear-EOS second-order load; physical/native attribution remains OPEN. |
| C2 spatial pressure and V0 force | Eta1=A sin(q)cos(ky)cos(omega*t); its squared load gives a cos(2ky) pressure coefficient N2*A²*sin²(q)/4*cos²(omega*t). Minus its y derivative gives a positive sin(Ky) force K times the coefficient. Splitting cos² gives Q(1+cos(nu*t)), Q=K*N2*A²*sin²(q)/8. Symbolic residuals and sign are checked. | CONFIRMED for canonical continuum force. |
| C3 balancing | H,g,N2>0 imply W>0, with diagonal entries 1/(2l+1), 1/(N2(2l+1)), g/H. Let D=sqrt(W). Then S=D Aop D^(-1), and S+S^T=D^(-1)(W Aop+Aop^T W)D^(-1)=0. | CONFIRMED for all pinned n and parameters. |
| C3 Hermitian spectral phase | For real skew S, (iS)^*=iS. The finite-dimensional spectral theorem supplies a unitary eigenbasis with real lambda; this is an additional standard mathematical step, not an imported native hypothesis. In that basis S acts as -i lambda, so a=-i lambda. The theorem can be justified by induction: an eigenvalue of a Hermitian matrix is real by v*Hv=lambda v*v; its orthogonal complement is invariant, and restriction repeats. | CONFIRMED mathematical extension of Target II; wrong a=+i lambda is caught. |
| C3 J/time convolution | Variation of constants for xdot=a x+h exp(bt), x(0)=0 gives J=(exp(bt)-exp(at))/(b-a), equal to exp(at)expm1((b-a)t)/(b-a). Differentiation verifies the ODE and initial condition; b=a gives t exp(at). Decomposing cos/sin into exp(±i nu t) yields the three terms in the consumer. All residuals and the coincident limit are executed symbolically. | CONFIRMED; no epsilon clipping is needed or licensed. |
| C3 planted constant/resonance/zero-a controls | The specified 2x2 constant and resonant sine solutions satisfy their exact ODEs and zero initial data. The 1x1 zero-a constant/cos/sin formulas differentiate to their forces. nu→0 gives t for cosine/constant and 0 for sine. Flipping the rotation sign at t=pi/4 gives vector error (0,-1), maxabs=1>1e-3. | CONFIRMED for control formulas and exact false-copy detection. Implementation tolerances remain OPEN until its own numerical controls execute. |
| Reconstruction E and B/(g alpha) at physical z | E is defined as the cos(Ky) eta coefficient. B=g alpha theta2_mode2 is a definition with g alpha nonzero for the actual primary contract. Evaluating the declared Legendre expansion at native physical z is mathematically defined. | CONFIRMED definition; OPEN for actual sampling, arrays, volume/time weighting or reconstruction implementation. |
| Numerical skew/imaginary/unforced-energy gates | Exact algebra licenses the expected invariant. It does not guarantee <=1e-12 in a particular floating-point assembly, eigensolver or reconstruction. | OPEN until numerical gates execute; no gate substituted by the exact proof. |
| n16/32/64 convergence and surface/pressure/bed/top gates | Target II gives projected energy consistency only. Highest-degree endpoint leakage is explicitly demonstrated above. The stated convergence and <=1e-7 boundary tests are additional consumer acceptance requirements. | OPEN until the predeclared grids and gates execute; finite-n exact boundary closure is not licensed. |
| Timing/prediction/comparison workflow | No consumer implementation, eigendecomposition timing, actual prediction, native residual comparison, FFT, parameter fitting, or adaptive n expansion was executed by this verifier. Frozen scores and promotion gates are not changed. | OPEN operational requirements for subsequent root work. |
| Transfer to native full nonlinear truth, exclusive mechanism, native energy certificate or global SOTA | Neither pin nor source derivation includes FD walls, filters, time/process splitting, finite capacities, moving volume or kinematic eta*v terms. | OPEN physical attribution; such stronger closure is not licensed by this record. |

## Independent C2 source derivation and provenance

The current worktree pressure/EOS/config/evaluation files match the saved 3c8 byte hashes. Its factory and benchmark source have since changed, so these two were not silently treated as frozen: seven exact blobs were extracted from commit `3c8e86bef2ee1fbfd50cca4343d6fc61694d9edc`, hashed against `.../material-point-v4-marginfix/01-candidate-primary-main/zhenmode/native-run.json` source-only metadata, and preserved under `frozen-source/`. Each passed byte equality. `frozen_source_audit.json` records paths, hashes and verbatim line locators. One WSL Git read failed because the worktree .git pointer uses a Windows absolute path; it was retained and relaunched against the primary Git directory, with no source mutation. Final bounded source audit true exit=0, wall=.608370511 s, peak aggregate sampled RSS=36,286,464 bytes.

Load-bearing retrieved statements, with frozen snapshot locators:

* `frozen-source/src/zhenmode/model/solver/physics/eos.py:18`: `return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))`.
* `.../model/config.py:12,16,128,129,140`: G_EARTH=9.81, ALPHA_T=2.0e-4, T_ref=15, S_ref=35, thermodynamics='linear'.
* `.../execution/wave_zhenmode.py:19,88`: the default PhysicsConfig receives only zero process coefficients, preserving those reference/EOS defaults; `.../model/solver/factory.py:560,621,630` passes the reference temperatures, thermodynamics, and selected surface scheme to parameters.
* `.../benchmarks/internal_wave.py:133,146,148,149`: eta and theta have the prescribed first-order amplitudes, T=T_C+gamma*z+theta and S=S_psu.
* `.../evaluation/internal_wave.py:37,41,44,45`: the time-dependent linear reference uses the stated omega, eta, temperature perturbation and constant salinity. Only this reference function was read for this inference; no numerical error values were read.
* `.../model/solver/dynamics/pressure.py:54–58`: the selected branch is 'linear_density', with `slope = (rho[..., 0] - rho[..., 1]) / p.dz_3d[..., 0]` and `load = G_EARTH * state.eta * (rho[..., 0] + .5 * slope * state.eta)`; lines 77–83 give the negative density-pressure gradient divided by RHO_0.
* Actual saved `.../material-point-v4-marginfix/01-candidate-primary-main/contract.json`: case='primary', control='main', T_C=15, S_psu=35, eos.name='linear', hydrostatic_surface_scheme='linear_density', gamma=.02, g=9.81, N2=3.924e-5. Fraction arithmetic gives alpha=1/5000, N2=g alpha gamma=981/25000000, and drho_dT/rho0=-1/5000.

The following is **derived here**, rather than quoted from source. With constant S35 and T0=15+gamma*z, rho_background(0)=0 and rho_background_z=-rho0*alpha*gamma. At z=0, prescribed theta1_surface=-gamma*eta1. Thus rho1_surface=-rho0 alpha theta1_surface=rho0 alpha gamma eta1. The continuous extrapolation integral has the second-order coefficient

load2/rho0 = g eta1 [rho1_surface/rho0 + (1/2)rho_background_z/rho0*eta1]
= (1/2)g alpha gamma eta1² = (1/2)N2 eta1².

Perturbations of the surface slope multiply eta1² and first appear at third order. A nonzero background surface anomaly would introduce a first-order eta1 load and an additional second-order eta2 load; those are absent only because the actual T15/S35 matches the EOS references. Nonlinear EOS, salinity perturbations, arbitrary ocean backgrounds, and equivalence of the finite top-node slope to a continuously evaluated perturbed slope are not covered. If those hypotheses are changed, C2 must be downgraded OPEN and any pressure deletion must follow the consumer's recorded erratum path. No strong native claim is rescued by fitting.

## Lineage and circularity

The mathematical checker imports only Python stdlib, Fraction, and SymPy. Its only file read is its own source for AST import audit. The source auditor imports stdlib/Fraction and reads only the saved contract, source-only run provenance, and explicitly allowed frozen physical source blobs. `circularity_grep.txt` preserves the literal import/read/assumption search. There is no project import or reference solver import and no oracle influenced a fit or tolerance. The deepest shared layer with other arms is the shared pin, standard exact arithmetic and SymPy; independence is at author/code/derivation/input-exposure level, not a claim of entirely distinct symbolic engines. Fresh Fraction matrix multiplication and the Rodrigues integral route reduce dependence on the shared CAS recurrence implementation. Planted mismatches demonstrate that this comparator can reject false copies.

No mathematical FIRE or erratum was found. All setup failures, checker normalization failure, source-read failure and uncapped preliminary source-only audit attempts and the superseded direct-child-only source monitor remain visible in the scoreboard. The final source monitor samples the recursive process tree including Git subprocesses; the exact mathematical script spawns no grandchildren. The final records establish the scoped cross-verification and interface trace; root's separate workflow and later numerical gates retain their own authority.

