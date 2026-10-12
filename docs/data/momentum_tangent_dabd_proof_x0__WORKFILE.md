# X0 hostile spec-only hunt

Regime: 1, pinned filter targets. Grade: RUNNING / not a proof closure.
Context: fresh nonauthor adversary; read TARGET_SPEC only, no other proof.
Scope: n>=4, real vectors, equal positive fixed weights, exact coefficient 1/16.

## Preregistered computation
- First run: control.py, SymPy exact, n=4 only. Rebuild R/L, verify T1/T2, both false 1/4 traps and Nyquist; time actual control before further calculations.
- Gates: all pinned exact control values match; false copies MUST be CAUGHT. Any in-domain contraction/preservation/symbol counterexample gives FIRE with exact witness.
- Resources: WSL Ubuntu-24.04 sourced suite env, CPU=1, wall=180 s, memory=4 GiB, sampled RSS; no licensed backend, no toolkit edits.
- Further run, conditional on first timing: hunt.py n=4,5,8 exact matrix sanity and finite principal-minor/low-order checks, plus exact sine residual checks. This is a bounded adversarial screen, not a global proof.

## Toolkit readiness and guide
PENDING consult README, exactn4 full guide and recorded readiness.

## First exact control
PENDING script, actual time, log and resources.

## Adversarial boundary hunt
PENDING inspect smallest n, endpoints, sine k=n, low-order tangent preservation, coefficient trap, and forbidden overclaims.

## Verdict
PENDING. NO-FIRE permits prover pipeline only; cannot close target.

Toolkit consultation completed before code: tools/README.md and rankscreen/GUIDE.md fully read; rankscreen is for large modular elimination and explicitly is not a generic exact matrix certificate. No exactn4-named installed tool/guide was found. SymPy core is the applicable exact arithmetic backend; these are one-shot pinned problem evaluators, not tool forks. Quick-start historical installation report read; it does not establish target truth. First control additionally checks rational known answer 1/2+1/3=5/6. No optional/licensed backend selected.

## First control executed (2026-10-13)
Command: sourced suite python run_bounded.py control.py control. Exit=0. Actual wrapper wall=0.712909690 s; sampled peakRSS=62748 KiB; CPU affinity [0]; 4 GiB AS fence and 180 s CPU/wall fences active. control.log and control.resources.json preserve exact output and samples.

### Measured follow-on plan
The exact n4 construction/rational controls fit under 1 second. Follow-on n4/n5/n8 may cost more because all principal minors and sine simplification are new operations, so their runtime is unknown rather than projected from rational n4. Keep the independently bounded 180 s fence; progress unit is one completed n row in hunt.log. No expansion if incomplete; retain timeout/unknowns. Predeclared finite scan only n4,n5,n8, all sine k1..n, no eigenvalue scan or floating data. Exact principal minors of Ft and I-Ft and Fn and I-Fn test Loewner constraints at these sizes. Tangent constants/affine/sum/nullspace, direct R-transpose route, and normal sine/B symbols form boundary attacks. Out-of-target positivity/TVD failures are separate witnesses, never FIRE.

## Finite hunt attempt 1 retained
Exit=1 after 3.778836120 s, sampled peakRSS=73000 KiB. n4 and n5 passed every declared check, but default SymPy simplify did not certify the n8,k1 sine residual zero. This is a CAS check-gate failure, not a FIRE: script only detected failure to normalize and did not establish any nonzero exact residual. Preserve hunt.py, hunt.log, hunt.results.json and hunt.resources.json unchanged.
Repair plan: retain rational matrix/minor route, replace n8 sine residual normalization with polynomial arithmetic in c=cos(pi/16), minimal polynomial 128c^8-256c^6+160c^4-32c^2+1. Chebyshev expressions for sin(m*pi/16)/sin(pi/16) and cos(m*pi/16) represent the pinned sine exactly; reduce each residual polynomial modulo that minimal polynomial. Same n4/n5/n8 cases, no enlargement, same resource fence. This is an independent exact representation repair; attempt1 does not count as clean.

## Repaired finite hunt executed
Exit=0 after 8.158259469 s; sampled peakRSS=83812 KiB. Every declared principal-minor, contraction quadratic-form, symmetry, tangent low-order/nullspace, direct-application, and normal sine/B symbol check passed at n4,n5,n8. Exact n8 representation repair normalized the stalled trig residual to zero, so no nonzero target witness existed in attempt1. hunt_repair.py/log/results/resources are the script of record for the full finite screen. Attempt1 retained as FAILED CERTIFIER, not counted clean.

## Scope/adversarial boundary ruling
- Smallest domain n4 and half-grid endpoint diagonals were attacked explicitly; k=n gives exact Fn Nyquist annihilation.
- T1 negative output refutes positivity; e0 also gives tangent TV increase from 1 to 17/16. T2 gives normal TV increase from 0 to 1. These refute excluded overclaims and DO NOT contradict pinned target.
- Constant-normal preservation is neither asserted nor true (T2). Generic tangential role does not license normal odd closure. No periodic wrapping was introduced.
- Fixed uniform quadrature only: W=diag(1000,1,1,1) yields ||Ft e1||_W^2=529/32 > ||e1||_W^2=1. This confirms arbitrary/dynamic weighted contraction is excluded, without introducing a new scientific claim or native data.
- Low-wave expansions refer only to the printed scalar symbols; no PDE energy, TVD, terrain, dynamic-volume, or physical qualification follows.

## Verbatim control output excerpts
```text
T1 Ft*[0,1,4,9]=[-1/8, 9/8, 33/8, 71/8] PASS negative=-1/8
T2 Fn*ones=[5/8, 9/8, 9/8, 5/8] PASS nonconstant
FALSE_COPY tangential=[0, 2, -2, 0] norm2=8 inputnorm2=4 quadratic=-4 CAUGHT
FALSE_COPY normal=[-3, 3, -3, 3] norm2=36 inputnorm2=4 CAUGHT; Fn*Nyquist=0 PASS
```

## Verdict: X0 NO-FIRE
No in-domain exact counterexample or pinned-spec domain defect found. Both wrong coefficient copies CAUGHT; T1/T2 independently reproduced. This finite adversarial screen permits the primary and two distinct diversity arms to begin. It does not prove the targets for all n, constitute a skeptical proof pass, certify a consumer, or close the BootLoops workflow. General-n targets remain OPEN pending the instructed proof, skeptical passes, nonauthor reconstruction, and interface trace.

### Append-only exact transcription correction
The weighted norm value in the scope paragraph above was mistyped as 529/32. The script-of-record value is **2069/128**, as printed verbatim below; no computation or verdict changes.
```text
OUT_OF_TARGET Ft e0=[15/16, 1/8, -1/16, 0] TV=17/16 inputTV=1; Fn ones TV=1 inputTV=0; W=diag(1000,1,1,1) Ft e1 weightednorm2=2069/128 input=1
SYMBOL_CHECK normal_departure=theta**4/16 + O(theta**6) original_departure=theta**2/2 - 5*theta**4/48 + O(theta**6)
X0_NO_FIRE finite cases only; general-n proof OPEN; total_seconds=7.398170273000005
```
