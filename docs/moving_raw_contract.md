# Manufactured moving raw means and characteristic ALE slice

## 2026-10-02: design before numerical execution

The base is merged main `7223a5d533f95bc289a35b2dab62901c2b2776ae`.
The next pilot uses a new, explicitly declared manufactured raw mean authority.
It does not identify the original global momentum control volumes. The patch
has two adjacent half-prisms, each with its own fourteen horizontal raw layers.
The first raw layer changes thickness; interior cuts and the bottom stay fixed.
This is different from the original moving top-three-band design.

Actual raw `h, IT, IS, Mu, Mv` remain authoritative. Velocity means are located
at x=d/4 and 3d/4. The old fixture stores endpoint velocities; interpreting those
as half-CV means changes the physical field. The new fixture integrates actual
half-CV means explicitly and uses independent declared physical parameters.

### Mean basis and moving frame

Let O be the actual physical-depth overlap between the two raw layer partitions,
D=diag(rho0*A*h), A=d*L/2, and

```text
K = [[0, O/hL], [O.T/hR, 0]]
Q = .75*I + .25*K
Psi = N*Q^-1
W_N = D*(2*I + K)/3
M = Q^-T * W_N * Q^-1
```

Q maps endpoint coefficients to actual half-prism means. D*K is symmetric and
the D-weighted spectrum of K lies in [-1,1], so Q is invertible without a fitted
constraint or regularization. Q*ones=ones and M*ones=D*ones; the same-field
momentum map R=M*D^-1 and its inverse conserve global impulse. This is a new
mean basis, not reuse of the old endpoint metric. In exact geometry arithmetic,
D <= M <= (4/3)*D. M-D is the actual within-CV reconstruction covariance.

The domain interior endpoint indicators N stay fixed, while the top shape moves.
Both inverse-side derivatives are required in Mdot. Qdot contains overlap motion
and the own-volume denominator derivative; Rdot also contains Ddot. Qdot*c=0
on a depth-uniform affine field does not eliminate every test-basis frame term.
Separate general-basis/shear geometry probes test the full matrices and frame
terms, but do not qualify a general sheared flow. Their finite-difference
truncation envelope is separate from roundoff identities and time quadrature.

### Restricted finite-volume method and independent oracle

The accepted family has common flat eta and bottom, T=Tref, inactive actual
T/S and density limiters, common stable affine density in z, globally affine
horizontal velocity, and constant transverse V. There is no horizontal density
gradient. Every actual mean and both endpoints of each auxiliary P1 reconstruction
must qualify the canonical affine field within the frozen primitive-operation
envelope. The original stocks are never changed to impose that qualification.

The candidate follows constant-pressure-gradient horizontal particle acceleration
and the incompressible vertical Jacobian. At every physical face it traces
inverse particle labels, integrates physical transport and Pa tractions, and
updates IT/IS/Mu/Mv only by signed FV fluxes and pressure impulses. The analytic
end-state array is never used to replace accepted stocks. The ALE grid trajectory
defines h geometrically before flux evaluation. Every raw water row must match
the consumed Q/R ledger; failures reject, without changing Q/R or moving the
mesh to remove a residual. Local, global and cumulative floating-point GCL
defects and their predetermined bounds are retained.

One shared horizontal union-depth face is consumed by both raw owners with
opposite signs. Each column's internal vertical ALE R is consumed by both
adjacent layers. Surface and bottom relative material fluxes vanish. Open
inflow is specified by the globally affine initial-field extension.

The spatial transport authority is explicitly the qualified canonical affine
reconstruction. This is not literal advection of a possibly discontinuous
initial auxiliary P1 profile. Initial endpoint deviations and primitive-operation
bounds yield separate reconstruction majorants. They cover both velocity
components, alpha/w, T/S, quadratic momentum flux, cubic KE flux, Pa, pressure
work, terminal stock quotients and inactive-P1 reconstruction. Absolute affine
weights cover open-inflow extrapolation. These errors are not added after seeing
an acceptance residual and do not become a physical loss or a hidden projection.

The oracle has its own Eulerian PDE derivation and does not import candidate
label, path, polynomial, consumer, mean-map or after-stock helpers. It binds
actual before/after stocks by independent end-CV volume integration, checks PDE
residuals, all absolute physical face quantities, local ledgers and full boundary
work. Agreement with the same analytic path alone would be insufficient.

### Energy and time integration envelopes

Physical KE uses actual M and actual mean velocities; raw mean KE uses D.
The difference is explicitly retained. For globally affine velocity,

```text
K_physical - K_raw = rho0*L*H*d^3*alpha^2/96.
```

Each raw CV consumes physical KE and gravitational PE transport and pressure
energy flux. Internal vertical p*R cancels between adjacent cells only after
both local budgets pass. The material top has R=0 but nonzero external Pa work
through its actual motion. Local rho0*g*z potential sums to the global rho0
free-surface PE after adding a fixed bottom constant; rho-prime gravity PE is
counted once. No vertical kinetic-energy claim is made for this hydrostatic slice.

The exact before/after mass-chain identity is checked independently for D and M:

```text
DeltaK = a_mid.T*DeltaMomentum - .5*a_after.T*DeltaMass*a_before.
```

This algebraic impulse work is distinct from true time-integrated horizontal
pressure work. Midpoint velocity times integrated pressure impulse is separately
reported too. Their nonlinear differences are not called dissipation or loss.

Candidate face integrands use canonical rational time polynomials. Numerator
degree and lambda denominator power are mechanically audited per execution
channel against the protocol's degree table. The absolute coefficient-operation
polynomial includes all physical length and area factors before cancellation.
The 16-point remainder is dimensional: 2*dt*sum_abs_coeff*T16. The independent
32-point oracle has its own coefficient majorants and T32 bound. Input
reconstruction, time quadrature and roundoff are separate budgets. Neither
minmod reconstruction nor Q/M/R top-thickness denominators are covered by a
lambda-only time-tail claim. No generic time-order claim follows from the
restricted exact-characteristic family.

### Frozen checks and limits

The positive and negative cases use alpha=+/-0.08, U=+/-0.03, V=-0.02,
external pressure [80,200] or [200,80] Pa, and successive 0.02 and 0.03 s steps.
Zero-pressure moving and alpha=0 fixed-eta controls are retained. Geometry,
mean/frame, input/material, absolute-face, energy and rollback controls are
frozen in `moving_raw_protocol.json` before numerical execution.

Every numerical invocation is serial, single CPU, hard wall 180 s and owned
process-tree memory 4 GiB. The first batch totals at most 20 minutes. Evidence
must bind its actual clean scientific source, with failed development runs
retained and not retrospectively rebound. Default production, real353, the
original CV, original top-three geometry and predict/12fast/replay remain
unqualified. Mixing, biharmonic, FCT/filter, wind, heat, rotation and drag are
not silently applied or qualified by this pilot. No merge or license change.

### Exact manufactured inputs

The base-7223 fixture partitions fourteen layers using normalized left weights
1 through 14 and right weights (14 through 1)^1.3. Initial a0=1 is the global
z intercept, so q_b=a0+s*b. Inherited inventory EOS is rho0=1025 kg/m3,
Tref=15, Sref=35, alpha=2e-4, beta=7.6e-4, g=9.81 m/s2.
The nonzero V direction is periodic extrusion; its paired transverse faces
cancel. These choices are frozen inputs, not inferred from candidate results.

### 2026-10-02 UTC: accepted numerical pilot

Scientific source `d3d29200cf0b09fccb5451e1a2b3e5aa7fb97401` produced
[the scalar witness](moving_raw_evidence.json): six actual raw commits in four
cases, of which five move the first raw slot. Positive and negative affine
flows each take successive 0.02 and 0.03 s steps. The zero-pressure moving
case and alpha=0 fixed-eta control also pass. The latter preserves actual eta
exactly. This is a restricted manufactured numerical pilot, not industrial,
original-CV or production qualification.

The accepted geometry is defined by the flow map before integration. Every
actual h row closes the ALE GCL within its predetermined envelope. Stocks
IT/IS/Mu/Mv are only changed by signed horizontal/vertical stock fluxes and
actual pressure impulse. No closed-form end stock is assigned. Cumulative
GCL bounds retain the sum of all prior per-step budgets and accumulation
roundoff; they are committed atomically with the raw state. Canonical
reconstruction errors use pre-step actual P1 endpoints, absolute open-inflow
extrapolation weights, a positive exact quotient denominator, neighbor/global
P1 bounds and Q-inverse bounds for local physical kinetic energy.

For the positive case, the observed actual changes are:

| dt (s) | eta change (m) | maximum deep IS change per area | maximum deep Mu change (kg/(m s)) | physical KE change (J) | raw KE change (J) | PE change (J) |
|---|---:|---:|---:|---:|---:|---:|
| 0.02 | -0.00447284345048 | 0.000379163377874 | 0.490432374867 | -1.39442561263 | -1.38343998537 | 27.3994944396 |
| 0.03 | -0.00668253503558 | 0.000568745066809 | 0.734056336952 | -2.04977899314 | -2.03343157572 | 42.0614650215 |

Actual M changes have Frobenius norms 11.5448651566 and 17.2514802878; actual
R changes have norms 0.00193173076488 and 0.00296452776997. Covariance KE
changes are -0.0109856272590 and -0.0163474174135 J. Internal pressure-energy
face integrals are nonzero, reaching 47.6838114637 and 71.1145219545 J;
moving-surface external pressure work is 1.87859424920 and 2.80666471494 J.
These are physical full-KE face and inventory accounts; no owner-dependent
variance face is falsely shared.

True pressure time work is -1.10113769189 and -1.62308735913 J. Actual
midpoint mean velocity times the integrated pressure impulse differs by
-2.34534057286e-7 and -7.74737985543e-7 J, against predeclared work bounds
6.16586991411e-12 and 9.26551290750e-12 J. The nonlinear difference is
resolved and is not called loss. Actual raw/physical variable-mass chains
separately retain nonzero mass work; replacing them by a fixed-mass identity
fails the frozen omission criterion.

The independent Eulerian oracle imports no candidate characteristic,
polynomial, mean-metric, face consumer or after-stock helper. It binds actual
end-CV inventories with seven-point volume rules, all absolute faces with
32-time/seven-space rules, and derives separate fsum water/TS/impulse and
local/global energy ledgers. The largest positive-case local GCL residual is
7.45931094670e-17 m3; two-step cumulative residual is at most
4.85722573274e-17 m3. Independent local total-energy residuals are at most
5.01554353605e-12 J. These finite physical ledgers use dimensional input,
roundoff and time-tail budgets; their residuals are not labeled roundoff-only
identities. The positive-case largest independent bound ratio is
0.000750823958750. Algebraic mass-chain identities use only operation
roundoff. There is no generic second-order or equal-error speed claim.

Executed horizontal rational numerators reach degree 7 and lambda power 4,
within the frozen per-channel table. The positive second-step maximum gravity
PE face tail bound is 1.15169085934e-53 J, separately reported from its
8.75534980943e-10 J input reconstruction bound and operation roundoff.
Agreement of the two Gauss rules is not substituted for the analytic tail
proof. The witness binds 256 actual source files by relative-path SHA256 and
records Python 3.12.14, NumPy 2.5.3, pytest 9.1.1 and Ruff 0.16.8.

### Validation, failures and reproduction

Independent design/code review passed before each scientific execution. The
final related regression passed 282 tests: 50 new geometry/time/actual-moving
tests and 232 related tests. It includes absolute paired-stock, common-Pa
offset, ALE R, internal/cap Pa energy and full KE face deletion controls;
raw KE substitution after valid physical faces; frame and changing-mass
omissions; invalid numerical scales; unsupported a1/shear/nonflat/nonaffine
or limiter-active states; and full rollback of every raw array, time/count,
receipt and cumulative GCL state/bound. Targeted Ruff and diff checks pass.

[The resource ledger](moving_raw_resources.json) retains all five serial
bounded invocations and their actual source commits. The TDD RED is retained
as tool-transcript-only without a fabricated logfile hash. The first
implementation run passed 44 and failed four witness checks because an
independent eta regrouping changed its last bits; that failure remains bound
to its original commit. Stable oracle eta and the frozen moving-top geometry
operation bound fixed it, while internal cuts retain strict equality. No
numerical threshold was enlarged after results. The later 49-test run,
282-test regression and source-bound witness all passed.

Total bounded wall time is 127.390 s, including both failures. Maximum owned
process-tree private memory is 78,422,016 bytes. The final scalar witness took
12.266 s, with 41,238,528-byte sampled interpreter RSS and 68,096,000-byte
owned-tree private memory. Each invocation used one CPU, hard 180 s and 4 GiB
limits. The existing bounded runner hash is unchanged from merged base7223.
No real archive step, GPU, production integration or alternate model was run.
Timings measure reproduction cost only.

Reproduce from the clean scientific commit in a local Python 3.12.14 venv
using `research/experiments/material_top_band/affine_requirements.lock`.
Create the ignored output directory before running the bounded witness:

```powershell
.venv\Scripts\python.exe scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.moving_raw_evidence --output logs/moving_raw/scientific_receipt.json
```

The related regression command is:

```powershell
.venv\Scripts\python.exe scripts/run_bounded_research_tests.py tests/research/contracts/test_raw_mean_geometry.py tests/research/contracts/test_rational_time_integral.py tests/research/contracts/test_moving_raw_characteristic.py tests/research/contracts/test_fixed_eta_raw.py tests/research/contracts/test_slope_dual_stock.py tests/research/contracts/test_affine_physical_pressure.py tests/research/contracts/test_pressure_force_geometry.py tests/research/contracts/test_inventory_pressure.py tests/research/contracts/test_material_real_geometry.py -q
```

The numerical modules/tests are portable; the resource launcher uses Windows
Job Objects. Other platforms need equivalent external limits. A later clean
docs-only head records its own commit on reproduction, while identical
scientific file hashes preserve the numerical source. Published receipts
contain manufactured scalar evidence and relative hashes, without private
arrays, local paths or credentials.

### Remaining scope

Original global momentum CV, general mean/inverse reconstruction, nonuniform
shear, horizontal density-gradient a1, nonflat/coast geometry, original moving
top-three band and predict/12fast/replay remain unsupported. Legal mixing,
biharmonic, FCT/filter, wind, bulkheat, rotation and drag adapters are absent.
The 8x4x6 end-to-end original task, mature-mode parity, generic second-order
fixed-endpoint convergence and equal-error speed qualification remain open.
Default production remains unchanged. No merge or license change is made.

### 2026-10-02 UTC: provenance counting correction

The witness has 256 source labels resolving to 216 distinct actual files;
legacy `src/` labels and reserved `checkout/src/` labels are logical archive
aliases, not literal checkout paths. The earlier phrase "256 actual source
files" should be read as 256 source hashes. Using the repository's canonical
`verify_current_source_hashes` resolver verifies all 256 with zero mismatches.
All published source hashes remain unchanged; this corrects counting and path
interpretation, not numerical evidence.

### 2026-10-02 23:44 UTC: exact execution-byte boundary

The archived scientific receipt hashes the actual Windows execution checkout
bytes, not Git's normalized blob bytes. Git reported a clean working tree with
`core.autocrlf=input`, but the protocol had been written with CRLF before its
frozen commit. Git normalized that text to LF when storing it. The scientific
commit and the published head retain the same protocol Git blob
`07e0f43cdfc04e3d282ae05907c5b182735cee78`; no scientific or protocol content
drift occurred.

The two independently verified SHA256 values for
`docs/moving_raw_protocol.json` are:

| Representation | Byte length | Line endings | SHA256 |
|---|---:|---|---|
| Actual Windows execution checkout, retained in the original receipt | 10672 | 166 CRLF | `e51975d35ccd14f777ebec5dba5d56db2d8d7b29fc07dac7f030995b9387f95d` |
| Scientific Git blob and GitHub raw content | 10506 | 166 LF | `f76bfd745cf52fac88812a6edff91d688082f8e70ed10ea87e051de5fdabf369` |

Converting the Git blob's LF to CRLF in memory produces exactly the retained
execution bytes and their recorded SHA256; the decoded JSON objects are also
equal. This diagnosis did not rewrite the protocol, alter the original
receipt, or rerun numerical work. The earlier zero-mismatch verification
describes the retained Windows execution checkout specifically. The earlier
byte-identical published-copy statement describes its local working-tree copy;
Git also normalizes text artifact line endings for the public repository.

Strict hash verification of the archived receipt will correctly fail for this
one label in a fresh LF checkout, including Linux and a Windows checkout using
Git's LF content. A clean Git status alone does not establish byte equality.
The numerical modules and tests remain portable, but that does not imply
platform-independent identity of archived source digests. A fresh execution
must record its own actual checkout-byte hashes and commit. It must not replace
the archived receipt or present new hashes as hashes of the old execution.
The original receipt, frozen protocol, repository-wide attributes and strict
hash checks remain unchanged.

### 2026-10-02 23:50 UTC: expanding-face majorant review correction

Independent review found that the original input-area factor used the initial
top-segment height plus input geometry perturbation, omitting canonical top
expansion for negative alpha. For alpha=-0.08 and dt=0.02 the smallest top
segment grows from 0.0266667 to 0.0311538 m. Its initial height is not the
required time-interval maximum. This finding limits the earlier claim that
the archived negative-case input majorants were fully proved. It does not
establish that an observed residual exceeded the overall old budget.

The original scientific receipt and resource ledger remain historical records
bound to their original source. The corrected factor will use the signed-q
analytic supremum H*max(0,-q/(1+q)) only for moving-top segments, plus input
geometry uncertainty and separately justified operation rounding. A new
frozen expansion protocol and independent tests cover negative alpha near
the legal duration/ratio limits, positive/zero alpha and fixed interior faces.
General positive-surface domains also require the PE-content z bound to include
the largest canonical eta. Corrected numerical evidence will have its own
precise source and new receipt, preserving the archived record.
