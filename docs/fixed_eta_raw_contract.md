# Manufactured fixed-eta raw pressure slice

## 2026-10-02: feasibility and contract before execution

Start from merged main `9ed44b18fa6af4edb18e20bcccd0d1b6e8dacadd`.
For fixed h, raw momentum r=D*u and physical projected momentum mu=W*u give
the unique same-velocity map R=W*D^-1. W being positive does not make R
conservative. The existing sloped fixture has incompatible raw and physical
total impulse, IS and PE. One cannot append conservation constraints to this
unique inverse and then repair its residual. If velocity is allowed to change,
total impulse/energy constraints alone leave many degrees of freedom; that is
not preservation of the original physical field.

Flat eta gives W*ones=D*ones and hence algebraic global conservation of R and
its inverse. This does not identify arbitrary endpoint coefficients as raw CV
means. A general manufactured half-prism mean basis would require a new Q/Psi
contract, and moving geometry would also require Qdot/Psidot. Neither is part
of this round. The accepted subspace uses uniform actual velocity at all raw
layers and both columns, where means and coefficients coincide.

This round defines an explicit manufactured raw CV patch: adjacent half-prisms
x=[0,d/2] and [d/2,d] of length L, each with its actual fourteen horizontal
layers, common flat eta and bottom. It does not claim these are the original
global solver's momentum control volumes. T=Tref and common stable affine
rho_prime(z) permit exact P1 TS face integration. Open side fluxes are nonzero;
surface, bottom and fixed internal horizontal interfaces are impermeable.
Linear external pressure provides a nonzero, depth-independent pressure force.

Every raw face segment has one pressure integral and one integrated water/TS/M
flux. Common segments use the union of physical depths and are consumed by both
neighbors with opposite signs. Outer faces independently close every raw row.
The uniform pressure force lies in the compatible subspace:
Fraw=-(P/rho0)*D*ones and Fmu=-(P/rho0)*W*ones=R*Fraw. Full-vector equality is
tested, not only total impulse or pressure power. A finite research pressure
step updates actual raw Mu, verifies all finite-time ledgers and commits the
complete copied ColumnStocks once. Rejection rolls back the whole state,
time, accepted count and receipt. Production qualification remains false.

For actual before/after momenta with changing symmetric mass matrices, the
required identity is

```text
DeltaK = u_mid^T DeltaM - 0.5*u_after^T DeltaMass*u_before.
raw_r_dot = D_dot*u + D*W^-1*(mu_dot - W_dot*u).
```

The accepted slice has fixed mass. Moving snapshots are independent obstruction
and algebra tests, not accepted manufactured evolution. Moving-alpha weighted
physical flux requires a separately defined local raw quadrature-storage
transfer; global conservation of the inverse cannot replace that raw row proof.
The changing raw-minus-physical KE metric difference is recorded separately,
never called physical loss. Alpha, horizontal density gradients, shear and
sloped eta remain rejected. Original global CV identification, moving geometry,
general transport, fast/replay, forcing/mixing adapters and industrial/equal-error
speed qualification remain open.
`n### Pre-execution time-flux and transverse-boundary clarification`n`nUse exact J1, J2 and J3 as frozen in the JSON, not Q times midpoint u for Mu. The sign-crossing control has zero water flux and positive transported Mu. The slice is xz extruded with periodic y; V has no unreported outer-wall source. All face fluxes are computed and consumed before checking the zero row divergence.

## 2026-10-02: readable time-flux clarification and absolute face gates

The preceding pre-execution clarification contains literal newline escape
characters from serialization. Its mathematical meaning is restated here;
the historical prefix is preserved.

Use the frozen exact J1, J2 and J3 time integrals. Mu flux is rho0*L*dz*J2,
including a sign crossing with J1=0 and J2>0. The slice is xz extruded with
periodic y, so opposing transverse transports and pressure forces cancel.

The actual left and right inventory P1 hydrostatic pressures are integrated
with two-point Gauss; the common pressure is their arithmetic mean, once.
Actual pressure integrals, water/TS/M transport and KE/gravitational PE flux
at each shared and each outer face are independently bound before any raw
commit using seven-point time/depth quadrature. This gate reads actual raw
P1 fields and integrates overlying raw layers, and does not call the candidate
pressure primitive or candidate polynomial time moments. Each raw row consumes
its separate signed outer and common fluxes, including KE/PE diagnostic fluxes.
The frozen common affine thermodynamic field is qualified against every actual
raw stock; roundoff in reconstructing that field is covered only by primitive
EOS operation bounds. No residual is fitted or repaired.

Common flux errors that preserve zero divergence are explicitly rejected:
shared and both outer stock fluxes set to zero or P0, all three pressure values
given a common offset, and shared/outer KE or PE jointly erased. The entire
state, time, count and last receipt remain unchanged on these failures.
KE/PE flux diagnostics do not add a new prognostic energy stock. The accepted
raw h/IT/IS/Mu/Mv state remains the single authority.

The accepted family also requires inactive actual TS and density limiters. Actual P1 T/S endpoints must bind the common affine EOS field; the independent face gate integrates the actual auxiliary T/S reconstruction separately from direct EOS density P1.

## 2026-10-02: measured fixed-eta raw commits

The implemented pilot accepts actual raw `ColumnStocks` on the new manufactured
two-half-prism patch, with 14 layers per half-prism. This is a restricted finite
pressure step, not a generic raw transport integrator. The independently reviewed
scientific source is `713a893d4e84b589eb5744ec3ce4a71d77d80fe6`, based on merged
main `9ed44b18fa6af4edb18e20bcccd0d1b6e8dacadd`. The pre-execution protocol commits
are `95e850989e5c3c2df1f61436e24daf9b89df5a84` and
`e6fd14818a73097dd59fd7196867a13d9d0f8dbe`.

Each positive/negative pressure case actually commits two successive steps,
0.01 and 0.02 s. The second step rebinds the returned raw stocks. Geometry,
thermodynamic stocks and transverse momentum remain byte-identical, while all
28 raw Mu rows, including deep rows, change. All five authority fields are
prepared on one copy and committed once per accepted step. Independent local
raw water/TS/M impulse ledgers, face pressure, inverse backward error, raw and
physical KE, unchanged PE, and consumed KE/PE flux ledgers pass before commit.

| Positive case | 0.01 s step | following 0.02 s step |
|---|---:|---:|
| Actual raw total x impulse change, kg m/s | -0.5040000000000036 | -1.0080000000000071 |
| Maximum actual deep Mu change, kg/(m s) | 0.0223999999999958 | 0.0447999999999933 |
| Actual raw KE change, J | -0.0151052487804879 | -0.0301219902439023 |
| Independent pressure boundary work, J | -0.0151052487804879 | -0.0301219902439026 |
| Actual PE change, J | 0 | 0 |
| Maximum shared water transport, m3 | 0.000123275885814303 | 0.000245829451985163 |
| Maximum shared IS transport | 0.00458718541085338 | 0.00914749278219004 |

Each case resolves 27 shared physical-depth segments. The negative case has the
opposite impulse sign. The zero-pressure case retains actual raw stocks and
nonzero verified face transport. In the sign-crossing case, maximum shared water
transport is 5.5744e-23 m3 while transported Mu reaches 9.63093e-9 kg m/s:
J2 remains positive when J1 is zero. No division by Q or false zero momentum
transport is used. Maximum inverse residual in actual total raw momentum is
1.4210854715202004e-14 kg m/s.

[The scalar evidence](fixed_eta_raw_evidence.json) records four cases, six
accepted manufactured raw steps, 32 case gates and 5928 comparisons. The largest
residual-to-local-bound ratio is 0.0016224057112475569. These are the frozen
512-eps primitive-operation roundoff gates; pressure difference cancellation
uses its own actual face-operation bounds. There is no physical truncation or
generic second-order claim. The receipt contains 248 portable source hashes,
Python/package versions and the clean scientific source commit. Later docs-only
publication commits do not change the scientific witness source.

The moving snapshots remain unaccepted. They expose local raw-minus-physical
storage change 0.0021150701988617016 kg m/s and a changing raw-minus-physical KE
gap -4.407614762769185e-5 J. Omitting Wdot or Rdot yields maximum raw derivative
defects 1.776368379499018 and 0.743168379499018 respectively. The separately
tested changing-mass KE identity has nonzero mass work for both actual D and W;
using a fixed-mass formula fails. The sloped same-velocity inverse still has
raw-vs-physical Mu gap 4.10000000000008 kg m/s, IS gap 50/779 and PE gap
629.1704812499999 J. These defects are reported as representation obstructions,
not physical losses.

### Validation, resources and reproduction

Independent code/math review passed before the scientific commit. The final
six-file related regression passed 229 tests in 25.42 pytest seconds; the final
three added receipt-scale controls passed separately in 0.22 s (41 deselected).
Together they cover the current 44 new-slice tests and 188 related tests.
Targeted Ruff and `git diff --check` passed. Full local production suite was not
run in this round; existing push/PR CI runs it on the exact published head.

[The resource ledger](fixed_eta_raw_resources.json) retains all nine serial
bounded numerical/test invocations, including the expected missing-module
collection failure, the implementation list-abs failure, and an incorrect
related-test filename collection failure. Total bounded wall time is 77.624 s.
Peak owned process-tree private memory is 75,214,848 bytes. The clean scalar
witness took 4.531 s, with 40,685,568-byte peak sampled interpreter RSS and
68,268,032-byte peak process-tree private memory. Every invocation used one CPU,
a 180 s hard wall and a 4 GiB process-tree memory cap. No real archive step,
large production integration, GPU or alternate model was run. These resource
figures are reproduction costs, not equal-error speed evidence.

From a clean checkout of the scientific commit, use a project-local Python
3.12.14 environment and `research/experiments/material_top_band/affine_requirements.lock`.
On Windows, the bounded witness command is:

```powershell
.venv\Scripts\python.exe scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.fixed_eta_raw_evidence --output logs/fixed_eta_raw/scientific_receipt.json
```

Create the ignored output directory first. The targeted test command is:

```powershell
.venv\Scripts\python.exe scripts/run_bounded_research_tests.py tests/research/contracts/test_fixed_eta_raw.py tests/research/contracts/test_slope_dual_stock.py tests/research/contracts/test_affine_physical_pressure.py tests/research/contracts/test_pressure_force_geometry.py tests/research/contracts/test_inventory_pressure.py tests/research/contracts/test_material_real_geometry.py -q
```

The pure scientific module and pytest tests are portable; the bounded launcher
uses Windows Job Objects. Other platforms must provide equivalent external
single-CPU, memory and wall limits. Reproduction on a later clean docs-only head
will record that later commit; numerical gates remain unchanged if the
scientific source hashes match. Raw source arrays, local private paths and
credentials are absent from the published receipts.

### Remaining qualification boundary

`qualification_passed`, `production_force_consumption_qualified` and
`original_global_CV_identified` remain false; moving accepted steps are zero.
Still missing are the original global momentum CV and its mean-preserving
Q/Psi reconstruction, general conservative inverse/local transport adapter,
moving raw metric/storage transfer, nonuniform shear and horizontal density
pressure gradients, and legal P0/FCT/filter/mixing/biharmonic/wind/heat/rotation/
drag adapters. The original 8x4x6 moving top-band predict/12fast/replay task is
not fulfilled by this restricted pilot. No full-stage, generic time-order,
industrial-quality, mature-model parity or equal-error speed claim is made.
Default production code and historical qualification records are unchanged.

Resource lineage: only the scientific_receipt invocation captures a clean source commit (713a893d4e84b589eb5744ec3ce4a71d77d80fe6). The other eight invocations are development runs without per-run source capture and are not retrospectively bound to that final commit. The resource ledger identifies the bounded runner with its portable relative path, SHA256 and scientific commit, and verifies that it is unchanged from the merged base.
