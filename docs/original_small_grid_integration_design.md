# Original small-grid handoff audit and integration gate

Source-audit baseline: `25258950905f9d1aa84509c4c99ebad9ef33ba2b`.
This is a stage-zero design freeze. No interface implementation, model run,
numerical acceptance, order or speed qualification is included.

## Native authority and earliest blocker

The original material restart family is
`FD_point_samples_material_top_mass_lumped_linear_momentum_v1`.
Its six fields are nodal u/v/T/S and eta/ice. The original
`material_thickness` and `_contents` use
h_tracer=h_ref+eta*e0 for tracer inventory. Momentum, layer-face transports,
barotropic averaging and the depth-uniform velocity lift still use h_ref.
The native restart stores u/v, not independent Mu/Mv; the reference momentum
below is a derived diagnostic quantity, not a new restart authority.

Reference momentum per horizontal area is rho0*h_ref*u. Treating it as
moving momentum and decoding it with h_tracer changes the velocity to
h_ref*u/h_tracer. For h_ref0=2.5 m and eta=-0.4 m the factor is exactly
25/21. This is an algebraic counterexample, not a measured model result.

Let B0 map original velocity DOFs to reference volume outflow per CV,
with units m3/s, and D0=diag(rho0*A*h_ref). For the original reference dynamic
pressure p and original acceleration a_P, the reference pairing is
D0*a_P=B0.T*p. Substituting D_h alone leaves force defect
(D_h-D0)*a_P=rho0*A*eta*e0*a_P. It does not vanish with roundoff.
Closed-y normal velocity constraints must use the same DOF projector in B0
and the force; unconstrained PGF samples are recorded separately.

Algebraically rescaling the old gradient by D_h^{-1}D0 repairs only that
fixed-flux pairing. It is not a physical moving operator. A real adapter
must define the moving face map B_h, its pressure transpose, current mass
and finite variable-mass work together.

## Geometry, pressure and clock differences

Native material geometry moves only the first dual interval. The requested
top-three band changes all three widths with
f=[1/9,1/3,5/9] for reference widths [2.5,7.5,12.5] m. Native and target
thickness rates differ by [8/9,-1/3,-5/9]*eta_dot. Their column sum is zero;
column closure cannot establish the local top-band GCL or ALE interface fluxes.

Actual barotropic averaging differs from reference averaging by
eta*(u_top-u_bar_ref)/(H_ref+eta), with the analogous v relation.
Current weights and the lifting/projection frame must change with momentum
mass, not just with tracer division.

The original hydrostatic function computes rho0*g*eta plus fixed-node
trapezoidal density-anomaly integration. It omits the moving-cap anomaly
integral g*integral_0^eta rho_prime(z) dz. For negative eta the z0=0 sample
may be above the physical cap. This Pa is labelled original linear
reference dynamic pressure, not qualified physical moving-domain pressure.

The original nonlinear predictor updates provisional tracers and momentum.
The twelve fast calls retain its shear and density forcing, use fixed
reference layer/column transports and return their macro-time mean. The
layer matcher explicitly adds a depth-weighted correction to match that mean.
Accepted material tracer replay starts from the retained first slow endpoint,
uses those matched faces and a linear eta path, and does not consume each
fast-state tracer history. Its final T/S do not retroactively update the
lagged fast density forcing. A stage named accepted_tracer_replay is an
intermediate state; final original result.valid must be recorded separately.
These stages are not a joint accepted h/IT/IS/Mu/Mv path.

The new Q/Psi/raw-step authority uses means on two half-prisms, a different
domain, DOF definition and metric. It cannot rename original point samples
or their reference stocks. On a same-layer pair, Q has rows [.75,.25] and
[.25,.75]. Interpreting native endpoint samples [0,1] as its mean DOFs
reconstructs endpoints [-.5,1.5]. Both have midpoint .5, so a single
mid-face check can miss the representation error. No such mapping is
performed or proposed as an identity adapter.

## Smallest implementation milestone

1. Import an immutable, validated view of the actual original state, grid
   and params on an 8x4x6 original factory case. Expose h_ref, h_tracer,
   original tracer stocks, reference momentum and metric, original EOS/Pa/PGF
   and native faces, with accurate authority labels. No raw-mean constructor,
   remap, fitting or second restart authority.
2. Connect the existing ten-stage observer and add an optional eager observer
   inside the actual original barotropic loop. Record its twelve calls,
   before/after mean states, actual first/mean faces and filter change.
   Default None must preserve original numerical results exactly. Python
   observation explicitly requires use_scan=False; requesting it with
   use_scan=True rejects before stages instead of silently editing a parameter.
   The original generic fast clock and advance function remain in control.
3. Add a scratch pressure-only probe using derived native reference momentum,
   fixed D0 and frozen original Pa. Set F0=B0.T*p and apply the actual
   before/after kick over 0.01 s. With u_mid=(u_before+u_after)/2, check
   DeltaK=dt*u_mid.T*F0=dt*p.T*B0*u_mid by independent assembly, and require
   nonzero impulse. The original wall projector is identical in both sides.
   This proves a reference discrete kick identity. It does not prove the
   combined rotation/wind/fast path, continuous time-integrated pressure
   work, moving physical Pa or total PE. Any later channel decomposition
   of an actual fast kick must consume that same actual midpoint and actual
   forcing; a separate pressure-only after-state cannot substitute for it.
4. Reject a joint moving-stock or raw-mean handoff with all six original
   fields byte-preserved. Executed diagnostic stages and original valid
   results remain distinct from accepted joint-stock steps, which stay zero.
   Late observer/operator failure also restores the complete entry snapshot.

This milestone exposes usable original operator and clock seams without
claiming the mathematical migration has already happened.

## Frozen tests and resource boundary

The companion `original_small_grid_integration_protocol.json` fixes one
original 8x4x6 all-wet flat-bottom factory case, nonzero eta/shear/density
contrast and active original diffusion, biharmonic, FCT/filter, wind,
bulk heat, rotation and drag. Convection is coefficient-enabled, with an
initial unstable density profile fixed by T increasing 0.2 C and S increasing
0.005 PSU per deeper node. The activity gate must observe a nonzero original
instability mask and nonzero applied mixing term; coefficients alone do not
establish activity. The existing material policy for
GM/Redi/ice/sponge is explicit; no coefficient is edited during preflight.

Proposed controls cover exact native field capture/restore, derived-stock
roundtrips under the primitive-operation roundoff bound and finite binding,
original EOS/reference Pa and independent native face assembly, reference
pressure transpose/fixed-mass work, default/observed identity, ten stages,
twelve actual fast calls and per-fast continuity, moving-mass/average and
top-three defects, forbidden half-prism reinterpretation and full rollback.
Identity bounds are frozen at 512 machine eps times absolute primitive
operands; omitted-operator controls must exceed their bound by a factor 100.
No observed residual sets a threshold. Physical truncation is separate.

Implementation source must be reviewed and cleanly committed before any
numerical invocation. Each invocation is serial, one CPU, hard180s,4GiB.
New source-bound receipts preserve historical frozen evidence. No original
user checkout, production default, license, real353 or dt300/600 is changed.

## Next mathematical decision

Choose a separately declared representation compatible with the original
grid: retain point DOFs with a geometry-derived consistent metric and face
basis, or introduce whole-native-CV means with explicit initialization and
restart transfer. The local half-prism Q cannot be an identity transfer for
either route.

Then jointly adapt top-three physical geometry, B_h/pressure transpose,
current-mass averaging/projection, finite mass work and accepted fast stocks;
connect replay to that same path and adapt conservative diffusion,
biharmonic, limiter/filter and source impulses. Moving-cap pressure and
density feedback need independent pressure/PE verification. Mature-mode
parity, fixed-endpoint order and equal-error speed remain downstream gates.

## Pinned source seams

All links below use the source-audit baseline, not a moving branch.

- [Material driver and restart authority](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/candidates/material/solver.py): material_thickness, _contents and _material_step.
- [Native geometry](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/geometry/columns.py) and [factory](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/fd/factory.py): nodal dual thickness, reference column weights and declared parameters.
- [Native transport](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/fd/transport.py) and [horizontal operators](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/fd/horizontal.py): native face map, explicit matcher and pressure adjoint.
- [Hydrostatic pressure](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/fd/pressure.py) and [original fast loop](https://github.com/TTAWDTT/ZhenMode/blob/25258950905f9d1aa84509c4c99ebad9ef33ba2b/src/ocean_solver/fd/barotropic.py): reference Pa, generic fast calls and actual mean faces.

## Design review scope

An independent inspector read the pinned original sources and passed the
mathematical and interface design gate. This review checked the counterexample
signs/units and parameter ownership, not implementation or numerical results.
Physical coefficients including T_ref/S_ref and bottom_friction belong to
PhysicsConfig; solver options belong to make_solver_global; the material
transport/momentum schemes belong to the material builder. Only nu_nsub is
an explicit factory subcycle keyword. adv_nsub/conv_nsub remain original
factory-derived fields (both 1 for this frozen case), and material linear
counts remain driver-derived. No _replace override may forge those counts.

## Execution status

Source audit and independent mathematical review were completed. The selected
Windows execution and file-write tools stopped returning during local
environment preparation; its process completion could not be confirmed.
No numerical job was started and no interface-code write or test result
was confirmed. This stage-zero Draft therefore contains documentation only.
It must not be described as an implemented or passed original-mode adapter.

## 2026-10-03: execution reconnected before implementation

The existing isolated checkout was found at the audited baseline with no
source edits and no active Python/test process. Its local package installation
was completed. The design protocol below remains the pre-implementation frozen
snapshot; subsequent implementation and execution status belongs to a separate
append-only evidence section and receipt, not a rewrite of these initial flags.

## 2026-10-03: implemented original-interface milestone and bounded evidence

The original native interface is now implemented in
[original_native.py](../research/experiments/material_top_band/original_native.py).
It captures the actual six original fields, first-slot material tracer stocks,
reference momentum/mass, original EOS/reference Pa/PGF and native faces.
Immutable byte-backed snapshots retain the original authority. The two optional
observer seams record the actual original driver and its generic fast loop;
their default None path retains the old call shape and numerical operators.

The pre-implementation flags in the frozen protocol above are a historical
snapshot. This section and the new
[receipt](original_small_grid_integration_receipt.json) record the implemented
milestone. They do not promote the unfinished joint-moving protocol.

### Source and validation

The scientific source is
`fddda45b39fcdcf752bef756d4c52e90541a40f1`, descended from the audited
main `25258950905f9d1aa84509c4c99ebad9ef33ba2b`. The original factory case
and independent signed-face/work assembly have their own
[support owner](../tests/support/original_native.py); the capture script imports
that owner normally rather than loading a test file dynamically.

The independent Code/Design Gate passed before the clean implementation commit.
The bounded GREEN passed 47 native-interface and directly relevant architecture
controls in 53.98 s pytest time, 55.109 s supervisor time. They cover:

- True native state/metric/tracer/reference-momentum binding and independent
  linear EOS, reference Pa, volume-face transpose and fixed-mass work.
- Ten actual original stages and twelve calls of the original generic fast
  function, with default/observed original output byte identity.
- Moving-mass-only reinterpretation, top-three local geometry and half-prism
  identity counterexamples; joint handoff refusal.
- Masked/nonfinite/large-finite derived-overflow inputs, actual rebuilt-factory
  inconsistent spherical metrics, forged area/mass and invalid durations.
- Early/late full six-field rollback and preservation of a previous successful
  diagnostic receipt after a later failure.
- Independent impulse-work, native transport-work and per-fast continuity
  bad-ledger refusals, including zero-bound and nonfinite-bound cases.

A deliberately huge finite-u input emitted one expected NumPy overflow warning;
the derived finite guard rejected that input. No valid-case numerical warning
or bound stop occurred. All reported witness identities are actively checked
against the frozen dimensional primitive-operation bounds, with no unit floor.

### Actual nonzero original diagnostic endpoint

The original material driver returned valid=true. It retained predictor,
twelve fast calls, explicit native layer matcher and accepted tracer replay
in their original order. The retained audit entry state stayed byte-identical
in all six fields; the separately labelled original diagnostic endpoint moved.

| Quantity | Observed value |
| --- | ---: |
| eta maximum absolute change | 1.64628050636e-8 m |
| u maximum absolute change | 9.65141245691e-7 m/s |
| T maximum absolute change | 2.47579310653e-5 C |
| S maximum absolute change | 5.88460359552e-7 PSU |
| deep IT maximum absolute change | 8.16033486117e-6 C m |
| deep IS maximum absolute change | 2.04012380323e-7 PSU m |
| deep reference Mu maximum absolute change | 6.47160763975e-4 kg/(m s) |
| per-fast independent continuity residual maximum | 2.77197166123e-17 m |
| per-fast continuity / primitive bound maximum | 3.05158695332e-4 |

The scratch reference-pressure kick had a nonzero maximum actual stock impulse
of 7.7875182625e7 kg m/s. Its independently assembled transpose residual / bound
maximum was 9.92844416878e-4. Actual before/after KE change was
2.437093244596e7 J; frozen-Pa force work was 2.437093226325e7 J. Their
0.182714369148 J difference is inside the 818.485379167 J roundoff envelope
derived from uncancelled before/after KE operands. It is not a physical loss,
PE balance, truncation estimate or relative-pressure-work accuracy claim.
Actual impulse work and native C(midpoint-u) transport work were separately
checked against the independent force work; their maximum bound ratios were
9.26031132917e-5 and 5.48304260641e-5.

The first mathematical blocker remains quantitative: at eta=-0.4 m and
reference top width 2.5 m, interpreting the original reference momentum as
moving momentum changes decoded u by factor 25/21. Native e0 and target
top-three f differ locally by [8/9,-1/3,-5/9]*eta_dot despite equal column sums.
The implementation exposes and refuses this incompatible handoff; it does
not repair moving momentum by changing a divisor.

### Reproduction and resource record

Use the project-local environment with Python 3.12.14, NumPy 2.5.3,
JAX/JAXlib 0.11.2, pytest 9.1.1 and Ruff 0.16.8. On Windows, explicitly bind
the checkout source paths when using the supervisor's underlying interpreter:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src') + [IO.Path]::PathSeparator + (Join-Path (Get-Location).Path 'src\compat')
.venv\Scripts\python.exe scripts\run_bounded_research_tests.py tests\research\contracts\test_original_native_integration.py tests\infrastructure\test_source_layout.py -q
.venv\Scripts\python.exe scripts\run_bounded_research_tests.py --module scripts.capture_original_native_contract --output docs\original_small_grid_integration_receipt.json
```

Numerical invocation requires a clean committed source tree. Commit or move any
locally produced artifact before another invocation. The new
[resource record](original_small_grid_integration_resources.json) includes
the initial environment import failure, corrected expected TDD RED, GREEN
and witness: total supervisor wall 75.515 s, peak owned private memory
638128128 bytes. Every invocation used one CPU, hard 180 s and 4 GiB; all
bound_stop flags were false. The unchanged bounded runner SHA256 is
844c81b1f8185d33bcb79872bd83213807ad0e1f708746ccee52f4a41e00ac03.

The witness binds 229 source labels to 189 actual files and separately records
actual executed byte hashes, Git blob hashes, lengths and line-ending counts.
This checkout's actual byte hashes all equal their committed Git blob hashes;
the equivalence was checked, not inferred from clean status. All resolved
parameters have scalar values or dtype/shape/hash descriptors, and the frozen
case settings and factory-derived counts are explicitly checked. Published
evidence contains generated-case scalar summaries and relative source labels,
not raw private input arrays, machine paths or raw local logs.

### Remaining migration gaps

Accepted joint moving-stock steps remain zero. Original point-sample/reference
momentum is still incompatible with the requested top-three moving-stock
authority and the half-prism raw-mean basis. A chosen original-grid DOF/basis,
its actual moving metric, common physical faces/pressure transpose and finite
variable-mass work must be derived together before a joint handoff can accept.

The original mean-face matcher still explicitly changes layer Q, predictor
density remains lagged through the fast clock, and tracer replay remains a
separate history. Current-mass conservative diffusion/biharmonic, shared limiter
and filter/source impulses, moving-cap Pa and total PE/ALE closure are not
qualified. The reported convection RHS entry preview is an activity diagnostic
only; the actual applied replay mixing ledger gate remains false. Physical
moving pressure, fixed-endpoint order >=1.9, mature-mode parity and equal-error
speed remain unqualified. Production defaults and the original user checkout
were not changed.

## 2026-10-03: final range-check correction and refreshed source binding

The final complete baseline-to-head whitespace check found one additional
terminal LF in each of the native audit, capture script and support module.
The earlier whitespace check covered only the uncommitted diff. Exactly those
three bytes were removed in scientific source
`f0d1cbc329967e659efc64edc149b803d193f4da`; independent inspection verified
that no other source byte changed. Ruff and the complete baseline-to-head
whitespace check now pass. This correction supersedes the current-source
binding above while preserving that earlier execution record.

The same 47 controls passed again in 53.18 s pytest / 54.172 s supervisor time,
with peak owned private memory 651431936 bytes. The same bounded witness
passed in 17.062 s with peak owned private memory 409767936 bytes. All
non-provenance receipt fields exactly equal the prior witness, including the
nonzero diagnostic endpoint, work identities and unqualified moving gates.
The refreshed receipt binds the corrected clean scientific source to all 229
labels / 189 actual files; actual bytes again equal the committed Git blobs.

The resource record retains all six invocations, including both initial
failures and both source bindings: total supervisor wall 146.749 s; maximum
owned private memory 651431936 bytes. Every numerical invocation remained
serial, one CPU, hard 180 s / 4 GiB with no bound stop. No new case or numerical
operator was added. The joint-moving and physical/order/speed gaps above
remain unchanged. The isolated branch's tracking ref was also repaired after
the first push; no original user checkout or persistent proxy setting changed.
