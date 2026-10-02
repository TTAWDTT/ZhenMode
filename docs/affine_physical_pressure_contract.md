# Affine fixed-physical pressure continuation

## 2026-10-02: protocol before numerical execution

This independent instantaneous probe starts at main
`87b956f388f6a3e13366184778a6aea010c0bdfd`. PR21's real force failure,
PR22's constant-density sigma result and all historical evidence are retained.
The new protocol fixes cases and bounds before execution. Qualification is
false for production, original fourteen-slot momentum consumption and real353.

Authority is copied h/IT/IS/Mu/Mv with the historical EOS. Pressure uses density
means in the exact stable affine family, never separate tracer limiting.
Affine secant ties permit only affine-preserving tangents. General P1,
endpoint clipping, stepped bottoms, dry/periodic patches and crossing reject.

The dual is the full physical trapezoid between two endpoints, extruded by L.
Its initial partition includes both irregular fourteen-layer columns. Interior
and virtual cut endpoints remain fixed physical coordinates for the tangent;
only the actual free surface moves. Auxiliary endpoint areas are dL/2. They
are not the original global momentum control volumes.

Every endpoint-layer velocity basis receives weighted four-edge traction plus
the volume pressure times basis-divergence term. Independent volume pressure
gradient quadrature and epsilon strip-load geometry perturbations verify every
basis. The pressure field is frozen during load perturbations and analytically
extended; it is not re-equilibrated on deformed geometry. Discontinuous shear
bases give strip-load variations, not a smooth global material deformation or
a general inventory-PE gradient. Pressure cannot depend on current velocity,
Q or an observed energy residual.

The manufactured material tangent has u=U+alpha*x, w=-alpha*(z-b). Density and
free-surface directions follow the continuum transport equations. Exact P1
physical ALE flux is new diagnostic algebra, not PR22's P0 upwind transport.
All cuts and open-side fluxes remain explicit. Internal virtual fluxes are
checked locally before cancellation. No unexplained residual is called loss.

PE contains rho0 free PE and anomaly gravity moment once. Physical volume
B(y)/Bdot and endpoint-stock PE are evaluated separately. Flat eta and the
restricted affine tangent give an inventory positive control. Unequal eta
gives a prederived endpoint-versus-volume PE gap and refusal. External pressure
is the declared linear lift: endpoint compensation does not invent a quadratic
zero-pressure-gradient state. Outward open-side pressure plus gz*rho flux and
top external-pressure work are counted once.

One-sided raw stock perturbations independently reconstruct scalar density P1
and integrate layer PE, including ds*h^3/12+s*h^2*hdot/4. Their truncation
majorant is separate from arithmetic roundoff. Scratch fixed-mass impulses use
real M_before/M_after and midpoint velocity. Their KE is the nodal lumped
metric; the interpolated volume KE difference is reported separately.

The new probe returns no accepted state and executes no Euler/full step. Full
current-mass KE, general P1/limiter derivatives, crossing, bottom steps,
predict/12fast/replay, real353 and MOM6/equal-error-speed adapters remain open.
Numerical checks are serial, one CPU, 180 seconds and 4 GiB per invocation.

## 2026-10-02 16:18 UTC: reviewed implementation and measured scalar gates

The protocol was committed before numerical execution at
`e5ade7aeeb8d1be8a4636ef033de62db0ef9e479`. Reviewed code is
`7fc3c6404fbc4edd32119bcad2434f5d94278109`. The independent reviewer confirmed the
physical contour, every basis load, full PE chain, finite-difference majorant,
local ALE content directions and provenance checks. Review was read-only and
did not launch duplicate numerical jobs.

The generated [scalar receipt](affine_physical_pressure_evidence.json) records
six cases, 27 physical strips and 28 endpoint-layer velocity bases per case.
All basis forces have independent seven-point physical-volume references and
three frozen geometric load perturbations. The 241 actual source labels were
also independently checked with `verify_current_source_hashes`; labels include
the real direct-file launchers, canonical modules, compatibility bridges,
test support, all material-top-band research modules, the new test, protocol
and direct probe dependency lock. No private arrays, machine paths or data
archives appear in either public receipt.

| Quantity | Flat positive manufactured tangent | Unequal-eta counterexample |
| --- | ---: | ---: |
| Pressure work, W | -2.572783920000082 | -882.9426959850001 |
| Independent physical PE direction, W | 136.474648128 | -60.084736316999994 |
| Explicit open-boundary power, W | 133.90186420800006 | -943.027432302 |
| Endpoint minus physical PE, J | 1.1368683772161603e-13 | 629.17048125 |
| Endpoint minus physical PE direction, W | 0 | -20.151052087500005 |
| Endpoint inventory pairing | Passed in restricted flat family | Refused |

Pressure work plus physical PE direction closes the declared boundary power.
This is an open dual; boundary pressure and advected PE remain explicit.
The unequal-eta gaps match the independent analytic B/Bdot discrepancy, including
the zero-flow refusal tested separately. A nonzero boundary lift is not treated
as a globally compensated zero-PG equilibrium.

Identical affine rho(z), eta and bottom with different irregular fourteen-layer
partitions has exactly zero computed basis force. The constant-density and
nonlinear-separate-TS static controls also have zero computed force; transport
of nonlinear T is explicitly outside this affine-T family. Across six cases,
the maximum basis discrepancy is `2.7284841053187847e-12 N`. Its bound is from
absolute local pressure operations before cancellation, not that residual.

The flat positive tangent has band relative-downward speed `0.02112 m/s`,
nonzero deep IS direction `0.004645050537481273` and deep Mu direction
`0.1170770313163047` in their actual inventory units per second. Every raw layer
direction agrees with an independent integral. All virtual cuts are checked
against the declared physical flow; each strip's four stocks separately close
volume specific-field direction plus actual-top shape against all four ALE
edge fluxes before internal cancellation. Removing an active internal edge
gives local residual `2.592512`, versus its frozen bound
`6.175851012812929e-13`. Removing the full slope-PE chain gives
`0.0041317121001895885 W`, independently separated from roundoff.

Scratch fixed-mass impulses implicitly last one second, using actual
Mu_before/Mu_after and midpoint velocity. The flat positive KE change is
`-2.3008500891505714 J`; independent pressure work differs by
`8.126832540256146e-14 J` against `1.4760962585569266e-12 J`. The endpoint metric
is auxiliary nodal lumped mass. Its initial KE exceeds interpolated physical
volume KE by `0.18368 J`; these metrics are not interchangeable. The scratch
impulse does not return an accepted state or implement a physical time step.

Raw-stock one-sided PE differences have physical perturbation error. At
epsilon `2^-8 s`, the flat case discrepancy is `0.029600615090998872 W`,
with an analytic truncation bound `0.029610393550346333 W` and total bound
`0.029610443673916253 W`. Halving epsilon reduces this directional-difference
error; this is not time-integrator order. Likewise the unequal-eta geometric
central differences include their explicit cubic remainder. Neither is
claimed to satisfy an absolute roundoff-only `1e-12` criterion. No fixed-endpoint
time evolution or second-order time-integrator claim was evaluated.

The [measured resource ledger](affine_physical_pressure_resources.json) retains
successful and failed bounded invocations. Each used an actual Windows Job
Object with one CPU, a 180-second wall limit and 4 GiB process-tree private-memory
limit. Final new suite: **36 passed**, 2.68 s pytest / 3.328 s job, peak
interpreter RSS 53,993,472 bytes and tree private memory 36,990,976 bytes.
Adjacent regression: **207 passed** (171 retained contracts plus 36 new),
118.76 s pytest / 119.391 s job, RSS 60,518,400 bytes and private 42,090,496 bytes.
The clean-source scalar CLI took 1.922 s, RSS 38,633,472 bytes and private
66,121,728 bytes including provenance subprocesses. No resource stop occurred.
Focused and repository Ruff both pass.

Failures were preserved and corrected without changing frozen bounds. Initial
TDD collection failed for the missing new module. Actual-momentum controls
found three missing refusals; zero-flow/virtual-crossing controls found two.
Review controls found missing common-slope/common-slope-direction, constant-T,
NaN and raw-ownership gates, plus four missing local-ledger checks. The first
raw-ownership implementation exposed an exact actual-top coordinate mismatch
in 25 tests: the candidate now uses the original eta at actual top, rather than
round-tripping it through bottom plus height. The zero-area control also failed
before its fix. Witness TDD failed for the absent CLI before implementation.
Earlier passing subsets are superseded by the reviewed 36-test and 207-test runs.
An initial copy-path setup attempt selected zero of 33 tests (exit 5); it was
corrected before the recorded finite-area red test and is not a scientific pass.

From the repository root with a project-local Python 3.12 environment, install
existing project dependencies with the probe version constraints:

```text
python -m pip install -e ".[dev]" -c research/experiments/material_top_band/affine_requirements.lock
python -m pytest tests/research/contracts/test_affine_physical_pressure.py -q
```

For the measured Windows bound, use a clean committed checkout (remove or move
an existing output receipt only within your own checkout before repeating):

```text
.venv\Scripts\python.exe scripts\run_bounded_research_tests.py --module research.experiments.material_top_band.affine_physical_evidence --output affine_witness.json
```

`affine_requirements.lock` pins NumPy 2.5.3, pytest 9.1.1 and Ruff 0.16.8 for this
new direct probe; it does not replace or claim to fully pin the existing solver
dependency environment. `affine_witness.json` contains public scalar summaries
and hashes, not original data. The CLI requires a clean committed source tree
and aborts on any failed frozen gate. Generated receipts remain bound to the
actual code commit above; later documentation commits do not rewrite that history.

Production code, old evidence, general P1/limiter derivatives, original
fourteen-slot momentum CVs, variable-mass KE, moving full-column sigma,
crossing/bottom-step/dry support and complete predict/12fast/replay remain
unqualified. No accepted step, real353 failure step, real dt300/600, GPU,
MOM6 same-condition run or equal-error speed comparison occurred. These are
the remaining pressure/geometry-to-industrial continuation requirements.

### Resource lineage clarification

The resource ledger's final_code_commit identifies the reviewed implementation.
Only scalar_witness.log ran from that clean commit and has its own source
manifest. The other retained logs record uncommitted implementation iterations
before commit, including earlier failures; they are not rebound to final code.
The final green and adjacent regression used the reviewed source subsequently
committed, with no separate per-run source manifest. The reproduction output
filename above is newly created in a clean checkout; existing tracked receipts
should stay in place. Omitting --output prints the scalar witness to stdout.
