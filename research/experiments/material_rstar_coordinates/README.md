# Original-nodal r-star metric controls

This isolated experiment addresses the `material_top_v1` thin-top restriction:
the actual 1-degree diagnostic retained a roughly 747.5 m water column while
its top nodal weight became negative at eta below -2.5 m. It does not change
the production solver, material factory, horizontal grid or restart semantics.

## Contract

For reference nodal widths `h_ref` with total `H`, stretch by `s=1+eta/H`:
`h=s*h_ref`, positive-downward node depth `d=-eta+s*d_ref`, and relative downward
transport `G[k+1]=G[k]-D[k]-(h_ref[k]/H)*eta_rate`, with `eta_rate=-sum(D)`.
Negative widths remain visible and invalidate the geometry; they are not clipped.
Columns must be contiguous and retain at least two nodes for derivatives.

Hydrostatic pressure uses actual node spacings. At a horizontal face, the
physical pressure difference is the coordinate difference minus
`g*mean(rho_prime)*delta(d)`. Diffusion uses the analogous physical gradient
and **the adjoint of that same gradient**, including its vertical metric terms.
The closed-domain volume operator is `K=-B.T*W*B`, with positive face measures.
The independent NumPy oracle assembles Lagrange derivatives and dense matrices,
not JAX autodifferentiation or calls to the kernel's gradient builder.

## Reproduce

From the repository root, using the configured float64 JAX environment:

```shell
python -m pytest tests/test_rstar_metric_controls.py -q
python research/experiments/material_rstar_coordinates/audit.py --output results/rstar_audit_new
```

The audit output must not exist. It freezes harness/kernel/reference source
hashes, saves both dense counterexample matrices and witnesses, and reports
dimensional errors and gates. Its fixtures come from the accompanying test
suite. For the actual CUDA backend, set `JAX_PLATFORMS=cuda` before launch in
a supported environment; GPU availability is not inferred from CPU results.

Controls include geometry, original donor/minmod constant-content transport,
affine resting density, dry sentinels, zero-eta reference agreement, quadratic
pressure coordinate-MMS, independent energy and conservation matrices, and
smooth-coordinate JVP/finite-difference/VJP checks. The MMS compares against
the original horizontal metric applied to an analytic surface potential; it
is not a manufactured solution for the complete moving ocean dynamics.

## Explicitly retained limitations

- Merely correcting horizontal flux gradients but keeping the old coordinate
  divergence has a positive-work eigenvector in the registered fixture.
- The full negative adjoint is conservative and dissipative, **not necessarily
  monotone**: a saved nonnegative unit pulse has a negative tendency at a zero
  node. Energy qualification does not establish a tracer maximum principle.
- Smooth operator gradients do not resolve the previously observed hard
  convection branch failure or qualify long-trajectory gradients.
- There is no accepted full-step API, actual moving-face momentum/transport
  qualification, complete-step diffusion bounds qualification, ice coupling, new-state
  restart migration, actual 1-degree rerun or industrial comparison here.

Before integration, freeze the moving-face and stage-content contracts, choose
and test the diffusion bounds policy without clipping residuals, then prove
complete-step source/budget/pressure-work consistency and time convergence.
Only afterward run the unchanged real 1-degree rejection protocol. Earlier
failed actual runs stay failed; these controls do not relabel them as passes.

The [bounds-policy review](../../literature/rstar_diffusion_bounds_20260930.md)
records primary sources, the retained maximum-principle counterexample, and
the next independent matrix experiment before a full-step implementation.

## Sparse graph and prescribed moving-content stages

`sparse_diffusion.py` coalesces all contributions to each physical off-diagonal
edge **before** separating positive graph diffusion and negative-edge
antidiffusion. Fixed-index grouped gathers and disjoint writes replace the
dense matrix. Shared pair limiters constrain Euler stages by graph-neighbor
content bounds. Fixed-mass Heun retains either-stage refusal. The unlimited
negative-adjoint counterexample above remains valid; it is not retrospectively
converted into a maximum-principle pass.

`moving_content.py` combines the **original donor** tracer operator and its
spherical face contract with relative vertical flux and this bounded diffusion.
Each Euler stage uses its actual old and endpoint volume weights. Heun averages
content and the corresponding moving geometry, rather than dividing new content
by old thickness. Sources shift the local allowed bounds only by independently
deposited source content divided by endpoint volume. Dimensioned heat deposition
uses `Q*area/(rho0*cp)`; signed source and exchange residuals remain observable.
These are prescribed transports, not a solved momentum/free-surface coupling.
The volume weights are proportional to mass under the constant reference density.
On refusal, content and surface roll back; exchange/source fields are retained
**attempt diagnostics**, not accepted cumulative ledger entries. Consumers must
check `valid` before accumulating them. The original minmod/FCT configurations
are not qualified by these donor-only coupled controls and are not disabled in
production.

Run the frozen controls and archive their sources, runtime, XML and numeric report:

```shell
python research/experiments/material_rstar_coordinates/sparse_audit.py --output results/rstar_sparse_new
```

The destination must be new. Actual CUDA uses the same command with
`JAX_PLATFORMS=cuda` set before launch in the GPU environment. The report identifies
the actual backend and float64 setting, refuses source changes during execution,
and always sets `production_promotion=false` even if its 28 isolated gates pass.

Controls include independent NumPy face-by-face donor fluxes and dense physical
diffusion; two-node columns, disconnected basins, dry bytes within the same
executable; constant, pulse and signed fields; local/global bounds, inventory,
fixed-mass variance, affine-depth equilibrium, original CFL rollback, dimensional
local/source ledgers, and smooth coordinate derivatives. Fixed-geometry Heun is
compared to the independent dense matrix exponential. Prescribed transport-only
moving Heun is compared to an independently refined variable-mass RK4 reference.
Neither artificial time window is a completed ocean integration, and neither
establishes complete moving advection/diffusion/momentum time order or energy.

Before a material factory cutover, independently qualify moving horizontal-face
overlap and fast/slow transports, momentum/pressure work, complete-stage budgets
and coupled time/space errors; specify a new state/restart contract. Then run the
original actual 1-degree acceptance protocol without relaxing the existing
capacity, geometry, gradient or cross-process byte failures. This experiment
does not qualify ice, century stability, climate/forecast accuracy or industrial
performance, and does not modify the production entry point.

The actual archived negative-top states can be inspected without advancing or
reinterpreting their tracer content:

```shell
python research/experiments/material_rstar_coordinates/real_geometry_audit.py --output results/rstar_actual_geometry_new --build-topology
```

This CPU-only audit freezes the explicit original grid and the **separate**
256-capacity diagnostic negative-top states. It preserves their rejected status,
compares thickness/interfaces with independent NumPy geometry, and optionally
counts actual sparse graph storage. The first 128-capacity stop is not the same
state and still has positive thickness. There are zero accepted integration
steps here, no full-step compile, remapped heat/salt inventory or GPU peak-memory
qualification.

## Pressure work versus physical resting state

`pressure_work.py` evaluates the original chain-rule force, its actual-width
negative-adjoint weighting, and a reversible energy-adjoint candidate on the
**same original node layout**. Transport uses a declared mean-thickness coordinate
band and original spherical metrics; this is not qualified physical face overlap.
The energy experiment integrates a piecewise-linear nodal density representation
and a **constant tail** from the last wet node to the bed. Its mass matches the
nodal widths; its depth first moment is not simply width times node position.
Centered density transport isolates reversible work. Original donor-minus-centered
potential transfer is reported separately, not removed or called external heating.

```shell
python research/experiments/material_rstar_coordinates/pressure_audit.py --output results/rstar_pressure_new
python research/experiments/material_rstar_coordinates/pressure_witness_audit.py --input results/rstar_pressure_new --output results/rstar_pressure_witness_new.json
```

The first command intentionally exits nonzero when candidate qualification fails,
while still saving all sources, cases and an affine-stair array witness. The
second command independently recomputes that witness with NumPy only; successful
audit means the saved **failure** is verified, not that physics passed. It checks
hash/dtype/finiteness, numerical values and claimed flags; its scope is one witness,
not all cases or a trajectory. Actual CUDA uses the same before-launch backend
selection as above; CPU/GPU qualified-source hashes must match.

Observed gates reject every full interface candidate: the width-weighted force
closes pure free-surface/constant-density work and preserves the affine resting
point field, but general pressure work is not qualified. The energy-adjoint force
closes the specified reversible work but creates a force on the affine resting
stair field. Fixed-domain full-wet vertical refinement also fails its registered
second-order diagnostic. Correct helper/counterexample tests do not convert these
physical failures into passes. The dry-surface energy conjugate was separately
fixed to match the energy derivative and ignore unused finite/NaN land values.

The affine nodal samples, constant-tail continuous reconstruction and a globally
affine continuous profile are different physical representations. Their inventory
and potential gaps are reported, not hidden inside a residual correction. Resolve
the point/content/pressure reconstruction and bed policy before introducing a new
factory or state/restart contract. These results neither prove r-star/A-grid
impossible nor invalidate the original historical integration or accepted 2-degree
thermal-source budgets. No force here is promoted to production.

## Bounded representation localization

`representation_protocol.json` freezes a bed-policy/consistent-mass factorial
control. `representation_audit.py --output results/rstar_representation_new`
archives eight cases, matrices, content and source identity. Truncating the bed
changes the domain; replacing lumped content changes individual content semantics.
Neither makes the unchanged coordinate-band flux a consistent weak transport.
All tested combinations retain the work/rest conflict. Successful audit execution
does not qualify a physical candidate or accept an ocean step.

`nodal_mass.py` supplies batched tridiagonal hat-mass application and inversion
as a prerequisite, not a transport scheme. Positive content does not imply
positive recovered node values; the negative control is mandatory. Caller-side
geometry rejection remains necessary, and no factory imports this module.
Independent Gauss, dense-column, dry-sentinel and local AD controls live in
`tests/test_rstar_representation.py` and `tests/test_rstar_nodal_mass.py`.
The first cancelled-dot comparison failures and its explicitly revised absolute
summand-based64eps scale are retained separately; the original pressure/rest
gates are unchanged. Next coupling scope and non-promotion constraints are in
[the representation decision](representation_decision_zh.md).

## Joint physical weak transport and completed-bed contract

`weak_transport.py` integrates the same nodal hats across actual shared fluid
apertures, including relative vertical transport from partial face volume flux.
Its pressure is paired with this transport and its declared lumped velocity mass.
An independent NumPy five-point face quadrature checks the RHS. This is an
instantaneous centered, unbounded experiment, not a complete ocean factory.

`bed_completion.py` registers `bed_complete_nodal_v1`: preserve original physical
beds and old wet nodes, reuse the first dry slot for a bed sample where needed,
and reject a missing slot. Shape is preserved but unknowns and local wet topology
change. Old dry values are not valid new samples; initialization, restart migration
and dependent operators are not implemented. The old sparse graph is not qualified
for the new topology. This is not truncation or a hidden cell-average rename.

```shell
python research/experiments/material_rstar_coordinates/weak_audit.py --complete-bed --output results/rstar_weak_completed_new
python research/experiments/material_rstar_coordinates/weak_witness_audit.py --input results/rstar_weak_completed_new --output results/rstar_weak_completed_witness_new.json
python research/experiments/material_rstar_coordinates/weak_audit.py --output results/rstar_weak_tail_new
```

The completed original-domain 16 controls pass the instantaneous gates on CPU and
actual CUDA/x64. The tail command deliberately exits nonzero: its original-domain
affine-stair rest failure remains. Truncated-bed controls are localization only.
The NumPy witness auditor checks every saved array and complete unique case matrix;
it does not independently rerun quadrature from geometry or accept an ocean step.
Positive regression counts include verified failures and do not qualify boundedness,
PDE order, full momentum/fast-slow coupling, ice or climate/forecast performance.
Local dense hat traces scale with column count times vertical-node-count squared;
actual-grid GPU throughput and peak memory are not qualified. No production imports
or default changes are made.

## Physical point pressure accuracy and consistent kinetic mass

`pressure_accuracy_protocol.json` registers all-node analytic hydrostatic controls,
with independent five-point Gauss force functionals. Vertical isolation uses the
exact centered angular derivative; horizontal isolation uses the exact hat
projection. Physical point errors are also saved without subtracting a bias.
Endpoints are never excluded from the qualification norm.

```shell
python research/experiments/material_rstar_coordinates/pressure_accuracy_audit.py --output results/rstar_pressure_lumped_new
python research/experiments/material_rstar_coordinates/pressure_accuracy_audit.py --consistent-velocity --output results/rstar_pressure_consistent_new
python research/experiments/material_rstar_coordinates/weak_momentum_audit.py --output results/rstar_weak_momentum_new
```

The first command intentionally fails the full-node vertical second-order gate
although its pressure work/rest controls passed. Its force functional is correct;
lumped velocity mass introduces endpoint point-force error. `weak_momentum.py`
uses the actual consistent hat kinetic mass and the **same** weak transpose.
This explicitly changes the local kinetic contract, not the pressure formula or
domain. Registered vertical/horizontal operator order and recomputed work/rest
controls pass on CPU and actual CUDA. Do not reuse the old diagonal power audit
for this new mass: the new raw packets include independently assembled kinetic
matrices. The old witness auditor is not an auditor for this kinetic contract.
These gates do not qualify moving-mass momentum, bounded tracers, complete PDE
order, fast-slow time coupling or a production factory.

## Bounded consistent content and full-grid capacity

`weak_sparse.py` assembles physical weak coefficients without a global dense
Jacobian. The graph contains adjacent vertical hats and all potential wet pairs
across physical column faces, not the old equal-index graph. Face quadrature
streams 16 original longitude columns and one actual periodic halo using a scan;
all latitude columns, beds, wet unknowns and the north wall remain unchanged.
This fixed graph permits changing vertical overlap but can be wider than the
currently nonzero overlap. Do not call its local-nz-squared storage globally dense.

`weak_bounded.py` decomposes high/low consistent-mass stages into antisymmetric
corrections, limits each pair against actual endpoint/source-adjusted node bounds,
then re-encodes **consistent** content. It checks decoded nodes too and rolls back
invalid stages without clipping states or correcting an inventory residual.
The declared source is applied identically to both paths. Limited-minus-high
thermal potential transfer is reported separately and need not be negative.

```shell
python research/experiments/material_rstar_coordinates/weak_bounded_audit.py --output results/rstar_bounded_new
python -m pytest tests/test_rstar_weak_bounded.py -q
python research/experiments/material_rstar_coordinates/weak_actual_capacity_audit.py --advance --output results/rstar_full_capacity_new
```

The last command requires the original saved grid, parameters and first-rejected
surface packets named in `real_geometry_protocol.json`; these large local inputs
are not bundled in a fresh clone. Omitting `--advance` only constructs the graph.
Use the same entry point on actual CUDA with `JAX_PLATFORMS=cuda` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. `weak_actual_protocol.json` records the
fields, gates and interpretation; every run freezes source/input hashes and raw
arrays, refuses existing output folders and records completed stages separately
from accepted ocean steps. It is not an integration command.

Eight prescribed-flow CPU/CUDA controls and two full original1degree real-grid
prescribed tracer stress stages pass, with independent quadrature/matrix/source
checks. The first full-grid CUDA OOM is retained: streaming fixes workspace, not
the domain or tolerance. The full-grid probe uses fresh analytic node fields and
100W/m2 heat, not a WOA restart or solved momentum; accepted ocean steps remain
zero. The first near-zero pulse stock comparison and the explicitly revised
point-precision-weighted oracle scale are retained too; runtime physical bounds,
inventory/decomposition and original pressure gates are not relaxed.
No time-order, full moving-mass momentum, full source/diffusion suite, production
factory/restart, real day7/30 or industrial qualification follows from these stages.
