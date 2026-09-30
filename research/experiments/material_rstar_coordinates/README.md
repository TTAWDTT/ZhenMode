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
