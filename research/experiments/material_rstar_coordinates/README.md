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
  qualification, explicit diffusion positivity policy, ice coupling, new-state
  restart migration, actual 1-degree rerun or industrial comparison here.

Before integration, freeze the moving-face and stage-content contracts, choose
and test the diffusion bounds policy without clipping residuals, then prove
complete-step source/budget/pressure-work consistency and time convergence.
Only afterward run the unchanged real 1-degree rejection protocol. Earlier
failed actual runs stay failed; these controls do not relabel them as passes.

The [bounds-policy review](../../literature/rstar_diffusion_bounds_20260930.md)
records primary sources, the retained maximum-principle counterexample, and
the next independent matrix experiment before a full-step implementation.
