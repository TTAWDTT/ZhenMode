# Float32 physical surface diagnosis: follow-up, not a relaxed gate

Registered 2026-09-29 after the `1d04acf` real-grid reference experiment and
independent precision attribution, before changing the arithmetic.

Four float64 100-step runs pass. All four float32 groups reject their first
step at 4.247e-6/4.396e-6, above the unchanged 2e-6 local surface gate, even
though donor outflow and wave bounds pass. The rejected step is not accepted
or counted as zero-error conservation. Original reference JSON/log and its
hashes remain at `extensive_transport_reference.*` in local results.

Independent float64 recomputation of the *same stored float32 output volume*
still gives 2.416e-6/2.585e-6: diagnostic arithmetic alone cannot fix the
problem. Incoming float32 V/A-h differs from the physical ratio by up to
1.192e-6 on the same relative scale. Nearest representable output volumes
could achieve 1.495e-6, and half a volume ULP corresponds to at most 3.04e-7m.
The two original compiled first-failure residuals reproduce exactly. These
are measured finite-precision identities, not pressure convergence or CFL
failures. The follow-up must keep the physical and representational causes
separate, and retain every old failure.

Primary arithmetic sources read 2026-09-29:

- [Goldberg floating-point tutorial](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html): ULP, nearest rounding, cancellation and guard precision. Stable algebra does not recover bits already lost in storage.
- [JAX default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html): float64 creation requires enabled X64; silently requesting 64 while it is disabled is not a reliable mixed-precision policy.

Candidate: expose a physical surface-height diagnosis from primary V using
float64 division/subtraction, returning the original state dtype. Use that
same physical diagnosis for the incoming pressure eta and the acceptance
check; preserve V, N, velocity and eta storage dtypes. Require X64 explicitly
for this float32 mixed-arithmetic path; do not change global JAX configuration
inside the module. No offset, volume refill, modified source, mean correction,
new depth threshold or tolerance relaxation. Public manifests and docs record
the mixed arithmetic; this is not a pure-float32 or GPU-speed claim.

Gates:

1. Independent NumPy float64 diagnosis agrees with the helper within one
   rounding to its returned dtype, on the same float32 V and area inputs.
2. A data-free 180x66 partial-cell geometry reproduces the old surface gate
   failure before implementation. Retain the old ratio, not only a missing API.
3. Repeat all eight original 100-step geometry/dtype/tracer groups at dt60;
   keep 2e-6/1e-12 surface, constant and content-budget thresholds unchanged.
   Original failures stay visible beside the new results. If stored-content
   or volume quantization still fails, report failure instead of automatically
   moving all storage to64 or redefining the residual normalization.
4. Source/dtype/fail-closed/gradient tests, configured lint, full suite and MMS
   remain valid. Independent installed imports expose the new core modules.

The production collocated solver is still not migrated. Passing this component
does not qualify full physical budgets, climate, century, GPU or adjoints.
