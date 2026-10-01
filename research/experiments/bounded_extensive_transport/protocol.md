# Bounded higher-order physical transport and explicit inventory precision

Registered 2026-09-29 against `b0d65fb`, before the precision controls or core
changes. The previous real-grid overall FAIL and all original thresholds remain.
The full industrial roadmap is unchanged; old production is not migrated yet.

## Research and hypotheses

Primary sources read 2026-09-29:

- [MITgcm transport schemes](https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html): donor is a diffusive low-order base, higher-order face fluxes require nonlinear limiters; multidimensional stability must not be inferred from separate directional CFLs.
- [MITgcm moving volume](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html): the same h/time-level Q must advance h and h*C, including explicit sources.
- [MOM6 tracer transport](https://mom6.readthedocs.io/en/main/api/generated/pages/Tracer_Advection.html): transport concentration and layer volume consistently; PPM/ALE is not a direct transplant into these fixed-z cells.
- [Zalesak 1979](https://doi.org/10.1016/0021-9991(79)90051-2), [paper](https://data.coaps.fsu.edu/pub/eric_back/OCP5930/Papers/FCT-JCPv31.pdf): sum positive/negative antidiffusive contributions over all directions; one face coefficient shared by its two neighbors, not clipping final cell contents.
- [Gottlieb, Shu and Tadmor 2001](https://drum.lib.umd.edu/items/78e24b83-01d8-4e55-935e-dc8bacaf910c/full): SSP time stepping is a convex combination of stable forward-Euler steps; it does not fix an unstable spatial operator or certify coupled momentum accuracy.
- [Mixed precision in NEMO/ROMS](https://gmd.copernicus.org/articles/12/3135/2019/): precision choices require application-level accuracy tests, not a blanket switch or emulated-operation timing as a production speedup.

Hypothesis: rounding V and N independently and losing small content increments
can break a mathematically bounded donor scheme. Separate arithmetic from stored
inventory precision on identical initial quantized inputs and prescribed Q.
Do not presume which field, or a physical operator, causes the failure.

Controls: native32; guard arithmetic with32 storage; V32/N64; V64/N32; V64/N64;
two32-component expansions reconstructed/updated in64. Same closed spherical
50/15m/land geometry, constant and patterned T/S, 100 dt60 steps, no sources.
Report local ranges, constant change, content/volume budgets, invalid steps,
stored bytes and isolated warmed CPU cost. The fixed-Q experiment is NOT a
bitwise replay or qualification of the earlier barotropic trajectory. If the
old numerical bound failure is not reproduced, do not claim a cause.

If higher inventory precision proves necessary, permit an explicit mixed
policy (V/N64, pressure eta and velocity32) rather than automatically changing
all state precision. Compare the two32 expansion cost/benefit; a pair reconstructed
in64 is not a pure32 accelerator. Record the increased inventory memory and
64-bit work; no GPU or full-model performance claim. Reject undocumented dtype
mixtures. Preserve legacy same-dtype donor controls, not relabel them qualified.

## Higher-order design

Implement metric-aware piecewise linear face reconstruction and conservative
multidimensional FCT around the physical donor Euler step. Bounds use wet local
incoming neighbors, extended by the forced low-order state when sources are
present. Multiply concentration bounds by NEW V to form extensive allowances.
The face antidiffusive amount is shared with opposite signs. No post-update
clamp, mean removal or V-dependent content refill. Use SSPRK2 for frozen Q and
constant-in-step sources; the final V still matches the mean Q used by eta.
Every intermediate Euler step must pass physical and outflow gates. This is
second-order tracer integration for the prescribed/frozen-Q problem, not a
proof of full momentum/tracer time coupling order.

Horizontal cell widths/center distances and current physical layer thickness
must enter reconstructions. Closed faces must not supply ghost/dry extrema.
New fields distinguish cell widths from distances between centers. All shape,
dtype, source, top/bottom, dry-cell and invalid-state gates remain fail-closed.

## Gates before qualification

1. Controls reproduce a native32 bound failure >2e-6. Precision comparisons
   retain the same input values/Q and do not alter bounds or select best regions.
2. Independent face exchange and forced totals close V/N budgets <=1e-12 in64;
   constant concentration relative change <=1e-12 for pure64, <=2e-6 mixed.
   Source-free ranges stay inside incoming extrema with unchanged absolute
   tolerance 1e-12 pure64 / 2e-6 mixed. Forced extrema are not mislabeled unforced.
3. Partial/unequal cells, coast/land/dry sentinels and all three flux directions
   remain conservative; FCT correction sums globally to zero without repair.
4. Periodic cell-average cosine transported one quarter period, nx32/64/128,
   CFL0.2, fully wet two latitude rows/one layer. Pure64 L2 error ratios >=3.2
   and fine-grid error less than donor at identical dt. Exact cell averages,
   analytic translation and budget comparisons are independent. A failure
   remains a failure; do not claim second order merely because SSPRK2 is used.
5. Local JVP/finite difference <=1e-6 and VJP dot identity <=1e-12 away from
   sign/limiter transitions; not full-model adjoint qualification.
6. Repeat both real-ETOPO geometries, constant/patterned T/S, pure64 and explicit
   velocity32/inventory64, 100 dt60 steps with four barotropic substeps. Keep
   the original surface/content/constant/bound thresholds; final state V/N
   inventories, geometry and source hashes are recorded. Add physical volume
   residual units, not only an inventory-normalized ratio hiding surface drift.
7. Keep same-dtype32 donor failures visible and preserve earlier archived tools
   with frozen Git sources. Full tests/lint/MMS and independent installation
   must pass. No bathymetry/timestep/tolerance change to manufacture success.

After component gates, next priorities remain C-grid 3D momentum/well-balanced
partial-cell pressure and production cutover (true sources/ice/mixing, budgets,
node-to-cell initialization and checkpoint migration). No century/climate/
GPU/distributed/full-adjoint PASS is implied by this phase.
