# 6h I0B0 actual N two-RHS decomposition

## Outcome

One paired original dt300 attempt passed. N reference kinetic change is **6743897896165952 J**. The component-work sum differs by -12.0 J within the frozen 855662 J absolute-product arithmetic bound. The residual is recorded and assigned to no physical source.

| Actual N work group | Work (J) |
|---|---:|
| pressure_net | +4.840104671306e+15 |
| advection_net | +9.342697273097e+10 |
| wind_net | +1.903699797887e+15 |
| horizontal_cancel | +0.000000000000e+00 |
| vertical_cancel | +0.000000000000e+00 |
| coriolis_cancel | +0.000000000000e+00 |
| bottom_cancel | +0.000000000000e+00 |
| final_residual_mask | +0.000000000000e+00 |

Net pressure and wind dominate this particular positive N increment. This is
operator work at the actual evolving state, not proof that the forces are
physically correct. It does not establish which input or solver change is
needed. In particular, positive N wind work and earlier negative fast-stage
wind work refer to different operator stages; fast-stage wind alone cannot
stand for total-step wind work. No compatible buoyancy PE or full mechanical
energy closure is claimed.

## Raw/filter and pressure terms

| Component | Work (J) | First actual RHS contribution (J) | Second actual RHS contribution (J) |
|---|---:|---:|---:|
| advection_raw | -3.568352315390e+11 | -1.531209013252e+11 | -2.037143302138e+11 |
| advection_dealias_difference | +4.502622042700e+11 | +2.214578065628e+11 | +2.288043977072e+11 |
| pressure_full | +5.772605128046e+15 | +2.886453052550e+15 | +2.886152075496e+15 |
| wind_surface | +1.892725676752e+15 | +9.463628383760e+14 | +9.463628383760e+14 |
| horizontal_diffusion_full | -2.252686469257e+15 | -1.119704895541e+15 | -1.132981573716e+15 |
| vertical_diffusion_full | -3.478606045260e+14 | -1.728850353669e+14 | -1.749755691591e+14 |
| coriolis_full | -4.169376368789e+09 | -4.601645621179e+13 | +4.601228683542e+13 |
| bottom_drag_full | -1.471955618851e+15 | -6.947305085835e+14 | -7.772251102675e+14 |
| horizontal_diffusion_removal | +2.252686469257e+15 | +1.119704895541e+15 | +1.132981573716e+15 |
| vertical_diffusion_removal | +3.478606045260e+14 | +1.728850353669e+14 | +1.749755691591e+14 |
| coriolis_removal | +4.169376368789e+09 | +4.601645621179e+13 | -4.601228683542e+13 |
| eta_mean_removal | +7.429801790437e+14 | +3.714900895218e+14 | +3.714900895218e+14 |
| density_mean_removal | -1.675480635783e+15 | -8.378125836469e+14 | -8.376680521366e+14 |
| wind_mean_removal | +1.097412113480e+13 | +5.487060567402e+12 | +5.487060567402e+12 |
| final_residual_mask | +0.000000000000e+00 | +0.000000000000e+00 | +0.000000000000e+00 |
| bottom_drag_compensation | +1.471955618851e+15 | +6.947305085835e+14 | +7.772251102675e+14 |

The advection-filter difference is positive here. The original tendency
filter is FFT longitudinal truncation plus a latitude binomial filter; it is
not an isolated velocity damping map. Raw and filtered values are compared
with the same wet mask. Neither raw nor filter work is charged with the
remaining work-identity roundoff. Full pressure and the positive eta-mean
removal and negative column-density-mean removal remain visible rather than
being silently renamed density PE conversion.

## Gates and budget

All 49 actual full-tendency/residual/compensated-RHS/Euler/final-increment/tracer/eta checks passed; maximum pointwise reconstruction error 2.22045e-16. Reference nodal mass normalization and wet column depth match exactly.
Both original and instrumented attempts were accepted; original boolean gates agree. All six attempted and returned fields are finite, maximum absolute difference 2.13163e-14, below the predeclared 1e-11 gate. This is tolerance equivalence, not byte identity.
Second RHS T/S are the actual tracer-updated fields: differences from incoming T/S reach 0.0589124674192 and 0.0168991154913, respectively. The Euler state is only an evaluation state; both RHS component works use the same full-N endpoint midpoint.
Total paired wall time 251.790495s, maximum sampled working set 2558889984 bytes, maximum sampled private commit 2519019520 bytes; one CPU affinity mask, below 600s/4GiB. No guard triggered. Projection trace calls: 0.

## Geometry and remaining gap

N holds eta fixed. Moving-top-minus-reference kinetic energy changes from -3.558064744647e+15 to -3.563721487184e+15 J, change -5.656742536922e+12 J. This distinct weight diagnostic changes with velocity at fixed eta; it is not an external source used to close Kref.

Next useful review is a compatible discrete pressure/buoyancy-PE work contract
at these actual states, and the physical admissibility of the existing forcing
and input. This record supplies no initialization repair, longer trajectory,
threshold relaxation or evidence of a 30h fix.

## Reproduction and integrity

Use `prepare_n_source.py` with the already verified historical source and original
`protocol_6h.json` to create a fresh isolated source. Run `n_probe.py` sequentially
for baseline and instrumented phases with archive/run directories materialized
locally by the authorized data owner; both workers bind source and input hashes.
The external local resource guard must enforce the frozen CPU/time/memory bounds.
Use `summarize_n_probe.py` on the resulting private arrays. The protocol, source
manifest, patch and scalar JSON contain identities; they do not contain raw arrays.

Frozen executed protocol SHA256: `1400c470b326c3b26bd76bd652fe7f9a79c2ea924792cde5c9d412eb1933ff05`.

The first scalar export failed because the arithmetic-bound factor remained a
NumPy scalar and produced a NumPy bool unsupported by JSON. The published
summary changes only that factor to the identical Python float; the frozen
protocol and solver worker are untouched. A negative-then-positive regression
test covers serialization. Existing raw arrays were reread, not recomputed.
The scalar JSON records both summary hashes and the failed new empty target;
the failed target was not overwritten.
