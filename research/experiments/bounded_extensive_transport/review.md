# Explicit64 inventories and bounded shared transport pass component gates

2026-09-29. Protocol `8803e73`, precision/method choice `71fc1c9`, core `2b9d632`.
This is NOT production cutover, a full ocean simulation, or century/climate/GPU
qualification. The complete industrial roadmap remains active.

## Evidence changes the implementation choice

The frozen `cb38ba7` prescribed-Q control completes twelve 100-step cases with
identical quantized incoming V/N and Q. Neither guard arithmetic nor promoting
only V or only N fixes the original32 bound failure. Both64 inventories and
reconstructed two32-component inventories do fix this fixture:

| Storage/arithmetic choice | Constant bound excursion | Patterned bound excursion | Inventory bytes |
| --- | ---: | ---: | ---: |
| Native32 | 3.101e-4 | 9.912e-5 | 36864 |
| Arithmetic64/store32 | 3.101e-4 | 9.912e-5 | 36864 |
| V32/N64 | 1.834e-4 | 7.260e-5 | 61440 |
| V64/N32 | 1.518e-4 | 6.871e-5 | 49152 |
| V64/N64 | 0 | 0 | 73728 |
| Reconstructed two32 expansion | 0 | 0 | 73728 |

This implicates independent inventory storage rounding on this fixed-Q control,
not a proven complete attribution of the earlier barotropic trajectories.
The selected explicit policy stores V/N64 and permits eta/velocities32. It does
not silently promote the full state. Inventory bytes double relative to32; the
pair expansion saves no bytes and still computes in64. No GPU or full-model
performance conclusion follows from the tiny warmed CPU control timings.

## Implemented method and direct gates

`bounded_transport` adds centered metric-weighted linear reconstruction and
all-direction FCT. One limited face amount is subtracted from its origin and
added to its neighbor; six-face positive/negative potentials are accumulated
before choosing the coefficient. Extensive allowances use NEW V. Reconstruction
distinguishes horizontal cell widths from center distances and uses current V/A
vertical thickness. Slopes vanish where either reconstruction neighbor is closed.

Two FCT Euler stages form SSPRK2 with frozen Q and constant-in-step sources;
BOTH V and N are convexly averaged. Explicit forcing may extend local bounds
to the forced low state; it is not mislabeled unforced boundedness. Every Euler
stage must be valid, even if an invalid second stage would be hidden by the final
average. No state clamp, global mean removal, content refill or timestep repair.

The coupled reference defaults to the original same-dtype donor. Explicit
`inventory_precision="float64"` and `transport_scheme="centered_fct"` select the
new policy; pressure surface checks retain the momentum dtype's original gate.
Undocumented dtype mixtures and unknown policies are rejected. This component
selection is not an alternative production-driver flag or a claim of migration.

Seventeen new tests pass, including an independent scalar-cell-loop FCT oracle,
three-direction partial-cell/coast exchange, forced freshwater/content totals,
constant preservation, wet/dry gates, intermediate depletion/CFL rejection,
irregular cell widths, explicit dtype policy and local JVP/FD/VJP away from
limiter/sign transitions. Existing thirty-three physical-transport tests pass.
The initial missing-module collection error is retained but is NOT a numerical
counterexample; the fixed-Q storage controls supply actual numerical failures.

Registered exact cell-average cosine, quarter-period translation at CFL0.2:

| nx | FCT/SSPRK2 L2 error | Refinement ratio |
| --- | ---: | ---: |
| 32 | 0.007342957151 | -- |
| 64 | 0.002177976873 | 3.371457816 |
| 128 | 0.000660276018 | 3.298585462 |

The registered >=3.2 ratios pass; donor at nx128 has error0.021474769168.
Observed orders are about1.75/1.72, NOT an asymptotic2.00 demonstration. Extrema
limiting can degrade smooth-peak accuracy. Formal RK order or this short sequence
does not prove second-order full momentum/tracer coupling, global climate
accuracy, or universal spatial convergence. Preserve this limitation rather than
market the candidate as already equivalent to mature PPM/ALE configurations.

## Eight real-grid trajectories and independent stored-state verification

Exactly the previous ETOPO geometries, 180x66x14 physical cells, edges +/-66deg,
100 dt60 steps with four dt15 linear-wave substeps, no altered bathymetry or
tolerance. Pure64 and momentum32/inventory64, constant/patterned T/S: **8/8 PASS**.
Physical construction uses momentum-quantized eta/C, promoted BEFORE inventory
construction; it is not bitwise continuation of old quantized32 V/N.

| Geometry/momentum | Maximum surface ratio | Worst content budget residual | Water inventory delta m3 |
| --- | ---: | ---: | ---: |
| Prior smoothed/64 | 8.341e-15 | 2.774e-14 | -0.007324219 |
| Prior smoothed/32 | 1.901e-7 | 3.075e-14 | -4.228515625 |
| Unsmoothed/64 | 7.858e-15 | 4.414e-14 | -0.001953125 |
| Unsmoothed/32 | 2.000e-7 | 5.030e-14 | +3.264160156 |

Water residual is the64 sum of per-cell stored V changes, not subtraction of two
huge rounded global inventories. Largest equivalent domain-mean surface change
is about1.36e-14m. Maximum source-free final range excursion is3.269e-13 across
all groups; this is observed on these synthetic fields, not an arbitrary tracer
or full-model guarantee. These references are linear waves and synthetic
two-tracer transport, not full3D momentum, real forcing, mixing or ice.

All eight initial/final V/N and geometry snapshots have hashes. The independent
`verify_reference.py` checks required runtime source hashes, fixed duration/dt,
snapshot hashes/shapes/dtypes/finite and wet/dry fields, physical column depths,
recomputed global ranges/constants/content budgets and physical water residuals.
It additionally requires inventory budgets <=1e-12 for the actual64 inventories;
these pass. Intermediate surface/CFL maxima are audited recorded diagnostics,
not reconstructed from final-only snapshots. Five negative metadata controls
reject incomplete source manifests, NaN diagnostics, changed timestep, failed
run status and a wrong snapshot hash. First strict manifest comparison rejected
Windows path separators; normalize separators before checking required keys,
then rerun the positive case and each distinct negative branch.

The CURRENT same-dtype donor control independently reruns all eight groups:
again6/8, the two patterned32 cases FAIL. Its repeated float32 trajectory is not
claimed bitwise equal to the archived one, since compilation/source argument
layout differs. The frozen `1d04acf` surface diagnostic still exactly reproduces
both archived first failures. Archived failures are not overwritten or relabeled.

## Reproduce and remaining qualification

```powershell
python research/experiments/bounded_extensive_transport/probe_precision.py
python -m pytest tests/test_bounded_extensive_transport.py tests/test_extensive_transport.py -q -s
python research/experiments/extensive_transport/run_reference.py --inventory-policy float64 --transport-scheme centered_fct --out results/industrial_alignment/bounded_transport_reference.json
python research/experiments/bounded_extensive_transport/verify_reference.py
python research/experiments/extensive_transport/run_reference.py --inventory-policy same --transport-scheme donor --out results/industrial_alignment/bounded_transport_retained_donor.json
```

Last command must return overallFAIL/exit1. Current default runner output was
renamed to extensive_transport_current.json so old evidence is not overwritten.
Verifier rejects shortened/fallback-dt runs even if the runner labels them PASS.
Raw JSON, NPZ, validation logs and installed artifacts remain local ignored
results, not falsely claimed as committed large data.

Full suite: **409 passed**,23 existing warnings. Configured and explicit research
ruff and diff-check pass. Old production MMS ALL PASS, derivative ratio4.30.
Wheel installs independently; isolated Python imports the installed new modules
and executes an installed FCT step. Only verified newly generated pip build/
copies were removed, not user source or result inventories.

Next deliver actual C-grid3D momentum, common-depth/well-balanced partial-cell
pressure and barotropic coupling, then true sources/mixing/ice, initialization,
restart and the production path. Do not keep this as an unused qualified
prototype. Legacy approximately2e21J proxy heat gap is NOT repaired by this
component table. Full global geometry, real-forcing spin-up/century and independent
climate/forecast, fair GPU/distributed cost and whole-model adjoint remain open.
