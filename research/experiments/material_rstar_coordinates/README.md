# Nodal r-star experiments: code and qualification map

Research only: neither the production CLI nor `material_top.py` imports this
directory. Read the [representation decision](representation_decision_zh.md)
for the joint state/operator contract and the
[repair evidence, sections 27–37](../../../docs/legacy_core_repair_status_zh.md)
for measured results, original failures and source-frozen witnesses.

## Current weak-form path

```text
bed_completion + kernel.rstar_geometry + nodal_mass
  -> weak_transport.physical_face_quadrature
  -> weak_sparse (potential wet-pair graph; streamed face coefficients)
  -> weak_bounded (consistent content, pair limiting, source and rollback)
  -> weak_time (two-stage candidate; moving-limiter time gate FAILED)

weak_transport + pressure_work + weak_momentum
  -> instantaneous consistent-kinetic pressure pairing
```

| Module | Responsibility / constraint |
| --- | --- |
| `bed_completion.py` | Preserve physical beds and original wet nodes; add a bed sample in a spare dry slot. New unknowns/topology, not valid values copied from an old dry restart |
| `kernel.py` | r-star geometry: `s=1+eta/H`, `h=s*h_ref`, physical depth `d=-eta+s*d_ref`; retain invalid widths, do not clip |
| `nodal_mass.py` | Batched consistent hat-mass application / inverse; positive content does not guarantee positive decoded nodes |
| `weak_transport.py` | Physical-aperture hat quadrature and weak content/volume rates; distinct from equal-index coordinate bands |
| `weak_oracle.py` | Independent NumPy five-point face integration; do not replace it with the JAX kernel |
| `pressure_work.py`, `weak_momentum.py` | Potential conjugates and matching consistent kinetic mass for instantaneous pressure; no complete moving-mass momentum budget |
| `pressure_accuracy.py` | Registered all-node physical pressure-order controls, including endpoints |
| `weak_sparse.py` | Fixed potential wet-pair graph, deterministic grouped reductions; 16-column streaming with a periodic halo, no cropped physical domain |
| `weak_bounded.py` | Antisymmetric high/low corrections, endpoint/source bounds, re-encoded consistent content and original-byte rollback; no tracer clipping or stock compensation |
| `weak_time.py` | Actual stage-time/state/geometry fields, content/surface combination and mean source ledger; candidate, not qualified moving time integration |

## Evidence status

- Completed-bed work/rest and specified consistent-kinetic pressure point-order
  controls pass on CPU/CUDA. Lumped-velocity endpoint-order failure remains.
- Bounded weak constant/pulse/signed/heat controls pass; unrestricted pulse
  negativity is retained. Limiter thermal potential transfer can be positive:
  boundedness is not full energy / entropy stability or external heating.
- Full 360x132x14 prescribed tracer stages pass on CPU/CUDA after streaming fixed
  the first CUDA OOM. They use fresh analytic fields and 100W/m2 heat on original
  grids and saved eta, **not** WOA restart continuation or solved momentum.
- Smooth Fourier time order is approximately 2, with an Euler order-1 negative
  control. Original moving-limiter signed/pulse orders approximately 0.77/1.16
  and 0.62/0.86 **fail** the registered 1.9 gate on both backends.
- Regression success is separate from candidate qualification. Accepted ocean
  steps remain zero for these weak-path audits; production is not promoted.

Full moving momentum/energy, fast-slow mean matching, complete source/diffusion
coupling, factory / initialization / restart migration, ice, full-step gradients,
real 1-degree day7/30, century and climate / forecast / industrial gates remain
open. Frozen-field local Euler probes do not isolate the time failure's cause;
they also do not use the complete failed trajectory's time/state field callback.
Do not replace failed cases or loosen their thresholds.

## Reproduction entry points

Run from the repository root in the configured float64 environment. Every
`--output` destination must be new. Select actual CUDA before launch with
`JAX_PLATFORMS=cuda` in a supported environment, not by relabeling CPU evidence.

```shell
python research/experiments/material_rstar_coordinates/weak_audit.py --complete-bed --output results/rstar_weak_new
python research/experiments/material_rstar_coordinates/pressure_accuracy_audit.py --consistent-velocity --output results/rstar_pressure_new
python research/experiments/material_rstar_coordinates/weak_momentum_audit.py --output results/rstar_momentum_new
python research/experiments/material_rstar_coordinates/weak_bounded_audit.py --output results/rstar_bounded_new
python research/experiments/material_rstar_coordinates/weak_time_audit.py --output results/rstar_time_new
```

The last audit currently exits **1** because moving time qualification fails;
a complete report / correct negative-control test is not a physics pass.
The matching source manifests, protocols and raw witnesses define each run.

```shell
python -m pytest tests/test_rstar_weak_bounded.py tests/test_rstar_weak_time.py -q
python research/experiments/material_rstar_coordinates/weak_actual_capacity_audit.py --advance --output results/rstar_capacity_new
```

The capacity audit requires the local archived grid, parameters and rejected eta
packets named in `real_geometry_protocol.json`, not bundled in a fresh clone.
Without `--advance` it only constructs the graph. Full CUDA capacity used
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, explicitly recorded in the report.
Neither command above is a complete ocean integration or a throughput benchmark.

## Retained controls: not parallel production solvers

These files remain because they define independent references, prerequisite
algebra or failed alternatives; absence from the current path does not make them
dead code. Their protocols/tests retain distinct physical contracts.

| Control family | Sources / entry points | Interpretation |
| --- | --- | --- |
| Coordinate metric and adjoint diffusion | `dense_oracle.py`, `audit.py`, `protocol.json` | Affine-rest / metric controls; unlimited negative-adjoint diffusion has a maximum-principle counterexample |
| Sparse diffusion and donor moving content | `sparse_diffusion.py`, `moving_content.py`, `sparse_audit.py` | Original donor / row-mass controls, not the new completed-bed weak graph or complete moving PDE |
| Actual rejected geometry | `real_geometry_audit.py`, `real_geometry_protocol.json` | CPU geometry inspection; 128-capacity stop differs from the later 256-cap negative-top state |
| Coordinate-band pressure | `pressure_audit.py`, `pressure_protocol.json` | Work/rest conflict and constant-tail failures retained; audit intentionally exits nonzero |
| Bed/mass factorial localization | `representation_audit.py`, `representation_protocol.json` | Truncated bed changes domain; changing mass alone does not fix incompatible coordinate-band transport |
| Original-tail weak pressure | `weak_audit.py` without `--complete-bed` | Original-domain affine-stair rest failure retained |
| Lumped-velocity point accuracy | `pressure_accuracy_audit.py` without `--consistent-velocity` | Instantaneous work/rest can pass while all-node vertical order fails |

The NumPy-only `pressure_witness_audit.py` and `weak_witness_audit.py` check
saved witness identity and the declared gates; their success does not rerun all
physics or turn failures into passes. `*_protocol.json` files register scope and
thresholds; `*_audit.py` files freeze sources and results. Do not merge either
kind into the numerical kernel.

## Evidence preservation

Detailed histories, including the first oracle comparison failures and their
explicitly revised scales, are in the repair record and frozen result packs.
Independent oracles, refused-stage witnesses, original OOM and time failures
are not cleanup targets. After source-only cleanup, historical hashes identify
their frozen commit/snapshots, not necessarily the current checkout. Do not
silently upgrade an old result to current-source qualification.
