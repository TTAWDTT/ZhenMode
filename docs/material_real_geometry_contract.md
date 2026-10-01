# Accepted-state geometry and inventory consumption contract

## 2026-10-01: scope and interpretation

`nodal_dual_stock_mean_bridge_v1` consumes the unchanged, accepted absolute
step **353** from the archived `1deg_dt300` case. It constructs a candidate
inventory geometry and evaluates static shared-face Q. **It does not execute
a timestep**, run a coastal model, switch a production entry point, qualify
the original failed integration, or enable biharmonic/filter terms. PR19's
dynamic research slice remains a separate method and commit.

The source is the complete `360 × 132 × 14` accepted state. All fourteen
slots remain represented. The bridge preserves every wet-column water,
IT, IS, reference Mu and reference Mv inventory; all slots below the top
three remain unchanged. Dry slots contribute no inventory, and their private
source values remain untouched. Original ice is retained byte for byte;
this bridge does not reinterpret or evolve ice.

The archive reader verifies all six exact file identities, accepted step
and case, and 35 archived source modules against git object
`212df951c351f82dba74fbc43db5e52b0ad34c47`. Archive members are read in memory;
historical code is never imported or executed. The reader rechecks the six
files and all accepted-state arrays after the audit. Published evidence
contains scalar receipts, shape/count metadata, and identities. It contains
no private arrays, machine paths, secrets, rejected-state arrays or generated
physical profiles.

## 2026-10-01: per-column geometry

Only `nodal_dual_v1` is accepted. From ordered positive-down nodes `d`, the
reference dual edges are `e = [0, midpoint(d[k],d[k+1]), d[-1]]`; archived
reference weights must match `diff(e)` exactly. A binary wet prefix determines
each column's own active-layer count `n`, discrete bottom `-e[n]`, and band
bottom `-e[min(3,n)]`. Raw ETOPO partial cells are not substituted for these
dual cells. Grid/parameter wet masks, terrain node cutoff, archived reference
spacing, bottom indicator and normalized reference depth are independently
checked. The historical dry `H_sw=1` division guard never becomes candidate
water: a dry column has zero layers, zero water/stocks and zero face Q.

Original material thickness is `w[k]` on each active layer, with `eta` added
only to layer zero. Every wet source thickness must be positive. Original
interfaces are `[eta,-e[1],...,-e[n]]`; inactive interfaces repeat the column
bottom. In the candidate, the complete top band's fixed reference fractions
`w[k]/e[min(3,n)]` partition `eta + e[min(3,n)]`. Deeper interfaces, thickness
and stocks remain fixed. A two-node or one-/two-layer shallow column uses
only its available complete cells. No wet support is invented on a dry column.

The original inventories per unit area are

```
IT = h_actual * T       IS = h_actual * S
Mu = rho0 * h_reference * u
Mv = rho0 * h_reference * v
```

The candidate is explicitly a **P0 mean-stock interpretation**. All four
stocks use the same physical overlap remap:
`N_new[j] = sum_i overlap(i,j) * N_old[i] / h_old[i]`.
In particular, momentum's denominator is actual source h. Replacing only a
mass divisor, reinterpreting reference M as actual-h M, and restoring column
totals after a nonconservative remap are not this contract. Column conservation
uses a fixed `256*eps*max(1,sum(abs(source_stock)))` arithmetic envelope.
The bridge rejects masked inputs instead of losing their support masks and
rejects integers that cannot be represented exactly by float64.

## 2026-10-01: three explicit consumption decisions

1. **Negative eta and node zero.** Node zero retains its original material
   inventory even when its FD sample lies above the physical surface.
   Inventory support and physical sample support are different facts. The
   physical support interval includes only original wet nodes lying at or
   below eta. Original-profile consumers reject an interval requiring a
   surface boundary value/reconstruction that the archive does not define.
2. **Reference momentum to actual mass.** Stocks are preserved, so actual
   velocity is `M/(rho0*h_actual)`. The conversion's kinetic cost is reported
   separately from the P0 remap's kinetic change. P0 overlap also checks its
   fixed-domain kinetic nonincrease within the arithmetic envelope. Neither
   quantity is declared a physical total-energy loss or a pressure-work proof.
3. **Dry west and staircase neighbors.** Each candidate shared face uses
   `[max(bottom_left,bottom_right), min(eta_left,eta_right)]`, partitioned at
   both columns' physical interfaces. A segment uses P0 stock-mean velocity
   and `Q = L*dz*dot((u_left+u_right)/2,normal)`. A dry neighbor has no segments
   and exactly zero Q. Unmatched deep water on a deeper neighbor cannot leak
   across the shallower discrete bottom.

`require_original_profile` rejects missing surface/bottom support. Passing
it only establishes bracketing physical wet samples; it does not choose an
interpolation/EOS, equate original node samples to candidate means, or qualify
pressure. Selected columns have **250/500 m** between their last wet node and
their discrete bottom. No T/S/u/v/pressure continuation is created there.

The minimum missing information is a surface boundary/reconstruction consistent
with node-zero inventory, a supported bottom profile or a newly specified
boundary closure, and original WOA missing-input masks/donor lineage. Finite
checkpoint values alone do not establish observational support. Therefore
original physical-profile recovery and pressure consumption remain rejected;
the new inventory bridge and candidate static P0 Q are usable under their
explicit interpretation.

## 2026-10-01: accepted-state static evidence

The wet-layer histogram is `0:15433, 11:1407, 12:2294, 13:13321, 14:15065`.
Independent scalar `fsum` global water/IT/IS/Mu/Mv residuals are all zero;
the largest column stock residual/envelope ratio is `0.00767849`.
Every deep h/stock slot is unchanged. The candidate minimum wet h is
`2.2231689717 m`; the original accepted minimum is `0.00852074535 m`.
This change is geometric redistribution, not evidence of coastal stability.

| Selected column | Active layers | Discrete bottom z (m) | Water (m) | Unsupported bottom tail (m) |
| --- | ---: | ---: | ---: | ---: |
| Center | 11 | -750 | 747.508520745 | 250 |
| East | 12 | -1500 | 1497.601216904 | 500 |
| South | 11 | -750 | 747.743659701 | 250 |
| North | 12 | -1500 | 1497.868449907 | 500 |
| West dry | 0 | 0 | 0 | 0 |

For the complete center column, reference kinetic energy is `31.7504648277`,
same-M actual-h kinetic energy is `388.9481264543`, and candidate P0-remapped
kinetic energy is `34.3407167237 J/m2`. The separately reported conversion cost
is `357.1976616266 J/m2`, and remap change is `-354.6074097306 J/m2`.

Static east/south/north Q is respectively `331754.776219`, `67853.041169`,
`-302503.488039 m3/s`. All three have common bottom `-750 m`, 13 shared
segments, and four segments lacking original physical sample support.
Dry west Q is exactly zero. Diagnostic face lengths use archived dual widths;
production face metrics/operator equivalence is not asserted. These values
do not advance any state.

## 2026-10-01: reproduction and boundaries

Use the project-local venv and the existing dependency lock
`research/experiments/material_top_band/requirements.lock`. Synthetic tests
require no private inputs. On Windows the owned job enforces one CPU, 180 s
and 4 GiB aggregate private memory, including descendant interpreters.

```powershell
& .\.venv\Scripts\python.exe scripts/run_bounded_research_tests.py -q tests/test_material_real_geometry.py tests/test_material_top_band_dynamic.py tests/test_conservative_top_band.py -p no:cacheprovider -W error
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff check --no-respect-gitignore research/experiments/material_top_band/real_geometry.py research/experiments/material_top_band/accepted_geometry_audit.py tests/test_material_real_geometry.py
```

The private archive audit is reproducible only by an authorized holder of
the six exact files and historical git object. Supply private paths as local
arguments; they are omitted from the output. `acceptedDir` and `metadataDir`
refer to the two accepted/provenance archive directories, and
`historicalRepository` provides read-only git objects. A different file or
rejected-state packet fails identity validation before NumPy/bridge execution.

```powershell
& .\.venv\Scripts\python.exe scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.accepted_geometry_audit --accepted-dir "$acceptedDir" --metadata-dir "$metadataDir" --git-repository "$historicalRepository" --output scratch/accepted_geometry_scalar_evidence.json
```

Remaining work includes a deliberately specified pressure/surface/bottom
closure, coastal ALE/transport/source adaptation with full budgets/rollback,
and current-mass conservative biharmonic/filter operators. In particular,
`kappa_bi=2e14` capacity must be derived on the candidate's own volume and
shared wet conductance graph; historical 42/47 substep counts cannot be reused.
No biharmonic capacity experiment or real dt300/dt600 integration is performed
in this change. There is no new order, speed or mature-mode-quality claim.

## 2026-10-01: final validation and independent review

The final source-bound accepted-353 static archive audit passed in **4.029 s**
owned-job wall time, with peak interpreter RSS **267,702,272 bytes** and peak
aggregate job private memory **261,517,312 bytes**. Its six archive snapshots
and all original arrays remained unchanged. The sanitized output is
[`material_real_geometry_evidence.json`](material_real_geometry_evidence.json).

The full focused geometry/dynamic/conservative batch passed **127 tests in
70.39 s**, with **72.123 s** owned-job wall time, **104,157,184 bytes** RSS and
**93,954,048 bytes** job private memory. That batch preceded the archive's final
independent-ledger guards; the numerical bridge did not change afterward.
After those guards, all **47 final geometry tests** passed in **0.34 s**,
**2.224 s** owned-job wall time, **94,932,992 bytes** RSS and **80,596,992 bytes**
job private memory. Warnings were errors, and repository plus explicit new-module
ruff checks passed. Each batch enforced one CPU, 180 s and 4 GiB; this stage's
bounded computation remained under the 20-minute aggregate budget.

Regression counterexamples were observed failing before correction: masked
support loss, two-node/default-band indexing, boolean wet-mask handling,
unused finite dry-sentinel overflow, integer-stock precision loss, and NaN
selected-column water/stock/eta ledgers. The independent ledger now checks
finiteness before comparisons. The archive explicitly requires both expected
surface/bottom refusals for each selected wet column, the dry-face zero-Q
receipt, and a density convention bound to the verified historical source
(also checked against an explicit manifest rho0, if supplied).

The independent inspector returned a **clean code gate** at source SHA256
`74cc4e757955ca910f2d85ac0e2abb5ee0c2e2fbf0326cb4f9ddd785a9f7dc2a`
for the numerical bridge, `7934e18aafbe132bd744058d959e4781033c620a0748a17889ae95077673d1a5`
for the archive reader, and `47a50336872127d106a02a13cb3b5220c0041f9a952c262efb25cd3f7bc8f76d`
for the tests. The inspector performed static review and **two brief synthetic
probes only**, totaling **0.524 s** owned-job wall time, with peak RSS
**35,885,056 bytes** and peak job private memory **31,084,544 bytes**.
The inspector did not rerun the real archive, run a model, or run the test
suite. The implementing agent ran the quoted archive/test batches.

Geometry branch: codex/material-real-geometry-contract; implementation base: bf03285c870a354442884d45f70a089fe84640b8.
