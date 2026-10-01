# Fixed-window four-cell diagnostic evidence

## 2026-10-01: corrected delivery and measured one-hour run

An earlier response accidentally reported the already-completed PR6 reader fix
instead of this run. The actual one-hour execution was subsequently checked
against its process exit, twelve common commit markers and 48 checkpoint hashes.
Only observed results appear in `evidence_1h.json` and `evidence_1h.md`. This branch
does not change PR6, the solver, scientific fill, acceptance gates or archived data.

The registered cells are archived input I0 versus the frozen paired layerwise
donor sensitivity I1, crossed with original closed domain B0 versus removing the
southernmost row and moving the closed wall B1. Exactly 202 perturbed nodes remain
in the cropped domain; no donor reselection. The 296 common observer nodes are
fixed from the prior archived low-eta mask, not selected using the new results.
I1 is not a validated initialization; B1 is not an open boundary.

The source is historical commit `212df951c351f82dba74fbc43db5e52b0ad34c47`, not
this branch's current solver. Thirty-five original module hashes are recorded;
only a separate temporary material_top copy is instrumented. The first step's
six-field differences versus prior uninstrumented results were exactly zero in
all four cells. All 48 one-hour steps accepted; returned and attempted states
were byte-identical. Numerical checks passing do not establish physical validity.

### Review artifacts and privacy

- `protocol_1h.json`: public-to-this-private-repository derivative of the frozen
  local protocol, with original SHA retained. Personal paths are replaced by CLI
  arguments; exported runner hashes differ from original portable-path edits.
- `instrumentation_1h.patch`: recorded scalar-observation patch, with line endings
  normalized for Git; patched historical module bytes remain hash-identical.
- `evidence_1h.json` / `.md`: per-step scalar checks, budgets, effects and limits;
  no original arrays or donor coordinates/values.
- `original_artifact_hashes_1h.json`: relative names and private checkpoint hashes,
  permitting a data owner to verify local artifacts; checkpoints are not included.
- `publication_manifest.json`: identity of the exported code/protocol/evidence.
- `scripts/controlled_window/`: portable-input workers, Windows bounded supervisor,
  and historical source materializer. No production module is modified.

The input owner must separately supply the authorized private archive files,
donor provenance, frozen observer masks, crop contract and prior full-step
reference states/summaries. See protocol filenames and hashes. The repository
does not contain or obtain those inputs; consumers cannot read the owner's local
paths from cloud environments. Current WOA identities are recorded but do not
establish the missing historical raw-file identities.

### Reproducing preparation

Use a local environment with the existing model dependencies (measured environment:
JAX0.11.2, NumPy2.5.3, SciPy1.18.1; CPU float64). Point to an existing authorized
clone containing the historical commit. Materialize into a **new** private output:

```text
python scripts/controlled_window/prepare_source.py \
  --repository <clone> --protocol docs/controlled_window/protocol_1h.json \
  --output <new-private-source-root> \
  --patch docs/controlled_window/instrumentation_1h.patch
```

For six hours, use `protocol_6h.json` and append
`--patch docs/controlled_window/instrumentation_6h.patch` after the one-hour patch.
The preparer reads Git objects and applies patches only in its new private output;
it neither checks out nor resets the source clone. Both original and patched
module hashes are checked before numerical execution. `supervisor.py --help`
lists the explicit input arguments; it requires the exact protocol SHA and
`--execute-reviewed`. The supervisor is Windows-specific because it bounds its
own worker's affinity and process memory using Win32. It never manages other jobs.

## 2026-10-01: conditionally authorized next fixed window

`protocol_6h.json` registers **model total6h from t0**, 72×dt300 per cell, unchanged
dt_fast25×12 and scientific gates. Do not concatenate it with another trajectory.
The prior one-hour process exited with no retained continuously executed next-step
reference. A checkpoint hash cannot prove state/clock/ledger restart equivalence;
therefore no one-hour restart is used or claimed. The explicitly authorized
alternative is fresh-t0 execution. Original historical modules and scheduler,
fixed January2023 forcing arrays and scalar parameters are revalidated; all
step counters, model time and accumulated diagnostic ledgers begin at zero.
No hidden evolving forcing clock is introduced: this is a frozen-forcing sensitivity
window, not a six-hour time-dependent reanalysis integration.

The one-hour measurement gives ~2.85s execution and ~3.48s including per-cell
diagnostics/checkpoint per warm step. With 288 attempts and two ~115s compilations,
the central cost estimate is ~1233s (~21min); planning allowance25min.
The hard **whole-run30min** includes input validation, compilation, diagnostics
and I/O, with one logical CPU and4GiB global working-set **and** commit limits.
The private output volume needs8GiB free for retained checkpoints; use an
independent temporary directory with enough space, without deleting old evidence.
The first complete warm round records cost and forecasts the remaining registered
work with120s margin. A forecast over budget stops the common prefix; it does not
silently choose a shorter trajectory. A from-t0 fixed3h fallback is preregistered
only if evidence before launch makes6h infeasible; current measurements favor6h.

### Same geographic sections and regional volume ledger

Freeze two common **internal** latitude faces, original j3/j4 at62°S and j7/j8
at58°S, spanning all360 longitude cells (0.5–359.5°E). The control volume is
original rows j4..j7, centers61.5–58.5°S. Both cuts avoid the new wall at oldj1.
All14 original depth nodes are retained; adjacent wet intersections and minimum
adjacent reference thickness are hashed and checked identical after crop. Those
are diagnostic support identities, not a replacement transport geometry or
depth-clipped flux. The actual solver's returned **step-mean** column transports
are measured directly; no face flux is inferred from a single eta observation.

Northward latitude flux is positive; integrate each step by its300s duration.
Latitude faces already contain spherical cos(latitude); multiply by the zonal
equatorial arc width, not the local cos-weighted width again. East/west flux at
the shared periodic seam is measured once with each boundary orientation; its
net contribution cancels. Land-blocked faces remain included as zero transport.
Every geographic-volume ledger records all four sides, area-weighted eta change,
the independent integrated divergence and nontransport eta changes. The archived
eta equation has no explicit volumetric water-source term. Heat/salt/brine source
budgets remain separate and are not relabeled as water sources.

Check `ΔΣAη + dt(Qnorth−Qsouth+Qeast−Qwest) − ΣA(nontransport)` against local
continuity error plus a declared cancellation-scale roundoff allowance; record
both terms, never fit a source to close the ledger. An unexpected ledger anomaly
stops the diagnostic extension without changing the scientific acceptance gate.
A single cut does not uniquely explain regional eta or establish physical causality.

### Reporting and stops

Only after all four accepted attempts and resource checks does a round commit.
The first scientific rejection, memory/wall guard or instrumentation mismatch
stops the entire batch. Keep rejected attempted and returned states privately;
record six-field rollback when a result is available. A killed in-flight result
is unavailable, never reconstructed as a rejection. Effects use only the maximum
complete common accepted prefix, including the2h/4h/6h checkpoints if reached.
Report conditional input/boundary effects and interactions through time for the
target, fixed296, south/reference bands and constrained161; do not credit deleted152.
Record signed and cumulative same-face fluxes and the regional ledger, original
checks/schedules/source budgets, first-step numerical equivalence and measured
warm memory/cost. Do not claim30h repair or infer physical boundary correctness.

Independent review conveyed conditional approval for this one fixed window after
preparation. No merge/auto-merge, reset card, original Goal/session change or
raw-array upload is authorized by these scripts. New source/state files are
temporary private outputs only; repository deliverables contain scalar evidence.

## 2026-10-01: fixed six-hour window completed

The conditionally authorized fresh-t0 alternative completed once:72 common rounds,
288 accepted attempts,21600 model seconds, normal exit and no guard. The frozen
executed protocol SHA is `9d9e756162c8f19ac1d3e32bbfb178d3c44d13c4f11ccf83e697b7e9c716f148`.
All288 checkpoint hashes and72 markers were checked. The first-hour48 checkpoint
archives are byte-identical to the prior one-hour run, without claiming any
continuous reference past that interval. Integration stopped at the registered
six-hour endpoint. No three-hour fallback or additional window was run.

Total wall1286.657092s (21min27s), peak working set3,377,913,856bytes and commit
3,355,398,144bytes, affinity1. First warm-round forecast including120s margin was
1384.282763s, below1800s. Warm execution means were2.964/2.977/3.021/3.017s in
cell order I0B0/I1B0/I0B1/I1B1. All scientific gates remained unchanged: worst
continuity1.894318036e-15m, face mismatch5.684341886e-14m²/s, inventory roundoff
ratio0.0356580724, minimum wet thickness1.008804956m, maximum speed1.199773351m/s.
No rejection occurred, so rollback was not triggered in this window.

See `evidence_6h.md` for2/4/6h conditional effects, interaction and geographic
ledger, `evidence_6h.json` for all per-step scalar records, and
`original_artifact_hashes_6h.json` for private checkpoint/marker identities.
These files contain scalar diagnostics and relative hashes only, not arrays.
At6h the fixed296 means are −1017.114/−1018.195/−1074.131/−1075.544mm; the
boundary main effects at2/4/6h are −56.651/−81.762/−57.183mm. Input sensitivity
and interaction vary with time and geography; neither contrast demonstrates repair.

The complete regional ledger includes both identical geographic cuts and the
periodic east/west seam, nontransport changes and the explicit absence of a
volumetric water source in this frozen equation. Per-step residual sums differ
slightly from a final ledger recomputed from accumulated totals. The latter
maximum is0.00600432m³ versus cumulative declared allowances85.1–89.2m³ at6h;
the added floating-point accumulation allowance is disclosed separately and is
**not** a new or relaxed scientific gate. Saved initial/final eta at1/2/4/6h
independently reconstruct volume change, differing from summed step changes by
at most0.001953125m³. No source was fitted to close the ledger.

Independent read-only review of code, completed scalar results, effect formulas
and ledger accounting passed. Historical raw-input identity, initialization
qualification, closed-boundary physics and the30h failure remain unresolved.


## 2026-10-01 correction and fixed-snapshot follow-up

The296-point mean is a subset statistic, not the whole wet-domain mean. The85–89m3 ledger allowance above is a posthoc arithmetic supplement, not a preregistered scientific acceptance gate. Existing I0B0 t0/2h/4h/6h snapshots are analyzed without any new integration in [evidence_postprocess.md](evidence_postprocess.md), with the prospectively frozen postprocessing definitions in [protocol_postprocess.json](protocol_postprocess.json). Their narrow arithmetic checks do not establish physical qualification.
