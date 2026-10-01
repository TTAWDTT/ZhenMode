# Moving-stock dynamic research slice — 2026-10-01

This work starts from PR18 head `d276ab8a6a585904d2a2b6072c4789476a6ce8ce`
and lives on the separate `codex/material-top-band-dynamics` branch. PR18's
resting-column seam remains separate. The new method is
`moving_stock_predict_fast_replay_v1`; it does not claim equivalence to
`symmetric_fast_v3`, `material_step`, or the production step. Production source,
defaults and licensing are unchanged. All cases below are synthetic prototypes,
not benchmark evidence or industrial quality/speed qualification.

## Executable state and geometry

The callable is `research.experiments.material_top_band.dynamic.advance`.
`State.h` is full-column thickness and `State.n[..., 0:4]` contains
`IT, IS, Mu, Mv` per horizontal area. `T=IT/h`, `S=IS/h`,
`u=Mu/(rho0*h)`, `v=Mv/(rho0*h)`, with `rho0=1025 kg/m3`.

The domain is exactly 8×4×6, all wet, periodic x, closed y, flat bottom.
The initial reference thicknesses are `[2.5, 7.5, 12.5, 42.5, 235, 200] m`.
The top band bottom is fixed at `b=-22.5 m`; its three masses obey
`h_k=f_k*(eta-b)`, `sum(f)=1`. The remaining three h are fixed, while every
deep tracer and momentum stock participates in horizontal/vertical transport,
pressure and mixing. Geometry/scalar types, consumed areas and derived metrics
are validated. Invalid or masked inputs return a full entry snapshot.

## Shared flux and pressure contracts

Each horizontal face is split at the union of physical-depth interfaces from
both incident columns. On each common wet segment, P0 velocity defines
`Q = L*dz*(u_left+u_right)/2`. The same segment Q transports water and all four
stocks. Layer Q and column Q are independently summed from these segments;
matching is a measured diagnostic with refusal, never a Q repair.

Horizontal layer divergence is D per horizontal area. Relative downward ALE
flux has `R_surface=R_bottom=0`, `eta_dot=-sum(D)`. Fixed deep cells use
`R_k=sum(D_l, l>=k)`. Top cells use
`R_(k+1)=R_k-D_k-f_k*eta_dot`. Both recurrences agree at the band bottom.
Using the fixed-deep recurrence for all cells is not allowed. ALE interface R
also transports all four stocks, including across the band/deep interface.

Density is mean-preserving limited P1 in physical z under the declared linear
EOS. Pressure jump is the common-segment Pa average of the integrated P1 density
profile, evaluated with two-point Gauss quadrature. The fixed-mass pressure kick
uses `-C.T*pressure_jump`, where C is the same effective P0 segment flux map.
Work uses C applied to the actual before/after momentum midpoint velocity.
Independent entry-face impulse and midpoint work checks validate each kick
before its receipt can be aggregated. Different affine density slopes provide
a quadratic-pressure witness that distinguishes Gauss averaging from midpoint
sampling.

## State progression and active processes

One slow first half-step is followed by a full predictor on a copy. Its predicted
slow endpoint is diagnostic and discarded. The accepted state then executes
twelve pressure-half / joint-transport / pressure-half fast substeps. Each
substep commits exactly one complete h/IT/IS/Mu/Mv state. The accepted endpoint
receives the second slow half-step once. Replay does not add predictor or
fast-end momentum and does not repeat already accepted tracer advection.
Only a fully successful macro step exposes these commits; any later rejection
returns the entire entry state and clears committed receipts. Executed receipts
remain available for diagnosis.

Joint transport is SSP-Heun with conservative P0 donor fluxes and optional FCT
antidiffusion. The FCT envelope is the current global tracer range, not the
original solver's limiter. It modifies stock corrections without changing Q or
C. Constant-specific T/S/u preservation and water closure are tested directly.
Current-mass horizontal/vertical diffusion exchanges conservative stock fluxes;
its capacity bound includes convection at every potentially active interface,
even when the initial density profile is stable. P0 remap integrates
`overlap*M_old/h_old` on exactly the same eta/bottom domain; it preserves deep
stocks and is tested against nonidentity overlap values and kinetic nongain.

The nonzero witness retains horizontal/vertical tracer and momentum mixing,
FCT, convection capability, rotation, bottom drag, wind stress and exact bulk
heat relaxation. Convection capability is dispatched; a separate inversion
regression exercises its capacity contract. Active biharmonic or momentum
filter settings reject before dynamics because their adapters are missing.
No coefficient is silently disabled to accept a call.

Heat and external impulse are independently derived from the accepted
pre-source state and parameter laws. Receipts and returned inventory changes
must agree with that oracle. Matching forged stock plus forged source receipt,
malformed/NaN pressure receipts, overflowed momentum, late replay rejection and
invalid derived geometry cannot manufacture acceptance.

## Nonzero end-to-end witness and limits

The reproducible 30 s witness has moving eta, vertical shear, horizontal density
contrast, nonzero common-depth pressure work and cross-band exchange. It changes
eta by `6.8070786483e-6 m`, crosses the band at `1.9052832981e-7 m/s`, and changes
deep IT/IS/Mu/Mv by maxima
`[5.9730858766e-6, 6.9030647865e-8, 18.0425450568, 6.5039259398]` in their stored
units. Absolute accepted pressure work is `1.9226838880e11 J`.

Independent water residual is `-32 m3` at the global area scale; its ratio to the
fixed arithmetic envelope is `0.0034194`. Independent IT/IS/Mu/Mv ratios are
`[0.0032036, 0, 0.0124739, 0.0063172]`. These are cancellation/roundoff identities,
not physical truncation errors. The fixed envelope is 256 machine eps times
the relevant absolute stock/energy scale, below a relative 1e-12 allowance.

PE contains only rho0 free-surface PE plus rho-prime gravity PE. Total PE/ALE
compatibility is unproved. The report exposes total energy change and an
unclosed remainder after pressure work; it includes transport/ALE, mixing and
sources, and is not labeled numerical loss. This witness has an unclosed
remainder about `-7.5562e12 J` and is not energy qualification.

An independent compact-grid fast-only fixed-endpoint witness uses endpoint
2 s and macro dt `[1, .5, .25, .125] s`. Successive endpoint norms are
`[1.0554656809e-6, 2.6393364368e-7, 6.6394255946e-8]`, giving observed orders
`[1.9996325012, 1.9910449247]`. This supports second-order time behavior only for
that fast-only case. The full active subset uses first-order diffusion updates
and has no measured order qualification. No spatial convergence, mature-mode
error comparison, speed claim or actual failure repair is established.

Still missing: conservative biharmonic, original momentum filter, original
process clocks and full-configuration replay, coast/variable bathymetry,
general default-size domains, total PE/ALE closure, active-subset order and
official benchmark quality/speed evidence. No actual dt300/dt600 step, standing
wave/global integration, GPU computation or production entry switch was run.

## Reproduce and inspect

From the checkout, use a project-local virtual environment. The slice's tested
NumPy/pytest/ruff environment is pinned in
`research/experiments/material_top_band/requirements.lock`; the normal project
metadata already declares these dependencies. Windows Python 3.13.5 was used.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r research/experiments/material_top_band/requirements.lock
& .\.venv\Scripts\python.exe -B scripts/run_bounded_research_tests.py -q tests/test_material_top_band_dynamic.py tests/test_conservative_top_band.py -p no:cacheprovider -W error
& .\.venv\Scripts\python.exe -B scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.dynamic_evidence --output docs/material_top_band_dynamic_evidence_20261001.json
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff check research/experiments/material_top_band/dynamic.py research/experiments/material_top_band/dynamic_evidence.py
```

The Windows runner binds the owned interpreter and descendants to one CPU,
180 s wall time, and a 4 GiB aggregate Job Object memory limit. It launches the
actual interpreter directly, rather than measuring a venv launcher that hides
the real child. The JSON artifact contains deterministic synthetic summaries,
parameters, stage receipts, order metrics and numerical source SHA256. It has
no original/private state arrays, machine paths or credentials.

## 2026-10-01 final validation record

Final complete focused batch: **87 tests passed in 69.06 s**. Windows Job wall
70.712 s; peak actual interpreter RSS 100,999,168 bytes; peak aggregate job
private memory 91,811,840 bytes. The final source-bound nonzero/order report was
regenerated in a separate 56.759 s batch, with RSS 37,556,224 bytes and job
private memory 32,141,312 bytes. Both used one CPU and stayed below 180 s/4 GiB.
Repository ruff, explicit research ruff and whitespace checks pass. Reported
local bounded runs total less than eight minutes; independent review probes
also remain within the overall twenty-minute budget.

An independent inspector reviewed the numerical contracts and separately ran
18 adversarial tests (6.50 s pytest, 8.25 s wall). Review closed newly activating
convection capacity, overflow/NaN fail-open, malformed receipt/scalar rollback,
matching forged source stock/receipt, exact P0 overlap, and genuine quadratic
P1 pressure quadrature coverage. No production source or default was changed.

## 2026-10-01 correction and final receipt-scalar validation

The preceding paragraph's attribution of an independently run 18-test batch
is withdrawn. The independent inspector's final execution record is read-only
static review plus three brief NumPy audit probes, totaling at most 30 s.
The 87-test batch and all other quoted bounded test batches were executed by
the implementing agent; independent review remains a separate code gate.

Final receipt guards also reject arbitrary-precision Python integers before
conversion or arithmetic. Fifteen targeted malformed-input/receipt tests
passed in 1.57 s (3.323 s bounded wall; RSS 95,088,640 bytes, job private
81,563,648 bytes). The final source-bound evidence was regenerated after this
last change in 58.269 s; RSS 37,605,376 bytes and job private 32,231,424 bytes.
The nonzero and observed-order values remain unchanged. Total local validation
including independent probes is below ten minutes, within the twenty-minute
budget; this supersedes the earlier under-eight-minute subtotal.

## Receipt-only follow-up: independent bulk heat quantity

The finite `bulk_heat_J` field previously escaped independent validation even
when external IT stock and the source law were correct. A targeted counterexample
left the returned stocks unchanged and doubled only this heat receipt; the
macro step incorrectly accepted it. This was a report-contract defect.

`_slow` now compares the reported J value with the independently derived
pre-source `expected[0]*rho0*Cp`. Its arithmetic scale includes the magnitudes
of air/water temperatures before subtraction, and uses the existing 256-epsilon
roundoff envelope. Trajectory formulas, stage order and thresholds are unchanged.
First-half and final-replay forgeries reject with full snapshot rollback; the
latter executes twelve fast steps but exposes no commits. Normal nonzero bulk
heat and the existing nonzero complete-step control still pass.

Four targeted tests pass in 4.37 s (6.14 s bounded wall, one CPU, interpreter
RSS 95,895,552 bytes, aggregate job private memory 82,178,048 bytes).
Repository and explicit research lint pass. No full order run or scan was
repeated. The numerical JSON retains its original source hash/head; its added
`receipt_validation_followup` binds this narrow guard and its targeted tests
without relabeling the historical numerical witness as a new full run.
