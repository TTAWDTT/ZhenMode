# Shared spatial score correction review

2026-09-29. Protocols4ae565c/540bee9 precede implementationcbfbb0b.
Whole industrial scope and required actual nonlinear/production work unchanged.

## Evidence and actual behavior

Initial pre-change16 witnesses fail,17 older metric/gate checks pass. The old
equal-cell two-latitude witness gives bias1.5 instead of wet-area1.0 and RMSE
sqrt(4.5) instead of sqrt(3). Missing new signatures/metadata are API failures,
not16 independently failed physical solvers. Retain
`results/industrial_alignment/climate_scoring_before.log`.

Four additional same-reference witnesses fail before the guard is added:
matching grid/areas alone permits different reference values. Two further
masked/constant witnesses fail: underlying masked values were treated as data,
and summing constants through unequal weights creates a false finite pattern
correlation from roundoff. Masked wet values now remain missing; masked wet
weights reject. Anchored weighted means and filters preserve exact constants
without widening a tolerance or assigning a fictitious perfect correlation.
Original failures remain in `climate_scoring_reference_before.log` and
`climate_scoring_masked_constant_before.log`.

Actual internal `score_npz`, external `score_external_field` and MOM6 wrapper
use the SAME v2 spatial scores.66 adjacent tests pass (0.76s,18 existing
NetCDF/NumPy deprecation warnings) in `climate_scoring_final_66.log`. They cover
independent NumPy weighted formulas, spherical edge areas/coordinate order,
provided nonuniform weights, periodic coordinate-window oracle, land sentinels,
missing values, constant correlation, shape/weight guards, source metadata,
same-reference/domain gates and table labels. These are spatial diagnostics,
not a new full-model physical experiment.

`metric_definition=area_weighted_angular_box_v2`; both angular half-widths are
2degrees. Wet-only area-weighted filtering uses sparse windows, not a fixed
number of grid cells. It does not provide constant-km smoothing or a proof of
eddy/forecast accuracy. Actual `cell_area_m2` takes precedence; inferred
midpoint edges are labelled and are not claimed to equal arbitrary real model
geometry. Empty/missing regions are not dropped to improve the gate.

Current comparison gate rejects missing/different v2 domain/reference hashes,
unknown/mixed definitions and incomplete coverage. Legacy/legacy comparison
remains explicitly an internal historical diagnostic. A PASS is never labelled
as independent climate or century qualification. Archived scores are unchanged;
the original baseline NPZ is not present in this checkout, so no historical
candidate is falsely claimed rescored or improved by this change.

## Full-validation state and remaining requirements

Specific full-suite61319/PID38224 is running; launch hashes are saved in
`climate_scoring_full_start_hashes.json`, and the wrapper will capture/check
end hashes, exit code and time in `climate_scoring_full_result.json`.
Do not infer completion from the adjacent suite. Ruff/diff and YAML pass.

Separately, all eight existing actual paired100x60s references and independent
LAST-step aggregate audits now pass, with9 corruptions rejected. Their hashed
runtime files were not changed while live. This does not fix the actual
zero-incoming-momentum rain failure or qualify nonlinear moving kinetic/
buoyancy/source dynamics. Source f52737a witness stays authoritative.

The current SST reference remains `T_init`, now visibly labelled as initial/
shared-initial state, not certified independence. Existing arithmetic record
averages and endpoint heat/salt percentages are explicitly NOT time-bounds
multi-year means or closed physical budgets. Initial-state MLD remains an
initial diagnostic. Formal evaluation still needs independent observation/QC/
epochs, tune/holdout/restoration separation, calendar/sample bounds, raw and
declared filtered multi-variable/basin/depth/season/variability/circulation
metrics and uncertainty. Complete production/century/forecast/GPU/distributed/
whole-adjoint/learning/engineering requirements stay in the full roadmap.
See `docs/century_climate_acceptance_zh.md`; no industrial completion claim.
