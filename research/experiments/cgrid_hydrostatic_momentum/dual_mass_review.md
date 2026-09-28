# Actual shared flux and moving-dual falsification

Registered BEFORE diagnosis/core edits in32a2548; implementation0eb56ed.
The previous implementation goal turn made concrete progress by exposing
actual shared Q and testing the ordinary-MAC hypothesis. The subsequent user
discussion of century proof did not qualify any longer or more complete run.
This review completes pending validation, without changing registered gates.

## Actual API and independent local audit

MomentumResult.fluxes is the SAME64 VolumeFluxes already passed to bounded
V/N transport. No second Q, endpoint reconstruction, damping, geometry change
or state repair. Four64/32 and explicit-source regressions fail before API
implementation, then pass;61 adjacent direct cases pass. Full suite at clean
0eb56ed:470 passed,23 existing warnings,441.80s. Legacy MMS passes with ratio
4.30; isolated installation imports installed modules and executes the active
step, checking actual Q and local V. Generated build artifacts were removed
only after resolved-path/tracked-file/reparse checks.

Same eight real ETOPO100x60s cases at clean0eb56ed pass, with unchanged
numerical results relative to the prior pressure repair. Hashed snapshots now
also contain previous V, actual last layer Q and fast mean Q. The current
independent verifier checks local stored-V increments, column sums, closed
faces and material boundaries, plus prior geometry/inventory/history metadata
and source/snapshot hashes. All eight pass; five metadata/hash and four local
flux negative controls reject. It independently recomputes final inventories
and the LAST local flux snapshot, not every recorded intermediate operator.
For representative disturbed groups local maximum V residual is about
1.53e-5m3; maximum64-eps inventory floors are1.41/1.57m3. The per-cell gate
also includes1e-12 of transported amount; maximum tolerance fraction about
0.00756. Floors account for subtracting large stored64 volumes, not relaxed
force, work or nonlinear physical gates.

Artifacts in results/industrial_alignment:
- cgrid_momentum_flux_reference.json/log and eight hashed NPZ snapshots;
- cgrid_shared_flux_before.log, cgrid_shared_flux_direct.log;
- cgrid_shared_flux_full_tests.log, cgrid_shared_flux_verification.log;
- cgrid_shared_flux_legacy_mms.log, cgrid_shared_flux_install.log.

## A proposed mapping fails, not a completed nonlinear solver

Actual steps use zero initial velocity/density, g=f=0 and explicit rain;
four clean0eb56ed cases share the preregistered geometry, dt60 andBT4.
Independent physical interval/half-sphere integration agrees with current
core dual masses. Scalar local source accounting and step validity pass.

| Geometry/source | Ordinary east/north relative change residual | Result |
| --- | --- | --- |
| Regular/uniform rain |4.64e-16/9.56e-16|PASS control|
| Regular/isolated rain |1/1|FAIL hypothesis|
| Irregular partial land/uniform rain |1.22e-15/1.43e-15|PASS control|
| Irregular partial land/isolated rain |1/1|FAIL hypothesis|

For isolated rain, actual common-wet dual mass change is zero: the neighbor
surface still limits the shared wet interval. Ordinary MAC predicts positive
1.45377e12m3 per component on regular geometry and4.14407e12m3 on irregular
geometry. This is not roundoff, nor evidence that implemented nonlinear
momentum has failed: nonlinear momentum is still absent from this linear API.

The independent kinematic delta(beta*V) identity passes when support exchange
is included. Its isolated-case UNFLOORED relative residual can also be1 because
the true change is zero; absolute local residuals0.00024--0.01270m3 are within
registered stored-inventory rounding floors. Do not report a tiny relative
identity error in those zero-change cases. Large geometry-exchange terms
cancel contraction, but cannot be injected as an arbitrary momentum source.
A separate scalar-index-loop wet-volume reconstruction rechecked four NPZ
hashes, actual Q/source accounting and physical masses; eight deliberate
exchange corruptions of1e8m3 reject. Its audit log is
cgrid_moving_dual_mass_independent.log. Original and clean diagnostics remain:
cgrid_moving_dual_mass.json and cgrid_moving_dual_mass_clean.json, their logs
and hashed snapshots. Overall registered ordinary-MAC hypothesis remainsFAIL,
all scalar/kinematic positive controlsPASS.

## Consequence for the actual mainline

Do not average the old Q onto old min-height masses and add nonlinear
convection. Select a physically specified moving staggered support, with
geometric boundary fluxes/complementary wet regions or a consistent new
coordinate representation. Pressure, Coriolis and fast/layer projection must
change together if their mass/face-area relation changes. Thin-face force,
local continuity, work, accuracy and adjoint gates must all be rechecked;
kinematic mass closure alone is insufficient. Actual production EOS/sources,
mixing/ice, initial/restart/driver migration remain required, followed by
global topology, century/sensitivities, independent climate/forecast, fair
GPU/distributed and whole adjoint/learning. No industrial completion claim.
