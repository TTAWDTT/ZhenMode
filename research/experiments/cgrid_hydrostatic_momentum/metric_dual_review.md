# Metric wet traces and actual paired dual Q: spatial qualification

Protocola3cad34 precedes tests/core edits; implementation4113bcc. Primary
state/source/force arithmetic remains unchanged. Current actual linear step
returns metric wet reconstruction AND full half-prism masses/paired Q and
includes their validity in acceptance. This is a necessary spatial interface
for nonlinear migration, not already implemented nonlinear momentum.

## Research, counterexample and repair

The protocol distinguishes normal integral/Piola mapping from physical
normal velocity, and stationary Cartesian cut-cell/dual mass theory from
moving spherical ocean dynamics. Project-derived arc weight w and its
primitive W correct the longitude-face transverse trace. A paired northward
interior bubble (T_E-T_W)*(mu-W) preserves divergence and zero external
normal correction. Existing vertical primitive and actual face Q unchanged.

Two old-code constant-U tests FAIL, with first sampled normal-speed relative
error about0.2013 on broad irregular latitude cells. This contradicts physical
constant-U interpretation, not the properly restricted prior mapped-flux
contract. New candidate passes on uniform/partial/land/moving tops and custom
radius, without a tolerance change. Independent Gauss contact/half-surface
quadrature and depth-break integration verify Q and center flux; JVP/FD
checks divergence. Omitting the north bubble is explicitly rejected.

Full north dual transverse east Q uses ARC halves=.5, while its mass and
vertical Q use actual spherical AREA halves. Northward center flux includes
(a-.5)*(F_E-F_W); combining it with old area-split transverse Q FAILS the
registered commutation control. Exterior wall halves remain in both stocks;
no bottom wedges removed. Independent physical sphere/half-prism integration
and local source/no-source32/64 actual-step increments pass. Shape/precision,
nonfinite/negative/closed flux and pole endpoints reject; there is no cosine
floor, clipping repair, arbitrary geometry source or hidden damping.

15 new regressions plus13 previous wet-trace tests pass. After final static
geometry validation and negative-control refinement, cleancore4113bcc runs
89 adjacent tests successfully in150.41s while other validation jobs run.
Earlier89-candidate/28-direct logs remain, not mislabeled as final-source runs.

## Actual short references and independent verification

At clean4113bcc SAME eight real ETOPO groups pass100x60s/BT4 with V/N/Q64
and momentum64/32. Source/input/snapshot hashes recorded at runtime. Largest
recorded normalized dual commutation4.43229e-16, wet bottom closure6.34720e-16.
Worst content budget4.95283e-18; volume change range-0.00537109375 to
+0.01318359375m3. No production, full nonlinear or longer climate integration.

Independent inventory/local-primary-Q verifier passes all8 and rejects9
existing corruptions. Host metric/trace/dual-mass/Q verifier passes all8 and
rejects5 original wet plus5 new dual/metric corruptions. It independently
integrates latitude arc halves by quadrature, checks center pairing/local
dual mass increments, snapshots and all recorded stage flags. Scope is LAST
snapshot and recorded flags, NOT independent every-step replay or full energy.

Four small source/precision fixtures have EXACT V/N/u/v in all16 comparisons
against the earlier immutable3a91aed baseline. Report hashes current sources
and that authoritative baseline manifest; do not promote equality to all
trajectories/configurations. Legacy MMS ALL PASS4.30 and isolated installed
active inventory64/Q64/momentum32/metric/dual coupling pass. The wheel is
loaded outside repo with installed paths asserted; own untracked build was
removed after resolved/tracked/reparse checks. Configured/research lint pass.
The specific full-suite handle completed exit0:498 passed,23 existing warnings,
514.28s. No duplicate test run was started because of an observation timeout.

## Scope and next physical implementation

The new returned mass represents ALL half-prisms, whereas D40/D41 rotation,
pressure and fast modes STILL use previous common-wet force masses. They are
not silently interchanged. The mapped field reproduces constant longitude-
normal U under the registered mapping, but is NOT a complete fluid point
velocity/ALE time reconstruction or full-vector accuracy proof. Latitude
pole topology remains unsupported here. Full dual mass/flux commutation does
not establish physical velocity basis, wall reaction or kinetic/buoyancy work.

Next implement actual nonlinear momentum only with specified wet/closed
velocity representation and wall reaction, paired mass/pressure/rotation,
C=sum(S^2/M) fast/layer correction for a DIAGONAL physical mass and moving
source/time coupling. Consider
whether a trace-compatible physical kinetic mass is diagonal or needs a
coupled basis; do not copy the new kinematic mass into forces by convenience.
If a non-diagonal kinetic mass is selected, its paired fast projection must
be re-derived; the diagonal column formula cannot be reused without proof.
Then complete actual EOS/real forcing/mixing/ice/initialization/checkpoint/
driver cutover, global geometry, century/sensitivities, independent climate/
forecast, fair GPU/distributed, whole adjoint/learning and engineering.
Industrial objective remains active and unmet; no indefinite unused kernel.

## Reproduction

At implementation4113bcc:
- python -m pytest tests/test_wet_flux_metrics.py tests/test_wet_flux_reconstruction.py tests/test_cgrid_hydrostatic_momentum.py tests/test_cgrid_pressure_work.py tests/test_momentum_shared_flux.py -q
- python research/experiments/cgrid_hydrostatic_momentum/run_reference.py
- python research/experiments/cgrid_hydrostatic_momentum/verify_reference.py --report results/industrial_alignment/cgrid_metric_dual_reference.json --negative-controls
- python research/experiments/cgrid_hydrostatic_momentum/verify_wet_trace_reference.py --negative-controls

Artifacts in results/industrial_alignment: metric_dual_before.log;
metric_dual_first_direct.log; metric_dual_direct.log; metric_dual_adjacent.log;
metric_dual_clean_adjacent.log; cgrid_metric_dual_reference.json/log and8 NPZ;
metric_dual_inventory_verification.log; metric_dual_independent_verification.log;
metric_dual_state_comparison.json/log; metric_dual_legacy_mms.log;
metric_dual_install.log; metric_dual_full_tests.log.
