# Actual nonlinear candidate: progress with real-reference FAIL retained

2026-09-29. Method registered1c2572d; implementatione085cd8, pinned verifier
d6dcf63. Full industrial objective is NOT achieved.

## Implemented actual mainline candidate

`src/nonlinear_dynamics.py:nonlinear_momentum_surface_step` advances3-D dual
momentum m*u, simultaneously solved moving surface and extensive V/N using
the SAME actual Q. Central/upwind momentum, spherical curvature/rotation,
common-contact held pressure FORCE, positive incoming velocity/kinetic flux,
negative bulk removal, inelastic mixing, wall impulse and momentum cast work
are explicit. Failed input/solve/FCT/work gates return invalid, not a repaired
energy state. The kinetic norm is FV full-half-prism, explicitly DIFFERENT
from the historical paired physical-field L2 norm. No old acceleration is
silently divided by new mass; pressure, time and rotation were rederived.

The production CLI remains legacy. Full EOS/buoyancy PE, mixing, real sources,
ice, initialization/restart and global/production cutover remain incomplete.

## Evidence

- Missing new API first fails collection, then28 final direct tests PASS42.07s.
  NumPy face loops include ALL horizontal/vertical fluxes and independent
  source maps, weighted-time identity, momentum rows, wall reactions and
  kinetic/surface/source work. Four deliberate defects reject; both actual
  nonlinear time refinements and fixed-physical-site metric refinement pass.
- First independent rotation audit used a different Earth rotation input;
  matching authoritative7.2921e-5 fixes input, not gate. Initial spatial test
  changed latitude scoring domain with grid resolution and FAILS2.891/3.404;
  original log retained. Register fixed seven nested locations before rerun;
  unchanged3.3 ratio gate passes without altering dynamics.
- Isolated installed wheel imports from an independent temp directory;
  same old rain geometry with top7m,600s,1e-6m/s and momentum32 dilutes correctly
  to0.09999143079604789m/s for stored float32 initial0.1. This is only the
  declared isolated-source/curvature-off test, not full nonlinear axial skill.
- Eight real180x66x14 groups run at cleand6dcf63 with unchanged launch/end
  source hashes. FOUR disturbed10x60s histories PASS; FOUR near-rest cases
  FAIL at the first step. Registered overall report remains **FAIL**.
- NumPy-only last-step audits of four disturbed cases PASS: worst actual
  momentum relative7.65344e-16; energy change/work relative4.57217e-16.
  Eleven deliberate state/time/Q/force/reaction/inventory/work/metric corruptions
  reject. This does NOT independently replay every step or derive full EOS PE.
- Independent near-rest probe identifies surface storage work, not an
  unexplained remainder: solved eta~6e-18m is lost on64-bit top volume.
  Measured surface representation work accounts for raw energy FAIL;
  unexplained8e-33 is below original5e-30 tolerance. No gate/core edit made.
- Preregistered compiled scalar precision witness: plain addition loses the
  entire registered source at100k/1M steps; compensated variants match
  Decimal exact binary target on signed fixtures. This is NOT an ocean run.
  On installed JAX0.11.2 scalar barrier gradient4 and vmap execute, unlike
  the current online documentation's stated transform limitation; this
  runtime observation does not qualify whole-model gradients/hardware.
- Legacy MMS ALL PASS, meridional ratio4.30. Specific full-suite56743/PID46616
  is terminal exit0:626 passed,18 existing deprecation warnings,490.27s.
  All src/test/pyproject launch/end hashes match. Ruff, YAML and21 checked
  README/acceptance/review local links pass. These tests do NOT undo real FAIL.

## Evidence paths and next action

`results/industrial_alignment/nonlinear_dual_corrected_direct_tests.log`,
`nonlinear_dual_real_10.json/log`, `nonlinear_dual_real_partial_host.json`,
`nonlinear_surface_storage_probe.json`, `nonlinear_surface_compensated_probe.json`,
`nonlinear_surface_storage_transformations.json`, `nonlinear_dual_installed.json`.
Source manifests and immutable snapshots stay unchanged.

Next follow [registered storage research](surface_storage_precision_protocol.md):
choose compensated physical state/geometry/Q/FCT/restart against explicit bounded
storage diagnostics, rather than make failing reports green by widening energy
tolerance. Repeat new eight-case evidence after a registered coherent repair.
Then complete buoyancy/thermodynamics and production plus ALL century/sensitivity,
independent climate/forecast, GPU/distributed and complete adjoint/learning/
engineering requirements; no narrow candidate PASS may finish the whole goal.
