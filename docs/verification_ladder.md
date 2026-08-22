# Verification Ladder for the Spectral Ocean Solver

This document records how the solver's physical accuracy is verified, following
the tiers that established numerical ocean models (e.g. POP, MOM, ROMS, MPAS-O)
use to demonstrate correctness before climatology skill is claimed.

The core principle: a physical model is only as trustworthy as its scrutiny.
We therefore move from *exact, analytic* checks, through *idealized dynamics*
with known solutions, up to *climatology-level* comparisons — each tier builds
on the one below it and demands evidence, not just a loss curve.

---

## Tier 1 — Analytic / manufactured-solution checks

The fastest, most decisive gate. Each test seeds a state with a known analytic
solution and verifies the discrete operator reproduces it to `~1e-8` (linear
operators) or a few percent (wave dynamics). Time cost: seconds to minutes.

| ID | Test | What it verifies | Threshold | Result |
|----|------|------------------|-----------|--------|
| T1a | Spectral derivative convergence | FFT-based `d/dx`, `d/dy`, `∇²` converge at spectral order across resolutions | worst err < 1e-8 | **PASS 1e-17** |
| T1b | Geostrophic balance | A balanced (U, eta) state does not spuriously accelerate (inviscid) | max\|dU/dt\| < 1e-3 m/s² | **PASS 3.4e-5** |
| T1c | Barotropic Rossby wave | Westward phase speed matches SW beta-plane dispersion `c = −βk/(k²+l²+f0²/gH)` | rel err < 10% | **PASS 1.44%** |
| T1d | Surface gravity wave | Barotropic mode speed `c = sqrt(gH)` | rel err < 5% | **PASS 0.54%** |
| T1e | Ekman / Sverdrup | Downwelling under cyclonic wind; southward barotropic transport; Sverdrup balance | all | **PASS (ALL TESTS PASSED)** |

### T1d note: why the test uses inviscid physics

The analytic surface-gravity-wave speed `c = sqrt(gH)` is a *linear inviscid*
result. The production `PhysicsConfig` carries Laplacian and biharmonic
stabilizers (`nu_bi = 1e12` etc.) calibrated for turbulent-closure stability.
Running the analytic test with those closures on, the measured speed shifts
**~5.9%** off the theory — an artifact of "running a linear inviscid check with
turbulent-closure physics," not a solver bug.

Isolated to inviscid physics the measurement is **0.54%**, and the residual is
the expected f-plane rotation correction `c_rot = sqrt(gH + f0²/k²)` (+0.31%).
This confirms the free-surface operator itself is exact; the viscosity is then
verified separately through the dissipation/energy checks rather than through a
linear wave-speed test.

The fix lives **at the test level** (`dataclasses.replace` on the test's physics),
not in the production defaults — the stabilizer is intentionally kept.

---

## Tier 2 — Idealized dynamics

Placeholder for the next step: canonical idealized flows with known behavior
(e.g. double-gyre wind forcing, spin-up circulation, inertial oscillation
frequency, geostrophic adjustment). These exercise the nonlinear and
dissipative pathways that linear Tier-1 checks cannot reach, and they must be
run with realistic (viscous) physics.

Status: **not yet implemented.**

---

## Tier 3 — Climatology-level skill

Placeholder: once Tiers 1–2 are green, compare model output (SST, SSH variability,
western-boundary-current position, eddy kinetic energy, stratification) against
observational climatologies (WOA, AVISO, OSCAR). This is where a reviewer
accepts real-world skill — and it cannot be claimed before Tiers 1–2 pass.

Status: **not yet implemented.**

---

## Running the suite

From `src/`:

```bash
python test_wave_speed.py   # T1d standalone (inviscid)
python test_forcing.py      # T1e standalone
python bench_accuracy.py    # full Tier-1 ladder (T1a–T1e)
```

`bench_accuracy.py` prints a per-test PASS/FAIL table and exits non-zero if any
test fails. Full output is logged to `logs/bench_accuracy.log`.

---

## Known open issue (not a Tier-1 defect)

A separate 90-day forced run develops genuine interior baroclinic instability
near i≈126–127 / ~30°N at ~2.5 months (quantitatively ruled *not* a seam
artifact). That is a long-horizon stability concern distinct from the accuracy
ladder above and is tracked separately.
