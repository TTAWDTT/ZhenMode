# Verification Ladder for the Spectral Ocean Solver

> **📦 已归档（Archived）** — 2026-09-20 仓库整理时移入 `docs/archive/`。
>
> 本文属于**已退役的区域谱模式（regional spectral solver）**时期的工作、
> 过程性工作日志，或已被后续文档取代的早期版本。保留它只是为了留存历史推理链，
> **不代表当前主线**。
>
> 当前主线是**全球有限差分模式**（`src/jax_solver_global.py`，见
> [`docs/solver_technical_report_zh.md`](../solver_technical_report_zh.md) 与
> [`docs/decisions.md`](../decisions.md)）。文档索引见 [`docs/README.md`](../README.md)。


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

Beyond the linear Tier-1 checks, Tier-2 exercises the nonlinear and
dissipative pathways the linear tests cannot reach, using flows with known
analytic behavior and forcing through the *production* (viscous) physics.

### T2-1 — Inertial oscillation frequency (`bench_t2_inertial.py`)

On the f-plane with uniform velocity, flat surface and uniform density, only
the Coriolis term acts: uniform Laplacian/biharmonic = 0, uniform divergence =
0 (no gravity wave), no PGF, no baroclinic coupling. The model must therefore
rotate the velocity vector at the analytic frequency `f0`. Northern-hemisphere
rotation is clockwise, so the `atan2(v,u)` phase decreases at `-f0`.
Bottom friction is isolated (set to `none`) so it does not contaminate the
amplitude, and the residual amplitude drift is RK2 truncation (O(dt²)).

Result (single Gaussian-bump seed, 8 inertial periods, DT=200 s):

```
  f_theory       = 8.3651534630e-05
  |f_measured|   = 8.3709891604e-05
  rel_err        = 6.976e-04   (0.07%)   -> PASS (<1e-3)
  amp_drift      = 3.457e-04            -> PASS (<1e-3)
```

### T2-2 — Geostrophic adjustment (`bench_t2_geostrophic.py`)

A Gaussian SSH bump released from rest adjusts toward geostrophic balance.
Because the solver is a constant-f-plane model (`f0` everywhere in both the
`_coriolis_rotation` and RK2 residual paths), the adjusted steady state must
satisfy `f0·v = +g·∂η/∂x`, `f0·u = −g·∂η/∂y` pointwise in the interior strip
(away from the boundary current and the bump core). Checks correlation
`corr>=0.95` and a residual ratio `<=0.15` after a 12-day integration.

Status: **PASS** — definitive (12-day run, `logs/t2_geostrophic_final.log`):

```
  ACTIVE region (7912 pts, |v| > 0.1·max):
    velocity RMS          = 0.1840 m/s
    ageostrophic resid RMS= 0.0096 m/s
    residual ratio        = 0.0519
    geostrophic corr      = 0.9984          -> PASS (>=0.95)
  [full interior strip: vel_rms=0.1345 res_rms=0.0078 ratio=0.058 corr=+0.998]
  eta_peak decay: 0.796 m → 0.503 m over 12 days (monotonic, waves radiated away)
  edge sponge: ON (tau=3600 s, outer 15% band) — absorbs wrap-around inertia-gravity
    waves on the periodic domain so the interior genuinely settles to balance
```

**Regime note (why the sponge is required):** with the production `H_sw=4000 m`
the deformation radius `LR≈2367 km` exceeds the ~1166×1423 km periodic domain, so
the flow cannot separate into a balanced component. Shallow `H_sw=200 m`
(`LR≈530 km` inside the domain) develops velocity but, without an edge sponge, the
periodic boundaries recycle emitted inertia-gravity waves indefinitely (ratio
stayed ~2.8, `eta_peak` oscillated non-monotonically). Adding the Newtonian edge
sponge at test level (production solver untouched) lets the interior radiate and
settle — a standard wave-radiation treatment for bounded/periodic domains.

---

## Tier 3 — Real-data (observed sea-level) skill

The "is the integrated result solid" answer against real observations. See
`bench_t3_realdata.py` for the implementation.

### Methodology (defensible, not a straw-man)

The model is **barotropic** (single wind-driven gyre, depth-averaged, free
surface). Observed daily sea-level anomaly (SLA) at 30–90 day scales is
dominated by baroclinic mesoscale eddies and steric effects the model cannot
represent — so a direct pointwise SSH-vs-SLA RMSE would be dominated by that
structural mismatch and would be methodologically indefensible. The defensible
check is **anomaly correlation of the large-scale, wind-driven sea-level
pattern**: demeaned and spatially smoothed (>50 km, the smallest scale the
0.25° observations support) model SSH vs observed SLA.

### Data sources (auth-free, verified fetchable)

- **SLA**: NOAA CoastWatch ERDDAP `nesdisSSH1day` (RADS-based, daily, 0.25°,
  2017–present), downloaded via urllib `.nc` + local netCDF4 (netCDF4 inline
  OPeNDAP constraints fail on Windows with `OSError(-75)`).
- **Wind forcing**: NCEP/NCAR R1 monthly means (NOAA PSL, auth-free) via
  `real_wind_forcing`, mapped to the same calendar month as the SLA window.

### T3-1 — Wind-driven SLA anomaly correlation

Integrate the real monthly-mean wind to a quasi-steady wind-driven state
(30-day spin-up), time-mean the observed SLA over the forcing month, regrid to
the model grid, smooth both to >50 km, demean, then compute the spatial anomaly
correlation and RMSE. Evidence bar: `corr > 0.25` for the large-scale
wind-driven pattern (modest by design — this is *pattern consistency*, not
mesoscale/eddy skill, and that limitation is stated explicitly).

Status: **FAIL** — recorded honestly (30-day spin-up, `logs/t3_realdata.log`):

```
  spatial anomaly correlation : -0.386
  RMS anomaly diff (RMSE)     : 0.2599 m
  model std / obs std         : 0.0040 / 0.2584 m
  FAIL: corr -0.386 > 0.25 (evidence bar not met)
```

**Diagnostic interpretation (not a solver defect).** Re-inspection of the model
spin-up field and the observed SLA shows the failure is the *declared* barotropic
vs. baroclinic structural limitation, now made empirical:

- The model's wind-driven SSH setup is **~4 mm** (std 0.004 m) — physically
  plausible for a barotropic layer adjusted to NCEP monthly wind through the
  strong biharmonic closure. Its large-scale sign is *correct*: SSH high sits
  under the mid-latitude **negative** wind-stress-curl band (subtropical-gyre
  interior), as Sverdrup balance requires.
- The observed SLA in this window is **dominated by baroclinic mesoscale eddies**
  (std 0.258 m ≈ 26 cm; the field stays eddy-dominated even after the 0.5°
  smoothing — within-band std ~0.2–0.29 m at every latitude). A barotropic model
  cannot produce this structure or amplitude by construction.
- Correlating the ~4 mm smooth gyre (demeaned) against the ~26 cm eddy field is
  therefore dominated by the eddy pattern. The **negative sign is noise-driven,
  not a reversed-wind/gyre bug**; it reports near-zero physical overlap between a
  weak large-scale setup and a strong mesoscale observation.

The pre-registered evidence bar (`corr > 0.25`) was intentionally modest — this
is *pattern consistency*, not mesoscale skill. It is met honestly: **FAIL**.
Possible follow-up (flagged, not yet run): a much stronger spatial smoothing
(≈2–3°, hundreds of km) would isolate the gyre-scale wind-driven pattern from the
eddy field and give a fairer test of *that* component — but any bar of this kind
must be fixed *before* re-running, not after, to stay defensible (R1/R4).

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
