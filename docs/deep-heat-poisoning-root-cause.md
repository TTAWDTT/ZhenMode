# Deep-heat poisoning: root cause

**Status: root cause identified.** Two independent defects, both now understood
(see the Defect 5 and "mixing sweep + one-step closure" sections at the end —
they supersede the earlier advection-only framing):

1. **Defect 5 (double-count)** — GM and Redi are the SAME operator, summed, so
   production runs the isopycnal skew flux at 2x the intended kappa.
2. **The downward heat pump** — convective adjustment + vertical diffusion + the
   skew flux all bury the surface bulk-flux heat in the abyss at ~100%
   efficiency, keeping SST ~0.8 K below the (pinned) T_atm and inflating the
   uptake rate ~10-40x.

The tracer advection operator is ALSO not discretely heat-conserving (its
vertical branch delivers a spurious heat flux; fixed under `project_adv_vel`),
but the one-step closure test shows that at the drifted state the residual OHC
drift IS the surface bulk source — i.e. the advective leak is no longer the
dominant term once projected. The observed warm-abyss / cold-upper dipole is
the signature of the downward pump acting on that surface source.

## The evidence

### 1. Advection alone does not conserve heat

Time-stepping the full solver from `ckpt_tenyr_ms_gm` with every tracer
closure off (`kappa_conv=kappa_h=kappa_v=kappa_gm=kappa_redi=kappa_bi=0`),
the bulk flux off (`lambda_bulk=0`), and no surface heat flux — so the ONLY
process acting on T is advection — the ocean's heat content changes:

| nstep | days | dH (ZJ) | rate (ZJ/yr) |
|-------|------|---------|--------------|
| 100   | 4.17 | −2.3349 | **−204.7**   |

In a closed domain with no-flux walls and no surface forcing, this must be
**exactly zero**. It is not.

### 2. The instantaneous operator integral is also nonzero

At the checkpoint state:

```
integral dV [ adv(T) ]  =  +66.578 ZJ/yr      (exact conservation requires 0)
```

while the same operator applied to a uniform field is annihilated:

```
max |adv(1)| over wet cells = 6.4e-20       (machine zero)
integral dV [ adv(1) ]      = 1.4e-9
```

So the operator is *internally* consistent — `sum_k (div_x + div_y + div_z)
= 0` pointwise for a constant tracer — and the failure is purely at the
domain boundary.

### 3. The failure is entirely in the vertical branch, at the top face

Per-level integral of `adv(T)`, and the decomposition:

| term | integral (ZJ/yr) |
|------|------------------|
| `div_x` (zonal)      | +0.000 |
| `div_y` (meridional) | +0.000 |
| `div_z` (vertical)   | **+66.578** |
| total                | **+66.578** |

Both horizontal branches telescope to machine zero, as they should
(periodic in x; closed N/S walls). The vertical branch does not, because it
carries the rigid-lid top-face closure
`Fz_top = Fz_in[:, :, :1] * T[..., :1]` (src/jax_solver_global.py:1074).

The top-face term alone accounts for the whole defect:

```
integral of -(Fz_top / dz) = -66.578 ZJ/yr
```

### 4. The per-level shape IS the observed dipole

```
k0  -52.4      k7   +7.4
k1   -0.6      k8  -14.6
k2  +13.8      k9  -20.1
k3  +18.4      k10 -25.5
k4  +16.8      k11 +20.3
k5  +15.8      k12 +63.0
k6   -4.0      k13 +28.2
```

Strong surface cooling, strong abyssal warming — the exact signature the
ten-year run shows. This is also why the 200-step ablations found the deep
trend *invariant to every mixing closure*: the source is in advection, not
in any closure.

## Why `Fz[0]` is nonzero

`Fz[0]` is the column-integrated horizontal divergence — the rigid-lid leak:

```
sum over wet columns of  Fz[0] * area  = +7.9e-09 m^3/s   (~0, net volume conserved)
sum over wet columns of |Fz[0]| * area = +1.3e+09 m^3/s   (NOT small at all)
rms Fz[0]                              = 1.28e-05 m^2/s
```

Net volume is conserved, but the *local* leak is large and spatially
correlated with SST. Because the net is ~zero, the heat defect reduces to

```
sum area * Fz[0] * T[0]  =  sum area * Fz[0] * (T[0] - Tbar)
```

which is nonzero exactly when `Fz[0]` correlates with `T[0]` — which it does
(warm surface convergence / cold upwelling).

The leak should be ~0: the barotropic subcycle enforces column-integrated
mass conservation every step. It isn't, because the two operators disagree:

- mass conservation is enforced on `ubt = sum_k u_interface_k * dz_norm_k`
  (src/jax_solver_global.py:783 `_barotropic_velocity`) — interface-centred
  `0.5*(u[k]+u[k+1])` averaging;
- the leak `Fz[0]` is diagnosed from `sum_k div_h(u_k) * dz_k`
  (src/jax_solver_global.py:521 `_vertical_transport_iface` → :475
  `_divergence_h`) — node-centred face-flux differences.

The depth-average of the divergence and the divergence of the depth-average
are not the same discrete operator, so the column-integrated divergence the
barotropic mode nulls is not the one the tracer budget sees. The residual is
`Fz[0]`, and the top-face closure converts it into spurious surface heat
flux.

## Consequence for the ten-year run

Retention of the applied surface heat, measured exactly over 200 steps:

```
SUM(Q dt) = +9.496 ZJ      dH = +3.948 ZJ       retained = 41.6%
```

58% of the surface heat input disappears. Over the full ten years the
archive shows the same thing at larger amplitude: `SUM(Q) = +4061 ZJ`,
`dOHC = +531 ZJ` — 87% unaccounted. Any equilibrium diagnosed from this
configuration (`gm_slope_max=0.005`, "frozen as production") is therefore
diagnosed against a broken heat budget, not a physical balance.

## Correction to two earlier conclusions

- The earlier framing that the ten-year run is "essentially conservative"
  (1.80 ZJ lost out of 21600 ZJ = 0.008%) was wrong. 1.80 ZJ over 200 steps
  is 79 ZJ/yr, which is the same order as the defect measured here. The
  error was treating a *rate* as a *fraction*.
- The earlier attribution of the abyssal trend to "wind-driven advection
  carrying the surface signal down" is half right: it is advection, but it
  is a numerical defect in the advection operator, not the physical
  wind-driven circulation.

## Also noted (secondary)

- src/jax_solver_global.py:1034 justifies donor-cell vertical advection
  with "added vertical diffusivity 0.5*|w|*dz <= 6e-3 m^2/s (below
  kappa_v=1e-5)". The comparison is inverted — 6e-3 is 600x *above* 1e-5.
  A two-to-three order of magnitude inversion of the intended diffusivity
  ratio. Worth re-deriving independently of the main fix.
- `_barotropic_velocity` (src/jax_solver_global.py:783) drops the bottom-most
  wet velocity: `u_avg = 0.5*(u[..., :-1] + u[..., 1:])` has nz-1 entries and
  weights by `dz_norm`, so the deepest level's u never enters `ubt`. For
  partial columns it also mixes in below-seafloor zeros.

## Correction: the leak is not what the stepper loses

The operator-level defect above is real, but it is **not** the dominant term
in the actual time integration. Instrumenting `_step_impl` stage by stage
(`_stage.out`, `_stage2.out`):

```
L half-step #1   dH  +0.00000000     (kappa_* = 0: identity on T)
N step           dH  -0.02306811     I(dT1)*dt = +0.00763743
L half-step #2   dH  +0.00000000
polar cap        dH  -0.00000000
mask/hold        dH  +0.00000000
TOTAL            dH  -0.02306811
```

The N step is the only sink, but the RK2 update `T_new = T + 0.5*(dT1+dT2)*dt`
is exactly linear, so `dH` must equal `I(0.5*(dT1+dT2))*dt`. It does not:

```
I(dT1)*dt         +0.00763743
I(dT2@old vel)*dt +0.00772523
I(dT2@pred vel)*dt -0.05343117     <- the real second stage
residual (dH vs I) -0.00000000     (the identity holds once dT2 is right)
```

The advection operator's heat integral is **numerically near-singular in
velocity**: the momentum update moves `u` by rms 2.6e-3 m/s (4% of the 6.2e-2
rms), and that is enough to swing `I(dT2)` from +0.008 to −0.053. The
operator is the residual of the large cancellation that `Fz[0]` embodies —
`I(adv(T)) = -sum_cells AREA*Fz[0]*T[0]` — so a small velocity change
rearranges two large terms. The stepper then integrates this ill-conditioned
tendency with RK2 and loses 0.0231 ZJ/step.

This is why the ten-year trend looked invariant to every mixing closure: it
is the advection operator's conditioning, and it enters through the RK2
second stage.

## Fix: project out the column-mean (barotropic) velocity

Advect with `u' = u - ubt`, the standard rigid-lid construction, so the
column integral of the divergence vanishes and `Fz[0] -> 0` by construction.
Normalizing by the **wet** column depth matters:

```
H_col = sum_k m_k*dz_node_k          (127.5 .. 5002.5, varies by column)
c     = sum_k u_k*m_k*dz_k / H_col
u'_k  = (u_k - c) * m_k
```

Using `sum(dz_node) = 5002.5` as the divisor (`shearL`, the earlier attempt)
leaves a residual because `H_col != 5002.5` in 49.8% of wet columns. The
`H_col` normalization zeroes `sum_k u'_k*dz_k` to 1.4e-14.

Measured, advection-only, 200 steps (`_zf0.out`, `_shearHcol.out`):

| variant | dH / 200 steps | rate | mass `I(adv(1))` | max&#124;S&#124; drift |
|---------|----------------|------|------------------|----------------|
| base    | −3.700676 ZJ   | −162.1 ZJ/yr | 1.8e-20 | none |
| `shearL` (÷5002.5) | −2.348138 ZJ | −102.8 ZJ/yr | 5.4e-20 | none |
| `shear` (÷H_col)   | **+0.152995 ZJ** | **+6.7 ZJ/yr** | **−3.4e-11** | none |

24x reduction, mass conserved exactly, tracers bounded.

**Rejected: zeroing the top face directly** (`Fz[...,0] := 0`). It is machine-
exact for conservation (dH = −0.000000 ZJ over 200 steps, `I(adv(1))` = 6.9e-20)
but it is not a valid discretization: it breaks the *pointwise* closure
`adv(1) == 0` at the surface cell, turning the top-face term into a
T-proportional source — precisely the pump this file already documents at
line 533 ("a 4%/step exponential pump"). Measured (`_zf0check.out`):
pointwise `max|adv(1)|` 6.7e-05/s at the surface (vs base 4.2e-21), and
`max|S|` runs 38.9 -> 148.4 in 60 steps. Global conservation is necessary
but not sufficient; the pointwise closure is what keeps the field bounded.

### Why the H_col projection is not exactly zero

`_algt.out` resolves it. `sum_k u'_k*dz_k` is 1.4e-14 (exact) and
`div_h(sum_k u'_k*dz_k)` is 1.7e-19 (exact), yet `Fz[0]` from
`_vertical_transport_iface` is rms 6.8e-6. The gap is `_divergence_h`'s
per-level wet gating (`_divergence_h` line 506-517): the gates are applied
*inside* the operator at each level, so pulling the k-sum through is invalid
at partial columns. I confirmed the mismatch is not a linearity failure —
`div_h(2.5*u) - 2.5*div_h(u)` is 5e-6, same order. Closing the last factor
means making the gate column-uniform (or excluding gated faces from the
k-sum), which is a larger change than the projection itself.

## Secondary findings (unchanged)

- src/jax_solver_global.py:1034 justifies donor-cell vertical advection
  with "added vertical diffusivity 0.5*|w|*dz <= 6e-3 m^2/s (below
  kappa_v=1e-5)". The comparison is inverted — 6e-3 is 600x *above* 1e-5.
- `_barotropic_velocity` (src/jax_solver_global.py:783) drops the bottom-most
  wet velocity: `u_avg = 0.5*(u[..., :-1] + u[..., 1:])` has nz-1 entries and
  weights by `dz_norm`, so the deepest level's u never enters `ubt`. For
  partial columns it also mixes in below-seafloor zeros.
- The 25.1% grid discrepancy: `p.H_sw = sum(grid.dz) = 4000.0` (13 interface
  spacings) but `sum(p.dz_node) = 5002.5` (14 node thicknesses). `H_sw` drives
  the barotropic mass/wind scaling; `dz_node` drives every tracer operator.

## A/B verification of the RK2 double-diffusion fix (2026-09-13)

Commit `fbdeaee` was written but **never exercised in a launched run** — the
cluster's `src/` was the 2026-09-10 copy, so every spinC/spinD run up to then
used the defect code. The current solver + runner (with `--freeze-adv-vel` /
`--conservative-kv` exposed, commit `2375166`) were staged to the cluster and
three runs were compared on an identical window (`ckpt_spinC_slope005x`, day
99645 → 118625, 52 yr, fp32, solo GPU, ~82 min each):

| tag | change | OHC start→end (ZJ) | drift | first-5yr slope | deepT drift |
|---|---|---|---|---|---|
| `spinD_f32` | defect code (baseline) | 33602 → 33903 | +5.79 ZJ/yr | +7.30 | +0.0015 C/yr |
| `spinE_fix_f32` | kv fix only | 33602 → 33348 | **−4.89 ZJ/yr** | **−11.87** | −0.0008 |
| `spinF_fix_frz` | kv fix + `freeze_adv_vel` | 33602 → 53454 | **+381.8 ZJ/yr** | +626 | +0.0857 |

**Result:** the fix removes a spurious ~+10.7 ZJ/yr heat source and flips the
ocean from warming to cooling. The separation is smooth and monotonic from yr 4
(E−D: −79 → −152 → −223 → … → −555 ZJ), i.e. a systematic source difference,
not multi-decadal chaotic branch divergence.

**`freeze_adv_vel` is catastrophic in production physics** (+382 ZJ/yr, deepT
+0.086 C/yr). Its earlier "partial win" had been measured with the bulk flux
off; there is no production case for it. Both flags stay DEFAULT OFF.

The fix is necessary but not sufficient: `spinE` still drifts −4.89 ZJ/yr. The
structural column-continuity leak (above) remains open.

## Defect 4 — the "equilibrium" is two ~700 ZJ/yr terms cancelling (2026-09-13)

Direct measurement at the `spinE_fix_f32` final state (bulk off, K=200 steps,
production physics) gives the **pure internal leak = −602.8 ZJ/yr**, with a
sharp vertical dipole:

| level | k0 | k1 | k2 | k3 | k4 | k5 | k6 | k7 | k8 | k9 | k10 | k11 | k12 | k13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZJ/yr | −359 | −200 | −150 | −123 | −85 | −45 | −2 | +16 | +88 | +117 | +91 | +35 | +10 | +3 |

The discretisation pumps heat from the surface layers INTO the abyss. This IS
the warm-deep/cold-upper dipole. The surface term (pinned `T_atm`) contributes
**+713.7 ZJ/yr**, so the two nearly cancel and the run only *looks* settled
(net +8 ZJ/yr). The earlier "−180 ZJ/yr" figure (Defect 3) came from short
windows on a less-drifted state; the leak grows as the deep stratification
poisons.

### Stage budget (1 step, bulk off, ZJ/yr)

| L1 (dt/2) | **N (dt)** | L2 (dt/2) | SUM = REAL step |
|---|---|---|---|
| −43.7 | **−618.6** | −43.4 | −705.7 |

The N step (`_explicit_full_step`) carries essentially all the leak. Its own
residual operator is conservative on the true state (`dT1` integral = −0.02),
so the leak is entirely in **stage 2**:

```
dT1 = resid(state)            # conservative
T_pred = T + dT1*dt
u_pred = u + du1*dt           # <-- FULL forward-Euler momentum predictor
dT2 = resid(u_pred, T_pred)   # <-- THIS leaks ~-619 ZJ/yr
```

`u_pred` is strongly divergent (the baroclinic PGF is undamped over a full dt
at the predictor), and flux-form advection is only conservative for a
divergence-free velocity, so `dT2`'s integral is large and negative. Freezing
the tracer stage-2 velocity at `u` (`freeze_adv_vel`) removes part of it
(−603 → ~−333) but not all, and by un-cancelling the surface term it makes the
net *warming* worse (+8 → +382 ZJ/yr) — hence DEFAULT OFF.

**Suspect fix (untested):** use the RK2 *midpoint* velocity `u + 0.5*du1*dt`
instead of the full predictor `u + du1*dt` in the tracer stage-2. A proper
midpoint stage is far less divergent. Tested next.

### The production "equilibrium" is a cancellation of two errors

Four configurations measured on the same state (K=200 steps, spinE final state):

| config | surface (ZJ/yr) | interior (ZJ/yr) | **net dOHC/dt** |
|---|---|---|---|
| (a) current production (pinned T_atm, freeze off) | +714 | −603 | **+8.2** |
| (b) freeze_adv_vel only | +714 | −68 | **+542** |
| (c) T_atm recentred to ocean mean, freeze off | 0 | **−456** | **−456** |
| (d) T_atm recentred + freeze_adv_vel | 0 | +79 | **+79** |

Reading:
- Current production's tiny +8 ZJ/yr is **not equilibrium** — it is a
  +714 surface source (Defect 2, pinned T_atm) almost exactly cancelling a
  −603 numerical sink (Defect 4). Remove either artifact and the run is far
  from settled: −456 (interior-only) or +542 (surface-only).
- `freeze_adv_vel` collapses the interior leak from −603 to −68 (a 9x
  reduction) — the leak scales with how far the stage-2 velocity departs from
  the true state velocity. But it leaves a +79 residual (the structural
  column-continuity leak) and, on its own, un-masks the surface source.
- No single flag fixes this. The interior leak (−603) and the pinned-T_atm
  surface source (+714) are **independent defects that happen to cancel**;
  each must be fixed on its own terms.

### Term attribution of the stage-2 leak (corrected scaling)

Integrating the residual operators as RATES (avoiding a spurious /dt in the
diagnostic) resolves the structure exactly:

| quantity | rate (ZJ/yr) |
|---|---|
| `dT1` residual @ true state | **−72** (small) |
| `dT2` residual @ predicted state | **−1188** (huge) |
| N = 0.5·(dT1+dT2) | **−630** (matches measured −618.6) |

and `_tracer_terms` at the predicted state gives `adv_T = −1188 ZJ/yr` — i.e.
**the entire leak is advection at the RK2 stage-2 predicted velocity**, with
every other term (diff_h/diff_v/conv/gm/redi) ~0. The column-integrated
horizontal divergence `Fz[0]` rms **doubles** from true u (1.99e-5) to the
predictor (3.69e-5). This is exactly the flux-form column leak
`−Σ AREA·Fz[0]·T[0]` of Defect 3, now attributed to the stage-2 velocity.

So the single root mechanism is: **`u_pred = u + du1*dt` (full forward-Euler
momentum predictor) is far more divergent than the true state, and flux-form
advection is not conservative for a divergent velocity.** Every candidate fix
(freeze the stage-2 velocity, use the RK2 midpoint `u+0.5du1*dt`, or make the
advection column-consistent) attacks this one mechanism; none completely
removes it, and all are entangled with the independent pinned-T_atm surface
source.

### Trajectory test — is `freeze_adv_vel`'s warming a transient or a runaway?

3000 steps (0.34 yr) each, from the spinE final state, production physics:

| step | freeze=False dOHC / SST | freeze=True dOHC / SST |
|---|---|---|
| 500  | +11.7 / 17.158 | +535.6 / 17.297 |
| 1000 | +18.7 / 17.177 | +537.8 / 17.326 |
| 1500 | +17.1 / 17.180 | +539.6 / 17.345 |
| 2000 | +17.1 / 17.172 | +542.8 / 17.346 |
| 2500 | +15.4 / 17.180 | +549.2 / 17.342 |
| 3000 | +14.3 / 17.178 | +555.8 / 17.339 |

- **Production (freeze off) sits at a STABLE but WRONG state**: dOHC ~+15 ZJ/yr,
  SST barely moving — the two-error cancellation holds over 0.34 yr.
- **freeze=True does not blow up, but does not settle either**: +540 ZJ/yr and
  *rising* over 3000 steps. SST rose only 0.21 K. This is the linear part of a
  long approach: OHC ~33600 ZJ needs ~3000 ZJ to lift SST 1.5 K to the pinned
  T_atm, i.e. ~5.5 yr / ~43000 steps at +540 ZJ/yr.
- So freeze's "+382/542 ZJ/yr" is neither a clean win nor a runaway — it is the
  surface source (Defect 2) marching the model toward the too-warm pinned T_atm.

**Complete fix needs BOTH defects addressed:** (1) the stage-2 interior leak
AND (2) the pinned T_atm. Fixing only the interior exposes the +540 surface
source; fixing only the surface leaves the −603 interior pump.

## Defect 5 — GM and Redi are the SAME operator, DOUBLE-COUNTED (2026-09-14, DECISIVE)

`_compute_tracer_residual` (jax_solver_global.py:1652-1675) computes BOTH
`gm_T = _redi_skew_flux_tendency(T, S_x_gm, S_y_gm, p, kappa=p.kappa_gm)` AND
`redi_T = _redi_skew_flux_tendency(T, S_x, S_y, p)` (kappa defaults to
`p.kappa_redi`), then `dTdt = ... + gm_T + redi_T`. Both calls use the SAME
operator with the SAME `_isopycnal_slope`, differing ONLY by the kappa scalar.
So production (kappa_gm=1000, kappa_redi=1000) applies the isopycnal skew flux
at **effective kappa = 2000 m^2/s** — exactly 2x.

**Proof (5-yr closure sweep, projection OFF, bulk ON, matched 4 yr, clean WOA):**

| run | kappa_gm | kappa_redi | eff. kappa | SST | deepT(1000m+) |
|---|---|---|---|---|---|
| spinN_gm0rd0 | 0 | 0 | 0 | 17.41 | 8.373 |
| spinO_rd0 | 1000 | 0 | 1000 | 17.24 | 8.393 |
| spinP_gm0 | 0 | 1000 | 1000 | 17.24 | 8.393 |
| spinQ_base | 1000 | 1000 | 2000 | 16.99 | 8.435 |

spinO_rd0 == spinP_gm0 **bit-identical to 4 decimals** — definitive proof GM==Redi.
A CPU probe (`_skew_probe.py`) showed the skew tendency rms scales exactly
linearly with the summed kappa (gm-only == redi-only == 3.366e-7, summed =
6.732e-7 = 2x). `_gm_tracer_transport` / `_gm_bolus_velocity` are DEAD CODE —
never called — so GM is not a bolus transport here, it is a second copy of Redi.

## Mixing sweep + one-step closure (2026-09-14)

**k-sweep, projection ON, matched window (`_drift_traj.csv`):**

| run | kappa_gm | kappa_redi | eff. kappa | drift ZJ/yr | deepT C/yr | SST |
|---|---|---|---|---|---|---|
| spinR_pj_k0 | 0 | 0 | 0 | +124 | +0.045 | 18.56 |
| spinS_pj_k1000 | 1000 | 0 | 1000 | +236 | +0.065 | 18.30 |
| spinT_pj_k2000 | 1000 | 1000 | 2000 | +405 | +0.101 | 17.91 |

**Mixing sweep** (projection ON, single skew kappa=1000 = spinS config):

| run | kappa_conv | kappa_v | drift ZJ/yr | SST @last |
|---|---|---|---|---|
| spinS_pj_k1000 (ref) | 0.05 | 1e-5 | +236 | 18.30 |
| spinW_conv0 | **0** | 1e-5 | +141 | 18.43 |
| spinX_kv0 | 0.05 | **0** | +194 | 18.41 |
| spinY_nomix | **0** | **0** | +125 | 18.58 |
| spinZ_alloff (kappa_gm=redi=v=conv=0) | — | — | ~flat @yr2 (OHC 21111, SST 18.74) | 18.74 |

Three comparable, ADDITIVE sequesterers: skew flux (κ), convective adjustment,
and κ_v. Removing any one roughly halves the drift; removing all of them drops
the drift toward zero AND lets SST rise to ~T_atm (18.74 vs 18.85; spinY 18.58,
production ~17.6). **The deep-heat chain is the surface bulk source being pumped
downward; the mixers are the pump, and the skew double-count is its biggest head.**

**One-step closure at the drifted state (spinK_proj50, κ=2000, proj ON):**
one `step_fn` from a copy of the spinK checkpoint gives actual +383 ZJ/yr vs the
single-state Euler residual estimate +528 ZJ/yr — same sign, same order (Euler
over-estimates ~1.4x at finite dt). SST_model 18.017 vs T_atm 18.851, gap
0.834 K: 0.834 x λ=40 W/m²/K x 3.16e14 m² = **333 ZJ/yr**, matching the actual
drift to ~15%. **So there is NO hidden time-integration leak at the drifted
state** — the OHC drift IS the surface bulk flux, sequestered downward at ~100%
efficiency. The fix is to stop the downward sequestration (single skew κ + the
mixer sweep), not to chase an advective leak.

**Candidate production config (NOT yet adopted):** `--kappa-gm 1000 --kappa-redi 0
--project-adv-vel` (= spinS_pj_k1000), optionally with reduced kappa_v/conv.
Do NOT change code defaults (both already 0.0 = closure off); the double-count is
a LAUNCH-CONFIG choice.

## Full mixing matrix, all PASS @50 yr (2026-09-15)

All 12 tags completed 18250 d with `VERDICT PASS` (max|u|~1.02, max|eta|~1.27,
0 NaN). Drift fitted on a **common yr10-49 window** (yr35-49 for the late-start
spinJ) from the checkpoint drift log; production/co-located GPUs are heavily
contended so wall times are not comparable.

| tag | kappa_v | gm | redi | conv | conv scope | yr10-25 | yr35-49 | deepT @49 | SST @49 |
|---|---|---|---|---|---|---|---|---|---|
| spinZ_alloff | 0 | 0 | 0 | 0 | col | +118 | +125 | 4.557 | 18.616 |
| spinR_pj_k0 | 1e-5 | 0 | 0 | 0.05 | col | +131 | +130 | 4.643 | 18.515 |
| spinAC_locc_twin | 0 | 1000 | 0 | 0.05 | **LOC** | +155 | +155 | 4.918 | 18.588 |
| spinAB_locc_kvh | 5e-6 | 1000 | 0 | 0.05 | **LOC** | +167 | +163 | 5.001 | 18.488 |
| spinY_nomix | 0 | 1000 | 0 | 0 | col | +166 | +168 | 5.021 | 18.552 |
| spinAA_locc | 1e-5 | 1000 | 0 | 0.05 | **LOC** | +176 | +171 | 5.075 | 18.431 |
| spinW_conv0 | 1e-5 | 1000 | 0 | 0 | col | +185 | +183 | 5.166 | 18.423 |
| spinX_kv0 | 0 | 1000 | 0 | 0.05 | col | +234 | +217 | 5.608 | 18.446 |
| spinS_pj_k1000 | 1e-5 | 1000 | 0 | 0.05 | col | +243 | +223 | 5.689 | 18.335 |
| spinAD_locc_k2000 | 1e-5 | 1000 | **1000** | 0.05 | **LOC** | +281 | +251 | 5.996 | 18.299 |
| spinT_pj_k2000 | 1e-5 | 1000 | **1000** | 0.05 | col | +415 | +328 | 7.158 | 18.147 |
| spinK_proj50 | 1e-5 | 1000 | **1000** | 0.05 | col | +414 | +327 | 7.154 | 18.143 |

**What the matrix settles:**

1. **κ-independent floor confirmed.** spinZ_alloff (every mixer off) still drifts
   **+122 ZJ/yr** — 45% of the spinS production candidate. So even with the
   sequesterers gone there is a large residual surface→abyss source. The mixers
   are not the whole story.
2. **Additivity holds, with the skew term dominant.** Single-lever removals from
   spinS (+223): kill conv → +183 (−40); kill κ_v → +217 (−6, small); kill both
   → +168 (−55). All three removals together (spinZ, plus gm=0) → +122 (−101).
   The ordering is **skew double-count > convection > κ_v**.
3. **Defect 5 is reproduced, not an artifact.** Redi=1000 on top of gm=1000
   (spinT/K, κ_eff=2000) nearly doubles the drift to +328 vs spinS +223; adding
   it under localized convection (spinAD +251 vs spinAA +171) shows the same
   ~1.5x step. **GM and Redi must not both be nonzero.**
4. **Localized convection is a real, modest win.** spinAA_locc +171 vs spinS
   +223 (−23%), spinAC +155 vs spinY +168 (−8%) — the column-wide pump was a
   genuine contributor but secondary to the skew double-count.
5. **spinX (kv=0, conv=0.05) > spinY (kv=0, conv=0)**: +217 vs +168, so residual
   convection still pumps ~50 ZJ/yr on top of pure skew.

**Remaining floor to attack:** spinZ's +122 ZJ/yr with zero explicit mixing is the
next target. It is *not* the skew/convection/κ_v chain (all off) — candidates are
the bulk surface flux's own downward penetration, horizontal/bolus numerics, or
the biharmonic ν. **This floor, not the mixers, now bounds how close to
equilibrium production can get.**

