# Deep-heat poisoning: root cause

**Status: root cause identified.** The tracer advection operator is not
discretely heat-conserving. Its vertical branch delivers a large spurious
heat flux whose signature is exactly the observed warm-abyss / cold-upper-
ocean dipole.

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
