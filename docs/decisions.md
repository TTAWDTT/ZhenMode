# Decision log — global FD solver

Why the solver looks the way it does: the failures that were diagnosed, the
measurements that settled them, and the choices that were rejected. The source
keeps the resulting invariant in a line or two and points here by D-number.

Sections are thematic, not chronological (`docs/archive/TIMELINE.md` has the
chronology). When a number below is quoted, it came from a run recorded in
`results/` or a doc cited in the section.

## D1 — Meridional derivative at the N/S walls: mirror ghost, not one-sided

`_d_dy` originally used the one-sided extrapolation stencil
`(-3u0 + 4u1 - u2)/(2 dy)` on the boundary rows. It is unbounded for advection
and drove a tracer blow-up (T → thousands on the boundary row). It was replaced
by a mirror ghost cell (`mode='edge'`: ghost = boundary value), which gives the
wall row a stable central difference and zero normal diffusive flux. The
advective flux through the wall is killed separately by the normal-velocity
mask (v = 0 on the boundary rows, applied in `_step_impl` and
`_free_surface_step_fd`).

## D2 — Spherical mass conservation: meridional flux needs cos(face)

Cell area on a lat-lon grid is `A_ij = R²·cos(lat_j)·dφ²`, so mass is an
area-weighted sum and divergence must telescope under that weight.

- Zonal term: with the per-cell `inv_dx_i = dy/A_i` the weighted sum
  `Σ_i A_i·div_x_i` telescopes exactly (verified: machine-zero residual).
- Meridional term: the scalar `inv_dy = 1/(R·dφ)` carries no `cos(lat)`, so
  `Σ_j A_j·div_y_j` does NOT telescope. That was a per-step mass leak of
  ~1.7e-7 m³ per unit velocity, which grows exponentially and drove the eta
  drift into the 15 m watchdog by day 9. The spherical form
  `div_y = (1/cos_j)·d(v·cos_face)/dy` with `cos_face = 0.5·(cos_j+cos_{j+1})`
  telescopes under the area weight (verified: machine-zero, vs 1.7e-7 before).

## D3 — Energy-neutral free surface: the gradient must be the exact adjoint

The forward-backward free-surface pair is energy-neutral only if
`<u, ∇η>_A = -<η, ∇·u>_A` holds discretely. The centered `_d_dx`/`_d_dy`
gradient is not the adjoint of the conservative divergence — it ignores both
the open-face gating and the per-cell `inv_dx` weight — and pairing the two
injected energy at basin scale: a pure free gravity wave grew ~22× in 40
steps. `_gradient_conservative` is the exact adjoint: a FACE-DIFFERENCE stencil
(the face AVERAGE was an earlier wrong guess) weighted by the cell `inv_dx`,
meridionally by the face cos and divided by the cell cos.

## D4 — Coastal flux gating for the 3D skew-flux divergence

`_divergence_conservative_3d` / `_gradient_conservative_3d` gate every face on
both adjacent cells being wet. The bare `_d_dx`/`_d_dy` divergence of the
isopycnal skew flux reads the land zeros inside its stencil and injects a
spurious coastal source; that drove `max|T|` from 30 to 47 C in 4 days. With the
gate, no flux crosses a coastline and integrated tracer is preserved.

## D5 — Laplacian face gating at wet/ghost faces

The bare central stencil reads the mask step at the coast directly. Ghost nodes
hold the `T_ref` sentinel (+15 C vs ~+1 C real deep water), so the inner
Laplacian spikes (+14 K/dx² at every wet/ghost face) and the OUTER Laplacian of
that spike is a large dipole. At `kappa_bi = 2e14` (an L-step term, invisible to
`terms_fn`) this warmed the wet coastal nodes toward +15 at ~+0.3 K/d
(`gpu365_fgate`, d150, k13: T[x=138] had drifted 1.2 -> 12.1 C) while advection
of the resulting zonal gradient cooled the interior at up to -13 K/d; by d365 it
was FAIL_DRIFT with `max|T| = 220`. Gating the face differences (zero across any
wet/ghost or wet/dry face, open iff both cells wet) makes the Laplacian see a
flat profile across closed faces — the physical no-flux BC — and removes the
halo at the source.

The earlier one-sided `(-2, -5, 4, -1)` y-stencil at the wall extrapolated and
amplified the same grid-scale mode that blew up the tracer on the boundary row;
it was replaced by the mirror ghost cell (ghost = boundary value), which gives
`d²u/dy²|_0 = (u1 - u0)/dy²` — zero normal gradient, stable.

## D6 — Face-flux divergence: the w diagnosis must see what advection sees

On fully-wet nodes the face-flux divergence and the central difference are
algebraically identical (the face averages telescope). At coastal-wall nodes
they disagree: the central form measures the node's SHEAR (`0.5*(u_east - 0)`,
with the land neighbour's u = 0), the face-flux form its real INFLOW through its
open face (`0.5*(u_node + u_east)`).

With the central form the w diagnosis and the tracer budget saw different
velocity fields at every wall. At the Peru-corner node (313.5E, 0.5S, k=11) a
~0.28 m/s depth-uniform inflow entered the tracer budget through the open east
face while the diagnosed w saw ~zero local divergence, so no vertical branch
existed to close the budget and S piled +0.7 PSU/step (38.6 -> 41.3 by step 8,
68 by step 30). The face-flux form makes w integrate exactly the convergence
advection sees: column convergence grows w upward to `w[0] = -∫div dz`, which is
the rigid-lid leak the barotropic subcycle is simultaneously absorbing, and the
Fz_top closure passes it through the surface node.

## D7 — Vertical transport from the exact discrete divergence inverse

The previous node-based w (trapezoid `div_avg` + cumsum + `_dealias_h_fd`) could
not invert the discrete divergence. The leftover residual reached 1.2e-5/s, and
`resid * T` is a T-PROPORTIONAL source — a 4%/step exponential pump (measured
`resid*S = 4.4e-4 PSU/s`; `max|T|` 29.6 -> 52.6 over 30 steps at the North
Brazil Current node). No CFL limit can remove it, because it is not a transport
term at all.

`_vertical_transport_iface` now returns the exact discrete inverse of
`_divergence_h`, cumulated from the seafloor up, weighted by the per-node cell
thickness `dz_node` (not `dz_3d`: that array is interface-centred, length nz-1,
and cannot multiply a per-layer divergence).

## D8 — Seafloor ghost fill for the closure stencils

Ghost layers below the seafloor hold `T_ref`/`S_ref` (15 C) from `init_state` and
never evolve. A centered `_d_dz` spans k-1..k+1, so at sills (bottom wet layer
k=12) it mixed that reservoir into the bottom wet layer: a spurious ~0.08 K/day
seafloor heat flux in the GM/Redi closure, and a NEGATIVE (inverted) `drho_dz`
that made the closure read "convective" at ~11.7k columns and pinned the
isopycnal slope at its clip. `_fill_ghost_bottom` replicates the bottom wet value
downward (the standard MOM/NEMO no-flux seafloor treatment): stencils then see a
zero gradient across the floor and the one-sided bottom derivative gets the
correct, stable sign. The background diffusion/convective BC is left as-is; its
ghost pull is ~0.006 K/day (separate, smaller issue).

One edge case worth stating: `_fill_ghost_bottom` locates the bottom wet
level as `max(k where wet)`, which is 0 for an entirely dry column, so such a
column is filled with its own node-0 value rather than left alone. Every caller
masks the result with `wet_mask_z` (or gates the face on both cells being wet),
so a dry column's fill never reaches the solution -- but the operator itself is
only meaningful on a column with at least one wet cell.

## D9 — Vertical diffusion must telescope (interface-flux form)

The node-form `_d2_dz2` integrated with `dz_node` weights does not telescope on
the non-uniform grid: its raw volume integral is ~-4.4e13 on the production
field, and pure `kappa_v*_d2_dz2` repeated 30x leaked -54 ZJ/yr. Vertical
diffusion can only redistribute tracer within a column, so this is a genuine
defect, not a small error. `_d2_dz2_flux` uses the interface-flux form, whose
column sum telescopes to `F[bot]-F[top] = 0` for ANY kappa and ANY field.
`conservative_kv=True` selects it in both the L half-step and the N residual;
the default (False) keeps legacy bit-exact traces.

The Strang split needs the matching SUBTRACTION. `_compute_tracer_tendency`
includes `kappa_v*d2/dz2` (it is a physical term), and the linear half-step
applies it over `dt/2` twice, so the N residual has to remove it or the step
runs vertical diffusion at 2x. Measured coefficient of `kappa_v*d2/dz2` per step
before the fix: 1.88-2.02 (and 2.12 for the momentum `nu_v` analogue). The
biharmonic is the mirror case: it is an L-step-only term, so it must NOT appear
in the residual -- a `+kappa_bi*biharm` there would be re-applied by `N(dt)` and
cancel the L-step damping exactly (`-dt/2 + dt - dt/2 = 0`), making it a silent
no-op.

## D10 — Convective adjustment must telescope, and should be localized

The node form (`kappa_conv * mask * _d2_dz2`) is not heat-conservative: a
quadratic-T unit test gives a column integral of -6.0e-3 (linear T conserves
only by accident). Gated by the time-varying full-column mask, every convective
episode therefore created net heat: +0.00025 K/day global mean,
polar-concentrated (+0.09 C/yr volume mean). Over the 10-yr run that homogenized
and warmed the polar columns by +1.13 C (below 2000 m: +7.4 C) with an
impossible implied +200 W/m² surface flux. `_conv_flux_tendency` reuses the
GM/Redi interface discretization, which conserves exactly for any mask and is
negative-semidefinite. Its top/bottom-node effective rate is half the old mirror
form's `2(C1-C0)/h0²` — same no-flux BC, standard FV discretization — so the
adjustment timescale changes only at the surface/bottom nodes.

`localize_conv=True` additionally gates the adjustment per-interface instead of
column-wide. The column-wide form is a downward heat pump: one unstable
near-surface interface forces abyssal mixing at `kappa_conv` (0.05 m²/s = 5000x
`kappa_v`), which the per-term decomposition shows pushing +300 ZJ/yr into each
of the three deepest layers while cooling the surface. Per-interface gating
confines the adjustment to where the instability actually is, so it can
redistribute but not ventilate the abyss.

## D11 — Hydrostatic pressure: mask the ghost water before integrating

`_compute_hydrostatic_pressure` zeroes the density anomaly below the seafloor
with `wet_mask_z` BEFORE the vertical integration, so ghost layers contribute
`dp = 0` and the cumulative pressure stays constant beneath the bottom. Without
that, columns of different ghost-water length had different bottom pressures and
the resulting spurious horizontal gradient blew up at steep topography. Fixing
the 3D PGF alone only delayed the day-50 equatorial-Atlantic blow-up by 5 days;
closing the coastal injection path in `_compute_pressure_gradient` (D4) removed
it.

## D12 — Mode split, and right-sizing the nu_h subcycle count

With `mode_split=True` the free surface leaves the Strang linear half-step and
runs as `n_subcyc` barotropic forward-backward subcycles of `dt_bt` at the end of
`_step_impl`, with the density-PGF coupling held fixed over the baroclinic step
(MOM-style forcing lag). That lifts the external-gravity-wave CFL off the
baroclinic dt — dt=3600 s is then safe at 1 deg — for ~10x wall-clock.

`nu_nsub` right-sizes the nu_h subcycling inside the split L half-steps. The
legacy default (`n_nu = n_subcyc` = 24 substeps at dt=3600 / dt_bt=150) puts the
CFL LHS at 0.136, a 3.7x margin under the 0.5 FTCS bound; the right-sized count
is ~6 substeps at the same margin, i.e. 4x fewer Laplacian pairs per half-step.

`nu_h` must act on the FULL 3D baroclinic velocity, not on the depth-averaged
barotropic state. The subcycle carries no `nu_h`: the baroclinic shear (which
`nu_h` is really there to damp) is invisible to the depth mean, so putting it
there damped only the depth-mean flow while the 3D shear grew unchecked
(real-grid +0.65 m/s per step at a trench node, 154.5E 11.5S k=12) and blew up
through tracer advection by step 15. It instead runs in the L half-steps on the
full 3D field, subcycled `nu_nsub` times, which satisfies the explicit diffusion
CFL while acting on the shear.

## D13 — Stage-2 advection velocity and the column heat leak

The raw stage-2 predictor `u_pred = u + du1*dt` has `col_div(u_pred) = O(dt)`,
because the barotropic subcycle projects only the FINAL u — after the tracer
stage. The flux-form column leak `-Σ AREA·Fz[0]·T[0]` is therefore exactly O(dt)
and monotonic (+63 ZJ/yr over 200 yr, a surface->abyss dipole).
`project_adv_vel=True` applies the 2D Euler velocity increment
`-dt·g·∇(eta_proj)`, with `eta_proj` solving the area-weighted Poisson problem
`∇·(H∇eta_proj) = col_div_h(u_pred)`; the residual is O(dt²). It removes ~93-99%
of the one-step interior heat change (-1123 ZJ/yr -> ~0).

`freeze_adv_vel=True` is the cheaper alternative for the same leak: it freezes
the tracer stage-2 velocity at the old (u, v), making the tracer RK2 consistent
with the momentum RK2 (both stages old-velocity).

`_project_column_divergence` uses the EXACT column divergence (the per-layer
masked `_divergence_conservative` summed with `dz_node`), not a constant-H 2D
divergence: the layer masking makes the two differ by ~40% at coastlines, and
that residual is what the closure exists to remove. The CG solves the
area-weighted Poisson problem
`Div_col(Grad_conservative(psi)) = col_div_h(u,v)/(dt*g)` for a surface-pressure
potential, then applies `(u,v) -= dt*g*grad(psi)`. The iteration count is fixed
rather than a tolerance loop, for predictable per-step cost: the stage-2 leak
saturates at -40 ZJ/yr (from -2238) for `n_iter >= 120`, so 150 is past the knee
and the corrected field's column divergence sits at the CG residual (~1e-11).

## D14 — Face-gated horizontal gradients for advection

The bare centered `_d_dx`/`_d_dy` differences across mask boundaries read the
ghost `T_ref` sentinel: ghost nodes hold +15 C while real 4000 m water is ~+1 C,
so the centered stencil sees `dT ~ 14 K` across EVERY wet/ghost face. A 6 cm/s
deep coastal current then felt `v*dT/dy ~ 0.3 K/d` of advection from a
temperature that does not exist — a linear pump `adv ∝ (15 - T_wet)` that drove
the k12/k13 warm/cold dipole at (130, 19) from d50 into the d360 blow-up with
`max|T| = 1247 C` (adv-form runs `gpu365_cap3d`, ctl290nogm: FAIL_DRIFT both,
only adv pumping; `terms_fn` showed adv -0.24 K/d at k13 growing linearly with
the anomaly, all other terms < 0.03). Zeroing the face difference at mask
boundaries is consistent with the already-gated mass flux and with
`_gradient_conservative_3d`; where the flow is parallel to the boundary the
cross-boundary flux is zero anyway, so only the unphysical part is removed.

## D15 — Scalar advection must be flux form

The advective form `-u·∇T` equals flux form minus `T·∇·u`, and that `+T·∂u/∂x`
part is a REAL anti-diffusion: at any discrete divergence (equatorial upwelling,
`∂w/∂z > 0` in the 5 m layer) the operator carries a genuine positive eigenvalue
`dt·δ` per step that grows the field exponentially. Subcycling slows the
per-step growth but the eigenvalue scales with `dt·δ` and survives any n: the
(310.5E, 6.5N) surface T runaway (+3.4e-3 K/s, growing 20x in 2 steps, NaN by
step 23) persisted through `adv_nsub=6`. The flux form has no anti-diffusive
term, is exactly conservative, and matches the continuity-consistent tracer
equation the free-surface subcycle already solves. Momentum keeps the advective
form, because the vector-invariant scheme needs it.

Momentum keeps the advective form for a different reason than the tracer: the
flux form `-div(u u)` carries a spurious `u*div_h` source. Its vertical gradient
stays the bare `_d_dz` on a ghost-filled field, because `w` is masked to zero in
ghost layers and the terms are multiplied by `wet_mask_z`, so a wet/ghost
vertical face carries no advective flux whatever `dT/dz` it reads.

## D16 — Donor-cell vertical tracer flux

Centered vertical face values fail twice:

1. RK2/forward-Euler integration of a CENTERED vertical transport is
   unconditionally unstable (`|λ|² = 1 + θ² > 1` for any θ). At dt=3600 the 5 m
   layer's `θ_v = dt·w/dz = 1.3-1.8` grew ~2x/step — the step-21 equatorial T
   runaway, which survived `adv_nsub=6` as a still-growing 1.08x/step.
   Donor-cell is monotone (stable for `θ <= 1` per substep) and adds only
   `0.5·|w|·dz <= 6e-3 m²/s` of implicit vertical diffusivity, below
   `kappa_v = 1e-5`.
2. The centered value multiplies the FULL tracer content into the surface
   node's flux (`w·T/dz ~ O(10 K/step)`) wherever the column-divergence leak
   gives `w[0] != 0`. The old advective form was protected by the small `dT/dz`
   factor; donor-cell keeps the flux one-sided (upwelling carries the DEEP value
   up, surface convergence carries the SURFACE value down).

The tendency must be written OUT-minus-IN (`dn - up`). The reverse silently
makes the donor-cell scheme ANTI-upwind, `T_new = (1+θ)·T - θ·T_donor`, a +θ
eigenvalue per substep: measured `T_w` grew 29.6 -> 2350 over 6 substeps, a
factor 2.45 = `1 + θ_v` with `θ_v = 1.453` in the 5 m layer.

The top face is special: `Fz[..., 0]` is the column-integrated horizontal
divergence — the rigid-lid leak the barotropic subcycle absorbs as the eta
tendency. Left out of the budget it is a T-INDEPENDENT inflow of the deep value
into the surface node wherever `Fz[0] != 0` (measured: +16.47 K per 600 s
substep, linear forever). The MOM-style closure carries the surface cell's own
value, `Fz_top = Fz[0]·T[0]`.

The advective term is subcycled `adv_nsub` times (frozen velocity and frozen
horizontal structure; only the advected field evolves, which is what the CFL
constrains) so that `dt*(|u|/dx + |v|/dy + w/dz) < ~0.5` per substep. The
binding term is `w/dz` in the 5 m surface layer: at dt=3600 s equatorial
upwelling `w ~ 2e-3 m/s` gives `dt*w/dz = 1.3-1.8 > 1`, and the real-grid split
run ran away +0.3 K/step at (310.5E, 6.5N) from step 21 and went to NaN by step
29. `adv_nsub` is sized from a nominal `w_max = 4e-3 m/s` (upwelling plus
western-boundary downwelling overshoot) at a 0.5 target; the horizontal terms
(~1e-6/s at |u| ~ 1 m/s) are negligible. At the historical dt=60-300 s the CFL
was 0.02-0.18, so `adv_nsub = 1` and that path is untouched. Diffusion
(`kappa_h*dt/dz^2 << 0.5`), the surface fluxes (dt-weighted, non-CFL), the
GM/Redi skew flux (CFL 0.58 at dt=3600) and convection (own `conv_nsub`) all stay
inside their bounds at dt=3600 and are evaluated once.

## D17 — Slope limiter: DM95 taper, not a tanh clip

The closure flux is multiplied by the Danabasoglu-McWilliams (1995) taper
`sigma = 1/(1+(|S|/S_lim)^4)`, which is ~1 for `|S| << S_lim` (stratified
interior: full GM) and ~0 for `|S| >> S_lim` (weakly stratified deep ocean,
steep fronts: closure suppressed). A tanh CLIP is the wrong treatment: it
SATURATES the slope at `S_lim`, holding the flux at a full `kappa*S_lim²` — an
effective 0.1 m²/s vertical diffusivity at `kappa_gm = 1000` — throughout the
weakly stratified deep ocean. That is a spurious diapycnal pump: it erodes deep
stratification, drives the Southern-Ocean deep-T runaway, and drains the
subtropical gyres (-6 m/yr eta trend, linear in `kappa_gm`; 365d FAIL at every
`kappa_gm` in {300, 1000}, with and without `kappa_redi`). The DM95 taper is the
standard OGCM treatment (also Large et al. 1997).

The denominator floor `_GM_RHOZ_FLOOR` preserves the SIGN of `drho_dz`, so the
bolus does not reverse in convective patches; only the magnitude is floored.

## D18 — GM/Redi: one skew-flux operator, in interface-flux form

The bolus-transport + isoneutral-diffusion pair is a single skew-flux tensor
(Griffies 1998). Its vertical term is DIFFUSIVE (CFL ~ `kappa*|S|²*dt/dz²`)
rather than advective (CFL ~ `|w*|*dt/dz`): on a grid with a thin surface layer
(dz = 5 m) the advective bolus form reaches CFL ~3.5 and blows up within ~12
steps, while the skew-flux form sits at ~0.5. The skew-flux form is what
MOM6/MITgcm/NEMO use. (Because GM and Redi are the SAME operator, enabling both
runs the closure at `kappa_gm + kappa_redi` — "Defect 5" in
`docs/deep-heat-poisoning-root-cause.md`. Do not enable both.)

The vertical term is discretized in INTERFACE flux form: flux on interfaces
`k+1/2`, one-sided cell values, interface slope `S(k+1/2) = 0.5*(S[k]+S[k+1])`,
tendency `(F[k-1/2]-F[k+1/2])/dz[k]`, zero flux at the material top/bottom. That
is a compact 3-point stencil, exactly diffusive (negative-semidefinite for the
`|S|²` term). The node-flux form it replaced (Fz at nodes from a centered
`dC/dz`, then `_d_dz` of Fz) is a 5-point/2-step stencil on which the even/odd
sublattices in z DECOUPLE: the sawtooth-in-z mode has eigenvalue exactly 0
(never damped), and the operator is asymmetric (max asym 2.7e-2) with mildly
positive symmetric-part eigenvalues on the stretched deep grid. It pumped a
deep-T runaway linear in `kappa_gm` and `gm_slope_max` (365d FAIL at d270-310
for tanh-clip / DM95-taper / slope-max 0.001 alike).

The explicit CFL cap uses the two-cell Gershgorin bound of the `|S|²` term,
`dt*D_v*(1/dz_k + 1/dz_k1)/dz_iface <= ~2` (RK2). On a stretched grid with a 5 m
surface layer, `k = 1000` with slopes at the taper cap and dt = 3600 s reaches
~14 at the default `gm_slope_max = 0.01`: the Gulf Stream front blew up at step
~857 of the mode-split run, and was stable for 1000+ steps at slope cap 0.001,
i.e. exactly when this CFL drops below ~0.9. The stratified interior
(`D_v << bound`) is untouched; only thin-layer steep-front corners are clipped.

The material boundaries are NOT `Fz_i[0]`/`Fz_i[-1]` — those are INTERIOR
interfaces. Zeroing them decouples the surface/bottom nodes from the vertical
skew transport entirely (found by an eigenvector test: the null vector was
`e_0`); the padded flux array carries the zero-flux BC instead.

## D19 — The sponge relaxes toward the MASKED initial state, not the raw one

`init_state` overwrites land and below-seafloor ghost cells with the sentinel
`T_ref` / `S_ref` (D8's 3D mask), because WOA interpolation puts meaningless
values there. The linear half-step's diffusion mask then holds those cells at
whatever the step came in with, so the sentinel is preserved indefinitely.

The lateral sponge broke that. Its target was `T_clim_3d = jnp.array(T_init)`
-- the RAW, unmasked initial field. In the polar band the damping is live, so
every land and ghost cell was pulled from the sentinel toward its raw WOA
value. Measured (`sponge_days=5`, `sponge_cells=4`, 5 half-steps at 1800 s, a
land block holding a deliberate 999 degC sentinel-junk): land T left 15 degC
and reached 32.3 degC, land S left 35 psu and reached 48.1 psu, still
diverging. That is exactly the coastline cliff the mask exists to prevent,
handed back to the hydrostatic pressure integral.

The same branch had a second failure: with `T_init=None` the target was
literally `jnp.zeros`, so switching the sponge on without an initial field
relaxed the polar band toward 0 degC instead of leaving the `T_ref` state
alone.

Fix: build the target the way `init_state` builds the state --
`T_clim*wet_mask_z + (1 - wet_mask_z)*T_ref` -- and use `T_ref`/`S_ref` when
there is no initial field. The sponge is then a strict no-op wherever the
state is already at the target, which is what makes `sponge_days=0` (the
production default) bit-exact and keeps the land invariant intact when it is
switched on. Regression: `tests/test_sponge.py` (dry + ghost sentinel, band
relaxation, interior bit-identical, no-init-field case).

## D20 — The node-form vertical diffusion: zero-flux boundary and seafloor fill

`_d2_dz2` is the legacy (default `conservative_kv=False`) vertical diffusion, and
the operator the momentum vertical diffusion always uses.

**Boundary nodes** use the zero-flux (ghost-point) form `2*(C1 - C0)/h0^2`: the
mirror ghost `Cg = C1` makes the centered curvature `(C1 - 2*C0 + Cg)/h0^2`
diffusive at the boundary. The centered-curvature form `(C2 - 2*C1 + C0)/h0^2`
it replaced is the curvature AT node 1 applied as the tendency of node 0 —
anti-diffusive there, it pushed a boundary anomaly AWAY from the interior value.
With `kappa_conv = 0.05` that feedback amplified the initial WOA salty-over-fresh
surface profiles into the ITCZ salinity runaway that NaN'd the 365 d run at day
135 (`conv_S = +84.5 PSU/day` at the worst cell; the zero-flux form gives -89.9
PSU/day, clearing the instability). Shared by convection, `kappa_v` and momentum
vertical diffusion, all of which want the same no-flux BC.

**Seafloor fill.** The centered stencil spans k-1..k+1, so at a column whose
seafloor is NOT the last grid level the bottom WET node read the ghost layers
— which hold the `T_ref`/`S_ref` sentinel from `init_state`, not the bottom
value. D8 fixed exactly this for the closure operators and left this one ("the
background diffusion/convective BC is left as-is; its ghost pull is ~0.006
K/day"). Measured on a 15 m shelf column at the production `kappa_v = 1e-5`,
the pull is an order of magnitude larger than that note:

| bottom wet T | ghost | legacy `kappa_v*d2T/dz2` | with the fill |
| --- | --- | --- | --- |
| +25.97 C | +15 C | **-0.0504 K/day** | +0.0001 K/day |
| +1.97 C | +15 C | **+0.0602 K/day** | +0.0001 K/day |
| +5.60 C (200 m) | +15 C | +0.0011 K/day | +0.0000 K/day |

It always pulls the seafloor toward `T_ref`, so it cools warm shelves and warms
cold ones — +-22 K/yr of spurious surface heat flux, concentrated on the
shallowest topography. Same defect class as D8.

Fix: `_d2_dz2` ghost-fills its input first, like every other vertical stencil in
the file (`_d_dz` call sites, `_d2_dz2_flux`, `_conv_flux_tendency`,
`_redi_skew_flux_tendency`). A column wet to the last grid level has no ghost
layer, so its stencil is unchanged bit-for-bit; a shallower one gets the
one-sided no-flux form. Regression: `tests/test_vertical_bc.py`.

This is the one intentional break of the `conservative_kv=False` "bit-exact
legacy trace" property: any run with `kappa_v > 0` or `nu_v > 0` changes in the
bottom wet layer of every column shallower than the deepest level.

## D21 — Polar cap: wet-point zonal mean, cos^2 taper, 3D mask, capped eta first

The lat-lon grid's `dx = R*cos(lat)*dlon` goes to zero at the poles, so the
explicit shallow-water CFL `dt < dx/sqrt(gH)` is unattainable on the edge rows.
The standard lat-lon OGCM remedy (MOM6 polar cap / Arctic fold) replaces the
unresolved polar dynamics with a zonally-uniform cap value, which
`_apply_polar_cap` builds for 2D and 3D fields alike.

Three things it must get right:

- **Average over WET points only, and write back to WET points only.** The polar
  rows are ~30% land. A naive all-column mean mixes ocean (`eta != 0`) with land
  (`eta == 0`), forcing a zonally-uniform value that creates a spurious PGF at
  EVERY coastline point in the band — wind-driven barotropic energy injection
  exactly there. This was the wind-forced blow-up nucleation (`max|u|` diverged
  at 79.5N 124.5E, a wet point flanked by land, sub-inertial, CFL-safe).
- **Use the 3D mask for a 3D field.** `wet_mask` is column-wide; with
  `polar_cap_rows=2` at 4000 m about a third of the band columns are ghost there,
  and their `T_ref` fill (15.0 C) dragged the cap T to ~+15 C — a meridional
  cliff against the -0.3 C WOA deep T that seeded the Southern-Ocean cold-pole
  blow-up (d170 collapse).
- **Taper the cap edge.** `_polar_cap_weights` gives weight 1.0 on the
  `polar_cap_rows` poleward rows and a cos^2 ramp to 0 over the next
  `polar_cap_taper` rows. A hard cutoff (`polar_cap_taper=0`) is a cliff at
  `j = ncap` that `_d_dy` amplifies exponentially — the pole-wall blow-up.

The cap is applied as a CONSISTENT TRIPLE: cap `eta` first, then drive the
barotropic momentum update from the CAPPED eta's pressure gradient, then cap the
resulting `ubt`/`vbt`. Capping all three to independent zonal means leaves `eta`
zonally uniform in the band while `ubt`/`vbt` carry a different zonal structure,
so `div(ubt)` and `grad(eta)` are dynamically inconsistent, the mismatch is a
spurious PGF, and the free mode gains energy (cap-ON NaN at step 534 vs cap-OFF
800). With the capped eta driving the PGF, `ubt`/`vbt` are consistent with `eta`
by construction and their cap is a CFL-safety smoothing rather than an
independent forcing.

Implementation note: `_polar_cap_weights` builds `jnp.ones`/`linspace` at
runtime, which are float64 under `jax_enable_x64`; they are cast to the field's
dtype or the blend promotes the whole field to f64 and silently defeats the fp32
state/params cast.

The 2D and 3D copies of the cap used different denominators for the same mean
(`max(sum(w), 1)` vs `sum(max(w, 1e-12))`). They now share `max(sum(w), 1)`,
which is exactly "divide by the number of wet cells" with a guard for an all-dry
row; the change is a 2.4e-13 relative difference.

## D22 — Free surface: forward-backward, implicit barotropic Coriolis, implicit drag

**Forward-backward (Sielecki), not forward-forward.** Forward-forward coupling
(both eta and ubt from the old state) has amplification
`|lambda| = sqrt(1 + (dt*c*k)^2) > 1` for ALL k — unconditionally unstable for
free gravity waves, and CFL does not save forward-Euler, only centered/leapfrog
schemes. At dt=60 s the basin-scale seiche grew ~1.0023/step, i.e. ~27x/day; a
no-wind 1 m eta bump reached 5.5 m in one day (mean ~0, so mass was conserved but
the amplitude grew). Forward-backward flips the trace of the amplification matrix
to `2 - dt^2*g*H*k^2`, giving `|lambda| = 1` (neutral) under CFL < 1, which is
the standard OGCM discretization (MOM6/ROMS/NEMO); bottom drag then decays the
free mode (`|lambda| ~ 0.97/step`) so a perturbed eta relaxes to the steady
wind-driven setup. No iterative Helmholtz solve is needed.

CFL: `dt < dx/sqrt(g*H) ~ 85-560 s`, so dt=60 s is safe on the global 1 deg grid.

**Implicit barotropic Coriolis.** The shallow-water momentum equation is
`du/dt - f*v = -g*grad(eta) + F`, `dv/dt + f*u = ...`. Treating Coriolis
implicitly (unconditionally stable, energy-neutral) gives
`u_new = (u_star + f*dt*v_star) / (1 + (f*dt)^2)` and the mirror for `v`.
Without it the barotropic PGF has no geostrophic balance: `F_rho ~ 4e-4 m/s^2`
drives a convergent `ubt` that grows eta monotonically (Coriolis was 180x too
small) into exponential eta/ubt growth and then advection overshoot. This is the
barotropic analogue of the 3D rotation in `_linear_half_step`.

**Implicit linear bottom drag, in split mode only.** The residual's `-r_bot*u`
is forward-Euler, whose amplitude `1 - r*dt` flips past the stability bound
`|1 - r*dt| <= 1` at `r*dt > 2` (and the RK2 form `1 - rdt + (rdt)^2/2` past
`r*dt = 2` as well). At the historical dt=60-300 s, `r*dt = 0.06-0.3` and
explicit was fine; the mode-split dt=3600 s gives `r*dt = 3.6`, and the bottom
layer amplified ~2.6x/step with sign alternation (smoke-test growth 3x/step at
8E 30N k=7). Both RK2 stages evaluate the residual at the OLD velocity, so the
drag part is identical in `du1`/`du2`: strip it from both stages and apply the
exact `exp(-r_bot*dt)` decay after the RK2 update.

## D23 — Mass-conserving sponge and eta relaxation

**The lateral sponge was adding volume.** A bare `eta *= exp(-rate*dt)` changes
global volume by `dV = sum(A*eta*(decay-1))` over the band. The wind setup makes
the band-mean eta NEGATIVE on both hemispheres (subpolar lows, and the
south-of-westerlies ACC minimum), so the sponge added volume every step: +0.377 m
of global mean-eta drift over 90 d, a steady ~10000 km^3/5d from day 5 that
matches the eta-decay leak budget. `_refill_volume` adds the removed volume back
UNIFORMLY over the wet domain: total volume is exactly conserved, the local
anomaly damping is unchanged, and a uniform eta offset has zero PGF so the
dynamics are untouched.

**Semi-enclosed-sea eta relaxation.** Gibraltar is 14 km wide — sub-grid on a
1 deg mesh. The one-cell strait cannot support the observed two-layer exchange,
and the residual pressure mismatch drives a spurious NET outflow that drains the
Mediterranean linearly (~+0.026 m/d; `max|eta| = 9.5 m` at d365 in `gpu365_glap`,
which would trip the 15 m watchdog at ~d500). This is not an instability but a
coarse-grid artifact that has to be BOUNDED for multi-year integrations. The
MOM-family remedy is a Rayleigh relaxation of eta toward the basin equilibrium
INSIDE the sea only (`eta *= exp(-rate*dt*mask)`, with eta = 0 the correct
equilibrium because the basin-mean PGF at Gibraltar then matches the Atlantic
open boundary at the same latitude), compensated by the same `_refill_volume`
uniform refill. The mask tapers 1 -> 0 over `eta_relax_buffer` degrees around the
box edge so the relaxation itself cannot seed a PGF cliff at the boundary.

## D24 — Surface boundary conditions: bulk heat flux and salinity restoring

Both surface fluxes are applied to the TOP NODE only, through `surface_mask`, and
both are divided by the top cell's heat capacity. `heat_factor =
1/(RHO_0*C_P*dz_surface)` turns a surface heat flux [W/m^2] into a tendency on
the surface node; the bulk flux is the Haney/Barnier form
`lambda_bulk*(T_atm - T[0])*heat_factor*surface_mask`, a genuine SST negative
feedback.

Surface salinity restoring is the same shape with a TRUE salt flux (psu/s, no
`heat_factor`, because salinity has no `rho*cp`):
`dS/dt += restore_coef_S*(S_ref_2d - S[0])*surface_mask`, i.e. Haney relaxation
of SSS to climatology with `tau = 1/restore_coef_S`. It replaces the v0.1
solver's missing surface salinity BC, which let SSS drift -0.9 psu/kyr. There is
no SST restoring — the bulk flux is the only surface thermal BC.

## D25 — Grid-scale de-aliasing: 2/3 FFT in lon, 5-pt binomial in lat

Nonlinear products in physical space (`u*du/dx`, ...) amplify the 2-dx grid-scale
mode. The lon axis is periodic and uniformly spaced, so a 2/3-rule FFT there is
exact and faithful. The lat axis is NOT periodic (closed no-flux N/S walls) and
not uniformly spaced, so an FFT there would mangle the field; a 5-pt binomial
low-pass `[1,4,6,4,1]/16` substitutes, with edge padding that mirrors the wall
ghost cell of `_d_dy`/`_laplacian_h`. Its transfer function is 0 at the Nyquist
(2-dx) wavenumber, ~0.01 near Nyquist and ~0.92 at 8-dx, so it kills the grid
scale while preserving resolvable structure.

`_dealias_h_fd` is applied at exactly THREE call sites, all pointwise nonlinear
products: `adv_u` and `adv_v` (`_advection_flux_form`) and the GM/Redi skew-flux
tendency. It is deliberately NOT applied to the diagnosed `w`
(`_compute_vertical_velocity`) or to the tracer advection (`_advection_scalar`):
both are flux-form operators whose whole point is an exact discrete telescoping
(D7, D15, D16), and a non-local FFT + low-pass filter would destroy it.

## D26 — Barotropic density PGF: transport-consistent wet-column form

The barotropic density forcing is the depth average, over the WET column, of the
SAME face-gated 3D baroclinic PGF the 3D momentum feels:
`F_rho = (1/H_sw) * SUM_k pgf3d_layer_k * dz_k * wet_iface_k`, normalized by
`H_sw` to match `ubt = transport/H_sw` in `_barotropic_velocity`.

The form it replaced was `F = -grad(p_bc_avg)/RHO_0` with `p_bc_avg` the
`H_sw`-normalized trapezoidal average of the baroclinic pressure over ALL 13
layers, ghost water included. Because `p_bc` is CONSTANT below the seafloor (rho
is masked before the cumsum, so `dp = 0` there), the ghost part contributes
`((H_sw-H)/H_sw) * grad(p_bc_bottom)`. At shelf breaks `grad(p_bc_bottom)` is
O(5e3 Pa / 1e5 m), so the ghost term alone is O(3e-5 m/s^2). It displaces the
implied equilibrium sea level `eta_eq = -p_bc_avg/(g*rho0)` by O(1-2 m) between
adjacent cells across every isobath step, and the free-surface step piles eta up
at each jump; near the equator (f -> 0) Coriolis cannot geostrophically cage the
pile-up, and eta ran 2.5 -> 12.25 m in 15 d at 47.5W 7.5N on the 2000 m isobath
off the Amazon fan (the "day-75 Amazon-fan eta blob"). The transport-weighted
form contains no ghost water and no below-bottom constant, so the spurious
isobath forcing vanishes; the remaining wet-column form stress is the physical
(JEBAR-type) coupling, ~10x smaller at the blob site. Measured at the d75 blob
state, cell (311,66), H=2000 m, H_sw=4000 m: `|F_old| = 4.6e-5` ->
`|F_new| = 4.9e-6 m/s^2`; global mean `|F|` 1.3e-5 -> 6.2e-6 m/s^2.
