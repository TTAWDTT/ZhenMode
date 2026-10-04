"""Split time integration, linear updates and coordinated fast substeps."""

from zhenmode.model.solver.dynamics.barotropic import (
    _barotropic_subcycle_transport,
    _free_surface_step_fd,
)
from zhenmode.model.solver.dynamics.pressure import (
    _compute_bt_rho_pgf,
)
from zhenmode.model.solver.dynamics.projection import _project_column_divergence
from zhenmode.model.solver.dynamics.tendencies import (
    _compute_momentum_residual,
    _compute_tracer_residual,
)
from zhenmode.model.solver.dynamics.transport import (
    _barotropic_velocity,
    _column_divergence,
    _layer_face_transports,
    _match_layer_face_transports,
    _reference_depth_divergence,
)
from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.numerics.horizontal import (
    _apply_polar_cap,
    _biharmonic_h,
    _horizontal_biharmonic_tracer,
    _horizontal_tracer_diffusion,
    _laplacian_h,
)
from zhenmode.model.solver.physics.surface import _dynamic_ice_closure
from zhenmode.model.solver.physics.vertical import (
    _effective_kappa_v,
    _vertical_diffusion,
    _vertical_momentum_diffusion,
)
from zhenmode.model.solver.state import JaxStateG
from zhenmode.model.solver.timestepping.subcycles import _subcycle


def _coriolis_rotation_2d(u, v, f, dt):
    """Exact Coriolis rotation on the full 2D f-field (not f-plane).

    Per-gridpoint rotation: u' = cos(f*dt)*u + sin(f*dt)*v, etc.
    f varies with latitude (2D), so this is exact everywhere, no beta-plane
    approximation.
    """
    angle = f[:, :, None] * dt     # broadcast 2D f to 3D velocity fields
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_new = cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new

def _rotate_baroclinic_shear(velocity_x, velocity_y, params, duration):
    """Rotate only the wet-depth anomaly; the fast mode rotates its own mean."""
    mean_x, mean_y = _barotropic_velocity(velocity_x, velocity_y, params)
    shear_x = (velocity_x - mean_x[..., None]) * params.wet_mask_z
    shear_y = (velocity_y - mean_y[..., None]) * params.wet_mask_z
    rotation = params.f
    if params.process_time_scheme == 'symmetric_fast_v3':
        rotation = rotation * params.interior_mask_z[..., 0]
    rotated_x, rotated_y = _coriolis_rotation_2d(shear_x, shear_y, rotation, duration)
    return (velocity_x + (rotated_x - shear_x) * params.wet_mask_z,
            velocity_y + (rotated_y - shear_y) * params.wet_mask_z)

def _linear_half_step(state, p, dt_half, budget=None, *, momentum_diffusion=None):
    """Linear half-step: FD diffusion + 2D Coriolis + free surface (or not, split).

    The default retains component-wise explicit diffusion; component CFL limits
    alone do not bound their combined update. An opt-in diffusion callback acts
    before masking, damping and rotation. The monolithic free surface remains
    forward Euler with CFL dt < dx/sqrt(gH). (D22)
    """
    # Explicit diffusion (horizontal Laplacian + vertical d2/dz2). nu_h CANNOT
    # be applied at dt_half in one explicit shot (nu_h*dt_half/dy^2 = 0.73 >
    # 0.25), but it MUST act on the full 3D baroclinic velocity: the N-step
    # residual subtracts nu_h*lap, so if L re-adds nothing the baroclinic SHEAR
    # (invisible to the depth-mean) loses all its damping. Apply nu_h here
    # subcycled (nu_sub_cyc substeps of dt_half/nu_sub_cyc, CFL 0.06 each) --
    # the same fixed-operator subcycling pattern as conv_nsub. Monolithic
    # dt=60-300 keeps the single-shot form (CFL-safe). (D12)
    if momentum_diffusion is not None:
        u, v = momentum_diffusion(state, p, dt_half)
    elif p.mode_split and p.n_subcyc > 0:
        # Subcycle count sized by the 0.5*LHS FTCS criterion. Legacy sizing
        # (nu_nsub=None) reuses n_subcyc; nu_nsub right-sizes it from the actual
        # metric worst case: nu_h*dt/(n*dy_min^2) <= 0.5 (see make_solver_global).
        # (D12)
        n_nu = max(1, int(p.n_subcyc if p.nu_nsub is None else p.nu_nsub))
        dts = dt_half / n_nu

        def _nu_sub(carry):
            cu, cv = carry
            return (cu + p.nu_h * _laplacian_h(cu, p) * dts,
                    cv + p.nu_h * _laplacian_h(cv, p) * dts), None

        (u, v), _ = _subcycle(_nu_sub, (state.u, state.v), n_nu, p)
    else:
        u = state.u + p.nu_h * _laplacian_h(state.u, p) * dt_half
        v = state.v + p.nu_h * _laplacian_h(state.v, p) * dt_half
    T = state.T + _horizontal_tracer_diffusion(state.T, p) * dt_half
    S = state.S + _horizontal_tracer_diffusion(state.S, p) * dt_half
    # Scale-selective biharmonic (nabla^4): damps grid-scale modes far more than
    # large-scale ones. Explicit forward-Euler; CFL nu_bi*dt/dx^4 < ~0.05.
    # docs/resolution_cfl_limits.md has the dx^4 auto-scaling.
    if p.nu_bi > 0.0 and momentum_diffusion is None:
        u = u - p.nu_bi * _biharmonic_h(state.u, p) * dt_half
        v = v - p.nu_bi * _biharmonic_h(state.v, p) * dt_half
    if p.kappa_bi > 0.0:
        T = T - p.kappa_bi * _horizontal_biharmonic_tracer(state.T, p) * dt_half
        S = S - p.kappa_bi * _horizontal_biharmonic_tracer(state.S, p) * dt_half
    if momentum_diffusion is None:
        u = u + _vertical_momentum_diffusion(state.u, p) * dt_half
        v = v + _vertical_momentum_diffusion(state.v, p) * dt_half
    T = T + _vertical_diffusion(state.T, _effective_kappa_v(p), p) * dt_half
    S = S + _vertical_diffusion(state.S, _effective_kappa_v(p), p) * dt_half
    # Mask: no diffusion updates over land or below seafloor (ghost water). Hold
    # land/ghost values at their ORIGINAL state (not zero): masking to zero
    # creates a T=0 cliff at every coastline that the (unmasked) _laplacian_h in
    # the N-step tracer residual reads as a huge gradient. Holding the pre-step
    # value preserves the no-flux (flat) land value the diffusive stencil already
    # assumes (mirror ghost = boundary value => no cliff). (D9)
    u = u * p.wet_mask_z + state.u * (1.0 - p.wet_mask_z)
    v = v * p.wet_mask_z + state.v * (1.0 - p.wet_mask_z)
    T = T * p.wet_mask_z + state.T * (1.0 - p.wet_mask_z)
    S = S * p.wet_mask_z + state.S * (1.0 - p.wet_mask_z)

    if budget is not None:
        diffusion_state = JaxStateG(u, v, T, S, state.eta, state.ice)
        budget.stage("linear_diffusion", state, diffusion_state)

    # Lateral sponge (Rayleigh damping): exponential decay, no damping CFL.
    # Applied in the linear half-step so Strang splitting gives total sponge time
    # = dt per full step. Absorbs the wind-driven barotropic energy that
    # Laplacian dissipation cannot arrest within its CFL cap, which otherwise
    # piles up at the polar edge rows.
    #
    # No-op when off: sponge_rate == 0 => decay == 1 and T_clim/S_clim == 0, so
    # every line below returns its input unchanged. When ON, T_clim/S_clim carry
    # the land sentinel on land and below-seafloor ghost cells, so the relaxation
    # leaves the land invariant the mask above restored intact. (D19, D23)
    decay = jnp.exp(-p.sponge_rate * dt_half)            # (nx,ny,1)
    u = u * decay
    v = v * decay
    T = p.T_clim_3d + (T - p.T_clim_3d) * decay
    S = p.S_clim_3d + (S - p.S_clim_3d) * decay

    if budget is not None:
        sponge_state = JaxStateG(u, v, T, S, state.eta, state.ice)
        budget.stage("sponge", diffusion_state, sponge_state)
        budget.sponge_sources(diffusion_state, decay)

    # Coriolis rotation (2D f-field, exact)
    if p.process_time_scheme != 'legacy':
        u, v = _rotate_baroclinic_shear(u, v, p, dt_half)
    else:
        u, v = _coriolis_rotation_2d(u, v, p.f, dt_half)

    # Semi-implicit free surface (with density barotropic PGF). mode_split: the
    # free surface runs ONCE per baroclinic step in _step_impl's barotropic
    # subcycle (dt_bt), not here -- the external-wave CFL forbids dt_half here.
    # The linear step keeps diffusion (nu_h subcycled above) + sponge + Coriolis
    # only. (D12, D22)
    if p.mode_split:
        return JaxStateG(u, v, T, S, state.eta, state.ice)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    updated = JaxStateG(u, v, T, S, eta, state.ice)
    if budget is not None:
        budget.stage("free_surface", sponge_state, updated)
    return updated

def _linear_bottom_drag_step(state, params, duration, budget=None):
    """Exact wet-bottom drag, separately audited from numerical face filtering."""
    if params.bottom_friction != 'linear' or params.r_bot == 0.:
        return state
    decay = jnp.exp(-params.r_bot * duration * params.bottom_mask * params.wet_mask_z)
    updated = state._replace(u=state.u * decay, v=state.v * decay)
    if budget is not None:
        budget.bottom_drag(state, updated)
    return updated

def _tracer_rk_subcycle(state, params, face_transport=None, budget=None):
    """Heun substeps of the instantaneous nonlinear RHS, with actual RK weights."""
    subcycles = max(int(params.adv_nsub), int(params.conv_nsub))
    subparams = params._replace(dt=params.dt / subcycles, adv_nsub=1, conv_nsub=1)

    def advance(current):
        temperature, salinity = current
        stage = state._replace(T=temperature, S=salinity)
        first = _compute_tracer_residual(stage, subparams, face_transport=face_transport,
                                         return_terms=budget is not None)
        predicted = stage._replace(T=temperature + subparams.dt * first[0],
                                    S=salinity + subparams.dt * first[1])
        second = _compute_tracer_residual(predicted, subparams, face_transport=face_transport,
                                          return_terms=budget is not None)
        updated = (temperature + 0.5 * subparams.dt * (first[0] + second[0]),
                   salinity + 0.5 * subparams.dt * (first[1] + second[1]))
        terms = (jax.tree.map(lambda start, end: 0.5 * (start + end), first[2], second[2])
                 if budget is not None else None)
        return updated, terms

    (temperature, salinity), terms = _subcycle(advance, (state.T, state.S), subcycles, params)
    if budget is not None:
        budget.tracer_transport(face_transport if face_transport is not None
                                else _layer_face_transports(state.u, state.v, params), weight=1.)
        budget.surface_sources(terms[0], duration=params.dt)
        budget.nonlinear_terms(terms[1], duration=params.dt, absolute_tendencies=terms[4])
        budget.advection_boundary_fluxes(terms[2], terms[3], duration=params.dt)
    return state._replace(T=temperature, S=salinity)

def _nonlinear_predictor_rk2(state, params, duration, budget=None):
    """Two actual momentum stages; accepted tracer transport is replayed later."""
    tracer = _tracer_rk_subcycle(state, params._replace(dt=duration), budget=budget)
    first_x, first_y = _compute_momentum_residual(state, params)
    implicit_drag = params.mode_split and params.bottom_friction == 'linear'
    bottom = params.bottom_mask * params.wet_mask_z
    if implicit_drag:
        first_x = first_x + params.r_bot * state.u * bottom
        first_y = first_y + params.r_bot * state.v * bottom
    predicted = tracer._replace(u=state.u + duration * first_x, v=state.v + duration * first_y)
    second_x, second_y = _compute_momentum_residual(predicted, params)
    if implicit_drag:
        second_x = second_x + params.r_bot * predicted.u * bottom
        second_y = second_y + params.r_bot * predicted.v * bottom
    velocity_x = state.u + 0.5 * duration * (first_x + second_x)
    velocity_y = state.v + 0.5 * duration * (first_y + second_y)
    if implicit_drag and params.process_time_scheme != 'symmetric_fast_v3':
        decay = jnp.exp(-params.r_bot * duration * bottom)
        velocity_x, velocity_y = velocity_x * decay, velocity_y * decay
    return tracer._replace(u=velocity_x, v=velocity_y)

def _explicit_full_step(state, p, dt, budget=None):
    """Forward-backward RK2 for nonlinear tendencies (FD).

    Tracers updated first (old velocity), then momentum uses predicted T
    for the baroclinic PGF — shifts internal-wave eigenvalues left of the
    imaginary axis for neutral stability.
    """
    if p.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3'):
        return _nonlinear_predictor_rk2(state, p, dt, budget=budget)
    dT1, dS1 = _compute_tracer_residual(state, p, budget=budget)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta, state.ice)

    du1, dv1 = _compute_momentum_residual(state_T, p)
    # ── Semi-implicit linear bottom drag (split mode only) ──
    # The residual's -r_bot*u is forward-Euler here: pure-drag amplitude
    # 1 - r*dt flips past |1-r*dt| <= 1 at r*dt > 2, and the mode-split dt=3600
    # gives r*dt = 3.6, so the bottom layer amplifies with sign alternation. Both
    # RK2 stages evaluate the residual at the OLD velocity, so the drag part is
    # identical in du1/du2: strip it from both and apply the EXACT linear-drag
    # decay exp(-r_bot*dt) after the RK2 update -- unconditionally stable and
    # exact for the pure-drag sub-operator. The monolithic path is untouched (its
    # dt range keeps the explicit form stable). (D22)
    implicit_drag = p.mode_split and p.bottom_friction == 'linear'
    if implicit_drag:
        bm = p.bottom_mask * p.wet_mask_z
        du1 = du1 + p.r_bot * state.u * bm
        dv1 = dv1 + p.r_bot * state.v * bm
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    # ── Tracer stage-2 advection velocity ──
    # RK2 evaluates the momentum residual at the OLD velocity in BOTH stages; the
    # tracer stage-2 advection was the lone exception, using the raw forward-Euler
    # predictor u_pred, a velocity the momentum scheme itself never reaches. That
    # velocity is strongly divergent (baroclinic PGF undamped over dt at the
    # predictor) and the flux-form advection of it leaks column heat at rate
    # -sum(AREA*Fz[0]*T[0]). Freezing it makes the tracer RK2 consistent with the
    # momentum RK2 (both stages at the old velocity) and changes the final T field
    # by only 0.037 K rms. Default False = historical behavior. (D13)
    if p.freeze_adv_vel:
        state_pred = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta, state.ice)
    else:
        # Column-divergence-consistent stage-2 velocity (project_adv_vel): the
        # raw predictor's column divergence is O(dt) and drives the flux-form
        # column heat leak; project it onto the divergence-free column space (the
        # same space the barotropic subcycle later enforces on the FINAL u) so
        # Fz_top = Fz[0]*T[0] conserves column heat. No-op when off. (D13)
        if p.project_adv_vel:
            u_adv, v_adv = _project_column_divergence(u_pred, v_pred, p, dt)
            if budget is not None:
                budget.column_projection(u_pred, v_pred, u_adv, v_adv)
        else:
            u_adv, v_adv = u_pred, v_pred
        state_pred = JaxStateG(u_adv, v_adv, T_pred, S_pred, state.eta, state.ice)

    dT2, dS2 = _compute_tracer_residual(state_pred, p, budget=budget)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = JaxStateG(state.u, state.v, T_new, S_new, state.eta, state.ice)

    du2, dv2 = _compute_momentum_residual(state_T_new, p)
    if implicit_drag:
        du2 = du2 + p.r_bot * state.u * bm
        dv2 = dv2 + p.r_bot * state.v * bm
    u_new = state.u + 0.5 * (du1 + du2) * dt
    v_new = state.v + 0.5 * (dv1 + dv2) * dt
    if implicit_drag:
        decay = jnp.exp(-p.r_bot * dt * bm)     # 1 away from the bottom
        u_new = u_new * decay
        v_new = v_new * decay
    return JaxStateG(u_new, v_new, T_new, S_new, state.eta, state.ice)

def _tracer_step_with_transport(state, params, column_transport, budget=None, advection_velocity=None):
    """Replay only the accepted tracer stages with the actual fast-mode mean.

    Both RK stages and every advection substep share matched open-layer faces.
    The provisional tracer update used for momentum forcing is discarded; its
    sources are not booked. This is a lagged predictor/corrector, not a claim of
    second-order coupled momentum or a moving-volume inventory formulation.
    """
    velocity_x, velocity_y = (state.u, state.v) if advection_velocity is None else advection_velocity
    faces = _match_layer_face_transports(velocity_x, velocity_y, column_transport, params)
    if params.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3'):
        return _tracer_rk_subcycle(state, params, face_transport=faces, budget=budget)
    temperature_1, salinity_1 = _compute_tracer_residual(state, params, budget=budget, face_transport=faces)
    predicted = state._replace(T=state.T + params.dt * temperature_1,
                               S=state.S + params.dt * salinity_1)
    temperature_2, salinity_2 = _compute_tracer_residual(predicted, params, budget=budget, face_transport=faces)
    return state._replace(T=state.T + 0.5 * params.dt * (temperature_1 + temperature_2),
                          S=state.S + 0.5 * params.dt * (salinity_1 + salinity_2))

def _step_impl(state, p, budget=None):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2).

    Default mode_split=True retains the L/N/L baroclinic core; linear half-steps
    carry diffusion/sponge/Coriolis, including subcycled nu_h, but no free surface. After the
    second L half-step, n_subcyc barotropic forward-backward subcycles of dt_bt
    evolve (eta, ubt, vbt), driven by the rho-PGF + wind forcing computed from the
    post-L/N/L predictor, held fixed over the subcycle. The final (ubt, vbt) is projected back onto the 3D
    velocity as a uniform-in-depth delta. The opt-in transport candidate replays
    accepted tracer stages with the actual fast-mode mean; momentum retains its
    provisional tracer forcing. Static nodal inventories remain approximate. (D12)
    """
    dt_half = p.dt / 2.0
    # Capture land/ghost values BEFORE the step. Final masking holds these
    # (not zero): masking land T to 0 builds a coastline cliff that the
    # unmasked _laplacian_h in the next step's tracer residual reads as a
    # huge gradient, seeding an exponentially-growing spurious PGF.
    land_u, land_v = state.u, state.v
    land_T, land_S = state.T, state.S
    if p.process_time_scheme == 'symmetric_fast_v3':
        state = _linear_bottom_drag_step(state, p, dt_half, budget=budget)
    state = _linear_half_step(state, p, dt_half, budget=budget)
    nonlinear_start = state
    predictor_budget = None if p.match_barotropic_transport else budget
    state = _explicit_full_step(state, p, p.dt, budget=predictor_budget)
    nonlinear_end = state
    if predictor_budget is not None:
        budget.stage("nonlinear", nonlinear_start, state)
    state = _linear_half_step(state, p, dt_half, budget=predictor_budget)
    barotropic_start = state

    # ── mode split: barotropic subcycle (free surface + nu_h) ──
    # Runs AFTER the L/N/L core on the un-masked final 3D state (before the
    # polar cap / land hold below, so the subcycle sees the same masked,
    # capped input the monolithic path fed _free_surface_step_fd).
    if p.mode_split and p.column_geometry == 'nodal_dual_v1' and (budget is not None or p.match_barotropic_transport):
        state, column_transport, filter_change = _barotropic_subcycle_transport(state, p)
        if budget is not None:
            budget.stage("free_surface", barotropic_start, state)
            budget.barotropic_transport(barotropic_start.eta, state.eta, column_transport, filter_change)
        if p.match_barotropic_transport:
            advection_velocity = ((0.5 * (nonlinear_start.u + nonlinear_end.u),
                                   0.5 * (nonlinear_start.v + nonlinear_end.v))
                                  if p.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3') else None)
            tracer_state = _tracer_step_with_transport(nonlinear_start, p, column_transport, budget=budget,
                                                      advection_velocity=advection_velocity)
            if budget is not None:
                budget.stage("nonlinear", nonlinear_start, tracer_state)
            tracer_state = _linear_half_step(tracer_state, p, dt_half, budget=budget)
            state = state._replace(T=tracer_state.T, S=tracer_state.S)
        if budget is not None:
            budget.before_transport_filter(state)
    elif p.mode_split:
        # Coupling forcing from the baroclinic state, held fixed over the
        # subcycle (MOM-style forcing lag at dt=3600 s is standard).
        F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
        ubt0, vbt0 = _barotropic_velocity(state.u, state.v, p)
        column_divergence_offset = None
        if p.column_geometry == 'nodal_dual_v1':
            column_divergence_offset = (_column_divergence(state.u, state.v, p)
                                        - _reference_depth_divergence(ubt0, vbt0, p))
        ubt, vbt = ubt0, vbt0
        eta = state.eta

        # lax.scan barotropic subcycle (when use_scan): one fused device-side loop
        # of n_subcyc forward-backward pairs instead of n_subcyc unrolled copies.
        # Numerically IDENTICAL; scan just compiles the body once, cutting XLA
        # graph size and host launch overhead. The Python-loop path stays default
        # (None) = bit-exact legacy trace.
        def _fb(carry):
            # One FB pair per call = dt_bt of evolution (caller passes
            # dt_half=dt_bt); each subcycle feeds the 2D triple forward.
            eta_i, ubt_i, vbt_i = carry
            return _free_surface_step_fd(
                eta_i, ubt_i, vbt_i, p, F_rho_x, F_rho_y, dt_half=p.dt_bt,
                column_divergence_offset=column_divergence_offset), None

        (eta, ubt, vbt), _ = _subcycle(_fb, (eta, ubt, vbt), int(p.n_subcyc), p)
        # Project the subcycle's net barotropic delta onto the 3D velocity
        # (uniform over depth) — same projection as the monolithic path.
        delta_ubt = (ubt - ubt0)[:, :, None]
        delta_vbt = (vbt - vbt0)[:, :, None]
        projection_mask = p.wet_mask_z if p.column_geometry == 'nodal_dual_v1' else 1.
        state = JaxStateG(state.u + delta_ubt * projection_mask, state.v + delta_vbt * projection_mask,
                          state.T, state.S, eta, state.ice)
        if budget is not None:
            budget.stage("free_surface", barotropic_start, state)

    if p.process_time_scheme == 'symmetric_fast_v3':
        state = _linear_bottom_drag_step(state, p, dt_half, budget=budget)
        if budget is not None:
            budget.before_transport_filter(state)

    # 3D polar-cap filter: zonally average the cap rows of u,v,T,S to kill
    # the cos(lat)->0 metric singularity in the diffusion/advection operators.
    u = _apply_polar_cap(state.u, p.wet_mask_z, p)
    v = _apply_polar_cap(state.v, p.wet_mask_z, p)
    T = _apply_polar_cap(state.T, p.wet_mask_z, p)
    S = _apply_polar_cap(state.S, p.wet_mask_z, p)
    # Final mask enforcement: hold land/ghost at the pre-step value (no cliff).
    wmask = p.wet_mask_z
    u = u * wmask + land_u * (1.0 - wmask)
    v = v * wmask + land_v * (1.0 - wmask)
    T = T * wmask + land_T * (1.0 - wmask)
    S = S * wmask + land_S * (1.0 - wmask)
    # No-flux wall: zero the NORMAL (meridional) velocity at the N/S boundary
    # rows so no flow crosses the closed wall. Combined with the mirror-ghost
    # stencils in _d_dy/_laplacian_h (zero normal gradient), this is the full
    # closed-boundary condition that replaces the unstable one-sided stencil.
    v = v * p.interior_mask_z
    masked = JaxStateG(u, v, T, S, state.eta, state.ice)
    if budget is not None:
        budget.stage("polar_cap_and_masks", state, masked)
    updated = _dynamic_ice_closure(masked, p, budget=budget)
    if budget is not None:
        budget.stage("dynamic_ice", masked, updated)
        budget.after_transport_filter(updated)
    return updated
