"""Canonical processes definitions; legacy operations unchanged."""
from ocean_solver.config.definitions import C_P, G_EARTH, RHO_0
from ocean_solver.dynamics.barotropic import _free_surface_step_fd
from ocean_solver.dynamics.pressure import (
    _compute_bt_rho_pgf,
    _compute_pressure_gradient,
    _reference_depth_gradient,
)
from ocean_solver.dynamics.transport import (
    _advection_flux_form,
    _advection_scalar,
    _barotropic_velocity,
    _compute_vertical_velocity,
    _layer_face_transports,
    _vertical_transport_iface,
)
from ocean_solver.numerics.backend import jax, jnp
from ocean_solver.numerics.horizontal import (
    _biharmonic_h,
    _d_dx,
    _d_dy,
    _gradient_conservative_3d,
    _horizontal_biharmonic_tracer,
    _horizontal_tracer_diffusion,
    _laplacian_h,
)
from ocean_solver.physics.isopycnal import _isopycnal_closure
from ocean_solver.physics.surface import _surface_heat_weights
from ocean_solver.physics.vertical import (
    _conv_flux_tendency,
    _convective_mask,
    _effective_kappa_v,
    _vertical_diffusion,
    _vertical_momentum_diffusion,
)
from ocean_solver.state.types import JaxStateG
from ocean_solver.timestepping.subcycles import _subcycle


def _compute_momentum_tendency(state, p):
    """du/dt, dv/dt for hydrostatic primitive equations (FD)."""
    w = _compute_vertical_velocity(state, p)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)

    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u

    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    diff_h_u = p.nu_h * _laplacian_h(state.u, p)
    diff_h_v = p.nu_h * _laplacian_h(state.v, p)
    diff_v_u = _vertical_momentum_diffusion(state.u, p)
    diff_v_v = _vertical_momentum_diffusion(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x_2d[:, :, None] * wind_factor * p.surface_mask
    wind_v = p.tau_y_2d[:, :, None] * wind_factor * p.surface_mask

    if p.bottom_friction == 'quadratic':
        speed = jnp.sqrt(state.u**2 + state.v**2)
        bot_u = -p.cd * speed * state.u * p.bottom_mask
        bot_v = -p.cd * speed * state.v * p.bottom_mask
    else:
        bot_u = -p.r_bot * state.u * p.bottom_mask
        bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    # Land: hold velocity at rest (no tendency over land).
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt

def _tracer_terms(state, p):
    """Diagnostic decomposition of dT/dt into physical terms.

    Returns the (6, nx, ny, nz) stack [adv, diff_h, diff_v, conv, gm, redi]
    of land-masked T-terms. Read-only diagnosis (blow-up attribution); NOT
    used in the time integration. Mirrors _compute_tracer_tendency.
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p)
    diff_h_T = _horizontal_tracer_diffusion(state.T, p)
    diff_v_T = _vertical_diffusion(state.T, _effective_kappa_v(p), p)

    conv_mask_3d, unstable_iface = _convective_mask(state, p)
    conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p,
                                 unstable_iface if p.localize_conv else None)

    gm_T, _, redi_T, _ = _isopycnal_closure(state, p)
    terms = [adv_T, diff_h_T, diff_v_T, conv_T, gm_T, redi_T]
    return jnp.stack([t * p.wet_mask_z for t in terms], axis=0)

def _compute_tracer_tendency(state, p, budget=None, face_transport=None, return_terms=False):
    """dT/dt, dS/dt (FD, land-masked). Includes bulk air-sea heat flux.

    Vertical advection is SUBCYCLED adv_nsub times when adv_nsub > 1 (frozen
    velocity and horizontal structure, fixed-operator subcycling like conv_nsub):
    the explicit advection CFL is dt*(|u|/dx + |v|/dy + w/dz) < ~1, dominated by
    w/dz in the thin surface layer. Only the ADVECTIVE term is subcycled --
    diffusion, surface fluxes, GM/Redi skew flux and convective adjustment are all
    inside their explicit bounds at dt=3600 and stay evaluated ONCE. (D16)
    """
    measure_terms = budget is not None or return_terms
    Fz = _vertical_transport_iface(state.u, state.v, p, face_transport=face_transport)
    if budget is not None and p.column_geometry == 'nodal_dual_v1':
        budget.tracer_transport(face_transport if face_transport is not None
                                else _layer_face_transports(state.u, state.v, p))
    n_a = int(p.adv_nsub)
    if n_a > 1:
        # The reported rate is the MEAN over the substeps, so the Strang
        # residual (which subtracts it and re-applies it over the full dt in
        # the N-step RK2) integrates the same subcycled operator over dt.
        dts = p.dt / n_a

        def _adv_sub(carry):
            tT, tS = carry
            if measure_terms:
                aT, top_T = _advection_scalar(tT, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
                aS, top_S = _advection_scalar(tS, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
                terms = (aT, aS, top_T, top_S)
            else:
                aT = _advection_scalar(tT, state.u, state.v, Fz, p, face_transport=face_transport)
                aS = _advection_scalar(tS, state.u, state.v, Fz, p, face_transport=face_transport)
                terms = (aT, aS)
            return (tT + aT * dts, tS + aS * dts), terms

        _, terms = _subcycle(_adv_sub, (state.T, state.S), n_a, p)
        adv_T, adv_S = terms[:2]
        if measure_terms:
            top_T, top_S = terms[2:]
    else:
        if measure_terms:
            adv_T, top_T = _advection_scalar(state.T, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
            adv_S, top_S = _advection_scalar(state.S, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
        else:
            adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p, face_transport=face_transport)
            adv_S = _advection_scalar(state.S, state.u, state.v, Fz, p, face_transport=face_transport)

    diff_h_T = _horizontal_tracer_diffusion(state.T, p)
    diff_h_S = _horizontal_tracer_diffusion(state.S, p)
    diff_v_T = _vertical_diffusion(state.T, _effective_kappa_v(p), p)
    diff_v_S = _vertical_diffusion(state.S, _effective_kappa_v(p), p)

    # The legacy subcycle scales both kappa and time. The candidate retains the
    # physical kappa and divides time only; neither path proves RK order. (D10)
    conv_mask_3d, unstable_iface = _convective_mask(state, p)
    iface_gate = unstable_iface if p.localize_conv else None
    n_c = int(p.conv_nsub)
    if n_c > 1:
        kappa_c = p.kappa_conv if p.process_time_scheme != 'legacy' else p.kappa_conv / n_c
        h_c = p.dt / n_c

        def _conv_sub(carry):
            tT, tS = carry
            ttT = _conv_flux_tendency(tT, conv_mask_3d, kappa_c, p, iface_gate)
            ttS = _conv_flux_tendency(tS, conv_mask_3d, kappa_c, p, iface_gate)
            return (tT + ttT * h_c, tS + ttS * h_c), (ttT, ttS)

        _, (conv_T, conv_S) = _subcycle(_conv_sub, (state.T, state.S), n_c, p)
    else:
        conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p, iface_gate)
        conv_S = _conv_flux_tendency(state.S, conv_mask_3d, p.kappa_conv, p, iface_gate)

    heat_factor = _surface_heat_weights(p) / (RHO_0 * C_P)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor
    # Bulk air-sea heat flux (Haney/Barnier): genuine SST negative feedback.
    bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1])
              * heat_factor)
    coastal_bulk_T = (p.coastal_bulk_lambda_2d[:, :, None]
                      * (p.T_atm_3d - state.T[:, :, 0:1])
                      * heat_factor)
    if getattr(p, 'dynamic_ice', False):
        heat_T = jnp.zeros_like(heat_T)
        bulk_T = jnp.zeros_like(bulk_T)
        coastal_bulk_T = jnp.zeros_like(coastal_bulk_T)

    # Surface salinity restoring (Haney): equivalent salt flux relaxing SSS
    # to climatology with timescale tau = 1/restore_coef_S. Same form as the
    # bulk heat flux — a surface tracer BC, not a body source (surface_mask).
    rest_S = (p.restore_coef_S * (p.S_ref_2d[:, :, None] - state.S[:, :, 0:1])
              * p.surface_mask)

    # Minimal thermodynamic sea-ice closure: add brine-rejection salt flux
    # where the live SST is at or below the freezing point.  This is opt-in
    # and does not alter the heat budget until enabled.
    ice_salt = (p.ice_salt_flux
                * (state.T[:, :, 0:1] <= p.ice_freeze_temp_c)
                * p.surface_mask)
    if getattr(p, 'dynamic_ice', False):
        ice_salt = jnp.zeros_like(ice_salt)

    # Diagnostic coastal surface-temperature restoring. Same Haney form as the
    # salinity restoring, but restricted to a land-adjacent mask and used only
    # to attribute how much of the SST error is local.
    rest_T = (p.coastal_restore_coef_2d[:, :, None]
              * (p.coastal_restore_T_2d[:, :, None] - state.T[:, :, 0:1])
              * p.surface_mask)

    gm_T, gm_S, redi_T, redi_S = _isopycnal_closure(state, p)

    dTdt = (adv_T + diff_h_T + diff_v_T + heat_T + bulk_T + coastal_bulk_T
            + conv_T + gm_T + redi_T + rest_T)
    dSdt = (adv_S + diff_h_S + diff_v_S + conv_S + gm_S + redi_S
            + rest_S + ice_salt)
    # Land: tracers held (no tendency over land).
    dTdt = dTdt * p.wet_mask_z
    dSdt = dSdt * p.wet_mask_z
    if budget is not None:
        budget.surface_sources((heat_T, bulk_T, coastal_bulk_T, rest_T, rest_S, ice_salt))
        budget.nonlinear_terms(((adv_T, adv_S), (conv_T, conv_S), (gm_T, gm_S), (redi_T, redi_S)))
        budget.advection_boundary_fluxes(top_T, top_S)
    if return_terms:
        processes = ((adv_T, adv_S), (conv_T, conv_S), (gm_T, gm_S), (redi_T, redi_S))
        return dTdt, dSdt, ((heat_T, bulk_T, coastal_bulk_T, rest_T, rest_S, ice_salt),
                            processes, top_T, top_S, jax.tree.map(jnp.abs, processes))
    return dTdt, dSdt

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

def _compute_tracer_residual(state, p, budget=None, face_transport=None, return_terms=False):
    """Tracer tendency minus the linear diffusion (handled by the linear step).

    The Strang split L(dt/2).N(dt).L(dt/2) handles ALL linear dissipation
    (Laplacian + biharmonic) in L, so the residual must SUBTRACT the linear terms
    the full tendency included or they are double-applied: dTdt -= kappa_h*lap and
    dTdt -= kappa_v*_d2_dz2 (the latter ran vertical diffusion at 2*kappa_v before
    the fix). The biharmonic is NOT in _compute_tracer_tendency (it is L-only), so
    it must not appear here at all -- a term here would be re-applied by N(dt) and
    cancel the L-step damping exactly (-dt/2 + dt - dt/2 = 0), a silent no-op. (D9)
    """
    result = _compute_tracer_tendency(state, p, budget=budget, face_transport=face_transport,
                                      return_terms=return_terms)
    dTdt, dSdt = result[:2]
    dTdt = dTdt - _horizontal_tracer_diffusion(state.T, p)
    dSdt = dSdt - _horizontal_tracer_diffusion(state.S, p)
    dTdt = dTdt - _vertical_diffusion(state.T, _effective_kappa_v(p), p)
    dSdt = dSdt - _vertical_diffusion(state.S, _effective_kappa_v(p), p)
    return (dTdt, dSdt, result[2]) if return_terms else (dTdt, dSdt)

def _compute_momentum_residual(state, p):
    """Momentum tendency minus linear parts (diffusion, Coriolis, bt PGF, bt wind).

    Biharmonic hyperviscosity is an L-only term (not in the full momentum tendency),
    so it must NOT appear here -- see _compute_tracer_residual for the no-op
    argument.
    """
    dudt, dvdt = _compute_momentum_tendency(state, p)
    # nu_h subtraction is unconditional: the full tendency ALWAYS includes
    # diff_h, so the residual must ALWAYS remove it -- the L step re-adds it.
    # Skipping the subtraction in split mode applied nu_h inside the N-step at
    # dt=3600, where nu_h*dt/dy^2 = 1.46 >> 0.25 exceeds the explicit CFL.
    # (D9, D12)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    # nu_v likewise: _linear_half_step applies nu_v*_d2_dz2 over dt/2 twice,
    # so omitting it here applied vertical momentum diffusion at 2*nu_v
    # (measured coefficient 2.12 per step).
    dudt = dudt - _vertical_momentum_diffusion(state.u, p)
    dvdt = dvdt - _vertical_momentum_diffusion(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # barotropic PGF from eta. The full tendency's 3D PGF contains the eta part
    # via p_bt = RHO_0*G_EARTH*eta (top-node pressure, _compute_hydrostatic_pressure)
    # differentiated by _gradient_conservative_3d with PER-LAYER wet/ghost face
    # gating. mode_split: cancel it with the SAME operator on the SAME field, so
    # the subtraction is EXACT and the external mode acts on 3D momentum solely
    # through the subcycle projection (a bare or column-gated 2D gradient leaves a
    # spurious eta-PGF wherever the layer gate differs from the column gate).
    # Monolithic: keep the historical bare _d_dx/_d_dy subtraction bit-exact to
    # the pre-split solver. (D12)
    if p.column_geometry == 'nodal_dual_v1':
        eta_pgf_x, eta_pgf_y = _reference_depth_gradient(state.eta, p)
        dudt = dudt + G_EARTH * eta_pgf_x[..., None]
        dvdt = dvdt + G_EARTH * eta_pgf_y[..., None]
    elif p.mode_split:
        eta_pgf_x, eta_pgf_y = _gradient_conservative_3d(state.eta[:, :, None], p)
        dudt = dudt + G_EARTH * eta_pgf_x
        dvdt = dvdt + G_EARTH * eta_pgf_y
    else:
        dudt = dudt + G_EARTH * _d_dx(state.eta[:, :, None], p)
        dvdt = dvdt + G_EARTH * _d_dy(state.eta[:, :, None], p)
    # barotropic PGF from density anomaly
    bt_rho_x, bt_rho_y = _compute_bt_rho_pgf(state, p)
    dudt = dudt - bt_rho_x[:, :, None]
    dvdt = dvdt - bt_rho_y[:, :, None]
    # barotropic wind
    bt_wind_x = p.tau_x_2d / (RHO_0 * p.H_sw)
    bt_wind_y = p.tau_y_2d / (RHO_0 * p.H_sw)
    dudt = dudt - bt_wind_x[:, :, None]
    dvdt = dvdt - bt_wind_y[:, :, None]
    # Mask the residual by wet_mask_z for consistency with the full tendency.
    # _compute_momentum_tendency masks dudt to 0 in ghost water, but the
    # barotropic subtractions above broadcast uniformly over ALL depths, so in
    # ghost layers the residual was -bt_pgf (a spurious PGF) instead of 0.
    # Masking closes the mismatch so the N-step residual equals the masked full
    # tendency minus its linear parts everywhere. (D12)
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt

def _linear_bottom_drag_step(state, params, duration, budget=None):
    """Exact wet-bottom drag, separately audited from numerical face filtering."""
    if params.bottom_friction != 'linear' or params.r_bot == 0.:
        return state
    decay = jnp.exp(-params.r_bot * duration * params.bottom_mask * params.wet_mask_z)
    updated = state._replace(u=state.u * decay, v=state.v * decay)
    if budget is not None:
        budget.bottom_drag(state, updated)
    return updated
