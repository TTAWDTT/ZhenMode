"""Momentum and tracer tendencies, with explicit transport and source terms."""

from zhenmode.model.config import G_EARTH, RHO_0
from zhenmode.model.solver.dynamics.pressure import (
    _compute_bt_rho_pgf,
    _compute_pressure_gradient,
    _reference_depth_gradient,
)
from zhenmode.model.solver.dynamics.transport import (
    _advection_flux_form,
    _advection_scalar,
    _compute_vertical_velocity,
    _layer_face_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.numerics.horizontal import (
    _d_dx,
    _d_dy,
    _gradient_conservative_3d,
    _horizontal_tracer_diffusion,
    _laplacian_h,
)
from zhenmode.model.solver.physics.eos import heat_capacity
from zhenmode.model.solver.physics.isopycnal import _isopycnal_closure
from zhenmode.model.solver.physics.surface import _surface_heat_weights
from zhenmode.model.solver.physics.vertical import (
    _conv_flux_tendency,
    _convective_mask,
    _effective_kappa_v,
    _vertical_diffusion,
    _vertical_momentum_diffusion,
)
from zhenmode.model.solver.timestepping.subcycles import _subcycle


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
    if budget is not None and p.column_geometry in {'nodal_dual_v1', 'fixed_partial_v1'}:
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

    heat_factor = _surface_heat_weights(p) / (RHO_0 * heat_capacity(p))
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
    if p.column_geometry == 'fixed_partial_v1':
        # Contact forces vary with node capacity. Cancel the complete eta
        # force, not its column mean; eta belongs to the external-mode step.
        eta_pgf_x, eta_pgf_y = _gradient_conservative_3d(state.eta[:, :, None], p)
        dudt = dudt + G_EARTH * eta_pgf_x
        dvdt = dvdt + G_EARTH * eta_pgf_y
    elif p.column_geometry == 'nodal_dual_v1':
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
