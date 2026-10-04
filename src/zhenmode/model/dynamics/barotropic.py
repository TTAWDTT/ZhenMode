"""Canonical barotropic definitions; legacy operations unchanged."""
from zhenmode.model.config.definitions import G_EARTH, RHO_0
from zhenmode.model.dynamics.pressure import _compute_bt_rho_pgf, _reference_depth_gradient
from zhenmode.model.dynamics.transport import (
    _barotropic_velocity,
    _column_divergence,
    _face_transport_divergence,
    _layer_face_transports,
    _reference_depth_divergence,
)
from zhenmode.model.numerics.backend import jnp
from zhenmode.model.numerics.horizontal import (
    _apply_polar_cap,
    _divergence_conservative,
    _gradient_conservative,
)
from zhenmode.model.timestepping.subcycles import _subcycle


def _refill_volume(eta_now, eta_before, area_cell, area_ocean, p):
    """Spread a decay-removed volume uniformly back over the wet domain.

    A Rayleigh decay changes global volume by sum(A*(eta_now - eta_before)).
    Adding that back as a uniform eta offset conserves total volume exactly,
    and a uniform offset has zero PGF so the dynamics are untouched. Used by
    the lateral sponge and the semi-enclosed-sea eta relaxation.
    """
    dV = jnp.sum(area_cell * (eta_now - eta_before))
    return eta_now + (-dV / area_ocean) * p.wet_mask

def _filter_barotropic_eta(eta, params, duration):
    """Retain the actual sponge/relaxation/cap changes, independently of transport."""
    area = params.dx_2d * params.dy
    ocean_area = jnp.maximum(jnp.sum(area * params.wet_mask), 1.0)
    decay = jnp.exp(-params.sponge_rate_2d * duration)
    updated = _refill_volume(eta * decay, eta, area, ocean_area, params)
    relaxation = jnp.exp(-params.eta_relax_rate * duration * params.eta_relax_mask)
    updated = _refill_volume(updated * relaxation, updated, area, ocean_area, params)
    return _apply_polar_cap(updated, params.wet_mask, params)

def _symmetric_free_surface_step(eta, velocity_x, velocity_y, params, duration,
                                 forcing_x=None, forcing_y=None, column_face_transport=None):
    """Drift-kick-drift with constrained midpoint rotation and actual half-step faces."""
    normal_mask = params.interior_mask_z[..., 0]
    velocity_x = velocity_x * params.wet_mask
    velocity_y = velocity_y * params.wet_mask * normal_mask
    if column_face_transport is None:
        layers = _layer_face_transports(velocity_x[..., None], velocity_y[..., None], params)
        first_faces = tuple(jnp.sum(flux, axis=-1) for flux in layers)
    else:
        first_faces = column_face_transport
    midpoint_eta = eta - (duration / 2.) * _face_transport_divergence(*first_faces, params)
    pressure_x, pressure_y = _reference_depth_gradient(midpoint_eta, params)
    source_x = params.tau_x_2d / (RHO_0 * params.H_sw)
    source_y = params.tau_y_2d / (RHO_0 * params.H_sw)
    if forcing_x is not None:
        source_x, source_y = source_x + forcing_x, source_y + forcing_y
    half_rotation = params.f * normal_mask * (duration / 2.)
    right_x = velocity_x + half_rotation * velocity_y + duration * (-G_EARTH * pressure_x + source_x)
    right_y = (velocity_y - half_rotation * velocity_x
               + duration * (-G_EARTH * pressure_y + source_y)) * normal_mask
    denominator = 1. + half_rotation * half_rotation
    next_x = ((right_x + half_rotation * right_y) / denominator) * params.wet_mask
    next_y = ((right_y - half_rotation * right_x) / denominator) * params.wet_mask * normal_mask
    delta_layers = _layer_face_transports((next_x - velocity_x)[..., None],
                                         (next_y - velocity_y)[..., None], params)
    second_faces = tuple(first + jnp.sum(delta, axis=-1)
                         for first, delta in zip(first_faces, delta_layers, strict=True))
    transported_eta = midpoint_eta - (duration / 2.) * _face_transport_divergence(*second_faces, params)
    mean_faces = tuple(0.5 * (first + second) for first, second in zip(first_faces, second_faces, strict=True))
    final_eta = _filter_barotropic_eta(transported_eta * params.wet_mask, params, duration)
    decay = jnp.exp(-params.sponge_rate_2d * duration)
    final_x = _apply_polar_cap(next_x * decay, params.wet_mask, params)
    final_y = _apply_polar_cap(next_y * decay, params.wet_mask, params) * normal_mask
    return (final_eta, final_x, final_y), mean_faces

def _free_surface_step_fd(eta, u, v, p, F_rho_x=None, F_rho_y=None, dt_half=None,
                          column_divergence_offset=None, column_face_transport=None):
    """Forward-backward (Sielecki) free-surface (shallow water) step on lat-lon FD.

        eta^{n+1} = eta^n - dt*H_sw*div_h(ubt^n)                   # eta from OLD u
        ubt^{n+1} = (ubt^n + dt*(-g*grad_h(eta^{n+1}) + F)) / (1 + r_bt*dt)

    with implicit linear barotropic bottom drag (unconditionally stable). The
    forward-forward pairing (both from the old state) is unconditionally unstable
    for free gravity waves -- |lambda| = sqrt(1+(dt*c*k)^2) > 1 for ALL k -- while
    forward-backward gives |lambda| = 1 under the external-wave CFL dt < dx/sqrt(gH).
    (D22)

    mode_split=True: called n_subcyc times per baroclinic step with dt_bt, one
    forward-backward pair per call; the 3D<->barotropic projection stays in the
    CALLER, so the subcycle sees only the 2D (eta, ubt, vbt) state. The subcycle
    carries NO nu_h: the baroclinic shear nu_h must damp is invisible to the
    depth-averaged BT state, so nu_h runs in the L half-steps on the full 3D field
    with its own subcycle. (D12)
    """
    if dt_half is None:
        dt_half = p.dt / 2.0
    if p.process_time_scheme == 'symmetric_fast_v3':
        return _symmetric_free_surface_step(eta, u, v, p, dt_half, F_rho_x, F_rho_y,
                                            column_face_transport)[0]
    if p.mode_split:
        ubt, vbt = u, v   # subcycle mode: caller passes BT velocity directly
    else:
        ubt, vbt = _barotropic_velocity(u, v, p)
        if p.column_geometry == 'nodal_dual_v1':
            column_divergence_offset = _column_divergence(u, v, p) - _reference_depth_divergence(ubt, vbt, p)

    F_x = jnp.zeros_like(ubt)
    F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x
        F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)

    r_bt = p.r_bot if p.bottom_friction == 'linear' and p.column_geometry == 'legacy' else 0.0
    drag = 1.0 / (1.0 + r_bt * dt_half)
    # Mass-conserving (flux-form) divergence: face fluxes zeroed at wet/dry
    # interfaces so the divergence telescopes to zero over the wet domain. The
    # centered (roll) form leaks volume at coastlines and closed walls. Mask
    # ubt/vbt to wet first so the face averages carry no land values. (D2)
    if p.column_geometry == 'nodal_dual_v1':
        if column_face_transport is not None:
            column_transport_divergence = _face_transport_divergence(*column_face_transport, p)
        else:
            column_transport_divergence = _reference_depth_divergence(ubt, vbt, p)
            if column_divergence_offset is not None:
                column_transport_divergence = column_transport_divergence + column_divergence_offset
    else:
        div_bt = _divergence_conservative(ubt * p.wet_mask, vbt * p.wet_mask, p)
        column_transport_divergence = p.H_sw * div_bt

    # Forward-backward (Sielecki) free-surface coupling: update eta FIRST (old
    # velocity), then update barotropic momentum using the NEW eta gradient.
    # Forward-forward (both from the old state) is unconditionally unstable for
    # free gravity waves, |lambda| = sqrt(1+(dt*c*k)^2) > 1 for ALL k -- CFL does
    # not save forward-Euler. Forward-backward flips the trace to
    # 2 - dt^2*g*H*k^2 => |lambda| = 1 under CFL < 1, the standard OGCM
    # discretization (MOM6/ROMS/NEMO); bottom drag then decays the free mode.
    # (D22)
    if p.column_geometry == 'nodal_dual_v1':
        eta_new = eta - dt_half * column_transport_divergence
    else:
        eta_new = eta - dt_half * p.H_sw * div_bt

    eta_new = eta_new * p.wet_mask
    # Lateral sponge on eta (2D). No-op when sponge_rate_2d == 0.
    sw_decay = jnp.exp(-p.sponge_rate_2d * dt_half)
    # MASS-CONSERVING SPONGE: the bare decay eta *= sw_decay changes global
    # volume by dV = sum(A*eta*(sw_decay-1)) over the band, and the wind setup
    # makes the band-mean eta NEGATIVE on both hemispheres, so the sponge was
    # ADDING volume every step. Correct by returning the removed volume
    # UNIFORMLY over the wet domain: total volume is exactly conserved, the local
    # anomaly damping is unchanged, and a uniform eta offset has zero PGF so the
    # dynamics are untouched. (D23)

    # ── Semi-enclosed-sea eta relaxation (Mediterranean artifact fix) ──
    # Gibraltar (14 km wide) is sub-grid on a 1 deg mesh: the one-cell strait
    # cannot support the observed two-layer exchange, and the residual pressure
    # mismatch drives a spurious NET outflow that drains the basin linearly.
    # Standard OGCM remedy (MOM-family): Rayleigh-relax eta toward the basin
    # equilibrium INSIDE the sea only, eta *= exp(-eta_relax_rate*dt_half) where
    # eta_relax_mask = 1 (eta == 0 is the correct equilibrium: the basin-mean PGF
    # at Gibraltar then matches the Atlantic at the same latitude).
    # MASS-CONSERVING COMPENSATION: the same uniform-refill trick as the sponge
    # returns the removed volume over the global wet domain; a uniform eta offset
    # has zero PGF, so the relaxation damps the basin anomaly, not its water
    # mass. (D23)

    # Polar-cap filter: zonally average the poleward rows to kill the
    # cos(lat)->0 metric singularity (dx->0 makes the explicit SW CFL
    # unattainable at the edge).
    #
    # CRITICAL: average over WET points only and write back to WET points only.
    # The polar rows are ~30% land, and a naive all-column mean mixes ocean
    # (eta != 0) with land (eta == 0), forcing a zonally-uniform value that
    # creates a spurious PGF at EVERY coastline point in the band. (D21)
    def _cap(field2d):
        return _apply_polar_cap(field2d, p.wet_mask, p)

    # CONSISTENT-TRIPLE CAP: cap eta FIRST, then drive the barotropic momentum
    # update from the CAPPED eta's pressure gradient, then cap the resulting
    # ubt/vbt. Capping all three to independent zonal means leaves eta zonally
    # uniform but ubt/vbt carrying a different zonal structure, so the next
    # step's div(ubt) and grad(eta) are dynamically inconsistent. Capping eta
    # first makes ubt/vbt consistent with eta by construction; their cap is then
    # a CFL-safety smoothing, not an independent forcing. (D21)
    eta_new = _filter_barotropic_eta(eta_new, p, dt_half)
    # Energy-consistent PGF: the exact adjoint of the conservative divergence
    # (area-weighted), so the FB pair is neutral on the masked non-uniform grid.
    # The centered _d_dx/_d_dy gradient is NOT the adjoint and injects energy.
    if p.column_geometry == 'nodal_dual_v1':
        grad_eta_x, grad_eta_y = _reference_depth_gradient(eta_new, p)
    else:
        grad_eta_x, grad_eta_y = _gradient_conservative(eta_new, p)
    # Barotropic momentum with PGF + forcing (intermediate star state).
    u_star = ubt + dt_half * (-G_EARTH * grad_eta_x + F_x)
    v_star = vbt + dt_half * (-G_EARTH * grad_eta_y + F_y)
    # Semi-implicit barotropic Coriolis. The shallow-water momentum eqn is
    #   du/dt - f*v = -g*grad(eta) + F ;  dv/dt + f*u = -g*grad(eta) + F
    # Legacy backward Euler is stable but damps kinetic energy:
    #   u_new = (u_star + f*dt*v_star) / (1+(f*dt)^2)
    #   v_new = (v_star - f*dt*u_star) / (1+(f*dt)^2)
    # Without it the barotropic PGF has no geostrophic balance: it drives a
    # convergent ubt that grows eta monotonically. This is the barotropic
    # analogue of the 3D rotation in _linear_half_step. (D22)
    if p.process_time_scheme != 'legacy':
        half_rotation = p.f * (dt_half / 2.)
        right_x = u_star + half_rotation * vbt
        right_y = v_star - half_rotation * ubt
        denominator = 1. + half_rotation * half_rotation
        ubt_new = (right_x + half_rotation * right_y) / denominator
        vbt_new = (right_y - half_rotation * right_x) / denominator
    else:
        fd = p.f * dt_half
        denom = 1.0 + fd * fd
        ubt_new = (u_star + fd * v_star) / denom
        vbt_new = (v_star - fd * u_star) / denom
    ubt_new = ubt_new * drag
    vbt_new = vbt_new * drag

    ubt_new = ubt_new * p.wet_mask
    vbt_new = vbt_new * p.wet_mask
    ubt_new = ubt_new * sw_decay
    vbt_new = vbt_new * sw_decay
    ubt_new = _cap(ubt_new)
    vbt_new = _cap(vbt_new)

    if p.mode_split:
        # Subcycle mode: caller works on the 2D barotropic state directly
        # (eta, ubt, vbt); the single 3D<->BT projection happens ONCE per
        # baroclinic step in _step_impl, not per subcycle.
        return eta_new, ubt_new, vbt_new

    # Project barotropic delta back to 3D velocity (uniform over depth)
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    projection_mask = p.wet_mask_z if p.column_geometry == 'nodal_dual_v1' else 1.
    u_new = u + delta_ubt * projection_mask
    v_new = v + delta_vbt * projection_mask
    # No-flux wall: enforce zero normal velocity at the N/S boundary rows on
    # the projected 3D v too, consistent with _step_impl's final mask.
    v_new = v_new * p.interior_mask_z
    return eta_new, u_new, v_new

def _barotropic_subcycle_transport(state, params):
    """Return actual OLD-face time mean and eta changes not caused by transport.

    The shear is frozen at the post-L/N/L predictor. Each OLD barotropic value
    is lifted onto that shear before recording the open-face volume flux. The
    optional transport-matched path drives eta with those same faces. Filtering,
    sponge and eta relaxation are recorded separately, not fitted into a flux.
    """
    if params.process_time_scheme == 'symmetric_fast_v3':
        state = state._replace(v=state.v * params.interior_mask_z)
    forcing_x, forcing_y = _compute_bt_rho_pgf(state, params)
    initial_u, initial_v = _barotropic_velocity(state.u, state.v, params)
    offset = (_column_divergence(state.u, state.v, params)
              - _reference_depth_divergence(initial_u, initial_v, params))
    zero = jnp.zeros_like(state.eta)

    def advance(carry):
        eta, mean_u, mean_v, total_x, total_y, filter_change = carry
        velocity_x = state.u + (mean_u - initial_u)[..., None] * params.wet_mask_z
        velocity_y = state.v + (mean_v - initial_v)[..., None] * params.wet_mask_z
        layers = _layer_face_transports(velocity_x, velocity_y, params)
        faces = tuple(jnp.sum(flux, axis=-1) for flux in layers)
        if params.match_barotropic_transport:
            divergence = _face_transport_divergence(*faces, params)
        else:
            divergence = _reference_depth_divergence(mean_u, mean_v, params) + offset
        if params.process_time_scheme == 'symmetric_fast_v3':
            updated, faces = _symmetric_free_surface_step(
                eta, mean_u, mean_v, params, params.dt_bt, forcing_x, forcing_y, faces)
            divergence = _face_transport_divergence(*faces, params)
        else:
            updated = _free_surface_step_fd(
                eta, mean_u, mean_v, params, forcing_x, forcing_y, dt_half=params.dt_bt,
                column_divergence_offset=offset,
                column_face_transport=faces if params.match_barotropic_transport else None)
        eta_new, mean_u_new, mean_v_new = updated
        nontransport_change = eta_new - (eta - params.dt_bt * divergence)
        return (eta_new, mean_u_new, mean_v_new, total_x + faces[0], total_y + faces[1],
                filter_change + nontransport_change), None

    final, _ = _subcycle(advance, (state.eta, initial_u, initial_v, zero, zero, zero), int(params.n_subcyc), params)
    eta, mean_u, mean_v, total_x, total_y, filter_change = final
    column_transport = (total_x / params.n_subcyc, total_y / params.n_subcyc)
    updated = state._replace(eta=eta,
                             u=state.u + (mean_u - initial_u)[..., None] * params.wet_mask_z,
                             v=state.v + (mean_v - initial_v)[..., None] * params.wet_mask_z)
    return updated, column_transport, filter_change
