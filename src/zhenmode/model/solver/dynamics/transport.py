"""Wet-face transport, continuity and tracer advection."""

from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.numerics.channel import channel_face_velocities
from zhenmode.model.solver.numerics.contacts import (
    contact_divergence,
    contact_material_derivative,
    contact_transports,
    neighbours,
)
from zhenmode.model.solver.numerics.horizontal import (
    _dealias_h_fd,
    _divergence_h,
    _face_thickness,
    _gradient_face_gated_3d,
)
from zhenmode.model.solver.numerics.vertical import _d_dz, _fill_ghost_bottom


def _column_divergence(u, v, p):
    """Column-integrated horizontal divergence (nx, ny).

    SUM_k div_h[k]*dz_node[k], built from the same per-layer face-flux operator the
    advection budget uses, so a zero here is exactly the discretely
    divergence-free condition the tracer flux operator needs. (D13)
    """
    return jnp.sum(_divergence_h(u, v, p) * p.dz_node, axis=-1) * p.wet_mask

def _layer_face_transports(velocity_x, velocity_y, params):
    """Static nodal volume flux per face width; y includes cos(face latitude)."""
    if getattr(params,'column_geometry','legacy')=='fixed_partial_v1':
        return contact_transports(velocity_x,velocity_y,params)
    if getattr(params, 'pressure_continuity_scheme', 'centered_second') == 'centered_fourth':
        fx, fy = channel_face_velocities(velocity_x, velocity_y)
        return fx * params.dz_node, fy * params.dz_node
    thickness_x = _face_thickness(params, 0)
    thickness_y = _face_thickness(params, 1)
    cosine_face = 0.5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    flux_x = 0.5 * (velocity_x + jnp.roll(velocity_x, -1, axis=0)) * thickness_x
    flux_y = 0.5 * (velocity_y + jnp.roll(velocity_y, -1, axis=1)) * thickness_y * cosine_face[None, :, None]
    return flux_x, flux_y.at[:, -1].set(0.)

def _face_transport_divergence(flux_x, flux_y, params):
    """Divergence of layer or column face transports, with closed y walls."""
    if flux_x.ndim==4:
        return contact_divergence(flux_x,flux_y,params)
    incoming_y = jnp.roll(flux_y, 1, axis=1).at[:, 0].set(0.)
    inverse_dx = params.inv_dx[..., :1] if flux_x.ndim == 3 else params.inv_dx[..., 0]
    cosine = params.cos_lat[None, :, None] if flux_y.ndim == 3 else params.cos_lat[None, :]
    return ((flux_x - jnp.roll(flux_x, 1, axis=0)) * inverse_dx
            + (flux_y - incoming_y) * params.inv_dy / cosine)


def _sum_layer_transports(flux):
    """Sum reference-node and contact axes into one physical column face."""
    return jnp.sum(flux,axis=(0,-1)) if flux.ndim==4 else jnp.sum(flux,axis=-1)

def _match_layer_face_transports(velocity_x, velocity_y, column_transport, params):
    """Match the fast-mode time mean at every open face, retaining layer shear."""
    fluxes = _layer_face_transports(velocity_x, velocity_y, params)
    corrected = []
    for axis, flux, target in zip((0, 1), fluxes, column_transport, strict=True):
        if getattr(params,'column_geometry','legacy')=='fixed_partial_v1':
            thickness=params.face_contacts[axis]
            depth=_sum_layer_transports(thickness)
            weights=thickness/jnp.where(depth>0.,depth,1.)[None,...,None]
            corrected.append(flux+weights*(target-_sum_layer_transports(flux))[None,...,None])
            continue
        thickness = _face_thickness(params, axis)
        depth = jnp.sum(thickness, axis=-1, keepdims=True)
        weights = thickness / jnp.where(depth > 0., depth, 1.)
        corrected.append(flux + weights * (target - jnp.sum(flux, axis=-1))[..., None])
    return tuple(corrected)

def _vertical_transport_iface(u, v, p, face_transport=None):
    """Interface volume transports Fz (nx, ny, nz+1), DOWNWARD-positive.

    The EXACT discrete inverse of _divergence_h, cumulated from the seafloor up:
    Fz[nz] = 0 and (Fz[k+1] - Fz[k])/dz_node[k] = -div_h[k], weighted by the
    per-node thickness dz_node (NOT dz_3d, which is interface-centred, length nz-1).
    Feeding this to the donor-cell vertical flux makes div_x + div_y + div_z cancel
    POINTWISE for a uniform tracer; a node-based w leaves a T-proportional residual
    no CFL limit removes. Fz[..., 0] is the column-integrated horizontal divergence,
    the rigid-lid leak the barotropic subcycle absorbs via Fz_top = Fz[0]*T[0]. (D7)
    """
    if face_transport is not None:
        integrand = _face_transport_divergence(*face_transport, p)
        accumulated = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
        return jnp.concatenate([accumulated, jnp.zeros_like(accumulated[..., :1])], axis=-1)
    div_h = _divergence_h(u, v, p)
    # div_h is per-LAYER (nz entries) — weight by the per-node cell thickness
    # dz_node, NOT dz_3d (interface-centred, length nz-1).
    integrand = div_h * p.dz_node                    # (nx, ny, nz)
    # cumsum from the bottom upward: Fz_int[k] = sum_{m=k..nz-1} integrand[m]
    Fz_int = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)

def _barotropic_velocity(u, v, p):
    """Depth-averaged (barotropic) horizontal velocity."""
    if p.column_geometry in {'nodal_dual_v1', 'fixed_partial_v1'}:
        return jnp.sum(u * p.dz_norm, axis=-1), jnp.sum(v * p.dz_norm, axis=-1)
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(u_avg * p.dz_norm, axis=-1)
    vbt = jnp.sum(v_avg * p.dz_norm, axis=-1)
    return ubt, vbt

def _reference_depth_divergence(ubt, vbt, p):
    """Divergence of common-wet nodal face transport, not H_cell * div(u)."""
    return _column_divergence(ubt[..., None], vbt[..., None], p)

def _compute_vertical_velocity(state, p):
    """Diagnose w from horizontal continuity. w=0 at bottom. Masked on land.

    w = -Fz, the exact interface transports of _vertical_transport_iface sign-flipped
    to the w>0 = DOWNWARD convention. A trapezoid node-w built from div_avg + cumsum
    could not invert the advection's discrete face-flux divergence pointwise, leaving
    a T-proportional source (an exponential pump); the exact inverse closes
    div_x + div_y + div_z to machine precision for a uniform tracer. No dealiasing:
    dealiasing w would re-open the non-closure the exact inverse just fixed.
    (D6, D7)
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    w = -Fz[..., :-1] * p.wet_mask_z
    return w

def _advection_flux_form(u, v, w, p):
    """3D advective-form momentum advection (FD, land-masked, dealiased).

    Advective form avoids the spurious u*div_h source; momentum keeps it while
    tracers use the flux form (D15). Horizontal gradients use
    _gradient_face_gated_3d (D14); the vertical gradient stays the bare _d_dz,
    because w is masked to zero in ghost layers and the terms are multiplied by
    wet_mask_z, so a wet/ghost vertical face carries no flux whatever dT/dz reads.
    The summed nonlinear tendency is de-aliased once by _dealias_h_fd, then
    land-masked. (D25)
    """
    du_dz = _d_dz(_fill_ghost_bottom(u, p), p)
    dv_dz = _d_dz(_fill_ghost_bottom(v, p), p)
    closed_faces = getattr(p, 'meridional_boundary_scheme', 'clamped_nodes') == 'closed_faces'
    if getattr(p, 'column_geometry', 'legacy') == 'fixed_partial_v1':
        faces = contact_transports(u, v, p)
        adv_u = -(contact_material_derivative(u, faces, p) + w * du_dz)
        adv_v = -(contact_material_derivative(v, faces, p) + w * dv_dz)
    else:
        du_dx, du_dy = _gradient_face_gated_3d(u, p)
        dv_dx, dv_dy = _gradient_face_gated_3d(v, p, normal=closed_faces)
        adv_u = -(u * du_dx + v * du_dy + w * du_dz)
        adv_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    adv_u = _dealias_h_fd(adv_u, p)
    adv_v = _dealias_h_fd(adv_v, p, normal=closed_faces)
    return adv_u * p.wet_mask_z, adv_v * p.wet_mask_z

def _limited_tracer_slope(tracer, wet, axis):
    """Minmod reconstruction using only wet neighbours and physical boundaries."""
    if axis == 1:
        padding = [(0, 0), (1, 1), (0, 0)]
        padded = jnp.pad(tracer, padding, mode='edge')
        padded_wet = jnp.pad(wet, padding, mode='edge')
        previous, following = padded[:, :-2], padded[:, 2:]
        previous_wet, following_wet = padded_wet[:, :-2], padded_wet[:, 2:]
    else:
        previous, following = jnp.roll(tracer, 1, axis=axis), jnp.roll(tracer, -1, axis=axis)
        previous_wet, following_wet = jnp.roll(wet, 1, axis=axis), jnp.roll(wet, -1, axis=axis)
    left_delta, right_delta = tracer - previous, following - tracer
    signs = jnp.sign(left_delta) + jnp.sign(right_delta)
    return (0.5 * signs * jnp.minimum(jnp.abs(left_delta), jnp.abs(right_delta))
            * wet * previous_wet * following_wet)

def _legacy_horizontal_tracer_fluxes(T,u,v,p,face_transport):
    wm=p.wet_mask_z
    if (face_transport is None
            and getattr(p, 'pressure_continuity_scheme', 'centered_second') == 'centered_fourth'):
        face_transport = _layer_face_transports(u, v, p)
    # ── Zonal flux at face (i+1/2), periodic in x ──
    # Fx = u_face * T_face (centered unless monotone_adv), face-gated on both
    # cells wet; +x-directed (u>0 carries T eastward).
    ux_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
    if face_transport is not None:
        ux_face = face_transport[0] / p.dz_node
    if p.fct_adv:
        # TVD/MUSCL flux limiter: reconstruct from the left and right cells with a
        # minmod slope, then choose the state consistent with the face velocity.
        # This is a compact local limiter, not a full Zalesak 3D FCT, but it is
        # flux-form, conservative, and removes the centered scheme's overshoot at
        # a sharp front while remaining second-order in smooth regions.
        slope_x = _limited_tracer_slope(T, wm, axis=0)
        Tx_face = jnp.where(
            ux_face >= 0.0,
            T + 0.5 * slope_x,
            jnp.roll(T, -1, axis=0) - 0.5 * jnp.roll(slope_x, -1, axis=0))
    elif p.monotone_adv:
        # Donor-cell face value: the upstream cell (positive ux takes T[i],
        # negative ux takes T[i+1]). Lower order than centered flux, but
        # conservative and positivity-preserving for Courant <= 1.
        Tx_face = jnp.where(ux_face >= 0.0, T, jnp.roll(T, -1, axis=0))
    else:
        Tx_face = 0.5 * (T + jnp.roll(T, -1, axis=0))
    gate_x = wm * jnp.roll(wm, -1, axis=0)
    Fx = ux_face * Tx_face * gate_x                    # (nx, ny, nz)
    # ── Meridional flux at face (j+1/2), closed N/S walls ──
    # The v*T flux carries the same cos(face_lat) factor as the mass flux in
    # _divergence_conservative, and the cell divergence divides by cos(cell).
    pad = [(0, 0), (1, 1), (0, 0)]
    T_pad = jnp.pad(T, pad, mode='edge')
    v_pad = jnp.pad(v, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    vy_face = 0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
    if face_transport is not None:
        cosine_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
        vy_face = face_transport[1] / (p.dz_node * cosine_face[None, :, None])
    if p.fct_adv:
        # Same TVD/MUSCL limiter as x, but on the padded row. The last face is
        # closed explicitly below, so the edge-padded value cannot enter the
        # budget.
        slope_y = _limited_tracer_slope(T, wm, axis=1)
        slope_y_pad = jnp.pad(slope_y, pad, mode='edge')
        Ty_face = jnp.where(
            vy_face >= 0.0,
            T_pad[:, 1:-1] + 0.5 * slope_y_pad[:, 1:-1],
            T_pad[:, 2:] - 0.5 * slope_y_pad[:, 2:])
    elif p.monotone_adv:
        # Same donor-cell choice as x; the last face is closed below, so the
        # edge-padded value cannot enter.
        Ty_face = jnp.where(vy_face >= 0.0, T_pad[:, 1:-1], T_pad[:, 2:])
    else:
        Ty_face = 0.5 * (T_pad[:, 1:-1] + T_pad[:, 2:])
    gate_y = wm_pad[:, 1:-1] * wm_pad[:, 2:]
    cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    Fy = vy_face * Ty_face * gate_y * cos_face[None, :, None]   # (nx, ny, nz)
    # The edge pad replicates wm at the last row, so Fy[:, -1] would carry a
    # flux through the closed wall (v is masked to 0 there, but vy_face reads
    # the interior neighbour). Close it explicitly, as
    # _divergence_conservative does.
    Fy = Fy.at[:, -1].set(0.0)
    return Fx,Fy


def _advection_scalar(T, u, v, Fz_in, p, return_boundary=False, face_transport=None):
    """3D FLUX-FORM scalar advection (FD, land-masked, NOT dealiased).

    Flux form (-div(uT)) rather than advective form: the two differ by +T*div(u), a
    REAL anti-diffusion whose positive eigenvalue dt*delta grows the field
    exponentially at any nonzero discrete divergence, and subcycling only slows it
    without removing it. Flux form is exactly conservative and matches the
    continuity-consistent tracer equation the free-surface subcycle solves; momentum
    keeps the advective form. (D15)

    No dealiasing here, unlike _advection_flux_form: the flux form's value is that
    its column divergence telescopes exactly into the boundary fluxes, which is what
    closes the column heat budget, and _dealias_h_fd would break it. Horizontal
    fluxes are face-gated at mask boundaries (D14); the vertical flux is donor-cell
    on Fz_in from _vertical_transport_iface, the exact discrete inverse of the
    horizontal divergence, so the column transport is exactly conservative and
    monotone. (D7, D16)

    ``fct_adv`` is a compact TVD/MUSCL flux limiter: it reconstructs the face
    state from the two donor cells with a minmod slope and chooses the state
    consistent with the face velocity. It is not yet a full Zalesak multidimensional
    FCT limiter, but it is conservative and removes the centered scheme's local
    overshoot at a sharp front.
    """
    wm = p.wet_mask_z
    partial = getattr(p, 'column_geometry', 'legacy') == 'fixed_partial_v1'
    if partial and face_transport is None:
        face_transport = _layer_face_transports(u, v, p)
    if partial:
        tracer_fluxes=[]
        for axis,flux in enumerate(face_transport):
            other=neighbours(T,axis)
            if p.fct_adv:
                slope=_limited_tracer_slope(T,wm,axis)
                left=T[None]+.5*slope[None]
                right=other-.5*neighbours(slope,axis)
                concentration=jnp.where(flux>=0.,left,right)
            elif p.monotone_adv:
                concentration=jnp.where(flux>=0.,T[None],other)
            else:
                concentration=.5*(T[None]+other)
            tracer_fluxes.append(flux*concentration)
        horizontal_div=contact_divergence(*tracer_fluxes,p)/p.dz_node
    else:
        Fx,Fy=_legacy_horizontal_tracer_fluxes(T,u,v,p,face_transport)
        Fx_up=jnp.roll(Fx,1,axis=0)
        Fy_up=jnp.roll(Fy,1,axis=1).at[:,0].set(0.)
        horizontal_div=((Fx-Fx_up)*p.inv_dx[...,0:1]
                        +(Fy-Fy_up)*p.inv_dy/p.cos_lat[None,:,None])
    # ── Vertical flux at interface (k+1/2): DONOR-CELL (upwind) ──
    # Fz is the DOWNWARD-positive interface transport (nx, ny, nz+1) from
    # _vertical_transport_iface; the donor is the SHALLOWER node when Fz > 0 and
    # the DEEPER node when Fz < 0. The face value must NOT be centered: centered
    # vertical transport is unconditionally unstable under RK2/forward-Euler,
    # and the centered value multiplies the full tracer content into the surface
    # node wherever Fz[0] != 0. Donor-cell is monotone and adds only
    # 0.5*|w|*dz of implicit diffusivity (below kappa_v). The bottom face reads
    # _fill_ghost_bottom(T) for the seafloor no-flux BC. (D16)
    T_deep = _fill_ghost_bottom(T, p)[..., 1:]        # node k (below iface k)
    T_shallow = T[..., :-1]                           # node k-1 (above iface k)
    wet_iface = wm[..., :-1] * wm[..., 1:]
    # Internal interfaces are Fz_in[..., 1:-1] (indices 1..nz-1, count nz-1):
    # index 0 is the surface (rigid-lid leak, handled below) and index nz is
    # the seafloor (zero transport).
    Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep) * wet_iface
    # Tendency is OUT-minus-IN: k increases DOWNWARD, so cell k's out-face is
    # its bottom (k+1/2) and its in-face the top (k-1/2). Swapping to (up - dn)
    # makes the donor-cell scheme ANTI-upwind. (D16)
    # Top face: Fz[..., 0] is the column-integrated horizontal divergence, the
    # rigid-lid leak the barotropic subcycle absorbs as the eta tendency. The
    # MOM-style rigid-lid closure carries the surface cell's own value,
    # Fz_top = Fz[0]*T[0], so upwelling equilibrates the surface node at the
    # deep value instead of growing without bound. (D7, D16)
    Fz_top = Fz_in[:, :, :1] * T[..., :1]
    up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
    dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
    div_z = (dn-up)/p.dz_node
    adv_T = -(horizontal_div+div_z)
    tendency = adv_T * p.wet_mask_z
    if return_boundary:
        return tendency, Fz_top[..., 0]
    return tendency
