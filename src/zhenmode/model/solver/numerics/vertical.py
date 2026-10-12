"""Nonuniform vertical stencils and no-flux interface balances."""

from zhenmode.model.solver.numerics.backend import jax, jnp


def _vertical_tracer_face_values(tracer, p):
    """Limited upwind candidates on midpoint faces of a nodal dual column.

    Slopes use physical node distances, preserve affine interior profiles and never
    reconstruct through a dry face. These are face bounds, not a claim of
    multidimensional TVD for the complete free-surface time integration.
    """
    wet = p.wet_mask_z > 0.5
    active = wet[..., :-1] & wet[..., 1:]
    secant = jnp.where(active, (tracer[..., 1:] - tracer[..., :-1]) / p.dz_iface, 0.)
    left, right = secant[..., :-1], secant[..., 1:]
    slope = jnp.where(left * right > 0., jnp.sign(left) * jnp.minimum(jnp.abs(left), jnp.abs(right)), 0.)
    # Boundary nodes are not cell centroids. One-sided extrapolation can
    # export positive inventory from a zero endpoint at arbitrarily small dt.
    # Retain donor states at these physical endpoints.
    slopes = jnp.concatenate((jnp.zeros_like(secant[..., :1]), slope, jnp.zeros_like(secant[..., -1:])), axis=-1)
    above = tracer[..., :-1] + .5 * p.dz_iface * slopes[..., :-1]
    below = tracer[..., 1:] - .5 * p.dz_iface * slopes[..., 1:]
    lo, hi = jnp.minimum(tracer[..., :-1], tracer[..., 1:]), jnp.maximum(tracer[..., :-1], tracer[..., 1:])
    return jnp.clip(above, lo, hi), jnp.clip(below, lo, hi)


def _conservative_endpoint_flux(tracer, mass_flux, flux, horizontal_div, p, minimum=None,
                                face_states=None, horizontal_numerator=None):
    """Convexly limited nodal endpoint consistency correction.

    End nodes are physical points. Impose their horizontal material transport
    plus the top vertical point derivative, and distribute the required flux
    correction with minimum interior volume-weighted tendency norm. A single
    column blend keeps the Euler state within the column/uncorrected envelope.
    This conserves boundary inventory exactly; it is not a global TVD proof.
    """
    widths = jnp.broadcast_to(p.dz_node, tracer.shape)
    step = p.dt / p.adv_nsub
    gradient_top = (tracer[..., 1] - tracer[..., 0]) / p.dz_iface[..., 0]
    # Surface inflow can supply values outside the resident column range.
    # Use the same one-step characteristic prediction as the endpoint target;
    # an explicitly bounded scalar can limit its extrapolated inflow. Temperature
    # has no arbitrary zero-Celsius bound, preserving additive unit invariance.
    surface_increment = -step * mass_flux[..., 0] * gradient_top
    if minimum is not None:
        surface_increment = jnp.maximum(surface_increment, minimum - tracer[..., 0])
    first = mass_flux[..., 1] * tracer[..., 0] - widths[..., 0] * surface_increment / step
    last = mass_flux[..., -2] * tracer[..., -1]
    offset_first, offset_last = first - flux[..., 1], last - flux[..., -2]
    if face_states is not None:
        # The physical endpoint and its donor state can coincide exactly.
        # Form their concentration difference before multiplication, avoiding a
        # roundoff-only outward correction that can zero the whole column blend.
        above, below = face_states
        upwind = jnp.where(mass_flux[..., 1:-1] > 0., above, below)
        offset_first = (mass_flux[..., 1]*(tracer[..., 0]-upwind[..., 0])
                        -widths[..., 0]*surface_increment/step)
        offset_last = mass_flux[..., -2]*(tracer[..., -1]-upwind[..., -1])
    # A stagnant mass face is also a material barrier for this correction.
    # Minimum-norm redistribution is solved separately on each connected span.
    faces = mass_flux[..., 1:-1]
    count = faces.shape[-1]
    indices = jnp.broadcast_to(jnp.arange(count), faces.shape)
    pinned = (faces == 0.) | (indices == 0) | (indices == count - 1)
    values = jnp.zeros_like(faces)
    values = values.at[..., 0].set(jnp.where(faces[..., 0] != 0., offset_first, 0.))
    values = values.at[..., -1].set(jnp.where(faces[..., -1] != 0., offset_last, 0.))
    if getattr(p, 'vertical_tracer_scheme', 'muscl_endpoint_v2') == 'muscl_band_v3':
        # The adjacent physical node needs a material derivative too. Its
        # incoming/outgoing target follows from continuity and the same limited
        # physical secant, not from copying a boundary value into another layer.
        secant = (tracer[..., 1:] - tracer[..., :-1]) / p.dz_iface
        a, b = secant[..., :-1], secant[..., 1:]
        gradient = jnp.where(a*b > 0., jnp.sign(a)*jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.)
        top_connected = (faces[..., 0] != 0.) & (faces[..., 1] != 0.)
        bottom_connected = (faces[..., -1] != 0.) & (faces[..., -2] != 0.)
        second = (first + tracer[..., 1]*(mass_flux[..., 2]-mass_flux[..., 1])
                  + .5*widths[..., 1]*(mass_flux[..., 1]+mass_flux[..., 2])*gradient[..., 0])
        penultimate = (last - tracer[..., -2]*(mass_flux[..., -2]-mass_flux[..., -3])
                       - .5*widths[..., -2]*(mass_flux[..., -3]+mass_flux[..., -2])*gradient[..., -1])
        pinned = pinned | ((indices == 1) & top_connected[..., None]) | ((indices == count-2) & bottom_connected[..., None])
        second_offset, penultimate_offset = second-flux[..., 2], penultimate-flux[..., -3]
        if face_states is not None:
            second_offset = (offset_first + mass_flux[..., 1]*(upwind[..., 0]-tracer[..., 1])
                             +mass_flux[..., 2]*(tracer[..., 1]-upwind[..., 1])
                             +.5*widths[..., 1]*(mass_flux[..., 1]+mass_flux[..., 2])*gradient[..., 0])
            penultimate_offset = (offset_last + mass_flux[..., -2]*(upwind[..., -1]-tracer[..., -2])
                                  +mass_flux[..., -3]*(tracer[..., -2]-upwind[..., -2])
                                  -.5*widths[..., -2]*(mass_flux[..., -3]+mass_flux[..., -2])*gradient[..., -1])
        values = values.at[..., 1].set(jnp.where(top_connected, second_offset, 0.))
        values = values.at[..., -2].set(jnp.where(bottom_connected, penultimate_offset, 0.))
    left = jax.lax.associative_scan(jnp.maximum, jnp.where(pinned, indices, 0), axis=faces.ndim - 1)
    right = jax.lax.associative_scan(jnp.minimum, jnp.where(pinned, indices, count - 1), axis=faces.ndim - 1, reverse=True)
    position = jnp.cumsum(widths, axis=-1)[..., :-1]
    lo, hi = jnp.take_along_axis(position, left, axis=-1), jnp.take_along_axis(position, right, axis=-1)
    fraction = (position - lo) / jnp.where(hi > lo, hi - lo, 1.)
    val_lo = jnp.take_along_axis(values, left, axis=-1)
    val_hi = jnp.take_along_axis(values, right, axis=-1)
    offset = val_lo + fraction * (val_hi - val_lo)
    zero = jnp.zeros_like(offset[..., :1])
    correction = jnp.concatenate((zero, offset, zero), axis=-1)
    delta = -(correction[..., 1:] - correction[..., :-1]) / widths
    original = -(horizontal_div + (flux[..., 1:] - flux[..., :-1]) / widths)
    if horizontal_numerator is not None:
        original = -(horizontal_numerator+(flux[..., 1:]-flux[..., :-1]))/widths
    low_increment = step * original
    # Work with increments rather than subtracting two near-equal absolute
    # temperatures; changing Celsius to Kelvin should not cancel small rates.
    surface_allowance = (tracer[..., :1] - tracer
                         + jnp.where(mass_flux[..., 0] > 0., surface_increment, 0.)[..., None])
    lower = jnp.minimum(jnp.minimum(jnp.min(tracer, axis=-1, keepdims=True) - tracer, surface_allowance), low_increment)
    upper = jnp.maximum(jnp.maximum(jnp.max(tracer, axis=-1, keepdims=True) - tracer, surface_allowance), low_increment)
    increment = step * delta
    positive = (upper - low_increment) / jnp.where(increment > 0., increment, 1.)
    negative = (low_increment - lower) / jnp.where(increment < 0., -increment, 1.)
    ratios = jnp.where(increment > 0., positive, jnp.where(increment < 0., negative, 1.))
    blend = jnp.clip(jnp.min(ratios, axis=-1, keepdims=True), 0., 1.)
    return flux + blend * correction


def _fill_ghost_bottom(u, p):
    """Seafloor no-flux fill: extend each column's bottom wet value downward.

    Ghost layers below the seafloor hold the T_ref/S_ref sentinel and never evolve,
    so any vertical stencil reaching them mixes that reservoir into the bottom wet
    layer. Replicating the bottom wet value is the standard MOM/NEMO no-flux
    treatment. Applied by every vertical operator. (D8)
    """
    wet = p.wet_mask_z
    idx = (wet > 0.5) * jnp.arange(u.shape[-1])
    kbot = jnp.max(idx, axis=-1)[..., None]        # (nx, ny, 1)
    kk = jnp.arange(u.shape[-1])[None, None, :]
    u_bot = jnp.take_along_axis(u, kbot, axis=-1)  # bottom wet value
    return jnp.where(kk >= kbot, u_bot, u)


def physical_tracer_gradient(tracer, p):
    """Minmod physical-depth slopes with one-sided endpoint secants."""
    secant = (tracer[..., 1:]-tracer[..., :-1])/p.dz_iface
    a, b = secant[..., :-1], secant[..., 1:]
    middle = jnp.where(a*b > 0., jnp.sign(a)*jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.)
    return jnp.concatenate((secant[..., :1], middle, secant[..., -1:]), axis=-1)


def point_vertical_antiflux(delta_rate, horizontal_anti, mass_flux, p):
    """Capacity-minimum residual on each connected vertical material span.

    Physical boundary and stagnant mass faces are pinned to zero anti-flux.
    Connected endpoint nodes retain their accepted baseline rate. Other nodes may retain a
    compatibility residual; a whole-column identity does not close each span.
    """
    h = jnp.broadcast_to(p.dz_node, delta_rate.shape)
    zero = jnp.zeros_like(delta_rate[..., :1])
    raw = jnp.concatenate((zero, -jnp.cumsum(h*delta_rate+horizontal_anti, axis=-1)), axis=-1)
    indices = jnp.broadcast_to(jnp.arange(raw.shape[-1]), raw.shape)
    n = delta_rate.shape[-1]
    pinned = (mass_flux == 0.) | (indices == 0) | (indices == n)
    values = jnp.zeros_like(raw)
    # These anti-fluxes cancel the new horizontal correction structurally at
    # physical endpoints. Recomputing an already accepted endpoint target can
    # create a roundoff-only outward increment and suppress the whole blend.
    first = jnp.where(mass_flux[..., 1] != 0., -horizontal_anti[..., 0], 0.)
    last = jnp.where(mass_flux[..., -2] != 0., horizontal_anti[..., -1], 0.)
    values = values.at[..., 1].set(first).at[..., -2].set(last)
    pinned = pinned | (indices == 1) | (indices == n-1)
    offset = values-raw
    left = jax.lax.associative_scan(jnp.maximum, jnp.where(pinned, indices, 0), axis=raw.ndim-1)
    right = jax.lax.associative_scan(jnp.minimum, jnp.where(pinned, indices, n), axis=raw.ndim-1, reverse=True)
    position = jnp.concatenate((zero, jnp.cumsum(h, axis=-1)), axis=-1)
    lo, hi = jnp.take_along_axis(position, left, -1), jnp.take_along_axis(position, right, -1)
    fraction = (position-lo)/jnp.where(hi > lo, hi-lo, 1.)
    offset_lo, offset_hi = jnp.take_along_axis(offset, left, -1), jnp.take_along_axis(offset, right, -1)
    corrected = raw+offset_lo+fraction*(offset_hi-offset_lo)
    return jnp.where(pinned, values, corrected)


def joint_scalar_blend(tracer, low_rate, delta_rate, mass_flux, p, minimum=None):
    """One shared-face coefficient for the whole scalar domain.

    Bounds describe the Euler correction relative to the existing baseline.
    If the baseline violates an absolute floor and no alpha can repair it,
    return the baseline and a false feasibility flag; do not claim positivity.
    """
    step = p.dt/p.adv_nsub
    low, delta = step*low_rate, step*delta_rate
    surface = -step*mass_flux[..., 0]*physical_tracer_gradient(tracer, p)[..., 0]
    if minimum is not None:
        surface = jnp.maximum(surface, minimum-tracer[..., 0])
    allowance = tracer[..., :1]-tracer+jnp.where(mass_flux[..., :1] > 0., surface[..., None], 0.)
    # Express the same bounds as margins around the baseline increment. The
    # literal zero represents its deliberate inclusion in the envelope. Forming
    # a bound including low and then subtracting low can produce wrong-sign
    # roundoff under fusion, falsely rejecting the valid zero correction.
    lower = jnp.minimum(jnp.minimum(jnp.min(tracer, -1, keepdims=True)-tracer-low,
                                    allowance-low), 0.)
    upper = jnp.maximum(jnp.maximum(jnp.max(tracer, -1, keepdims=True)-tracer-low,
                                    allowance-low), 0.)
    if minimum is not None:
        lower = jnp.maximum(lower, minimum-tracer-low)
    denominator = jnp.where(delta != 0., delta, 1.)
    one, two = lower/denominator, upper/denominator
    lower_alpha = jnp.where(delta != 0., jnp.minimum(one, two), 0.)
    upper_alpha = jnp.where(delta != 0., jnp.maximum(one, two), 1.)
    lo, hi = jnp.maximum(0., jnp.max(lower_alpha)), jnp.minimum(1., jnp.min(upper_alpha))
    feasible = ((lo <= hi) & jnp.all(lower <= upper)
                & jnp.all(jnp.where(delta == 0., (lower <= 0.) & (upper >= 0.), True)))
    return jnp.where(feasible, jnp.clip(hi, 0., 1.), 0.), feasible


def integrated_reference_scalar_rate(tracer, mass_flux, horizontal_shift_numerator,
                                     horizontal_mass_numerator, p, minimum=None):
    """Flux-form RHS with a global concentration reference and one division.

    Horizontal inputs are capacity-integrated numerators, before division by h.
    Retain the reference times the complete mass residual: this also detects an
    inconsistent/reversed continuity flux rather than clipping constant fields.
    The supported physical bed has zero mass and scalar boundary flux.
    """
    reference = tracer.reshape(-1)[0]
    shifted = tracer-reference
    left, right = shifted[..., :-1], shifted[..., 1:]
    if getattr(p, 'vertical_tracer_scheme', 'donor_cell') in ('muscl_nodal_v1', 'muscl_endpoint_v2', 'muscl_band_v3'):
        left, right = _vertical_tracer_face_values(shifted, p)
    interior = mass_flux[..., 1:-1]*jnp.where(mass_flux[..., 1:-1] > 0., left, right)
    surface = mass_flux[..., :1]*shifted[..., :1]
    flux = jnp.concatenate((surface, interior, jnp.zeros_like(surface)), axis=-1)
    mass_residual = horizontal_mass_numerator+(mass_flux[..., 1:]-mass_flux[..., :-1])
    horizontal_effective = horizontal_shift_numerator+reference*mass_residual
    if getattr(p, 'vertical_tracer_scheme', 'donor_cell') in ('muscl_endpoint_v2', 'muscl_band_v3'):
        shifted_minimum = None if minimum is None else minimum-reference
        flux = _conservative_endpoint_flux(shifted, mass_flux, flux,
                                           horizontal_effective/p.dz_node, p, minimum=shifted_minimum,
                                           face_states=(left, right), horizontal_numerator=horizontal_effective)
    numerator = horizontal_effective+(flux[..., 1:]-flux[..., :-1])
    rate = -numerator/p.dz_node*p.wet_mask_z
    return rate, (mass_flux[..., 0]*tracer[..., 0])


def _d_dz(u, p):
    """Vertical first derivative, non-uniform grid."""
    du_interior = (u[..., 2:] - u[..., :-2]) / p.dz_denom_interior
    du_top = (u[..., 1:2] - u[..., 0:1]) / p.dz_bnd_top
    du_bot = (u[..., -1:] - u[..., -2:-1]) / p.dz_bnd_bot
    return jnp.concatenate([du_top, du_interior, du_bot], axis=-1)


def _d2_dz2(u, p):
    """Vertical second derivative, non-uniform grid.

    Top and bottom nodes use the zero-flux (ghost-point) form 2*(C1 - C0)/h0^2: the
    mirror ghost Cg = C1 makes the centered curvature diffusive at the boundary,
    where the plain centered-curvature form was anti-diffusive and drove the ITCZ
    salinity runaway. The field is ghost-filled first (_fill_ghost_bottom), so a
    seafloor that is not at the last grid level is a no-flux boundary too: without
    the fill the bottom wet node reads the T_ref/S_ref sentinel below the floor, a
    spurious seafloor flux up to 0.05 K/day. (D20)
    """
    u = _fill_ghost_bottom(u, p)
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = 2.0 * (u[..., 1:2] - u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = 2.0 * (u[..., -2:-1] - u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return jnp.concatenate([d2u_top, d2u_interior, d2u_bot], axis=-1)


def _iface_flux_divergence(Fz_i, p):
    """Divergence of an interface flux: tend[k] = (F[k-1/2] - F[k+1/2])/dz_node[k].

    Fz_i holds one flux per INTERFACE between nodes k and k+1 (length nz-1);
    the material top and bottom carry F = 0, which is the zero-flux BC. The
    column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 exactly, for
    any flux and any mask (D9, D10, D18).
    """
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node


def _diffusive_interface_flux(tracer, kappa, p):
    """Downward diffusive flux between ghost-filled nodes, before process gating."""
    tracer = _fill_ghost_bottom(tracer, p)
    Cm = tracer[..., :-1]
    Cp = tracer[..., 1:]
    return -kappa * (Cp - Cm) / p.dz_iface


def _d2_dz2_flux(tracer, kappa, p):
    """Vertical diffusion in INTERFACE-flux form (exactly column-conservative).

    F[k+1/2] = -kappa*(C[k+1]-C[k])/dz_iface[k], zero at the material top and
    bottom; the column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 for any
    kappa and any field. The node form does not telescope on a non-uniform grid, and
    vertical diffusion can only redistribute within a column. (D9)
    """
    # kappa may be scalar or a (nx, ny, nz) coastal-enhanced field; only the
    # scalar-zero shortcut can be branched on at trace time.
    if getattr(kappa, "ndim", 0) == 0 and kappa == 0.0:
        return jnp.zeros_like(tracer)
    Fz_i = _diffusive_interface_flux(tracer, kappa, p)
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_iface_f, Fz_i, 0.0)
    return _iface_flux_divergence(Fz_i, p) * p.wet_mask_z
