"""Mechanically preserved FD horizontal implementation."""
from config import OMEGA, R_EARTH

from .backend import jnp


def _d_dx(u, p):
    """Zonal derivative du/dx, 2nd-order central, longitude-periodic.

    Land points are differentiated like any other node; the callers mask. (D2)
    """
    return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) * (0.5 * p.inv_dx)

def _d_dy(u, p):
    """Meridional derivative du/dy, 2nd-order central with NO-FLUX WALLS.

    The lat axis does not wrap: a mirror ghost cell (ghost = boundary value) gives
    the wall row a stable central difference with zero normal diffusive flux, and
    the normal velocity is masked to 0 there so no advective flux crosses either.
    Index 0 is the SOUTH row. (D1)
    """
    # Mirror-pad the lat axis by 1 cell each side (ghost = boundary value).
    # ndim-agnostic: pad only axis 1.
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    return (u_pad[:, 2:] - u_pad[:, :-2]) * (0.5 * p.inv_dy)

def _divergence_conservative(ubt, vbt, p):
    """Mass-conserving horizontal divergence of the barotropic velocity.

    Flux form: a face is open only when BOTH adjacent cells are wet, and the N/S
    truncation walls are closed. The zonal term uses the per-cell inv_dx = dy/A_i
    and the meridional term the spherical form (1/cos_j)*d(v*cos_face)/dy, so the
    divergence telescopes to zero under the area weight A_ij = R^2*cos(lat)*dphi^2.
    The bare centered form leaks volume at coastlines and walls. (D2)
    """
    wm = p.wet_mask                                   # (nx, ny)
    cos_lat = p.cos_lat                               # (ny,)
    # Zonal (axis 0, periodic): face (i+1/2,j) open iff cell i and i+1 both wet.
    uface_open = wm * jnp.roll(wm, -1, axis=0)        # 1 where both sides wet
    uface = 0.5 * (ubt + jnp.roll(ubt, -1, axis=0)) * uface_open
    div_x = (uface - jnp.roll(uface, 1, axis=0)) * p.inv_dx[..., 0]
    # Meridional (axis 1, closed N/S walls): face (i,j+1/2) open iff j and j+1
    # both wet. The roll wraps j=ny-1 -> j=0 across the pole, so explicitly
    # close the two boundary faces (no flow through the truncation walls).
    vface_open = wm * jnp.roll(wm, -1, axis=1)
    vface_open = vface_open.at[:, -1].set(0.0)        # north wall closed
    vface = 0.5 * (vbt + jnp.roll(vbt, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    fcos = vface * cos_face[None, :]                     # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)                  # flux at face j-1/2
    fcos_in = fcos_in.at[:, 0].set(0.0)                  # south wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :]
    return div_x + div_y

def _gradient_conservative(eta, p):
    """Adjoint of _divergence_conservative, for the free-surface pressure gradient.

    Face differences weighted by the cell inv_dx (meridionally by the face cos and
    divided by the cell cos). The centered _d_dx/_d_dy is NOT the adjoint: pairing
    it with the conservative divergence injects energy into the free mode. (D3)
    """
    wm = p.wet_mask
    cos_lat = p.cos_lat                               # (ny,)
    inv_dx = p.inv_dx[..., 0]                          # (nx, ny)
    # Zonal (axis 0, periodic): open face iff both neighbors wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)            # face (i+1/2,j)
    open_xm = wm * jnp.roll(wm, 1, axis=0)             # face (i-1/2,j)
    d_eta_xp = (jnp.roll(eta, -1, axis=0) - eta) * open_xp   # eta_{i+1}-eta_i
    d_eta_xm = (eta - jnp.roll(eta, 1, axis=0)) * open_xm    # eta_i-eta_{i-1}
    grad_x = inv_dx * 0.5 * (d_eta_xp + d_eta_xm)
    # Meridional (axis 1, closed walls): adjoint of the spherical divergence,
    # face cos weighted, divided by the cell cos.
    open_yp = wm * jnp.roll(wm, -1, axis=1)
    open_ym = wm * jnp.roll(wm, 1, axis=1)
    # Close the two boundary faces (adjoint of the closed truncation walls).
    open_yp = open_yp.at[:, -1].set(0.0)               # north wall
    open_ym = open_ym.at[:, 0].set(0.0)                # south wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    d_eta_yp = (jnp.roll(eta, -1, axis=1) - eta) * open_yp * cos_face_p[None, :]
    d_eta_ym = (eta - jnp.roll(eta, 1, axis=1)) * open_ym * cos_face_m[None, :]
    grad_y = (p.inv_dy / cos_lat[None, :]) * 0.5 * (d_eta_yp + d_eta_ym)
    return grad_x, grad_y

def _gradient_conservative_3d(field, p):
    """Masked, cos(lat)-weighted adjoint gradient, face-gated like the divergence.

    No flux reaches into land zeros across a coastline. (D4)
    """
    wm = p.wet_mask_z                                 # (nx, ny, nz)
    cos_lat = p.cos_lat                               # (ny,)
    inv_dx = p.inv_dx[..., 0:1]                        # (nx, ny, 1)
    # Zonal (axis 0, periodic): open face iff both neighbors wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)
    open_xm = wm * jnp.roll(wm, 1, axis=0)
    d_fp = (jnp.roll(field, -1, axis=0) - field) * open_xp
    d_fm = (field - jnp.roll(field, 1, axis=0)) * open_xm
    grad_x = inv_dx * 0.5 * (d_fp + d_fm)
    # Meridional (axis 1, closed walls): adjoint of the spherical divergence.
    open_yp = wm * jnp.roll(wm, -1, axis=1)
    open_ym = wm * jnp.roll(wm, 1, axis=1)
    open_yp = open_yp.at[:, -1, :].set(0.0)           # north wall
    open_ym = open_ym.at[:, 0, :].set(0.0)            # south wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    shp = [1, p.ny, 1]
    d_yp = (jnp.roll(field, -1, axis=1) - field) * open_yp * cos_face_p.reshape(shp)
    d_ym = (field - jnp.roll(field, 1, axis=1)) * open_ym * cos_face_m.reshape(shp)
    grad_y = (p.inv_dy / cos_lat[None, :, None]) * 0.5 * (d_yp + d_ym)
    return grad_x, grad_y

def _divergence_conservative_3d(Fx, Fy, p):
    """3D divergence whose faces are gated on BOTH adjacent cells being wet.

    The bare _d_dx/_d_dy divergence of a coastal flux reads the land zeros inside
    its stencil and injects a spurious coastal source. (D4)
    """
    wm = p.wet_mask_z                                 # (nx, ny, nz)
    cos_lat = p.cos_lat                               # (ny,)
    # Zonal (axis 0, periodic): face (i+1/2,j,k) open iff cell i and i+1 wet.
    uface_open = wm * jnp.roll(wm, -1, axis=0)
    uface = 0.5 * (Fx + jnp.roll(Fx, -1, axis=0)) * uface_open
    div_x = (uface - jnp.roll(uface, 1, axis=0)) * p.inv_dx[..., 0:1]
    # Meridional (axis 1, closed walls): face (i,j+1/2,k) open iff j,j+1 wet.
    vface_open = wm * jnp.roll(wm, -1, axis=1)
    vface_open = vface_open.at[:, -1, :].set(0.0)    # north wall closed
    vface = 0.5 * (Fy + jnp.roll(Fy, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    shp = [1, p.ny, 1]
    fcos = vface * cos_face.reshape(shp)                  # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)
    fcos_in = fcos_in.at[:, 0, :].set(0.0)                 # south wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :, None]
    return div_x + div_y

def _laplacian_h(u, p):
    """Horizontal Laplacian on the sphere, in physical (meter) space:

        grad^2 u = 1/cos(phi) d/dphi(cos(phi) du/dphi) + 1/cos^2(phi) d2u/dlam2
                 = d2u/dy2 - (tan(phi)/R) du/dy + (1/cos^2(phi)) d2u/dx2

    The nodal candidate uses closed wet-face cosine fluxes. The legacy branch
    retains its expanded metric correction for production compatibility. (D5)
    """
    if getattr(p, 'column_geometry', 'legacy') == 'nodal_dual_v1':
        return _horizontal_diffusion_flux(u, jnp.ones_like(p.coastal_kappa_h_2d)[:, :, None], p)
    # ∂²u/∂x²: FACE-GATED conservative form (open iff BOTH cells wet). The bare
    # central stencil reads the mask step at the coast, where ghost nodes hold
    # the T_ref sentinel, and the spike/dipole it produces drifts whole coastal
    # cells. (D5)
    wm = p.wet_mask_z                                   # (nx, ny, nz)
    d2u_dx2 = ((jnp.roll(u, -1, axis=0) - u) * wm * jnp.roll(wm, -1, axis=0)
               - (u - jnp.roll(u, 1, axis=0)) * wm * jnp.roll(wm, 1, axis=0)) * p.inv_dx2
    # ∂²u/∂y²: same wet/wet face gate in y, plus the mirror ghost cell of
    # _d_dy (ghost = boundary value), giving zero normal gradient at the N/S
    # wall: d²u/dy²|_0 = (u1 - 2u0 + u_ghost)/dy² = (u1 - u0)/dy². (D1, D5)
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    d2u_dy2 = ((u_pad[:, 2:] - u_pad[:, 1:-1]) * wm_pad[:, 2:] * wm_pad[:, 1:-1]
               - (u_pad[:, 1:-1] - u_pad[:, :-2]) * wm_pad[:, 1:-1] * wm_pad[:, :-2]) * p.inv_dy2
    # spherical metric correction: -(tanφ/R) ∂u/∂y
    # tan(lat) = sin(lat)/cos(lat); signed sin(lat) from f = 2*Omega*sin(lat)
    # (constant along longitude, so row 0 of f gives the per-row sin).
    du_dy = _d_dy(u, p)
    sin_lat_row = p.f[0, :] / (2.0 * OMEGA)           # (ny,) signed sin(lat)
    tan_over_R = (sin_lat_row / p.cos_lat) / R_EARTH   # (ny,) tan(lat)/R
    shp = [1] * u.ndim
    shp[1] = p.ny
    corr = -du_dy * tan_over_R.reshape(shp)
    return d2u_dx2 + d2u_dy2 + corr

def _horizontal_diffusion_flux(tracer, diffusivity, p):
    """Wet-face variable-coefficient diffusion with closed latitude walls."""
    wet = p.wet_mask_z
    zonal_flux = (0.5 * (diffusivity + jnp.roll(diffusivity, -1, axis=0))
                  * (jnp.roll(tracer, -1, axis=0) - tracer) * p.inv_dx
                  * wet * jnp.roll(wet, -1, axis=0))
    padding = [(0, 0), (0, 1), (0, 0)]
    next_tracer = jnp.pad(tracer, padding, mode='edge')[:, 1:, :]
    next_diffusivity = jnp.pad(diffusivity, padding, mode='edge')[:, 1:, :]
    next_wet = jnp.pad(wet, padding, mode='edge')[:, 1:, :]
    cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    meridional_flux = (0.5 * (diffusivity + next_diffusivity)
                       * (next_tracer - tracer) * p.inv_dy
                       * wet * next_wet * cos_face[None, :, None])
    meridional_flux = meridional_flux.at[:, -1, :].set(0.)
    incoming_y = jnp.roll(meridional_flux, 1, axis=1).at[:, 0, :].set(0.)
    return ((zonal_flux - jnp.roll(zonal_flux, 1, axis=0)) * p.inv_dx
            + (meridional_flux - incoming_y) * p.inv_dy / p.cos_lat[None, :, None])

def _horizontal_tracer_diffusion(tracer, p):
    """Conservative wet-face diffusion for background and enhanced coefficients."""
    diffusivity = p.kappa_h + p.coastal_kappa_h_2d[:, :, None]
    return _horizontal_diffusion_flux(tracer, diffusivity, p)

def _horizontal_biharmonic_tracer(tracer, p):
    """Square the self-adjoint wet-face Laplacian; negative sign dissipates."""
    unit_diffusivity = jnp.ones_like(p.coastal_kappa_h_2d)[:, :, None]
    laplacian = _horizontal_diffusion_flux(tracer, unit_diffusivity, p)
    return _horizontal_diffusion_flux(laplacian, unit_diffusivity, p)

def _biharmonic_h(u, p):
    """Square the geometry-specific horizontal Laplacian. (D5)"""
    return _laplacian_h(_laplacian_h(u, p), p)

def _divergence_h(u, v, p):
    """Horizontal divergence from FACE-FLUX differences, as advection sees it.

    Algebraically identical to the centered difference on fully-wet nodes, but at a
    coastal wall the face-flux form measures the real inflow through the open face
    where the centered form measures the shear against the land neighbour's u = 0.
    Only the former lets the diagnosed w close the advective budget. Meridionally it
    mirror-pads the closed N/S walls and carries the same cos(face)/cos(cell)
    spherical factors as _divergence_conservative. (D6)
    """
    wm = p.wet_mask_z
    # Zonal (axis 0, periodic): face (i+1/2) velocity, open iff both wet.
    Fx = 0.5 * (u + jnp.roll(u, -1, axis=0)) * wm * jnp.roll(wm, -1, axis=0)
    div_x = (Fx - jnp.roll(Fx, 1, axis=0)) * p.inv_dx[..., 0:1]
    # Meridional (axis 1, closed N/S walls): edge-pad, face (j+1/2).
    pad = [(0, 0)] * v.ndim
    pad[1] = (1, 1)
    v_pad = jnp.pad(v, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    Fy = (0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
          * wm_pad[:, 1:-1] * wm_pad[:, 2:]
          * 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))[None, :, None])
    Fy = Fy.at[:, -1].set(0.0)                       # north truncation wall
    div_y = (Fy - jnp.roll(Fy, 1, axis=1)) * p.inv_dy / p.cos_lat[None, :, None]
    return div_x + div_y

def _dealias_h_fd(field, p):
    """2-dx grid-scale filter for pointwise-nonlinear products (FD analogue of the
    spectral 2/3-rule dealias).

    Lon is periodic and uniformly spaced, so a 2/3 FFT rule is exact there. Lat is
    closed and non-uniform, so an FFT would mangle the field and a 5-pt binomial
    [1,4,6,4,1]/16 substitutes: 0 at the Nyquist wavenumber, ~0.92 at 8-dx. Edge
    padding mirrors the wall ghost cell, as in _d_dy/_laplacian_h.

    Applied at exactly THREE call sites, all pointwise nonlinear products: adv_u,
    adv_v, and the GM/Redi skew-flux tendency. Deliberately NOT applied to the
    diagnosed w or to the flux-form tracer advection, whose exact discrete
    telescoping it would destroy (D7, D15). (D25)
    """
    # 2/3 FFT dealias in lon (axis 0, periodic + uniform -> exact).
    f_hat = jnp.fft.fft(field, axis=0)
    f_hat = f_hat * p.dealias_lon_mask
    f_lon = jnp.real(jnp.fft.ifft(f_hat, axis=0))
    # 5-pt binomial low-pass [1,4,6,4,1]/16 in lat (axis 1, edge-padded wall).
    fp = jnp.pad(f_lon, ((0, 0), (2, 2), (0, 0)), mode='edge')
    f_sm = (fp[:, :-4] + 4.0 * fp[:, 1:-3] + 6.0 * fp[:, 2:-2]
            + 4.0 * fp[:, 3:-1] + fp[:, 4:]) / 16.0
    return f_sm

def _gradient_face_gated_3d(field, p):
    """Face-gated horizontal gradient for advection: d/dx, d/dy with each face
    difference zeroed at every wet/ghost (or wet/dry) interface.

    The bare centered _d_dx/_d_dy differences across mask boundaries: ghost nodes
    hold the T_ref sentinel (+15 C) against ~+1 C deep water, so the centered
    stencil reads dT ~ 14 K across every wet/ghost face and a deep coastal current
    feels advection from a temperature that does not exist. Zeroing the face
    difference matches the already-gated mass flux and _gradient_conservative_3d;
    where the flow is parallel to the boundary the cross-boundary flux is zero
    anyway, so only the unphysical part is removed. (D14)

    Returns (dF/dx, dF/dy) on wet nodes; ghost nodes keep whatever the caller masks.
    """
    wm = p.wet_mask_z                                   # (nx, ny, nz)
    # Zonal (axis 0, periodic): face (i+1/2) open iff cells i, i+1 both wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)
    open_xm = wm * jnp.roll(wm, 1, axis=0)
    d_fp = (jnp.roll(field, -1, axis=0) - field) * open_xp
    d_fm = (field - jnp.roll(field, 1, axis=0)) * open_xm
    grad_x = p.inv_dx[..., 0:1] * 0.5 * (d_fp + d_fm)
    # Meridional (axis 1, closed N/S walls): mirror-ghost edge padding as in
    # _d_dy (zero normal gradient at the wall) PLUS the wet/wet face gate.
    pad = [(0, 0), (1, 1), (0, 0)]
    f_pad = jnp.pad(field, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    open_yp = wm_pad[:, 2:] * wm_pad[:, 1:-1]           # face (j+1/2)
    open_ym = wm_pad[:, 1:-1] * wm_pad[:, :-2]          # face (j-1/2)
    d_yp = (f_pad[:, 2:] - f_pad[:, 1:-1]) * open_yp
    d_ym = (f_pad[:, 1:-1] - f_pad[:, :-2]) * open_ym
    grad_y = p.inv_dy * 0.5 * (d_yp + d_ym)
    return grad_x, grad_y

def _polar_cap_weights(ncap, ntaper):
    """Blend weights for the tapered polar cap, in POLE-INWARD order.

    1D array of length ncap+ntaper, the blend fraction of the zonal mean applied to
    each row from the pole inward: rows [0:ncap] get weight 1.0 (full zonal average,
    killing the cos(lat)->0 metric singularity), rows [ncap:ncap+ntaper] ramp 1->0
    via a cos^2 taper. A hard cutoff (ntaper=0) leaves a cliff at j=ncap that _d_dy
    amplifies exponentially. (D21)

    The order is a caller contract, not a convenience: the weight at index 0
    belongs on the POLE ROW. The south band slices pole-inward and uses this
    array as-is; the north band slices pole-LAST and must reverse it. Applying
    the north band unflipped leaves the wall row effectively uncapped -- see
    _apply_polar_cap.
    """
    if ncap <= 0:
        return jnp.zeros(0)
    full = jnp.ones(ncap)
    if ntaper <= 0:
        return full
    # cos^2 ramp from 1 (at the cap edge) to 0 (into the interior).
    x = jnp.linspace(0.0, jnp.pi / 2.0, ntaper + 2)[1:-1]   # open interval
    ramp = jnp.cos(x) ** 2
    return jnp.concatenate([full, ramp])

def _apply_polar_cap(field, wm, p):
    """Tapered wet-point zonal-average polar cap on a 2D or 3D field.

    Blends the wet-point zonal mean of the poleward rows in over polar_cap_taper
    transition rows with a cos^2 ramp (a hard cutoff is a meridional cliff _d_dy
    amplifies). Land and ghost nodes keep their masked value, so wm must match the
    field's rank: the 3D mask matters, since the 2D one is column-wide and would let
    ghost nodes below a shallow seafloor into the deep-level mean. (D21)

    Non-legacy float32 candidates reduce and blend only the cap bands in float64,
    then return the original dtype. Incremental blending preserves constant wet
    fields exactly. This does not make rounded tracer inventories conservative.
    """
    ncap = p.polar_cap_rows
    if ncap <= 0:
        return field
    nb = ncap + p.polar_cap_taper
    # Runtime-built jnp.ones/linspace are float64 under jax_enable_x64; cast
    # to the field's dtype or the blend promotes the whole field to f64
    # (silently defeats the fp32 state/params cast downstream).
    wts = _polar_cap_weights(ncap, p.polar_cap_taper).astype(field.dtype)

    def _cap_band(f, w, wts_band):
        if field.dtype == jnp.float32 and getattr(p, 'process_time_scheme', 'legacy') != 'legacy':
            wet64 = w.astype(jnp.float64)
            values64 = f.astype(jnp.float64) * wet64
            count64 = jnp.maximum(jnp.sum(wet64, axis=0, keepdims=True), 1.)
            mean64 = jnp.sum(values64, axis=0, keepdims=True) / count64
            fraction64 = wts_band.astype(jnp.float64).reshape((1, nb) + (1,) * (field.ndim - 2))
            return (values64 + fraction64 * (mean64 * wet64 - values64)).astype(field.dtype)
        # Wet-point zonal mean over the FULL band (one value per (row, ...)),
        # broadcast back; land/ghost stays at its masked value.
        s = f * w
        wsum = jnp.maximum(jnp.sum(w, axis=0, keepdims=True), 1.0)
        zmean = jnp.sum(s, axis=0, keepdims=True) / wsum
        zmean = jnp.broadcast_to(zmean, f.shape) * w
        wb = wts_band.reshape((1, nb) + (1,) * (field.ndim - 2))
        return wb * zmean + (1.0 - wb) * s

    # _polar_cap_weights orders the blend pole-inward (weight 1 on the POLE
    # row, tapering into the interior). The south band already slices inward
    # from its pole, so the order matches; the north band slices POLE-FIRST
    # (j=ny-1 back toward the interior), so its weights must be flipped.
    # Unflipped, the north cap spends its full zonal mean nb-1 rows INSIDE
    # the wall and leaves the wall row itself at weight ~0.15 of it -- i.e.
    # effectively uncapped. The wall row then grows a 2dx zonal checkerboard
    # (measured on the real ETOPO relief: peak |u| pinned to j=ny-1, 1.0 ->
    # 24.8 m/s between day 1.0 and day 2.0). (D21)
    south = _cap_band(field[:, :nb], wm[:, :nb], wts)
    north = _cap_band(field[:, -nb:], wm[:, -nb:], jnp.flip(wts))
    return jnp.concatenate([south, field[:, nb:-nb], north], axis=1)
