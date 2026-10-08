"""Model state, computational parameters and complete state byte identities."""

from __future__ import annotations

import hashlib
from collections import namedtuple

from zhenmode.model.solver.numerics.backend import np

JaxStateG = namedtuple('JaxStateG', ['u', 'v', 'T', 'S', 'eta', 'ice'])

JaxStateG.__new__.__defaults__ = (0.0,)

FDParams = namedtuple('FDParams', [
    # Spherical metric
    'dx_2d',            # (nx, ny) zonal spacing [m] = R*cos(lat)*dlon
    'dy',               # scalar meridional spacing [m]
    'cos_lat',          # (ny,) cos(lat)
    'inv_dx',           # (nx, ny) 1/dx_2d
    'inv_dy',           # scalar 1/dy
    'inv_dx2',          # (nx, ny) 1/dx_2d^2
    'inv_dy2',          # scalar 1/dy^2
    # Coriolis
    'f',                # (nx, ny) full 2D Coriolis
    # Land
    'wet_mask',         # (nx, ny) 1=ocean, 0=land
    'wet_mask_z',       # (nx, ny, nz) TRUE vertical wet mask: 1 where layer is
                        # above the seafloor, 0 below (ghost water excluded).
                        # Used in pressure integration to kill the spurious PGF
                        # at steep topography (ghost-water-column bug fix).
    'interior_mask_z',  # (nx, ny, 1) 1 in the interior, 0 on the N/S boundary
                        # rows (j=0, j=ny-1). Enforces the no-flux wall: the
                        # normal (meridional) velocity v is zeroed here so no
                        # flow crosses the closed N/S truncation wall.
    # Vertical grid (non-uniform z-levels)
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node',
    'surface_mask', 'bottom_mask',   # (1,1,nz)
    # Dimensions
    'nx', 'ny', 'nz',
    'face_contacts', 'contact_depths_m', 'node_depth_m', 'column_geometry',
])
FDParams.__new__.__defaults__ = (None, None, None, 'legacy')

FDPhysParams = namedtuple('FDPhysParams', [
    # metric + grid (from FDParams)
    'dx_2d', 'dy', 'cos_lat', 'inv_dx', 'inv_dy', 'inv_dx2', 'inv_dy2',
    'f', 'wet_mask', 'wet_mask_z',
    'interior_mask_z',
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node', 'surface_mask', 'bottom_mask',
    'nx', 'ny', 'nz',
    # physics
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v', 'kappa_conv',
    'nu_bi', 'kappa_bi',
    'T_ref', 'S_ref', 'r_bot', 'cd', 'bottom_friction',
    # forcing (2D physical-space; no FFT pre-compute in the FD solver)
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d',
    # free surface
    'H_sw', 'dz_norm', 'dt',
    # bulk air-sea heat flux
    'T_atm_3d', 'lambda_bulk',
    # Surface salinity restoring (Haney): relax SSS toward S_clim_surf with an
    # equivalent salt flux, dSdt += -restore_coef_S*(S_surf - S_ref_2d)*surface_mask.
    # (D24)
    'S_ref_2d', 'restore_coef_S',
    # Diagnostic coastal surface-temperature restoring (land-adjacent band)
    'coastal_restore_coef_2d',  # (nx, ny) 1/s restoring coefficient
    'coastal_restore_T_2d',     # (nx, ny) target SST climatology
    # Diagnostic coastal extra bulk heat exchange
    'coastal_bulk_lambda_2d',   # (nx, ny) extra W/m^2/K
    # Diagnostic coastal horizontal tracer diffusivity
    'coastal_kappa_h_2d',       # (nx, ny) extra m^2/s
    # Diagnostic coastal vertical tracer diffusivity
    'coastal_kappa_v_2d',       # (nx, ny) extra m^2/s
    # lateral sponge (polar-edge Rayleigh damping; absorbs the wind-driven
    # barotropic energy that Laplacian dissipation can't arrest within its
    # CFL cap, which otherwise piles up at the polar edge rows)
    'sponge_rate',       # (nx, ny, 1) damping rate [1/s]; 0 interior
    'sponge_rate_2d',    # (nx, ny) 2D damping for the free-surface step
    'T_clim_3d', 'S_clim_3d',   # (nx, ny, nz) climatology the sponge relaxes to
    'polar_cap_rows',    # int: poleward rows fully zonally averaged per step (metric singularity)
    'polar_cap_taper',   # int: extra rows over which the cap blend cos^2-tapers 1->0 (cap-edge cliff)
    'dealias_lon_mask',  # (nx, 1, 1) 2/3-rule FFT dealias mask for the periodic lon axis
    # Gent-McWilliams eddy closure (sub-grid baroclinic transport)
    'kappa_gm',          # m²/s GM eddy diffusivity (bolus transport); 0 = off
    'gm_slope_max',      # dimensionless isopycnal-slope limiter
    'kappa_redi',        # m²/s Redi isopycnal diffusivity (skew-flux); 0 = off
    # Semi-enclosed-sea SSH (eta) relaxation (Mediterranean artifact fix)
    'eta_relax_mask',    # (nx, ny) 1 inside the semi-enclosed sea, 0 elsewhere
    'eta_relax_rate',    # 1/s Rayleigh relaxation rate; 0 = off
    # Mode split: with mode_split=True the free surface runs in n_subcyc
    # barotropic forward-backward subcycles of dt_bt at the end of _step_impl,
    # with the density-PGF coupling held fixed over the baroclinic step
    # (MOM-style forcing lag). Lifts the external-gravity-wave CFL off the
    # baroclinic dt (dt=3600 s is then safe at 1 deg). (D12)
    'mode_split',        # bool: barotropic-subcycle free surface
    'dt_bt',             # s: ACTUAL subcycle dt (= dt / n_subcyc, exact fill)
    'n_subcyc',          # int: barotropic subcycles per baroclinic step
    'conv_nsub',         # int: convective-adjustment subcycles per bc step
    'adv_nsub',          # int: vertical-advection subcycles per bc step
    # nu_h subcycle count in the split L half-steps. None = legacy
    # (n_nu = n_subcyc); an int right-sizes it from the actual metric at the
    # same FTCS margin, i.e. ~4x fewer Laplacian pairs per half-step (D12).
    'nu_nsub',
    # True: run the barotropic subcycle as a jax.lax.scan (fused device loop,
    # body compiled once) instead of an unrolled Python loop. Numerically
    # identical; cuts XLA graph size and host launch overhead.
    'use_scan',
    # True: freeze the tracer stage-2 advection velocity at the old (u, v)
    # instead of the raw forward-Euler predictor u_pred. Makes the tracer RK2
    # consistent with the momentum RK2 and removes the column heat leak. (D13)
    'freeze_adv_vel',
    # True: use the exactly-column-conservative interface-flux form of vertical
    # diffusion (_d2_dz2_flux) instead of kappa_v*_d2_dz2. Default False =
    # legacy bit-exact traces. (D9)
    'conservative_kv',
    # True: project the stage-2 tracer advection velocity so its column
    # divergence matches the final (subcycle-projected) u -- the continuity the
    # Fz_top = Fz[0]*T[0] closure needs to conserve column heat. Default False =
    # legacy bit-exact traces. (D13)
    'project_adv_vel',
    # True: gate the convective adjustment PER-INTERFACE (mix only across
    # interfaces that are actually unstable) instead of the column-wide mask,
    # which mixes the whole column whenever any interface is unstable. Default
    # False = legacy bit-exact traces. (D10)
    'localize_conv',
    # True: first-order donor-cell face values for the HORIZONTAL tracer fluxes
    # (the vertical tracer flux is already donor-cell), making the full 3D
    # tracer transport conservative and monotone for a divergence-free velocity.
    # Default False = historical centered path. (D16)
    'monotone_adv',
    # Experimental TVD/MUSCL flux limiter for horizontal tracer transport.
    # It reconstructs the face value from the two donor cells with a minmod
    # slope and chooses the state consistent with the face velocity. This is a
    # first local bounded-flux step toward a full Zalesak FCT scheme.
    'fct_adv',
    # Optional mixed-layer heat capacity: when set, surface heat flux is
    # distributed over this depth instead of the surface grid-cell thickness.
    # 0/None keeps the legacy bit-exact surface-node treatment.
    'mixed_layer_depth_m',
    # Optional 2D mask (1=apply mixed-layer depth, 0=legacy surface-node)
    'mixed_layer_mask_2d',
    # Optional 2D mixed-layer depth [m].  When present it overrides the scalar
    # mixed_layer_depth_m cell-by-cell while still respecting mixed_layer_mask_2d.
    'mixed_layer_depth_2d',
    # Minimal thermodynamic sea-ice closure: a constant brine-rejection salt
    # flux (psu/s) applied only where the surface temperature is at or below
    # the freezing point. 0 = off.
    'ice_freeze_temp_c',
    'ice_salt_flux',
    # Minimal dynamic ice: stateful thickness with latent-heat growth/melt,
    # conductivity insulation, and brine-rejection salinity flux.
    'dynamic_ice',
    'ice_insulation_scale_m',
    'projection_niter',
    'projection_rtol',
    'projection_preconditioner',
    'projection_inv_diagonal',
    'projection_niter_source',
    'projection_max_refinements',
    'column_geometry',
    'match_barotropic_transport',
    'process_time_scheme',
    'thermodynamics',
    'eos_pressure_dbar',  # explicitly supplied sea pressure, fixed during a run
    'face_contacts', 'contact_depths_m', 'node_depth_m',
    'surface_pressure_pa',  # optional atmosphere/ice load at ocean surface, Pa
])

FDPhysParams.__new__.__defaults__ = (None, False, False, False, False, False, False, False, None, None, None, -1.8, 0.0, False, 1.0, 150, None, 'none', None, 'default', 2, 'legacy', False, 'legacy', 'linear', None, None, None, None, None)

def _state_identity(state):
    # Incoming invalid states still need a failure report; do not pass them
    # through restart's finite-only contract encoder.
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "finite": bool(np.isfinite(value).all()),
        }
        for name, field in zip(state._fields, state, strict=True)
        for value in [np.asarray(field)]
    }

def _same_state_bytes(left, right):
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).shape == np.asarray(b).shape
        and np.asarray(a).tobytes() == np.asarray(b).tobytes()
        for a, b in zip(left, right, strict=True)
    )
