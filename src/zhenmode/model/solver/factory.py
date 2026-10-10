"""Validate and assemble the finite-difference solver and compiled entry points."""

import os

from zhenmode.model.config import RHO_0, finite_number, integer_count
from zhenmode.model.solver.dynamics.pressure import _compute_hydrostatic_pressure
from zhenmode.model.solver.dynamics.projection import _column_projection_diagonal
from zhenmode.model.solver.dynamics.tendencies import _tracer_terms
from zhenmode.model.solver.dynamics.transport import _compute_vertical_velocity
from zhenmode.model.solver.geometry.fd_metrics import make_fd_params
from zhenmode.model.solver.geometry.grid import validate_grid
from zhenmode.model.solver.numerics.backend import jax, jnp, np
from zhenmode.model.solver.numerics.horizontal import nu_nsub_for_2d_cfl
from zhenmode.model.solver.physics.eos import _density_anomaly
from zhenmode.model.solver.state import FDPhysParams, JaxStateG
from zhenmode.model.solver.timestepping.step import _step_impl


def _masked_coefficient(mask, coefficient, shape):
    """A nonnegative coastal increment; absent masks keep the zero field."""
    if mask is not None and coefficient > 0.0:
        return jnp.array(mask, dtype=jnp.float64) * coefficient
    return jnp.zeros(shape)


def make_solver_global(grid, physics, dt, forcing=None,
                       T_atm=None, lambda_bulk=0.0,
                       mixed_layer_depth_m=None, mixed_layer_mask=None,
                       mixed_layer_depth_2d=None,
                       ice_freeze_temp_c=-1.8, ice_salt_flux=0.0,
                       dynamic_ice=False, ice_insulation_scale_m=1.0,
                       S_ref_surf=None, sss_restore_days=0.0,
                       coastal_restore_mask=None, coastal_restore_days=0.0,
                       coastal_bulk_mask=None, coastal_bulk_lambda=0.0,
                       coastal_kappa_h_mask=None, coastal_kappa_h=0.0,
                       coastal_kappa_v_mask=None, coastal_kappa_v=0.0,
                       coastal_restore_T=None,
                       sponge_days=0.0, sponge_cells=0,
                       T_init=None, S_init=None,
                       polar_cap_rows=2, polar_cap_taper=3, return_params=False,
                       eta_relax_days=0.0, eta_relax_box=None,
                       eta_relax_buffer=1.0, dynamic_forcing=False,
                       mode_split=False, dt_bt=150.0, nu_nsub=None,
                       dtype='float64', use_scan=False, freeze_adv_vel=False,
                       conservative_kv=False, project_adv_vel=False,
                       localize_conv=False, monotone_adv=False,
                       fct_adv=False, projection_niter=None,
                       projection_rtol=None, projection_preconditioner='none',
                       projection_max_refinements=2, column_geometry='legacy',
                       match_barotropic_transport=False, process_time_scheme='legacy',
                       eos_pressure_dbar=None, surface_pressure_pa=None,
                       external_mode_scheme='forward_backward',
                       meridional_boundary_scheme='clamped_nodes'):
    """Create a JIT-compiled global FD ocean solver.

    Key properties:
      - Pure finite-difference operators; no wavenumber / decay factors. Forcing is
        physical-space 2D (tau_x, tau_y, Q_heat); no pre-FFT.
      - Free surface is forward-backward (Sielecki) + polar-cap filter, so dt must
        sit below the external-gravity-wave CFL unless mode_split=True (barotropic
        subcycling at dt_bt; off = bit-exact to the pre-split solver). (D12, D22)
      - Lateral sponge at the POLAR EDGE rows: the y axis is a closed boundary, not
        a periodic seam, so wind-driven barotropic energy piles up there. (D19, D23)
      - No SST restore (bulk flux only). Surface SALINITY restoring is available
        (S_ref_surf + sss_restore_days>0): Haney relaxation of SSS to climatology,
        mirroring the bulk-heat-flux form. Optional eta_relax adds a mass-conserving
        Rayleigh SSH relaxation inside a semi-enclosed sea. (D23, D24)
      - nu_nsub: None = legacy nu_h subcycle count (n_subcyc); 'cfl' = right-sized
        from the explicit-diffusion CFL; int = verbatim. (D12)
      - dtype='float32' casts params/state to fp32 (1:64 FP64:FP32 on Ada GPUs --
        the biggest kernel-time lever); default 'float64' = bit-exact.
      - use_scan=True runs the barotropic subcycle as lax.scan (numerically
        identical, smaller XLA graph / fewer host launches).
      - projection_niter overrides OCEAN_PAV_NITER, resolved once at construction.
        projection_rtol has a dtype floor; optional Jacobi scales the exact native
        Poisson diagonal. projection_max_refinements bounds actual-transport
        corrections with an original-RHS stopping floor. Neither a cap nor
        refinement is a convergence or physical conservation guarantee.
      - monotone_adv=True switches horizontal tracer face values to first-order
        donor-cell (upwind). ``fct_adv=True`` instead uses a local bounded
        centered face value; it takes precedence over monotone_adv.
      - column_geometry='nodal_dual_v1' is an opt-in M1 repair candidate:
        static nodal dual cells, an explicit staircase/deep truncation policy,
        common-wet depth fluxes and their volume-weighted adjoint. Requires
        conservative_kv/localize_conv. It does not yet match the actual fast-mode
        time-averaged tracer transport or define true moving-volume inventories.
        Default 'legacy' retains the frozen original numerical path.
      - column_geometry='fixed_partial_v1' instead closes every fixed reference
        column at its actual bed, including one-node shallow water. Horizontal
        faces use the overlap of adjacent reference cells; flux differences
        divide by local capacity and gradients use its weighted adjoint.
        Requires conservative_kv/localize_conv; currently rejects GM/Redi and
        dynamic_ice pending their separate qualification. This is a geometry
        and operator candidate, not an eligible complete OMIP experiment.
      - match_barotropic_transport=True is an opt-in M2 transport candidate,
        requiring mode_split and nodal_dual_v1. Retains the original momentum
        predictor; replays accepted tracer stages with the actual OLD-face
        barotropic time mean. No source is counted twice. Retains static nodal
        thickness and the original surface tracer flux; no exact moving-volume
        heat/salt conservation, full temporal order or speedup is claimed.
      - process_time_scheme='consistent_split_v1' is an opt-in M3 candidate:
        convection subcycles retain physical kappa; slow rotation acts only on
        shear, and fast mean rotation uses implicit midpoint. Requires M2.
        The forward-backward gravity update and other process time errors remain;
        this is not a claim of second-order accuracy for the complete step.
      - process_time_scheme='subcycled_rk2_v2' additionally uses actual Heun
        nonlinear tracer substeps, predictor-velocity momentum stages and
        midpoint slow shear for matched tracer faces. Linear and gravity
        splitting errors remain; no whole-model second-order claim is made.
      - process_time_scheme='symmetric_fast_v3' additionally uses two actual
        half-continuity steps around midpoint fast momentum, constrained walls,
        and exact linear bottom drag around the full step. Diffusion, filters
        and coupled forcing errors remain unqualified; not whole-model RK2.
      - external_mode_scheme='symmetric' changes only the unsplit reference
        external pressure/continuity step and assigns mean rotation to it.
        Other processes keep legacy timing. This opt-in requires nodal_dual_v1
        and zero bottom drag; it does not establish complete-model accuracy.
      - meridional_boundary_scheme='closed_faces' retains velocity at every
        cell centre; outer mass/tracer faces stay closed. Normal momentum uses
        odd wall reflection, and mean rotation acts at all wet centres.
        Currently limited to the symmetric external mode on an all-wet, flat,
        Cartesian inviscid channel without polar filtering or dynamic ice.
    """
    finite_number('dt', dt, positive=True)
    finite_number('dt_bt', dt_bt, positive=True)
    if external_mode_scheme not in ('forward_backward', 'symmetric'):
        raise ValueError('unknown external_mode_scheme')
    if external_mode_scheme == 'symmetric':
        if mode_split or process_time_scheme != 'legacy' or column_geometry != 'nodal_dual_v1':
            raise ValueError('symmetric external mode requires unsplit legacy with nodal_dual_v1')
        active_drag = physics.cd if physics.bottom_friction == 'quadratic' else physics.r_bot
        if active_drag != 0:
            raise ValueError('symmetric external candidate requires separate bottom-drag qualification')
    validate_grid(grid)
    if meridional_boundary_scheme not in ('clamped_nodes', 'closed_faces'):
        raise ValueError('unknown meridional_boundary_scheme')
    if meridional_boundary_scheme == 'closed_faces':
        if external_mode_scheme != 'symmetric':
            raise ValueError('closed_faces requires the symmetric external mode')
        if (not np.all(np.asarray(grid.wet_mask) == 1)
                or (grid.wet_mask_3d is not None and not np.all(np.asarray(grid.wet_mask_3d) == 1))
                or not np.all(np.asarray(grid.depth) == -np.asarray(grid.z)[-1])
                or not np.all(np.asarray(grid.cos_lat) == 1)
                or not np.all(np.asarray(grid.dx_2d) == np.asarray(grid.dx_2d)[0, 0])):
            raise ValueError('closed_faces currently requires an all-wet flat Cartesian channel')
        if (any(getattr(physics, name) != 0 for name in (
                'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
                'kappa_conv', 'kappa_gm', 'kappa_redi'))
                or polar_cap_rows != 0 or polar_cap_taper != 0 or dynamic_ice
                or coastal_kappa_h != 0 or coastal_kappa_v != 0):
            raise ValueError('closed_faces requires separate mixing, polar-filter and ice qualification')
    if dtype not in {'float32', 'float64'}:
        raise ValueError('dtype must be float32 or float64')
    if physics.thermodynamics not in {'linear', 'teos10_reference'}:
        raise ValueError('unknown thermodynamics definition')
    if physics.thermodynamics == 'teos10_reference':
        from zhenmode.model.solver.physics.teos10 import validate_state
        validate_state(physics.S_ref,physics.T_ref,0.)  # also keep dry/ghost sentinels valid
        if eos_pressure_dbar is None:
            raise ValueError('TEOS reference variant requires explicit sea pressure in dbar')
        pressure = np.asarray(eos_pressure_dbar)
        if (pressure.shape != (grid.nx, grid.ny, grid.nz) or pressure.dtype.kind not in 'fiu'
                or not np.isfinite(pressure).all() or np.any((pressure < 0) | (pressure > 8000))
                or np.any(np.diff(pressure,axis=-1) < 0) or np.any(pressure[...,0] != 0)):
            raise ValueError('TEOS sea pressure must be finite ordered dbar on the full grid with surface zero')
        if dynamic_ice or ice_salt_flux != 0 or lambda_bulk != 0 or coastal_bulk_lambda != 0:
            raise ValueError('TEOS reference variant cannot use legacy ice or bulk temperature sources')
        if ice_freeze_temp_c != -1.8:
            raise ValueError('TEOS reference variant does not accept a legacy freezing-temperature override')
    elif eos_pressure_dbar is not None:
        raise ValueError('linear EOS does not accept an unused TEOS pressure field')
    for name in ('nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
                 'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'):
        finite_number(name, getattr(physics, name), nonnegative=True)
    for name, value in (('lambda_bulk', lambda_bulk), ('sss_restore_days', sss_restore_days),
                        ('coastal_restore_days', coastal_restore_days), ('coastal_bulk_lambda', coastal_bulk_lambda),
                        ('coastal_kappa_h', coastal_kappa_h), ('coastal_kappa_v', coastal_kappa_v),
                        ('sponge_days', sponge_days), ('eta_relax_days', eta_relax_days)):
        finite_number(name, value, nonnegative=True)
    for name, value in (('polar_cap_rows', polar_cap_rows), ('polar_cap_taper', polar_cap_taper),
                        ('sponge_cells', sponge_cells)):
        integer_count(name, value)
    if nu_nsub is not None and not (isinstance(nu_nsub, str) and nu_nsub == 'cfl'):
        integer_count('nu_nsub', nu_nsub, minimum=1)
    if match_barotropic_transport and (not mode_split or column_geometry not in {'nodal_dual_v1', 'fixed_partial_v1'}):
        raise ValueError("match_barotropic_transport requires mode_split=True and reference column geometry")
    if process_time_scheme not in ('legacy', 'consistent_split_v1', 'subcycled_rk2_v2', 'symmetric_fast_v3'):
        raise ValueError("unknown process_time_scheme")
    if process_time_scheme != 'legacy' and not match_barotropic_transport:
        raise ValueError(f"{process_time_scheme} requires match_barotropic_transport=True")
    if column_geometry in {'nodal_dual_v1', 'fixed_partial_v1'}:
        if not conservative_kv or not localize_conv:
            raise ValueError("reference column geometry requires conservative_kv=True and localize_conv=True")
    if column_geometry == 'fixed_partial_v1':
        if physics.kappa_gm > 0 or physics.kappa_redi > 0 or dynamic_ice:
            raise ValueError('fixed_partial_v1 requires separately qualified GM/Redi and sea-ice operators')
    if polar_cap_rows > 0 and 2 * (polar_cap_rows + polar_cap_taper) > grid.ny:
        raise ValueError('polar cap bands must not overlap; reduce rows/taper or disable the cap')
    base = make_fd_params(grid, column_geometry=column_geometry)
    nx, ny, nz = base.nx, base.ny, base.nz
    if surface_pressure_pa is not None:
        load = np.asarray(surface_pressure_pa)
        if load.shape != (nx, ny) or not np.isfinite(load).all():
            raise ValueError('surface pressure must be a finite (nx, ny) Pa field')
    if projection_niter is None:
        legacy_cap = os.environ.get('OCEAN_PAV_NITER')
        projection_niter_source = 'default' if legacy_cap is None else 'environment:OCEAN_PAV_NITER'
        try:
            projection_niter = 150 if legacy_cap is None else int(legacy_cap)
        except ValueError as error:
            raise ValueError('projection_niter environment value must be an integer') from error
    else:
        projection_niter_source = 'explicit'
    if isinstance(projection_niter, (bool, np.bool_)) or not isinstance(projection_niter, (int, np.integer)) or projection_niter <= 0:
        raise ValueError('projection_niter must be a positive integer')
    if projection_rtol is not None and (not np.isfinite(projection_rtol) or projection_rtol <= 0.):
        raise ValueError('projection_rtol must be finite and positive')
    if projection_preconditioner not in {'none', 'jacobi'}:
        raise ValueError('projection_preconditioner must be none or jacobi')
    if isinstance(projection_max_refinements, (bool, np.bool_)) or not isinstance(projection_max_refinements, (int, np.integer)) or not 0 <= projection_max_refinements <= 2:
        raise ValueError('projection_max_refinements must be an integer from 0 to 2')
    projection_dtype = jnp.float32 if dtype == 'float32' else jnp.float64
    projection_rtol = max(float(projection_rtol or 1e-12), 32. * float(jnp.finfo(projection_dtype).eps))
    if mixed_layer_depth_m is not None:
        if not np.isfinite(mixed_layer_depth_m) or mixed_layer_depth_m < 0.:
            raise ValueError("mixed_layer_depth_m must be finite and nonnegative")
    if mixed_layer_depth_2d is not None:
        depths = np.asarray(mixed_layer_depth_2d)
        if depths.shape != (nx, ny) or not np.all(np.isfinite(depths) & (depths > 0.)):
            raise ValueError("mixed_layer_depth_2d must have grid shape and positive finite depths")
    if dynamic_ice and (not np.isfinite(ice_insulation_scale_m) or ice_insulation_scale_m <= 0.):
        raise ValueError("ice_insulation_scale_m must be finite and positive")

    if forcing is None:
        tau_x_2d = jnp.zeros((nx, ny))
        tau_y_2d = jnp.zeros((nx, ny))
        Q_heat_2d = jnp.zeros((nx, ny))
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (jnp.array(f) for f in forcing)

    # Effective shallow-water depth = vertical grid span
    H_sw = float(jnp.sum(jnp.array(grid.dz)))
    dz_norm = (jnp.array(grid.dz).reshape(1, 1, -1) / H_sw)
    if column_geometry in {'nodal_dual_v1', 'fixed_partial_v1'}:
        column_thickness = np.asarray(base.dz_node) * np.asarray(base.wet_mask_z)
        column_depth = np.sum(column_thickness, axis=-1)
        H_sw = jnp.asarray(np.where(column_depth > 0., column_depth, 1.))
        dz_norm = jnp.asarray(column_thickness) / H_sw[..., None]

    if T_atm is not None and lambda_bulk > 0.0:
        T_atm_3d = jnp.array(T_atm)[:, :, None]
    else:
        T_atm_3d = jnp.zeros((nx, ny, 1))
        lambda_bulk = 0.0

    # ── Surface salinity restoring (Haney) ──
    # dSdt += -(SSS - S_ref)/tau at wet surface cells. tau=0 => off (bit-exact to
    # the pre-restoring solver). S_ref_surf is a (nx, ny) climatological SSS field
    # (e.g. WOA surface salinity); the restoring is a TRUE salt flux (psu/s, no
    # heat_factor -- salinity has no rho*cp). (D24)
    if S_ref_surf is not None and sss_restore_days > 0.0:
        S_ref_2d = jnp.array(S_ref_surf)
        restore_coef_S = 1.0 / (sss_restore_days * 86400.0)
    else:
        S_ref_2d = jnp.zeros((nx, ny))
        restore_coef_S = 0.0

    # ── Diagnostic coastal extra vertical tracer diffusion ──
    coastal_kappa_v_2d = _masked_coefficient(coastal_kappa_v_mask, coastal_kappa_v, (nx, ny))

    # ── Diagnostic coastal extra horizontal tracer diffusion ──
    coastal_kappa_h_2d = _masked_coefficient(coastal_kappa_h_mask, coastal_kappa_h, (nx, ny))

    # ── Diagnostic coastal extra bulk heat exchange ──
    # Same air-sea form as the bulk flux, but only in the land-adjacent band.
    # This is a more physical alternative to direct SST restoring.
    coastal_bulk_lambda_2d = _masked_coefficient(coastal_bulk_mask, coastal_bulk_lambda, (nx, ny))

    # ── Diagnostic coastal surface-temperature restoring ──
    # dTdt += -(SST - T_ref)/tau in the land-adjacent band. This is not a
    # physical closure; it is a narrowly scoped attribution experiment for the
    # 0..3-cell cold-bias population.
    if coastal_restore_mask is not None and coastal_restore_days > 0.0 \
            and coastal_restore_T is not None:
        coastal_restore_coef_2d = (
            jnp.array(coastal_restore_mask, dtype=jnp.float64)
            / (coastal_restore_days * 86400.0))
        coastal_restore_T_2d = jnp.array(coastal_restore_T)
    else:
        coastal_restore_coef_2d = jnp.zeros((nx, ny))
        coastal_restore_T_2d = jnp.zeros((nx, ny))

    # ── Lateral sponge (polar-edge Rayleigh damping) ──
    # Applied at both lat edges (the polar cap rows). Cosine-tapered from
    # r_max at the edge to 0 at sponge_cells into the interior.
    if sponge_days > 0.0 and sponge_cells > 0:
        r_max = 1.0 / (sponge_days * 86400.0)
        nc = int(sponge_cells)
        j = np.arange(ny)
        taper = np.zeros(ny)
        edge = np.minimum(j, ny - 1 - j)   # distance to nearest N/S edge
        in_band = edge < nc
        taper[in_band] = 0.5 * (1.0 + np.cos(np.pi * edge[in_band] / nc))
        sponge_2d_np = (r_max * taper).reshape(1, ny)
        sponge_rate_2d = jnp.array(np.broadcast_to(sponge_2d_np, (nx, ny)))
        sponge_rate = sponge_rate_2d[:, :, None]   # (nx, ny, 1)
        # The relaxation target must be the initial state AS THE MODEL SEES IT,
        # i.e. carrying the same land / below-seafloor sentinel that init_state
        # installs (T_ref / S_ref). Relaxing land toward the raw, unmasked T_init
        # drags the land value off that sentinel and breaks the "land and ghost
        # hold their pre-step value" invariant the diffusion mask establishes,
        # re-opening the coastline cliff the pressure integral then reads. (D19)
        wm3 = jnp.asarray(base.wet_mask_z)
        if T_init is not None:
            T_clim_3d = jnp.array(T_init) * wm3 + (1.0 - wm3) * physics.T_ref
            S_clim_3d = (jnp.array(S_init) if S_init is not None
                         else jnp.full_like(T_clim_3d, physics.S_ref))
            S_clim_3d = S_clim_3d * wm3 + (1.0 - wm3) * physics.S_ref
        else:
            # No initial field: init_state fills T_ref / S_ref everywhere, so
            # that is the only target that leaves the sponge a no-op.
            T_clim_3d = jnp.full((nx, ny, nz), physics.T_ref)
            S_clim_3d = jnp.full((nx, ny, nz), physics.S_ref)
    else:
        sponge_rate_2d = jnp.zeros((nx, ny))
        sponge_rate = jnp.zeros((nx, ny, 1))
        T_clim_3d = jnp.zeros((nx, ny, nz))
        S_clim_3d = jnp.zeros((nx, ny, nz))

    # ── Semi-enclosed-sea eta relaxation mask (Mediterranean artifact fix) ──
    # A parameterized lat/lon box; 0 rate = off (bit-exact to the old runs). The
    # buffer band (mask tapers 1 -> 0 over eta_relax_buffer degrees around the box
    # edge) keeps the PGF smooth at the mask boundary so the relaxation itself
    # cannot seed a new PGF cliff at Gibraltar. (D23)
    if eta_relax_days > 0.0 and eta_relax_box is not None:
        lon0, lon1, lat0, lat1 = eta_relax_box
        rate = 1.0 / (eta_relax_days * 86400.0)
        # The global grid lon is in [-180, 180) (WOA convention); a
        # Mediterranean box does not straddle the seam, so plain comparisons
        # suffice (a straddling box would need lon wrap handling — not used).
        lon2d, lat2d = np.meshgrid(np.asarray(grid.lon), np.asarray(grid.lat),
                                   indexing='ij')
        core = ((lon2d >= lon0) & (lon2d <= lon1)
                & (lat2d >= lat0) & (lat2d <= lat1))
        # Box-coordinate distance (degrees) outside the box, for the taper.
        dlon = np.maximum(np.maximum(lon0 - lon2d, lon2d - lon1), 0.0)
        dlat = np.maximum(np.maximum(lat0 - lat2d, lat2d - lat1), 0.0)
        d_out = np.maximum(dlon, dlat)
        buf = float(eta_relax_buffer)
        mask_np = np.where(core, 1.0,
                           np.where(d_out < buf, 0.5 * (1.0 + np.cos(
                               np.pi * d_out / buf)), 0.0))
        mask_np = mask_np * np.asarray(base.wet_mask)   # ocean points only
        eta_relax_mask = jnp.array(mask_np)
        eta_relax_rate = rate
    else:
        eta_relax_mask = jnp.zeros((nx, ny))
        eta_relax_rate = 0.0

    # Nonlinear products (u*du/dx, ...) amplify the 2-dx grid-scale mode. Lon is
    # periodic -> a 2/3 FFT rule in lon is exact; the closed lat wall can't use
    # FFT, so the lat 2-dx is killed by a 5-pt binomial low-pass in _dealias_h_fd.
    # That helper is applied to adv_u/adv_v and to the GM/Redi skew-flux tendency
    # ONLY -- not to w or to the flux-form tracer advection, whose exact
    # telescoping it would destroy. (D7, D15, D25)
    _kmax = nx // 2
    _keep = max(1, int(_kmax * 2 / 3))
    _kidx = np.fft.fftfreq(nx) * nx            # 0..nx/2, -nx/2..-1
    _lon_mask_np = (np.abs(_kidx) <= _keep).astype(np.float64).reshape(nx, 1, 1)
    dealias_lon_mask = jnp.array(np.broadcast_to(_lon_mask_np, (nx, 1, 1)))

    # ── Mode split (baroclinic/barotropic) ──
    # n_subcyc = exact number of barotropic subcycles filling dt: dt_bt_eff =
    # dt/n_subcyc (rounded so each subcycle is a uniform dt_bt). With
    # mode_split=False all fields are inert (n_subcyc=0) and _step_impl takes the
    # bit-exact monolithic path. (D12)
    if mode_split:
        n_subcyc = max(1, int(round(dt / dt_bt)))
        dt_bt_eff = dt / n_subcyc
    else:
        n_subcyc = 0
        dt_bt_eff = 0.0

    # ── Compute dtype ──
    # 'float64' (default) = legacy bit-exact path. 'float32' casts every params
    # array and zeros template to fp32 -- on Ada-class GPUs (FP64:FP32 = 1:64)
    # this is the single biggest kernel-time lever. The runner still reads/writes
    # float64 npz (I/O casts at the boundary). The cast of params happens right
    # after FDPhysParams(...).
    state_dtype = jnp.float32 if dtype == 'float32' else jnp.float64

    # ── nu_h subcycle right-sizing (split L half-steps) ──
    # Legacy (None): n_nu = n_subcyc (24 at dt=3600/dt_bt=150) -- comfortably
    # inside the bound, but ~1.6x more Laplacian pairs than the bound needs.
    # nu_nsub='cfl' right-sizes from the true worst metric point (see
    # nu_nsub_for_2d_cfl). An int is honored verbatim (probe/benchmark). (D12)
    if nu_nsub == 'cfl':
        nu_nsub = nu_nsub_for_2d_cfl(physics.nu_h, dt, grid.dx_2d, grid.dy)
    elif nu_nsub is not None:
        nu_nsub = max(1, int(nu_nsub))

    conv_nsub = max(1, int(np.ceil(
        physics.kappa_conv * dt / (0.4 * float(jnp.min(jnp.array(grid.dz))) ** 2))))
    adv_nsub = max(1, int(np.ceil(dt * 4.0e-3 / (0.5 * float(jnp.min(jnp.array(grid.dz)))))))
    if column_geometry == 'nodal_dual_v1':
        widths = np.asarray(base.dz_node).ravel()
        distances = np.asarray(base.dz_iface).ravel()
        interface_rate = 1. / distances
        row_rate = (np.pad(interface_rate, (1, 0)) + np.pad(interface_rate, (0, 1))) / widths
        conv_nsub = max(1, int(np.ceil(physics.kappa_conv * dt * np.max(row_rate) / 0.4)))
        adv_nsub = max(1, int(np.ceil(dt * 4.0e-3 / (0.5 * np.min(widths)))))
        if physics.nu_v * (dt / 2.) * np.max(row_rate) > 0.4:
            raise ValueError("nodal_dual_v1 vertical momentum diffusion exceeds its explicit CFL margin")
        maximum_kappa_v = physics.kappa_v + float(np.max(np.asarray(coastal_kappa_v_2d)))
        if maximum_kappa_v * (dt / 2.) * np.max(row_rate) > 0.4:
            raise ValueError("nodal_dual_v1 vertical tracer diffusion exceeds its explicit CFL margin")

    if column_geometry == 'fixed_partial_v1':
        widths = np.asarray(base.dz_node)
        wet = np.asarray(base.wet_mask_z) > .5
        interface_rate = (wet[..., :-1] & wet[..., 1:]) / np.asarray(base.dz_iface)
        row_rate = (np.pad(interface_rate, ((0,0),(0,0),(1,0)))
                    + np.pad(interface_rate, ((0,0),(0,0),(0,1)))) / widths
        maximum_rate = float(np.max(row_rate))
        conv_nsub = max(1, int(np.ceil(physics.kappa_conv * dt * maximum_rate / .4)))
        adv_nsub = max(1, int(np.ceil(dt * 4.e-3 / (.5 * np.min(widths[wet])))))
        maximum_kappa_v = physics.kappa_v + float(np.max(np.asarray(coastal_kappa_v_2d)))
        if max(physics.nu_v, maximum_kappa_v) * (dt/2.) * maximum_rate > .4:
            raise ValueError('fixed_partial_v1 vertical diffusion exceeds its explicit CFL margin')

    params = FDPhysParams(
        dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
        inv_dx=base.inv_dx, inv_dy=base.inv_dy,
        inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
        f=base.f, wet_mask=base.wet_mask, wet_mask_z=base.wet_mask_z,
        interior_mask_z=(jnp.ones_like(base.interior_mask_z)
                         if meridional_boundary_scheme == 'closed_faces' else base.interior_mask_z),
        dz_denom_interior=base.dz_denom_interior,
        dz_bnd_top=base.dz_bnd_top, dz_bnd_bot=base.dz_bnd_bot,
        d2z_hm=base.d2z_hm, d2z_hp=base.d2z_hp, d2z_denom=base.d2z_denom,
        d2z_h0_top=base.d2z_h0_top, d2z_h0_bot=base.d2z_h0_bot,
        dz_3d=base.dz_3d, dz_surface=base.dz_surface, dz_iface=base.dz_iface,
        dz_node=base.dz_node,
        surface_mask=base.surface_mask, bottom_mask=base.bottom_mask,
        nx=nx, ny=ny, nz=nz,
        nu_h=physics.nu_h, nu_v=physics.nu_v,
        kappa_h=physics.kappa_h, kappa_v=physics.kappa_v,
        kappa_conv=physics.kappa_conv,
        nu_bi=physics.nu_bi, kappa_bi=physics.kappa_bi,
        T_ref=physics.T_ref, S_ref=physics.S_ref,
        r_bot=physics.r_bot, cd=physics.cd,
        bottom_friction=physics.bottom_friction,
        tau_x_2d=tau_x_2d, tau_y_2d=tau_y_2d, Q_heat_2d=Q_heat_2d,
        H_sw=H_sw, dz_norm=dz_norm, dt=dt,
        T_atm_3d=T_atm_3d, lambda_bulk=lambda_bulk,
        S_ref_2d=S_ref_2d, restore_coef_S=restore_coef_S,
        coastal_restore_coef_2d=coastal_restore_coef_2d,
        coastal_bulk_lambda_2d=coastal_bulk_lambda_2d,
        coastal_kappa_h_2d=coastal_kappa_h_2d,
        coastal_kappa_v_2d=coastal_kappa_v_2d,
        coastal_restore_T_2d=coastal_restore_T_2d,
        sponge_rate=sponge_rate, sponge_rate_2d=sponge_rate_2d,
        T_clim_3d=T_clim_3d, S_clim_3d=S_clim_3d,
        polar_cap_rows=int(polar_cap_rows),
        polar_cap_taper=int(polar_cap_taper),
        dealias_lon_mask=dealias_lon_mask,
        kappa_gm=physics.kappa_gm,
        gm_slope_max=physics.gm_slope_max,
        kappa_redi=physics.kappa_redi,
        eta_relax_mask=eta_relax_mask,
        eta_relax_rate=eta_relax_rate,
        mode_split=bool(mode_split),
        dt_bt=dt_bt_eff,
        n_subcyc=n_subcyc,
        conv_nsub=conv_nsub,
        nu_nsub=nu_nsub,
        use_scan=bool(use_scan),
        # Vertical-advection subcycles: keep dt*(|u|/dx + |v|/dy + w/dz) below
        # ~0.5 per substep. The dominant constraint is w/dz in the 5 m surface
        # layer (equatorial upwelling w~3e-3 m/s -> dt*w/dz ~ 2.2 at dt=3600); the
        # horizontal terms (~1e-6/s at |u|~1 m/s) are negligible. Sized from a
        # nominal w_max=4e-3 m/s and a 0.5 target CFL per substep. Monolithic
        # dt=60-300: dt*w/dz <= 0.18 -> adv_nsub=1, path untouched. (D16)
        adv_nsub=adv_nsub,
        freeze_adv_vel=bool(freeze_adv_vel),
        conservative_kv=bool(conservative_kv),
        project_adv_vel=bool(project_adv_vel),
        localize_conv=bool(localize_conv),
        monotone_adv=bool(monotone_adv),
        fct_adv=bool(fct_adv),
        mixed_layer_depth_m=float(mixed_layer_depth_m or 0.0),
        mixed_layer_mask_2d=(jnp.asarray(mixed_layer_mask, dtype=jnp.float64)
                              if mixed_layer_mask is not None
                              else jnp.ones((nx, ny), dtype=jnp.float64)),
        mixed_layer_depth_2d=(jnp.asarray(mixed_layer_depth_2d,
                                           dtype=jnp.float64)
                              if mixed_layer_depth_2d is not None else None),
        ice_freeze_temp_c=float(ice_freeze_temp_c),
        ice_salt_flux=float(ice_salt_flux),
        dynamic_ice=bool(dynamic_ice),
        ice_insulation_scale_m=float(ice_insulation_scale_m),
        projection_niter=int(projection_niter),
        projection_rtol=projection_rtol,
        projection_preconditioner=projection_preconditioner,
        projection_inv_diagonal=None,
        projection_niter_source=projection_niter_source,
        projection_max_refinements=int(projection_max_refinements),
        column_geometry=column_geometry,
        match_barotropic_transport=bool(match_barotropic_transport),
        process_time_scheme=process_time_scheme,
        thermodynamics=physics.thermodynamics,
        eos_pressure_dbar=(None if eos_pressure_dbar is None else jnp.asarray(eos_pressure_dbar)),
        face_contacts=base.face_contacts, contact_depths_m=base.contact_depths_m,
        node_depth_m=base.node_depth_m,
        surface_pressure_pa=(None if surface_pressure_pa is None else jnp.asarray(surface_pressure_pa)),
        external_mode_scheme=external_mode_scheme,
        meridional_boundary_scheme=meridional_boundary_scheme,
    )

    if projection_preconditioner == 'jacobi':
        diagonal = _column_projection_diagonal(params)
        params = params._replace(projection_inv_diagonal=1. / jnp.where(diagonal > 0., diagonal, 1.))

    # fp32 cast: params was just built in float64 (numpy defaults); when
    # computing in float32 every array field must be cast too, or XLA inserts
    # implicit f64 promotion kernels that silently kill the fp32 speedup.
    # Scalars (int/float/str/None) are passed through unchanged.
    if dtype == 'float32':
        params = params._replace(**{
            f: jax.tree.map(lambda a: a.astype(state_dtype)
                            if isinstance(a,jnp.ndarray) and a.dtype==jnp.float64 else a,v)
            for f in params._fields for v in [getattr(params, f)]})

    return _compile_solver(params, physics, (nx, ny, nz), state_dtype,
                           dynamic_forcing=dynamic_forcing, return_params=return_params)


def _compile_solver(params, physics, shape, state_dtype, *, dynamic_forcing, return_params):
    """Bind one parameter set to step, initial-state and diagnostic entry points."""
    nx, ny, nz = shape
    @jax.jit
    def step(state):
        return _step_impl(state, params)

    # ── Dynamic forcing path ──
    # Same compiled graph as step() except the 2D forcing (tau_x, tau_y, Q_heat)
    # is passed as RUNTIME arguments instead of baked constants. params._replace
    # on the namedtuple swaps the three fields for traced placeholders; XLA
    # compiles ONE graph shared by all 12 monthly snapshots (no 12x memory). With
    # dynamic_forcing=False nothing is built (bit-exact to the pre-dynamic path).
    step_dyn = None
    if dynamic_forcing:
        @jax.jit
        def step_dyn(state, tau_x, tau_y, q_heat, T_atm_3d=None, surface_pressure_pa=None):
            updates = {
                'tau_x_2d': tau_x,
                'tau_y_2d': tau_y,
                'Q_heat_2d': q_heat,
            }
            # Optional runtime bulk target lets monthly atmospheric forcing
            # reuse the single compiled dynamic-forcing graph. Existing
            # callers that omit this argument remain bit-compatible.
            if T_atm_3d is not None:
                updates['T_atm_3d'] = T_atm_3d
            if surface_pressure_pa is not None:
                updates['surface_pressure_pa'] = surface_pressure_pa
            return _step_impl(state, params._replace(**updates))

    @jax.jit
    def diagnostics(state, surface_pressure_pa=None):
        current = (params if surface_pressure_pa is None else
                   params._replace(surface_pressure_pa=surface_pressure_pa))
        rho_prime = _density_anomaly(state.T, state.S, params)
        rho = RHO_0 + rho_prime
        pressure = _compute_hydrostatic_pressure(state, current)
        w = _compute_vertical_velocity(state, current)
        return rho, pressure, w

    @jax.jit
    def terms_fn(state):
        return _tracer_terms(state, params)

    def init_state(T_init=None, S_init=None):
        if params.thermodynamics == 'teos10_reference':
            if T_init is None or S_init is None or np.shape(T_init) != shape or np.shape(S_init) != shape:
                raise ValueError('TEOS initialization requires full-grid CT and reference salinity')
            from zhenmode.model.solver.physics.teos10 import validate_state
            validate_state(S_init, T_init, params.eos_pressure_dbar)
        u = jnp.zeros((nx, ny, nz), dtype=state_dtype)
        v = jnp.zeros((nx, ny, nz), dtype=state_dtype)
        eta = jnp.zeros((nx, ny), dtype=state_dtype)
        if T_init is not None:
            T = jnp.array(T_init, dtype=state_dtype)
            S = jnp.array(S_init, dtype=state_dtype) if S_init is not None \
                else jnp.full_like(T, physics.S_ref)
        else:
            T = jnp.full((nx, ny, nz), physics.T_ref, dtype=state_dtype)
            S = jnp.full((nx, ny, nz), physics.S_ref, dtype=state_dtype)
        # Mask land + ghost water (layers below seafloor): set to a sentinel.
        # wet_mask_z is the TRUE 3D mask (1 where water exists, 0 on land AND
        # below seafloor). This discards WOA-interpolated T in ghost layers
        # before it can enter the pressure integral or advection.
        T = T * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.T_ref
        S = S * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.S_ref
        return JaxStateG(u, v, T, S, eta, jnp.zeros_like(eta))

    # Return arity unchanged when dynamic_forcing=False (all existing
    # callers unpack 3-/5-tuples); with dynamic_forcing=True step_dyn is
    # appended as the last element.
    if dynamic_forcing:
        if return_params:
            return step, init_state, diagnostics, params, terms_fn, step_dyn
        return step, init_state, diagnostics, step_dyn
    if return_params:
        return step, init_state, diagnostics, params, terms_fn
    return step, init_state, diagnostics
