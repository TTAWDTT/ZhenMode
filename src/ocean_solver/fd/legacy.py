"""
Global Finite-Difference Ocean Solver — hydrostatic primitive equations, JAX.

Numerics:
  - 2nd-order finite differences on a global lat-lon grid (lon-periodic),
  - spherical metric factors (dx = R*cos(lat)*dlon, varies with latitude),
  - a real wet_mask for no-flux land boundaries,
  - full 2D Coriolis f = 2*Omega*sin(lat).

Convention (shared with grid.py):
  - 3D fields: (nx, ny, nz), axis 0 = lon (periodic), axis 1 = lat, axis 2 = z
  - 2D fields: (nx, ny)
  - z negative downward, z=0 at surface

Every FD operator is a pure function of (field, params); `params` carries the
precomputed metric fields. The reasoning behind each closure, limiter and
subcycle count — with the measurements that settled it — lives in
docs/decisions.md, keyed by the D-numbers referenced below. Keep it there:
this file is the solver, not the lab notebook.
"""

from collections import namedtuple as namedtuple

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.fd.backend import jax as jax
from ocean_solver.fd.backend import jnp as jnp
from ocean_solver.fd.backend import np as np
from ocean_solver.fd.barotropic import (
    _barotropic_subcycle_transport as _barotropic_subcycle_transport,
)
from ocean_solver.fd.barotropic import _filter_barotropic_eta as _filter_barotropic_eta
from ocean_solver.fd.barotropic import _free_surface_step_fd as _free_surface_step_fd
from ocean_solver.fd.barotropic import _refill_volume as _refill_volume
from ocean_solver.fd.barotropic import _symmetric_free_surface_step as _symmetric_free_surface_step
from ocean_solver.fd.closures import _GM_RHOZ_FLOOR as _GM_RHOZ_FLOOR
from ocean_solver.fd.closures import _REDI_CFL_TARGET as _REDI_CFL_TARGET
from ocean_solver.fd.closures import _isopycnal_closure as _isopycnal_closure
from ocean_solver.fd.closures import _isopycnal_slope as _isopycnal_slope
from ocean_solver.fd.closures import _redi_skew_flux_tendency as _redi_skew_flux_tendency
from ocean_solver.fd.eos import _density_anomaly as _density_anomaly
from ocean_solver.fd.factory import make_solver_global as make_solver_global
from ocean_solver.fd.geometry import make_fd_params as make_fd_params
from ocean_solver.fd.horizontal import _apply_polar_cap as _apply_polar_cap
from ocean_solver.fd.horizontal import _biharmonic_h as _biharmonic_h
from ocean_solver.fd.horizontal import _d_dx as _d_dx
from ocean_solver.fd.horizontal import _d_dy as _d_dy
from ocean_solver.fd.horizontal import _dealias_h_fd as _dealias_h_fd
from ocean_solver.fd.horizontal import _divergence_conservative as _divergence_conservative
from ocean_solver.fd.horizontal import _divergence_conservative_3d as _divergence_conservative_3d
from ocean_solver.fd.horizontal import _divergence_h as _divergence_h
from ocean_solver.fd.horizontal import _gradient_conservative as _gradient_conservative
from ocean_solver.fd.horizontal import _gradient_conservative_3d as _gradient_conservative_3d
from ocean_solver.fd.horizontal import _gradient_face_gated_3d as _gradient_face_gated_3d
from ocean_solver.fd.horizontal import (
    _horizontal_biharmonic_tracer as _horizontal_biharmonic_tracer,
)
from ocean_solver.fd.horizontal import _horizontal_diffusion_flux as _horizontal_diffusion_flux
from ocean_solver.fd.horizontal import _horizontal_tracer_diffusion as _horizontal_tracer_diffusion
from ocean_solver.fd.horizontal import _laplacian_h as _laplacian_h
from ocean_solver.fd.horizontal import _polar_cap_weights as _polar_cap_weights
from ocean_solver.fd.integration import _explicit_full_step as _explicit_full_step
from ocean_solver.fd.integration import _nonlinear_predictor_rk2 as _nonlinear_predictor_rk2
from ocean_solver.fd.integration import _step_impl as _step_impl
from ocean_solver.fd.integration import _tracer_rk_subcycle as _tracer_rk_subcycle
from ocean_solver.fd.integration import _tracer_step_with_transport as _tracer_step_with_transport
from ocean_solver.fd.pressure import _compute_bt_rho_pgf as _compute_bt_rho_pgf
from ocean_solver.fd.pressure import _compute_hydrostatic_pressure as _compute_hydrostatic_pressure
from ocean_solver.fd.pressure import _compute_pressure_gradient as _compute_pressure_gradient
from ocean_solver.fd.pressure import _reference_depth_gradient as _reference_depth_gradient
from ocean_solver.fd.processes import _compute_momentum_residual as _compute_momentum_residual
from ocean_solver.fd.processes import _compute_momentum_tendency as _compute_momentum_tendency
from ocean_solver.fd.processes import _compute_tracer_residual as _compute_tracer_residual
from ocean_solver.fd.processes import _compute_tracer_tendency as _compute_tracer_tendency
from ocean_solver.fd.processes import _coriolis_rotation_2d as _coriolis_rotation_2d
from ocean_solver.fd.processes import _linear_bottom_drag_step as _linear_bottom_drag_step
from ocean_solver.fd.processes import _linear_half_step as _linear_half_step
from ocean_solver.fd.processes import _rotate_baroclinic_shear as _rotate_baroclinic_shear
from ocean_solver.fd.processes import _tracer_terms as _tracer_terms
from ocean_solver.fd.projection import _column_projection_diagonal as _column_projection_diagonal
from ocean_solver.fd.projection import _project_column_divergence as _project_column_divergence
from ocean_solver.fd.projection import projection_config as projection_config
from ocean_solver.fd.sources import _dynamic_ice_closure as _dynamic_ice_closure
from ocean_solver.fd.sources import _surface_heat_weights as _surface_heat_weights
from ocean_solver.fd.stability import nu_nsub_for_2d_cfl as nu_nsub_for_2d_cfl
from ocean_solver.fd.subcycles import _subcycle as _subcycle
from ocean_solver.fd.transport import _advection_flux_form as _advection_flux_form
from ocean_solver.fd.transport import _advection_scalar as _advection_scalar
from ocean_solver.fd.transport import _barotropic_velocity as _barotropic_velocity
from ocean_solver.fd.transport import _column_divergence as _column_divergence
from ocean_solver.fd.transport import _compute_vertical_velocity as _compute_vertical_velocity
from ocean_solver.fd.transport import _face_transport_divergence as _face_transport_divergence
from ocean_solver.fd.transport import _layer_face_transports as _layer_face_transports
from ocean_solver.fd.transport import _limited_tracer_slope as _limited_tracer_slope
from ocean_solver.fd.transport import _match_layer_face_transports as _match_layer_face_transports
from ocean_solver.fd.transport import _reference_depth_divergence as _reference_depth_divergence
from ocean_solver.fd.transport import _vertical_transport_iface as _vertical_transport_iface
from ocean_solver.fd.types import FDParams as FDParams
from ocean_solver.fd.types import FDPhysParams as FDPhysParams
from ocean_solver.fd.types import JaxStateG as JaxStateG
from ocean_solver.fd.vertical import _conv_flux_tendency as _conv_flux_tendency
from ocean_solver.fd.vertical import _convective_mask as _convective_mask
from ocean_solver.fd.vertical import _d2_dz2 as _d2_dz2
from ocean_solver.fd.vertical import _d2_dz2_flux as _d2_dz2_flux
from ocean_solver.fd.vertical import _d_dz as _d_dz
from ocean_solver.fd.vertical import _effective_kappa_v as _effective_kappa_v
from ocean_solver.fd.vertical import _fill_ghost_bottom as _fill_ghost_bottom
from ocean_solver.fd.vertical import _iface_flux_divergence as _iface_flux_divergence
from ocean_solver.fd.vertical import _vertical_diffusion as _vertical_diffusion
from ocean_solver.fd.vertical import _vertical_momentum_diffusion as _vertical_momentum_diffusion
from ocean_solver.geometry.columns import nodal_control_thickness as nodal_control_thickness
from ocean_solver.validation.mms import _mms_convergence as _mms_convergence
from ocean_solver.validation.mms import _mms_grid as _mms_grid
from ocean_solver.validation.mms import _mms_run as _mms_run

preserve_legacy_names(globals(), 'jax_solver_global')

if __name__ == "__main__":
    print("=== Global FD Solver — MMS verification ===")
    res = _mms_run()
    all_pass = True
    for name, (err, ok) in res.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:30s} = {err:.3e}  [{status}]")
    print()
    print("--- 2nd-order convergence check (d/dy) ---")
    conv = _mms_convergence()
    print(f"  coarse (ny=14)  d/dy rel L2 = {conv['ddy_coarse']:.3e}")
    print(f"  fine   (ny=28)  d/dy rel L2 = {conv['ddy_fine']:.3e}")
    print(f"  convergence ratio (expect ~4 for 2nd-order) = {conv['convergence_ratio']:.2f}  "
          f"[{'PASS' if conv['passes'] else 'FAIL'}]")
    if not conv['passes']:
        all_pass = False
    print()
    print("ALL PASS" if all_pass else "SOME FAILED")
    raise SystemExit(0 if all_pass else 1)
