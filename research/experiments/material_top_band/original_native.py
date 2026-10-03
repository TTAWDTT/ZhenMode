"""Audit the original nodal authority without accepting a moving-stock handoff.

Original u/v remain point samples with reference-mass momentum diagnostics.
Tracer inventory uses the original first-slot moving thickness. This module
records the actual original clocks and refuses incompatible joint authorities.
"""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from ocean_solver.candidates.material import solver as material
from ocean_solver.configuration import OMEGA, R_EARTH, RHO_0
from ocean_solver.fd.eos import _density_anomaly
from ocean_solver.fd.geometry import make_fd_params
from ocean_solver.fd.pressure import _compute_hydrostatic_pressure, _compute_pressure_gradient
from ocean_solver.fd.transport import _face_transport_divergence, _layer_face_transports
from ocean_solver.fd.types import JaxStateG
from ocean_solver.geometry.columns import nodal_control_thickness
from ocean_solver.provenance.archives import current_source_files
from research.experiments.material_top_band.integration import parameter_digest, state_digest

AUTHORITY = "FD_point_samples_material_top_mass_lumped_linear_momentum_v1"
STAGES = ("bottom_drag_first", "linear_first", "nonlinear_predictor",
          "predictor_linear_second", "barotropic", "transport_match",
          "accepted_tracer_replay", "linear_second", "bottom_drag_second", "closed_wall")


def _immutable(value):
    array = _finite(value, "derived native value")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _snapshot(state):
    return JaxStateG(*(_immutable(value) for value in state))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, str):
        return value
    return _immutable(value)


def _finite(value, name):
    if np.ma.getmaskarray(value).any():
        raise ValueError("masked native input: "+name)
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("nonfinite or non-real native input: "+name)
    return array


def _source_hashes():
    root = Path(__file__).resolve().parents[3]
    selected = [
        "research/experiments/material_top_band/original_native.py",
        "research/experiments/material_top_band/integration.py",
        "research/experiments/fd_static_bridge/bridge.py",
        "tests/research/contracts/test_original_native_integration.py",
        "docs/original_small_grid_integration_protocol.json",
        "scripts/run_bounded_research_tests.py",
        "scripts/capture_original_native_contract.py",
    ]
    return MappingProxyType({name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in current_source_files(root, selected).items()})


@dataclass(frozen=True)
class NativeReferenceView:
    state: object
    h_ref: np.ndarray
    h_tracer: np.ndarray
    tracer_stocks: np.ndarray
    reference_momentum: np.ndarray
    mass: np.ndarray
    area: np.ndarray
    rho_prime: np.ndarray
    pressure: np.ndarray
    acceleration: np.ndarray
    faces: tuple
    parameter_digest: str
    source_hashes: object
    authority: str = AUTHORITY


def capture_native(state, params, grid):
    """Bind actual original state, factory metrics and clock; never remap DOFs."""
    shape = (grid.nx, grid.ny, grid.nz)
    if shape != (8, 4, 6):
        raise ValueError("native audit currently requires the original 8x4x6 grid")
    for name, value in state._asdict().items():
        array = _finite(value, name)
        expected = shape[:2] if name in {"eta", "ice"} else shape
        if array.shape != expected or array.dtype != np.float64:
            raise ValueError("native state shape/float64 binding: "+name)
    for name, value in params._asdict().items():
        if np.ma.getmaskarray(value).any():
            raise ValueError("masked native parameter: "+name)
        if value is not None and np.asarray(value).dtype.kind not in "US":
            _finite(value, "params."+name)
    for name in ("dx_2d", "dy", "cos_lat", "f", "z", "dz", "depth",
                 "wet_mask", "wet_mask_3d", "lon", "lat"):
        _finite(getattr(grid, name), "grid."+name)
    material._validate_params(params)
    latitude = np.array([-30., -10., 10., 30.])
    cosine = np.cos(np.radians(latitude))
    prescribed = {"z": np.array([0., -5., -15., -30., -50., -80.]),
                  "lat": latitude, "lon": (np.arange(8)+.5)*360./8,
                  "cos_lat": cosine,
                  "dx_2d": np.broadcast_to(R_EARTH*cosine*np.radians(45.), (8, 4)),
                  "dy": R_EARTH*np.radians(20.),
                  "f": np.broadcast_to(2*OMEGA*np.sin(np.radians(latitude)), (8, 4))}
    for name, expected_value in prescribed.items():
        if not np.array_equal(np.asarray(getattr(grid, name)), expected_value):
            raise ValueError("native frozen spherical geometry binding: "+name)
    if (not np.all(np.asarray(grid.wet_mask_3d) == 1.)
            or not np.all(np.asarray(grid.wet_mask) == 1.)
            or np.any(np.asarray(grid.depth) != -np.asarray(grid.z)[-1])
            or np.asarray(grid.z)[0] != 0.
            or not np.array_equal(grid.dz, -np.diff(grid.z))):
        raise ValueError("native all-wet flat-bottom geometry binding")
    expected = make_fd_params(grid, column_geometry="nodal_dual_v1")
    for name, value in expected._asdict().items():
        if not np.array_equal(np.asarray(getattr(params, name)), np.asarray(value)):
            raise ValueError("native derived metric binding: "+name)
    widths = nodal_control_thickness(grid.z)
    href = np.broadcast_to(widths, shape)
    depth = widths.sum()
    if (np.any(np.asarray(params.H_sw) != depth)
            or not np.array_equal(np.asarray(params.dz_norm), href/depth)
            or np.any(np.asarray(grid.dx_2d) <= 0.) or grid.dy <= 0.
            or np.any(np.asarray(grid.cos_lat) <= 0.)):
        raise ValueError("native reference column/horizontal metric binding")
    counts = (params.n_subcyc, params.adv_nsub, params.conv_nsub, params.nu_nsub)
    if any(value is None or isinstance(value, (bool, np.bool_)) or int(value) != value or not 1 <= value <= 16
           for value in counts):
        raise ValueError("native bounded integral subcycle binding")
    if (params.n_subcyc != 12 or params.dt != .12
            or params.dt_bt != params.dt/params.n_subcyc):
        raise ValueError("native twelve-call clock binding")
    adv_count = max(1, int(np.ceil(params.dt*.004/(.5*widths.min()))))
    interface_rate = 1./np.abs(np.diff(grid.z))
    row_rate = (np.pad(interface_rate, (1, 0))+np.pad(interface_rate, (0, 1)))/widths
    conv_count = max(1, int(np.ceil(params.kappa_conv*params.dt*row_rate.max()/.4)))
    if params.adv_nsub != adv_count or params.conv_nsub != conv_count:
        raise ValueError("native factory-derived count binding")
    snapshot = _snapshot(state)
    if (np.any(snapshot.ice != 0.) or np.any(snapshot.T < -5.) or np.any(snapshot.T > 45.)
            or np.any(snapshot.S < 0.) or np.any(snapshot.S > 50.)
            or np.any(snapshot.v*np.asarray(params.interior_mask_z) != snapshot.v)):
        raise ValueError("native tracer/ice/constrained-wall state binding")
    original = JaxStateG(*(jnp.asarray(value) for value in snapshot))
    h = np.asarray(material.material_thickness(original.eta, params))
    if np.any(h <= 0.):
        raise ValueError("native moving tracer thickness must be positive")
    area = np.asarray(params.dx_2d)*params.dy
    mass = RHO_0*area[..., None]*href
    if not np.isfinite(mass).all() or np.any(mass <= 0.):
        raise ValueError("native reference mass must be finite and positive")
    acceleration = np.stack(_compute_pressure_gradient(original, params))
    acceleration[1] *= np.asarray(params.interior_mask_z)
    return NativeReferenceView(
        snapshot, _immutable(href), _immutable(h), _immutable(material._contents(original, params)),
        _immutable(RHO_0*href[..., None]*np.stack((snapshot.u, snapshot.v), axis=-1)),
        _immutable(mass), _immutable(area), _immutable(_density_anomaly(original.T, original.S, params)),
        _immutable(_compute_hydrostatic_pressure(original, params)), _immutable(acceleration),
        tuple(_immutable(value) for value in _layer_face_transports(original.u, original.v, params)),
        parameter_digest(params), _source_hashes())


@dataclass(frozen=True)
class ReferencePressureKick:
    before_velocity: np.ndarray
    after_velocity: np.ndarray
    before_momentum: np.ndarray
    after_momentum: np.ndarray
    force: np.ndarray
    impulse_work: float
    pressure_work: float
    transport_work: float
    kinetic_change: float
    physical_moving_pressure_qualified: bool = False


def reference_pressure_kick(view, params, *, duration=.01):
    """Actual scratch fixed-Dref stock kick with frozen original dynamic Pa."""
    value = _finite(duration, "pressure probe duration")
    if (value.shape != () or value.dtype.kind == "b" or not 0. < float(value) <= .01
            or parameter_digest(params) != view.parameter_digest):
        raise ValueError("native pressure probe duration/parameter binding")
    if view.authority != AUTHORITY:
        raise ValueError("native pressure probe authority binding")
    original = JaxStateG(*(jnp.asarray(value) for value in view.state))
    href = np.broadcast_to(np.asarray(params.dz_node), original.u.shape)
    expected_mass = RHO_0*(np.asarray(params.dx_2d)*params.dy)[..., None]*href
    acceleration = np.stack(_compute_pressure_gradient(original, params))
    acceleration[1] *= np.asarray(params.interior_mask_z)
    expected = {"h_ref": href, "mass": expected_mass,
                "area": np.asarray(params.dx_2d)*params.dy,
                "pressure": np.asarray(_compute_hydrostatic_pressure(original, params)),
                "acceleration": acceleration}
    for name, actual in expected.items():
        supplied = _finite(getattr(view, name), "view."+name)
        if not np.array_equal(supplied, actual):
            raise ValueError("native pressure probe stock/metric/operator binding: "+name)
    if dict(view.source_hashes) != dict(_source_hashes()):
        raise ValueError("native pressure probe source binding")
    duration = float(value)
    mass = np.broadcast_to(view.mass, view.acceleration.shape)
    force = mass*view.acceleration
    before_momentum = mass*np.stack((view.state.u, view.state.v))
    before_velocity = before_momentum/mass
    after_momentum = before_momentum+duration*force
    after_velocity = after_momentum/mass
    midpoint = .5*(before_velocity+after_velocity)
    impulse_work = math.fsum((midpoint*(after_momentum-before_momentum)).ravel())
    pressure_work = math.fsum((duration*midpoint*force).ravel())
    faces = _layer_face_transports(jnp.asarray(midpoint[0]), jnp.asarray(midpoint[1]), params)
    outflow = view.area[..., None]*np.asarray(_face_transport_divergence(*faces, params))
    transport_work = duration*math.fsum((view.pressure*outflow).ravel())
    kinetic_change = math.fsum((.5*mass*(after_velocity**2-before_velocity**2)).ravel())
    for value in (impulse_work, pressure_work, transport_work, kinetic_change):
        if not math.isfinite(value):
            raise ValueError("nonfinite native pressure work")
    return ReferencePressureKick(
        *(_immutable(value) for value in (before_velocity, after_velocity, before_momentum,
                                          after_momentum, force)),
        impulse_work, pressure_work, transport_work, kinetic_change)


@dataclass(frozen=True)
class OriginalDiagnosticReceipt:
    original_state: object
    original_valid: bool
    stages: tuple
    fast_calls: tuple
    convection_rhs_abs_max: float
    pressure_impulse_abs_max: float
    checks: object
    source_hashes: object
    parameter_digest: str
    accepted_joint_steps: int = 0


class OriginalNativeAudit:
    """Retain entry authority while observing actual original diagnostic stages."""
    def __init__(self, state, params, grid):
        self.view = capture_native(state, params, grid)
        self.params, self.grid = params, grid
        self._last_receipt = None

    @property
    def state(self):
        return self.view.state

    @property
    def accepted_joint_steps(self):
        return 0

    @property
    def last_receipt(self):
        return self._last_receipt

    def require_handoff(self, target):
        raise ValueError("incompatible native reference/first-slot authority for "+str(target))

    def diagnose(self, *, fast_observer=None):
        if self.params.use_scan:
            raise ValueError("native eager observation requires use_scan=False")
        rebound = capture_native(self.state, self.params, self.grid)
        if (rebound.parameter_digest != self.view.parameter_digest
                or dict(rebound.source_hashes) != dict(self.view.source_hashes)
                or state_digest(rebound.state) != state_digest(self.view.state)):
            raise ValueError("native state/parameter/source binding changed")
        stages, fast_calls = [], []

        def observe_stage(name, current, diagnostics):
            stages.append(MappingProxyType({"name": name, "state": _snapshot(current),
                                            "diagnostics": _freeze(diagnostics)}))

        def observe_fast(row):
            frozen = _freeze(row)
            fast_calls.append(frozen)
            if fast_observer is not None:
                fast_observer(frozen)

        original = JaxStateG(*(jnp.asarray(value) for value in self.state))
        result = material._material_step(
            original, self.params, "actual_geometry_v2", 16, "joint_heun_v1",
            stage_observer=observe_stage, fast_observer=observe_fast)
        if tuple(row["name"] for row in stages) != STAGES or len(fast_calls) != 12:
            raise ValueError("native original stage/clock count binding")
        faces = _layer_face_transports(original.u, original.v, self.params)
        vertical = material._vertical_transport_iface(original.u, original.v, self.params, face_transport=faces)
        convection = material._nonlinear_rhs(original, self.params, faces, vertical)[3]
        activity = float(np.max(abs(np.asarray(convection))))
        probe = reference_pressure_kick(self.view, self.params)
        receipt = OriginalDiagnosticReceipt(
            _snapshot(result.state), bool(result.valid), tuple(stages), tuple(fast_calls), activity,
            float(np.max(abs(probe.after_momentum-probe.before_momentum))),
            _freeze(result.checks), self.view.source_hashes, self.view.parameter_digest)
        self._last_receipt = receipt
        return receipt

