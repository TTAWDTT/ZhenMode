"""Predeclared rotating hydrostatic internal-wave holdout; selected numerics are immutable."""

from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import contract as wave_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.provenance.sources import load_json

DEVELOPMENT_SCHEMA = "vertical-tracer-development-v1"

SCHEMA = "frozen-internal-wave-v1"
SELECTION_SHA256 = "479ed6b1140dd3f33cb0057ecad0a275b83b34dad54ba3314c2dc97b4ad543dc"
# Arb sign-certified decimal bracket, rounded to binary64; q tan(q)=1/2500 on (pi,3pi/2).
Q = 3.1417199723837514
PROFILES = {"primary": (32, 16), "horizontal-coarse": (16, 16), "vertical-fine": (32, 32)}
MODELS = ("zhenmode", "mom6", "oceananigans")
MODEL_NAMES = dict(zip(MODELS, ("ocean-solver", "MOM6", "Oceananigans"), strict=True))


def selection():
    value = load_json(Path(__file__).with_name("frozen_selection.json"))
    if digest(value) != SELECTION_SHA256:
        raise ValueError("selected numerical configurations changed")
    return value


def contract(model, profile="primary", control="main"):
    if model not in MODELS or profile not in PROFILES or control not in ("main", "half", "null"):
        raise ValueError("unknown frozen internal-wave row")
    if control == "null" and profile != "primary":
        raise ValueError("null control is frozen on primary grid only")
    frozen = selection()
    chosen = frozen["models"][model]
    ny, nz = PROFILES[profile]
    h = np.full(nz, 100 / (nz - 1))
    h[[0, -1]] *= 0.5
    c = wave_contract()
    c.pop("period_s")
    c.update(schema=SCHEMA, benchmark="internal-wave", frozen_model=model,
             case=profile, control=control, nx=8, ny=ny, nz=nz,
             Lx_m=20000.0, Ly_m=20000.0, f=1e-4, dt=chosen["dt_s"],
             amplitude_m={"main": 0.01, "half": 0.005, "null": 0.0}[control],
             duration_s=6000.0 if control == "null" else 60000.0, output_s=1500.0,
             initial_h_m=h.tolist(),
             selection_sha256=SELECTION_SHA256, selected_index=chosen["selected_index"],
             selected_contract_sha256=chosen["old_contract_sha256"],
             selection_commit=frozen["report_commit"],
             internal_wave=dict(dT_dz_C_m=0.02, q=Q, N2_s_2=3.924e-5),
             reference=dict(kind="linear_rotating_hydrostatic_free_surface_first_internal_mode",
                            pressure_units="p_prime/rho0 in m2/s2",
                            branch="pi < q < 3*pi/2; q*tan(q)=1/2500",
                            provenance="derived hydrostatic rotating normal mode; linear reference, full nonlinear native trajectories",
                            upstream="https://ocw.mit.edu/ans7870/textbooks/Wunsch/Edited/Chapter10.pdf"))
    for name, key in (("zhenmode", "ocean_options"), ("mom6", "mom_time_options"),
                      ("oceananigans", "oceananigans_options")):
        c[key] = frozen["models"][name]["options"]
    scale = c["amplitude_m"] or 0.01
    c["thresholds"] = dict(error=0.1, T_error_C=0.1 * scale * 0.02,
                           S_error_psu=1e-10, volume=1e-10, tracer_inventory=1e-8,
                           null_absolute=5e-11)
    return c


def validate_contract(c):
    if not isinstance(c, dict) or c.get("schema") not in (SCHEMA, DEVELOPMENT_SCHEMA):
        raise ValueError("unknown internal-wave protocol")
    if c.get("schema") == DEVELOPMENT_SCHEMA:
        if not isinstance(c.get("development"), dict):
            raise ValueError("development declaration must be an object")
        expected = development_contract(c.get("case"), c.get("control"), c["development"].get("vertical_tracer_scheme"), c["development"].get("problem"), c["development"].get("hydrostatic_integral_scheme", "trapezoidal"), c["development"].get("column_mean_scheme", "nodal_capacity"), c["development"].get("tracer_flux_arithmetic", "legacy"), c["development"].get("momentum_vertical_velocity_scheme", "dual_face_v1"), c["development"].get("material_point_gradient_scheme", "minmod_v1"), c["development"].get("momentum_tangent_filter_scheme", "binomial_v1"), c["development"].get("tracer_transport_scheme", "centered_second"))
    else:
        expected = contract(c.get("frozen_model"), c.get("case"), c.get("control"))
    if digest(c) != digest(expected):
        raise ValueError("internal-wave contract differs from frozen definition")


def development_contract(profile="primary", control="main", scheme="muscl_nodal_v1", problem="internal-wave", integral="trapezoidal", mean="nodal_capacity", arithmetic="legacy", momentum="dual_face_v1", gradient="minmod_v1", tangent_filter="binomial_v1", horizontal="centered_second"):
    """New development experiment, never a replacement for the frozen holdout."""
    if scheme not in ("donor_cell", "muscl_nodal_v1", "muscl_endpoint_v2", "muscl_band_v3", "material_point_v4"):
        raise ValueError("unknown vertical development scheme")
    if integral not in ("trapezoidal", "cubic_uniform_v1"):
        raise ValueError("unknown pressure development integral")
    if mean not in ("nodal_capacity", "cubic_consistent_v1"):
        raise ValueError("unknown column development mean")
    if arithmetic not in ("legacy", "integrated_reference_v1"):
        raise ValueError("unknown tracer development arithmetic")
    if momentum not in ("dual_face_v1", "point_primitive_v1"):
        raise ValueError("unknown momentum vertical-velocity scheme")
    if gradient not in ("minmod_v1", "fourth_uniform_v1"):
        raise ValueError("unknown material-point gradient scheme")
    if gradient != "minmod_v1" and scheme != "material_point_v4":
        raise ValueError("fourth_uniform_v1 gradient requires material_point_v4")
    if tangent_filter not in ("binomial_v1", "gram_fourth_v1"):
        raise ValueError("unknown momentum tangent-filter scheme")
    if tangent_filter != "binomial_v1" and (scheme != "material_point_v4" or gradient != "fourth_uniform_v1"
            or integral != "cubic_uniform_v1" or mean != "nodal_capacity"
            or arithmetic != "integrated_reference_v1" or momentum != "dual_face_v1"):
        raise ValueError("gram_fourth_v1 requires the qualified material-point fourth-gradient dual-face family")
    if horizontal not in ("centered_second", "paired_product_fourth"):
        raise ValueError("unknown horizontal tracer development scheme")
    if horizontal != "centered_second" and (scheme != "material_point_v4"
            or gradient != "fourth_uniform_v1" or mean != "nodal_capacity"
            or arithmetic != "integrated_reference_v1" or momentum != "dual_face_v1"
            or tangent_filter != "binomial_v1"):
        raise ValueError("paired_product_fourth requires the integrated material-point fourth-gradient family with default momentum owners")
    if problem == "internal-wave":
        c = contract("zhenmode", profile, control)
    elif problem == "thermal-wind" and profile == "coarse" and control == "main":
        from zhenmode.benchmarks.configuration_scan import combination_contract

        c = combination_contract(9)
    else:
        raise ValueError("unknown vertical-development physical row")
    original = digest(c)
    if problem == "thermal-wind":
        c.update(frozen_model="zhenmode", control="main")
    c.update(schema=DEVELOPMENT_SCHEMA,
             development=dict(base_contract_sha256=original, vertical_tracer_scheme=scheme, problem=problem))
    c["ocean_options"]["vertical_tracer_scheme"] = scheme
    if integral != "trapezoidal":
        c["ocean_options"]["hydrostatic_integral_scheme"] = integral
        c["development"]["hydrostatic_integral_scheme"] = integral
    if mean != "nodal_capacity":
        c["ocean_options"]["column_mean_scheme"] = mean
        c["development"]["column_mean_scheme"] = mean
    if arithmetic != "legacy":
        c["ocean_options"]["tracer_flux_arithmetic"] = arithmetic
        c["development"]["tracer_flux_arithmetic"] = arithmetic
    if momentum != "dual_face_v1":
        c["ocean_options"]["momentum_vertical_velocity_scheme"] = momentum
        c["development"]["momentum_vertical_velocity_scheme"] = momentum
    if gradient != "minmod_v1":
        c["ocean_options"]["material_point_gradient_scheme"] = gradient
        c["development"]["material_point_gradient_scheme"] = gradient
    if tangent_filter != "binomial_v1":
        c["ocean_options"]["momentum_tangent_filter_scheme"] = tangent_filter
        c["development"]["momentum_tangent_filter_scheme"] = tangent_filter
    if horizontal != "centered_second":
        c["ocean_options"]["tracer_transport_scheme"] = horizontal
        c["development"]["tracer_transport_scheme"] = horizontal
    return c


def matrix():
    """21 rows; nulls first, then main/half at three declared physical resolutions."""
    return [contract(m, "primary", "null") for m in MODELS] + [
        contract(m, p, a) for p in PROFILES for a in ("main", "half") for m in MODELS
    ]


def initial_native(c, model):
    """Integrate analytic t=0 scalar/layer means; velocities retain native horizontal nodes."""
    validate_contract(c)
    if model != MODEL_NAMES[c["frozen_model"]]:
        raise ValueError("wrong model for frozen row")
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    k, m = np.pi / c["Ly_m"], c["internal_wave"]["q"] / c["H_m"]
    y = (np.arange(ny) + 0.5) * c["Ly_m"] / ny
    horizontal = np.broadcast_to(np.cos(k * y), (nx, ny))
    A, gamma = c["amplitude_m"], c["internal_wave"]["dT_dz_C_m"]
    factor = 1.0 if model == "ocean-solver" else np.sinc(k * c["Ly_m"] / ny / (2 * np.pi))
    eta = A * np.sin(m * c["H_m"]) * horizontal * factor
    if model == "ocean-solver":
        z = np.broadcast_to(-np.linspace(0, c["H_m"], nz), (nx, ny, nz))
        h = None
        vertical_factor = 1.0
    else:
        h = np.broadcast_to(c["initial_h_m"], (nx, ny, nz)).copy()
        if model == "Oceananigans":
            h *= 1 + eta[..., None] / c["H_m"]
        else:
            h[..., 0] += eta
        z = eta[..., None] - np.cumsum(h, axis=-1) + h / 2
        vertical_factor = np.sinc(m * h / (2 * np.pi))
    theta = -gamma * A * np.sin(m * (z + c["H_m"])) * vertical_factor * horizontal[..., None] * factor
    u = -A * c["f"] * m / k * np.cos(m * (z + c["H_m"])) * vertical_factor * np.sin(k * y)[None, :, None]
    result = dict(eta=eta, T=c["T_C"] + gamma * z + theta,
                  S=np.full(z.shape, c["S_psu"]), u=u, v=np.zeros(z.shape))
    if h is not None:
        result["h"] = h
    return result
