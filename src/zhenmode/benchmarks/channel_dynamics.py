"""Frozen rotating adjustment and stratified thermal-wind benchmark definitions."""

import numpy as np

from zhenmode.benchmarks.standing_wave import contract as wave_contract
from zhenmode.benchmarks.standing_wave import digest

SCHEMA = "channel-dynamics-v1"
CASES = ("geostrophic-adjustment", "thermal-wind")


def contract(case, level="coarse", method="baseline"):
    if case not in CASES or level not in ("coarse", "medium", "fine"):
        raise ValueError("unknown frozen channel benchmark")
    if case == "thermal-wind" and level != "coarse":
        raise ValueError("thermal-wind currently freezes one measured-size grid")
    if method not in ("baseline", "symmetric-external-mode", "symmetric-closed-faces"):
        raise ValueError("unknown channel method")
    c = wave_contract()
    c.update(schema=SCHEMA, benchmark=case, case=level, dt=50.0, f=1e-4)
    c["oceananigans_options"]["coriolis"] = "FPlane"
    for name in ("DT", "DTBT", "DT_THERM", "DT_FORCING"):
        c["mom_time_options"][name] = 50.0
    c.pop("period_s")
    c["duration_s"] = 32000.0
    if case == "geostrophic-adjustment":
        c.update(
            nx=8,
            ny={"coarse": 64, "medium": 128, "fine": 256}[level],
            Lx_m=100000.0,
            Ly_m=float(32000 * np.sqrt(981.0) / np.sqrt(3)),
            f=float(np.pi / 32000),
            oscillation_period_s=32000.0,
        )
        c["reference"] = dict(
            kind="linear_rotating_shallow_water_cosine",
            rotating_fraction=0.25,
            provenance="classic geostrophic-adjustment equation family; sinusoidal closed-channel variant, not a reproduction of MOM6 frontal IC",
            upstream="https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/user/adjustment_initialization.F90",
        )
    else:
        nz = 16
        h = np.full(nz, 100 / (nz - 1))
        h[[0, -1]] *= 0.5
        c.update(
            nx=32,
            ny=32,
            nz=nz,
            Lx_m=100000.0,
            Ly_m=100000.0,
            duration_s=86400.0,
            output_s=2700.0,
            initial_h_m=h.tolist(),
            thermal_wind=dict(dT_dz_C_m=0.02, delta_T_C=0.5, front_width_m=25000.0, spice_T_C=0.1),
        )
        c["reference"] = dict(
            kind="steady_thermal_wind_with_density_neutral_advected_spice",
            provenance="MOM6 baroclinic-zone sinusoidal front/linear stratification, rotated into closed y; add bottom-pressure-balanced SSH/velocity and density-neutral T/S wave",
            upstream="https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/user/baroclinic_zone_initialization.F90",
        )
    c["thresholds"] = dict(
        error={"coarse": 0.1, "medium": 0.05, "fine": 0.02}[level],
        volume=1e-10,
        tracer_inventory=1e-8,
    )
    c["thresholds"].update(T_error_C=1e-10, S_error_psu=1e-10)
    if case == "thermal-wind":
        c["thresholds"]["T_error_C"] = c["thresholds"]["error"] * c["thermal_wind"]["spice_T_C"]
        c["thresholds"]["S_error_psu"] = (
            c["thresholds"]["T_error_C"] * (-c["eos"]["drho_dT"]) / c["eos"]["drho_dS"]
        )
    if method != "baseline":
        c["schema"] = "channel-dynamics-method-v1"
        c["method"] = method
        c["ocean_options"]["external_mode_scheme"] = "symmetric"
    if method == "symmetric-closed-faces":
        c["ocean_options"]["meridional_boundary_scheme"] = "closed_faces"
    return c


def validate_contract(c):
    if isinstance(c, dict) and c.get("schema") == "real-wind-channel-v1":
        from zhenmode.benchmarks.forced_channel import validate_contract as validate_forced

        return validate_forced(c)
    if not isinstance(c, dict) or c.get("schema") not in (SCHEMA, "channel-dynamics-method-v1"):
        raise ValueError("unknown channel protocol")
    expected = contract(c.get("benchmark"), c.get("case"), c.get("method", "baseline"))
    if digest(c) != digest(expected):
        raise ValueError("channel configuration differs from frozen definition")


def front(c, y):
    """Physical initial front, independent of the evaluator's implementation."""
    o = c["thermal_wind"]
    position = (np.asarray(y) - c["Ly_m"] / 2) / o["front_width_m"]
    clipped = np.clip(position, -1, 1)
    anomaly = o["delta_T_C"] * np.sin(np.pi * clipped / 2)
    gradient = np.where(
        np.abs(position) < 1,
        o["delta_T_C"] * np.pi / (2 * o["front_width_m"]) * np.cos(np.pi * position / 2),
        0,
    )
    return anomaly, gradient


def surface(c, y):
    if c["benchmark"] == "geostrophic-adjustment":
        return c["amplitude_m"] * np.cos(np.pi * np.asarray(y) / c["Ly_m"])
    anomaly, _ = front(c, y)
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    q = 1 - alpha * anomaly
    root = np.sqrt(q * q - 2 * alpha * alpha * c["thermal_wind"]["dT_dz_C_m"] * c["H_m"] * anomaly)
    # Small quadratic root holds bottom hydrostatic pressure constant, including SSH terms.
    return 2 * alpha * c["H_m"] * anomaly / (q + root)


def initial_points(c, x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    if c["benchmark"] == "geostrophic-adjustment":
        return dict(
            T=np.full_like(x, c["T_C"]),
            S=np.full_like(x, c["S_psu"]),
            u=np.zeros_like(x),
            v=np.zeros_like(x),
        )
    o = c["thermal_wind"]
    delta, gradient = front(c, y)
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    beta = c["eos"]["drho_dS"] / c["rho0"]
    spice = o["spice_T_C"] * np.cos(2 * np.pi * x / c["Lx_m"])
    return dict(
        T=c["T_C"] + o["dT_dz_C_m"] * z + delta + spice,
        S=c["S_psu"] + alpha / beta * spice,
        u=-c["gravity"] * alpha * gradient * (z + c["H_m"]) / c["f"],
        v=np.zeros_like(x),
    )


def initial_native(c, model):
    """Point-valued FD or native C-grid layer initial data; no model or oracle calls."""
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    if model == "ocean-solver":
        z = np.broadcast_to(-np.linspace(0, c["H_m"], nz), (nx, ny, nz))
        return dict(eta=surface(c, y), **initial_points(c, x[..., None], y[..., None], z))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    yq = y[..., None] + dy * nodes / 2
    eta = np.sum(surface(c, yq) * weights / 2, axis=-1)
    h = np.broadcast_to(c["initial_h_m"], (nx, ny, nz)).copy()
    if model == "Oceananigans":
        h *= 1 + eta[..., None] / c["H_m"]
    else:
        h[..., 0] += eta
    z = eta[..., None] - np.cumsum(h, axis=-1) + h / 2
    result = initial_points(c, x[..., None], y[..., None], z)
    # Scalars are cell means in x/y and layer means in z. Linear z integrates exactly.
    if c["benchmark"] == "thermal-wind":
        delta, _ = front(c, yq)
        delta = np.sum(delta * weights / 2, axis=-1)
        o = c["thermal_wind"]
        alpha = -c["eos"]["drho_dT"] / c["rho0"]
        beta = c["eos"]["drho_dS"] / c["rho0"]
        spice = o["spice_T_C"] * np.cos(2 * np.pi * x / c["Lx_m"]) * np.sinc(1 / nx)
        result["T"] = c["T_C"] + o["dT_dz_C_m"] * z + delta[..., None] + spice[..., None]
        result["S"] = np.broadcast_to(c["S_psu"] + alpha / beta * spice[..., None], h.shape).copy()
    result.update(eta=eta, h=h)
    return result
