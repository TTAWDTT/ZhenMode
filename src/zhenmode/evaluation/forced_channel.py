"""Forced trajectory observables and independently assembled water/salt/heat/work budgets."""

import numpy as np

from zhenmode.benchmarks.forced_channel import validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.channel_dynamics import validate as validate_initial


def score(c, a):
    validate_initial(c, a, definition_validator=validate_contract)
    volume = a["h"] * a["area"][None, :, None]
    water = volume.sum(axis=(1, 2))
    heat = (volume * a["T"]).sum(axis=(1, 2))
    salt = (volume * a["S"]).sum(axis=(1, 2))
    ke = 0.5 * c["rho0"] * sum((a["volume_" + q] * a[q] ** 2).sum(axis=(1, 2)) for q in ("u", "v"))
    pe = 0.5 * c["rho0"] * c["gravity"] * (a["eta"] ** 2 * a["area"]).sum(axis=1)
    power = np.zeros_like(a["time"])
    mean_square = {}

    def time_mean(values):
        return float(
            np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(a["time"]))
            / (a["time"][-1] - a["time"][0])
        )

    for q in ("u", "v"):
        # v wall dual cells carry half area; velocity and hence work there are zero.
        if a["metadata"]["velocity_layout"] == "cgrid" and q == "v":
            area = np.full((c["nx"], c["ny"] + 1), c["Lx_m"] / c["nx"] * c["Ly_m"] / c["ny"])
            area[:, [0, -1]] *= 0.5
            area = area.ravel()
        else:
            area = a["area"]
        mean = (a["volume_" + q] * a[q]).sum(axis=-1) / a["volume_" + q].sum(axis=-1)
        mean_square[q] = time_mean(np.sum(mean**2 * area, axis=1) / area.sum())
        stress_component = "tau_x_N_m2" if q == "u" else "tau_y_N_m2"
        power += c["wind_stress"][stress_component] * np.sum(a[q][:, :, 0] * area, axis=1)
    work = np.zeros_like(power)
    work[1:] = np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(a["time"]))
    energy = ke + pe
    denominator = max(float(energy[0]), float(np.max(abs(work))), 1.0)
    metrics = dict(
        eta_rms_m=float(
            np.sqrt(time_mean(np.sum(a["eta"] ** 2 * a["area"], axis=1) / a["area"].sum()))
        ),
        depth_mean_u_rms_m_s=float(np.sqrt(mean_square["u"])),
        depth_mean_v_rms_m_s=float(np.sqrt(mean_square["v"])),
        volume_drift=float(np.max(abs(water / water[0] - 1))),
        T_inventory_drift=float(np.max(abs(heat / heat[0] - 1))),
        S_inventory_drift=float(np.max(abs(salt / salt[0] - 1))),
        work_residual_relative=float(np.max(abs(energy - energy[0] - work))) / denominator,
    )
    limits = {
        "volume_drift": c["thresholds"]["volume"],
        "T_inventory_drift": c["thresholds"]["tracer_inventory"],
        "S_inventory_drift": c["thresholds"]["tracer_inventory"],
    }
    failures = [key for key, limit in limits.items() if metrics[key] > limit]
    m = a["metadata"]
    return dict(
        schema=c["schema"],
        contract_sha256=digest(c),
        model=m["model"],
        source_sha=m["source_sha"],
        executable_sha256=m["executable_sha256"],
        metrics=metrics,
        failed_metrics=failures,
        engineering_screen_pass=not failures,
        industrial_qualified=False,
        performance_comparison=False,
        accuracy_ranking=False,
        data_kind=c["data_kind"],
        trajectories=dict(
            time_s=a["time"].tolist(),
            kinetic_J=ke.tolist(),
            surface_potential_J=pe.tolist(),
            wind_power_W=power.tolist(),
            wind_work_J=work.tolist(),
        ),
        limitations=[
            "RMS values describe trajectories; no observed/analytic trajectory error reference.",
            "Work residual includes numerical dynamics, prescribed surface representation and snapshot quadrature error.",
            "Real source weather is sampled at one point; stress is frozen to its six-hour mean, SST=15 C/current=0 for derivation.",
        ],
    )
