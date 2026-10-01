"""Static local top bridge; original deep arrays are carried byte-for-byte."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "fd_static_bridge"))
from bridge import integrate  # noqa: E402
from discrete_audit import audit  # noqa: E402

B = -22.5


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)
    elif not isinstance(value, (str, type(None))):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite derived stage")


@np.errstate(over="raise", invalid="raise", divide="raise")
def column(p):
    """Return new top arrays and unchanged deep records, or untouched snapshot."""
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in p.items()}
    try:
        require_finite(audit(**p))
        d, w, wet, v = map(
            np.asarray, [p["depth"], p["reference_weights"], p["wet_mask"], p["values"]]
        )
        eta = float(p["eta"])
        rho = float(p["rho0"])
        g = float(p["gravity"])
        if (
            len(d) < 4
            or not np.all(wet[:3] == 1)
            or not np.array_equal(w[:3], [2.5, 7.5, 12.5])
            or not np.array_equal(d[:4], [0.0, 5.0, 15.0, 30.0])
            or eta > 0
        ):
            raise ValueError("unsupported original top band/coverage")
        z = np.array([eta, -2.5, -10.0, B])
        h = -np.diff(z)
        u = np.finfo(float).eps / 2
        if np.any(h <= 8 * u * (abs(z[:-1]) + abs(z[1:]))):
            raise ValueError("nonpositive or unresolved original geometry")
        n = h[:, None] * v[:3]
        n[:, 2:] = rho * w[:3, None] * v[:3, 2:]
        require_finite(n)
        target = np.r_[eta, eta - (eta - B) * np.cumsum([1 / 9, 1 / 3, 5 / 9])]
        target[-1] = B
        new = np.array([integrate(z, n, target[k + 1], target[k]) for k in range(3)])
        require_finite(new)
        gamma = 512 * u / (1 - 512 * u)
        bound = gamma * (abs(n).sum(axis=0) + abs(new).sum(axis=0))
        residual = new.sum(axis=0) - n.sum(axis=0)
        require_finite([bound, residual])
        if np.any(abs(residual) > bound):
            raise ValueError("inventory roundoff budget exceeded")
        density = (
            rho * (-p["alpha"] * (v[:, 0] - p["Tref"]) + p["beta"] * (v[:, 1] - p["Sref"])) * wet
        )
        original = (
            np.r_[0, np.cumsum(g * 0.5 * (density[:-1] + density[1:]) * np.diff(d))] + rho * g * eta
        )
        require_finite([density, original])
        oldbase = float(np.interp(-B, d, original))

        def top_pressure(depth):
            q = integrate(target, new, depth, eta)
            length = eta - depth
            require_finite(q)
            value = rho * g * eta + rho * g * (
                -p["alpha"] * (q[0] - p["Tref"] * length) + p["beta"] * (q[1] - p["Sref"] * length)
            )

            require_finite(value)
            return value

        newbase = top_pressure(B)
        ht = -np.diff(target)
        deep = w[3:] * wet[3:]
        deepstock = deep[:, None] * v[3:]
        deepstock[:, 2:] *= rho
        record = dict(
            depth=d[3:].copy(),
            values=v[3:].copy(),
            reference_weights=w[3:].copy(),
            wet_mask=wet[3:].copy(),
            inventory=deepstock.copy(),
            pressure_increment=(original[3:] - oldbase).copy(),
        )
        scan = np.linspace(B, eta, 101)
        difference = [top_pressure(x) - np.interp(-x, d, original) for x in scan]
        report = dict(
            band_water_residual_m=float((-np.diff(target)).sum() - h.sum()),
            qualification_passed=False,
            band_water_m=float(h.sum()),
            inventory_residual=residual.tolist(),
            inventory_roundoff_bound=bound.tolist(),
            band_bottom_delta_pressure_Pa=float(newbase - oldbase),
            top_pressure_difference_max_Pa=float(max(abs(np.array(difference)))),
            original_reference_mass_K_J_per_m2=float(
                0.5 * rho * np.sum(w[:3, None] * v[:3, 2:] ** 2)
            ),
            reference_momentum_on_original_actual_mass_K_J_per_m2=float(
                0.5 * np.sum(n[:, 2:] ** 2 / (rho * h[:, None]))
            ),
            candidate_actual_mass_K_J_per_m2=float(
                0.5 * np.sum(new[:, 2:] ** 2 / (rho * ht[:, None]))
            ),
            original_velocity_actual_mass_K_J_per_m2=float(
                0.5 * rho * np.sum(h[:, None] * v[:3, 2:] ** 2)
            ),
            inventory_units=["m K", "m psu", "kg/(m s)", "kg/(m s)"],
            accepted_scope="local_top_band_only; full geometry requires external binding",
            original_reference_to_actual_velocity_factor_max=float(max(w[:3] / h)),
            candidate_velocity_max_m_s=float(abs(new[:, 2:] / (rho * ht[:, None])).max()),
            deep_byte_preserved=True,
            eos_identity={
                k: float(p[k]) for k in ["alpha", "beta", "rho0", "gravity", "Tref", "Sref"]
            },
        )
        require_finite([record, report, difference])
        return (
            dict(
                eos={k: float(p[k]) for k in ["alpha", "beta", "rho0", "gravity", "Tref", "Sref"]},
                z=target,
                n=new,
                deep=record,
                original_depth=d.copy(),
                original_pressure=original.copy(),
                original_base=oldbase,
                new_base=newbase,
                eta=eta,
                wet_bottom_node=float(d[np.flatnonzero(wet)[-1]]),
            ),
            True,
            report,
        )
    except (ValueError, TypeError, IndexError, FloatingPointError) as error:
        return before, False, {"rejection_reason": str(error), "qualification_passed": False}


@np.errstate(over="raise", invalid="raise", divide="raise")
def pressure(result, depth):
    require_finite(result)
    if (
        np.ndim(depth) != 0
        or not np.isfinite(depth)
        or depth > result["eta"]
        or depth < -result["wet_bottom_node"]
    ):
        raise ValueError("outside defined domain")
    if depth < B:
        return (
            result["new_base"]
            + np.interp(-depth, result["original_depth"], result["original_pressure"])
            - result["original_base"]
        )
    rho = result["eos"]["rho0"]
    g = result["eos"]["gravity"]
    q = integrate(result["z"], result["n"], depth, result["eta"])
    length = result["eta"] - depth
    return rho * g * result["eta"] + rho * g * (
        -result["eos"]["alpha"] * (q[0] - result["eos"]["Tref"] * length)
        + result["eos"]["beta"] * (q[1] - result["eos"]["Sref"] * length)
    )


@np.errstate(over="raise", invalid="raise", divide="raise")
def pair(a, b, distance):
    require_finite([a, b])
    if a["eos"] != b["eos"]:
        raise ValueError("pair EOS identities differ")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("positive distance required")
    bottom = max(-400.0, -a["wet_bottom_node"], -b["wet_bottom_node"])
    top = min(a["eta"], b["eta"])
    depths = np.linspace(bottom, top, 201)
    change = (a["new_base"] - a["original_base"]) - (b["new_base"] - b["original_base"])
    report = dict(
        common_domain_m=[float(bottom), float(top)],
        bottom_delta_difference_Pa=float(change),
        deep_gradient_acceleration_change_m_s2=float(-change / (a["eos"]["rho0"] * distance)),
        common_pressure_difference_max_Pa=float(
            max(abs(pressure(a, x) - pressure(b, x)) for x in depths)
        ),
    )

    require_finite(report)
    return report
