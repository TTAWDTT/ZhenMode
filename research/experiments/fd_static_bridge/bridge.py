"""Read-only nodal-dual column bridge, no production solver imports."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

RHO, G = 1025.0, 9.81
LOW = np.array([-5.0, 0.0])
HIGH = np.array([45.0, 50.0])


def reconstruct(z, n):
    h = -np.diff(z)
    c = n / h[:, None]
    x = 0.5 * (z[:-1] + z[1:])
    sec = np.diff(c, axis=0) / np.diff(x)[:, None]
    slope = np.empty_like(c)
    slope[0] = sec[0]
    slope[-1] = sec[-1]
    a, b = sec[:-1], sec[1:]
    slope[1:-1] = np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0)
    excursion = 0.5 * h[:, None] * abs(slope[:, :2])
    cap = np.maximum(0, np.minimum(c[:, :2] - LOW, HIGH - c[:, :2]))
    theta = np.ones_like(cap)
    np.divide(cap, excursion, out=theta, where=excursion > 0)
    slope[:, :2] *= np.minimum(1, theta)
    return c, slope, x


def integrate(z, n, lo, hi):
    c, m, x = reconstruct(z, n)
    out = np.zeros(4)
    for k in range(len(c)):
        a, b = max(lo, z[k + 1]), min(hi, z[k])
        if b > a:
            out += (b - a) * (c[k] + m[k] * (0.5 * (a + b) - x[k]))
    return out


def bridge(depth, h0, eta, values, target, Tref, Sref, alpha, beta, rho0, gravity):
    for value in (eta, Tref, Sref, alpha, beta, rho0, gravity):
        if np.asarray(value).shape != () or not np.isfinite(value):
            raise ValueError("finite scalar EOS/eta required")
    if min(alpha, beta, rho0, gravity) <= 0:
        raise ValueError("positive EOS constants required")
    depth, h0, values, target = map(
        lambda a: np.asarray(a, dtype=float), (depth, h0, values, target)
    )
    if (
        depth.ndim != 1
        or depth.size < 3
        or h0.shape != depth.shape
        or values.shape != (depth.size, 4)
    ):
        raise ValueError("shape")
    if (
        not all(np.isfinite(a).all() for a in [depth, h0, values, target])
        or not np.isfinite([eta, Tref, Sref]).all()
    ):
        raise ValueError("nonfinite")
    if depth[0] != 0 or np.any(np.diff(depth) <= 0) or eta > 0:
        raise ValueError("domain/extrapolation")
    z = np.r_[eta, -0.5 * (depth[:-1] + depth[1:]), -depth[-1]]
    expected = -np.diff(np.r_[0.0, z[1:]])
    if not np.allclose(h0, expected, rtol=0, atol=1e-12):
        raise ValueError("not nodal_dual_v1")
    h = -np.diff(z)
    if (
        np.any(h <= (8 * np.finfo(float).eps / 2) * (abs(z[:-1]) + abs(z[1:])))
        or target.ndim != 1
        or len(target) < 3
        or np.any(-np.diff(target) <= 0)
        or not np.array_equal(target[[0, -1]], z[[0, -1]])
    ):
        raise ValueError("nonpositive/different domain")
    if np.any(values[:, :2] < LOW) or np.any(values[:, :2] > HIGH):
        raise ValueError("tracer bounds")
    old = h[:, None] * values
    old[:, 2:] = h0[:, None] * values[:, 2:]
    new = np.array([integrate(z, old, target[k + 1], target[k]) for k in range(len(target) - 1)])
    means = new / (-np.diff(target))[:, None]
    tolerance = (
        1100 * np.finfo(float).eps / 2 * (abs(means[:, :2]) + np.maximum(abs(LOW), abs(HIGH)))
    )
    if np.any(means[:, :2] < LOW - tolerance) or np.any(means[:, :2] > HIGH + tolerance):
        raise ValueError("candidate tracer excess")
    # Independent interpolation inventory on same physical domain; never residual-filled.
    cuts = np.sort(np.r_[eta, -depth[depth > -eta]])
    geometry = np.array(
        [np.trapezoid(np.interp(cuts, -depth[::-1], values[::-1, f]), cuts) for f in range(4)]
    )
    scans = np.linspace(-depth[-1], eta, 401)
    rho = rho0 * (-alpha * (values[:, 0] - Tref) + beta * (values[:, 1] - Sref))
    original = (
        np.r_[0, np.cumsum(gravity * 0.5 * (rho[:-1] + rho[1:]) * np.diff(depth))]
        + rho0 * gravity * eta
    )
    pold = np.interp(-scans, depth, original)
    pnew = []
    for d in scans:
        stock = integrate(target, new, d, eta)
        length = eta - d
        pnew.append(
            rho0 * gravity * eta
            + rho0
            * gravity
            * (-alpha * (stock[0] - Tref * length) + beta * (stock[1] - Sref * length))
        )
    return new, dict(
        eos_identity=dict(
            alpha=float(alpha),
            beta=float(beta),
            rho0=float(rho0),
            gravity=float(gravity),
            Tref=float(Tref),
            Sref=float(Sref),
        ),
        qualification_passed=False,
        water_m=sum(h),
        original_material_T_S_reference_u_v=old.sum(axis=0).tolist(),
        candidate_inventory=new.sum(axis=0).tolist(),
        interpolation_geometry_inventory=geometry.tolist(),
        max_original_pressure_difference_Pa=float(abs(np.array(pnew) - pold).max()),
        momentum_units="rho0 times u/v stocks = kg m/s per unit area",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    snapshot = Path(args.input).read_bytes()
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as p:
        required = {
            "depth",
            "h0",
            "eta",
            "values",
            "target",
            "Tref",
            "Sref",
            "source_sha",
            "alpha",
            "beta",
            "rho0",
            "gravity",
        }
        if set(p.files) != required:
            raise ValueError("exact canonical keys required")
        if p["source_sha"].shape != () or p["source_sha"].dtype.kind != "U":
            raise ValueError("source_sha must be Unicode scalar")
        source_sha = str(p["source_sha"].item())
        if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
            raise ValueError("declared historical source_sha required")
        _, report = bridge(**{key: p[key] for key in required - {"source_sha"}})
        report["declared_source_sha"] = source_sha
        report["historical_source_independently_verified"] = False
    report["input_sha256"] = hashlib.sha256(snapshot).hexdigest()
    with open(args.output, "x") as f:
        json.dump(report, f, indent=2)
