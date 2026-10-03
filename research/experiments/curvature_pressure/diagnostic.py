"""Pre-frozen smooth polynomial accuracy diagnostic, no production caller."""

import argparse
import json
from pathlib import Path

import numpy as np

RHO, G, ALPHA = 1025.0, 9.81, 2e-4
PROFILES = {"quadratic": [20.0, 0.1, 0.001], "cubic": [20.0, 0.1, 0.0, 0.00001]}


def grid(n, shift=False):
    z = np.linspace(0.0, -20.0, n + 1)
    if shift:
        z[1:-1:2] += 0.2 * 20 / n
    return z


def means(z, coef):
    primitive = np.polynomial.polynomial.polyint(coef)
    return (
        np.polynomial.polynomial.polyval(z[:-1], primitive)
        - np.polynomial.polynomial.polyval(z[1:], primitive)
    ) / (-np.diff(z))


def reconstruct(z, c):
    h = -np.diff(z)
    x = 0.5 * (z[:-1] + z[1:])
    sec = np.diff(c) / np.diff(x)
    m = np.empty_like(c)
    m[0] = sec[0]
    m[-1] = sec[-1]
    a, b = sec[:-1], sec[1:]
    m[1:-1] = np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0)
    excursion = 0.5 * h * abs(m)
    capacity = np.maximum(0, np.minimum(c + 5, 45 - c))
    theta = np.ones_like(c)
    np.divide(capacity, excursion, out=theta, where=excursion > 0)
    m *= np.minimum(1, theta)
    return m


def pressure_bound(z, coef, depths):
    h = -np.diff(z)
    x = 0.5 * (z[:-1] + z[1:])
    c = means(z, coef)
    m = reconstruct(z, c)
    second = np.polynomial.polynomial.polyder(coef, 2)
    curvature = max(abs(np.polynomial.polynomial.polyval([-20.0, 0.0], second)))
    distance = abs(np.diff(x))
    secerr = curvature * (distance / 2 + (h[:-1] ** 2 + h[1:] ** 2) / (24 * distance))
    slopeerr = np.r_[secerr[0], np.maximum(secerr[:-1], secerr[1:]), secerr[-1]]
    error = curvature * h**2 / 6 + slopeerr * h / 2
    lo = np.maximum(depths[:, None], z[None, 1:])
    length = np.maximum(0, z[None, :-1] - lo)
    mid = 0.5 * (z[None, :-1] + lo)
    content = length * (c - 20 + m * (mid - x))
    u = np.finfo(float).eps / 2
    gamma = 256 * u / (1 - 256 * u)
    rounding = (
        gamma
        * RHO
        * G
        * ALPHA
        * np.sum(length * (abs(c) + 20 + abs(m) * (abs(mid) + abs(x))), axis=1)
    )
    return -RHO * G * ALPHA * content.sum(axis=1), RHO * G * ALPHA * (length * error).sum(
        axis=1
    ) + rounding


def run():
    depths = np.linspace(-20.0, 0.0, 401)
    rows = []
    for name, coef in PROFILES.items():
        anomaly = np.array(coef)
        anomaly[0] -= 20
        p = np.polynomial.polynomial.polyint(anomaly)
        exact = (
            -RHO
            * G
            * ALPHA
            * (
                np.polynomial.polynomial.polyval(0.0, p)
                - np.polynomial.polynomial.polyval(depths, p)
            )
        )
        previous = None
        for n in [8, 16, 32, 64, 128]:
            pairs = [pressure_bound(grid(n, shift), coef, depths) for shift in [False, True]]
            errors = [float(abs(value - exact).max()) for value, _ in pairs]
            assert all(np.all(abs(value - exact) <= bound) for value, bound in pairs)
            force = -(pairs[1][0] - pairs[0][0]) / (RHO * 1000)
            assert np.all(abs(force) <= (pairs[0][1] + pairs[1][1]) / (RHO * 1000))
            ratios = None if previous is None else [a / b for a, b in zip(previous, errors)]
            if n in [64, 128]:
                assert min(ratios) >= 3
            rows.append(
                dict(
                    profile=name,
                    layers=n,
                    max_pressure_error_Pa=errors,
                    refinement_ratio=ratios,
                    max_pseudo_acceleration_m_s2=float(abs(force).max()),
                    max_envelope_Pa=[float(bound.max()) for _, bound in pairs],
                )
            )
            previous = errors
    return {"rows": rows, "qualification_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write a new local report instead of source-tree evidence")
    args = parser.parse_args()
    result = run()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
