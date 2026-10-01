"""Isolated unit-area two-column P1 mean-state ALE contract, no solver import."""

from dataclasses import dataclass

import numpy as np

RHO = 1025.0
G = 9.81
VERSION = 2
# Isolated declared physical contract; velocities have no imposed bound.
TRACER_LOWER = np.array([-5.0, 0.0])
TRACER_UPPER = np.array([45.0, 50.0])


@dataclass
class State:
    z: np.ndarray  # descending physical interfaces (2,4)
    n: np.ndarray  # extensive T,S,u,v (2,3,4)
    step: int = 0

    def copy(self):
        return State(self.z.copy(), self.n.copy(), self.step)


def geometry_valid(s):
    return (
        type(s.step) is int
        and s.step >= 0
        and s.z.shape == (2, 4)
        and s.n.shape == (2, 3, 4)
        and s.z.dtype == s.n.dtype == np.float64
        and np.isfinite(s.z).all()
        and np.isfinite(s.n).all()
        and np.all(s.z[:, -1] == -20.0)
        and np.all(-np.diff(s.z) > 0)
    )


def valid(s):
    if not geometry_valid(s):
        return False
    means = s.n[..., :2] / (-np.diff(s.z))[..., None]
    return bool(np.all(means >= TRACER_LOWER) and np.all(means <= TRACER_UPPER))


def target(eta):
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (2,) or not np.isfinite(eta).all() or np.any(eta <= -10):
        raise ValueError("top band exhausted or invalid eta")
    return np.stack((eta, eta - 0.25 * (eta + 10), np.full(2, -10.0), np.full(2, -20.0)), axis=1)


def reconstruction(s):
    """Mean-preserving limited physical-z P1; affine exact, not curvature exact."""
    if not valid(s):
        raise ValueError("invalid state")
    h = -np.diff(s.z)
    c = s.n / h[..., None]
    x = 0.5 * (s.z[:, :-1] + s.z[:, 1:])
    secant = np.diff(c, axis=1) / np.diff(x)[..., None]
    slope = np.empty_like(c)
    slope[:, 0] = secant[:, 0]
    slope[:, -1] = secant[:, -1]
    a, b = secant[:, 0], secant[:, 1]
    slope[:, 1] = np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0)
    # Symmetric endpoint excursions shrink, while the integral/mean is unchanged.
    # Both cell endpoints are inside the declared physical tracer interval.
    excursion = 0.5 * h[..., None] * abs(slope[..., :2])
    capacity = np.minimum(c[..., :2] - TRACER_LOWER, TRACER_UPPER - c[..., :2])
    theta = np.ones_like(capacity)
    np.divide(capacity, excursion, out=theta, where=excursion > 0)
    slope[..., :2] *= np.minimum(1.0, theta)
    return c, slope, x


def integral(s, col, cell, lo, hi):
    c, m, x = reconstruction(s)
    return (hi - lo) * (c[col, cell] + m[col, cell] * (0.5 * (hi + lo) - x[col, cell]))


def remap(s, z):
    out = State(np.asarray(z, dtype=float).copy(), np.zeros_like(s.n), s.step)
    if (
        not valid(s)
        or not geometry_valid(out)
        or not np.array_equal(z[:, [0, -1]], s.z[:, [0, -1]])
    ):
        raise ValueError("invalid target or different water domain")
    for j in range(2):
        for k in range(3):
            for src in range(3):
                lo, hi = max(z[j, k + 1], s.z[j, src + 1]), min(z[j, k], s.z[j, src])
                if hi > lo:
                    out.n[j, k] += integral(s, j, src, lo, hi)
    if not valid(out):
        raise ValueError("remap tracer means outside declared bounds")
    return out


def pressure(s, depth):
    """Same P1 T/S reconstruction as flux; common physical-depth evaluation."""
    if not valid(s) or np.ndim(depth) != 0 or not np.isfinite(depth):
        raise ValueError("invalid state or nonfinite scalar depth")
    if depth < -20 or np.any(depth > s.z[:, 0]):
        raise ValueError("depth outside common wet domain")
    p = RHO * G * s.z[:, 0].copy()
    for j in range(2):
        for k in range(3):
            lo, hi = max(depth, s.z[j, k + 1]), s.z[j, k]
            if hi > lo:
                content = integral(s, j, k, lo, hi)
                p[j] += (
                    RHO
                    * G
                    * (-2e-4 * (content[0] - 20 * (hi - lo)) + 8e-4 * (content[1] - 35 * (hi - lo)))
                )
    return p


def ledger(s):
    h = -np.diff(s.z)
    c, m, x = reconstruction(s)
    ke = 0.5 * RHO * np.sum(h[..., None] * c[..., 2:] ** 2)
    reconstructed_ke = ke + RHO / 24 * np.sum(h[..., None] ** 3 * m[..., 2:] ** 2)
    # Exact physical height integral of reconstructed density anomaly.
    density = RHO * (-2e-4 * (c[..., 0] - 20) + 8e-4 * (c[..., 1] - 35))
    gradient = RHO * (-2e-4 * m[..., 0] + 8e-4 * m[..., 1])
    pe = G * np.sum(h * x * density + h**3 / 12 * gradient)
    return {
        "water_m3": float(h.sum()),
        "stocks": s.n.sum(axis=(0, 1)),
        "mean_ke_J": float(ke),
        "reconstructed_ke_J": float(reconstructed_ke),
        "anomaly_pe_J": float(pe),
        "T_S_second_moment": np.sum(
            h[..., None] * c[..., :2] ** 2 + h[..., None] ** 3 / 12 * m[..., :2] ** 2, axis=(0, 1)
        ),
    }


def advance(s, segments, eta, sources, dt=1.0):
    """Segments=(lo,hi,q) physical wet face bands, q m3/s, donor P1 average.

    Volume and material use EXACTLY the same overlap partition. Fixed shape;
    positive intermediate layer volumes required before conservative remap.
    """
    before = s.copy()
    try:
        if not valid(s) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid input")
        tz = target(eta)
        source = np.asarray(sources, dtype=float)
        if source.shape != s.n.shape or not np.isfinite(source).all():
            raise ValueError("invalid source")
        dh, dn, outgoing = np.zeros((2, 3)), np.zeros_like(s.n), np.zeros((2, 3))
        intervals = []
        for lo, hi, q in segments:
            if not np.isfinite([lo, hi, q]).all() or hi <= lo or lo < -20 or hi > min(s.z[:, 0]):
                raise ValueError("face outside common wet overlap")
            if any(max(lo, a) < min(hi, b) for a, b in intervals):
                raise ValueError("overlapping prescribed segments")
            intervals.append((lo, hi))
            cuts = sorted({lo, hi, *[float(v) for v in s.z.ravel() if lo < v < hi]})
            for a, b in zip(cuts[:-1], cuts[1:]):
                mid = 0.5 * (a + b)
                cells = [
                    int(np.flatnonzero((s.z[j, :-1] >= mid) & (s.z[j, 1:] <= mid))[0])
                    for j in range(2)
                ]
                amount = dt * q * (b - a) / (hi - lo)
                donor = 0 if q >= 0 else 1
                value = integral(s, donor, cells[donor], a, b) / (b - a)
                for j, sign in ((0, -1), (1, 1)):
                    dh[j, cells[j]] += sign * amount
                    dn[j, cells[j]] += sign * amount * value
                outgoing[donor, cells[donor]] += abs(amount)
        h = -np.diff(s.z)
        if np.any(outgoing > 0.5 * h):
            raise ValueError("outflow CFL")
        nh = h + dh
        if np.any(nh <= 0):
            raise ValueError("nonpositive intermediate")
        nz = np.concatenate(
            (np.asarray(eta)[:, None], np.asarray(eta)[:, None] - np.cumsum(nh, axis=1)), axis=1
        )
        if np.max(abs(nz[:, -1] + 20)) > 1e-12:
            raise ValueError("eta not consistent with shared volume flux")
        nz[:, -1] = -20.0  # enforce exact fixed bed only after roundoff consistency check
        intermediate = State(nz, s.n + dn + dt * source, s.step + 1)
        if not valid(intermediate):
            raise ValueError("intermediate tracer means outside declared bounds")
        out = remap(intermediate, tz)
        return (
            out,
            True,
            {
                "transport_source_ke_change_J": ledger(intermediate)["mean_ke_J"]
                - ledger(s)["mean_ke_J"],
                "remap_mean_ke_change_J": ledger(out)["mean_ke_J"]
                - ledger(intermediate)["mean_ke_J"],
                "remap_PE_change_J": ledger(out)["anomaly_pe_J"]
                - ledger(intermediate)["anomaly_pe_J"],
                "remap_T2_change_m3_K2": float(
                    ledger(out)["T_S_second_moment"][0]
                    - ledger(intermediate)["T_S_second_moment"][0]
                ),
                "remap_S2_change_m3_psu2": float(
                    ledger(out)["T_S_second_moment"][1]
                    - ledger(intermediate)["T_S_second_moment"][1]
                ),
            },
        )
    except (ValueError, IndexError, TypeError) as error:
        return before, False, {"rejection_reason": str(error)}


def fd_to_means(nodes, values, z):
    """Specified piecewise-linear FD interpolation; reject extrapolation.

    Mapping is conservative integration of interpolant, NOT invertible generally.
    """
    nodes, values = np.asarray(nodes), np.asarray(values)
    if nodes.ndim != 1 or values.shape != (nodes.size, 4) or np.any(np.diff(nodes) <= 0):
        raise ValueError("invalid point profile")
    if z.min() < nodes[0] or z.max() > nodes[-1]:
        raise ValueError("unsupported extrapolation")
    result = np.zeros((3, 4))
    for k in range(3):
        lo, hi = z[k + 1], z[k]
        cuts = np.array(sorted({lo, hi, *nodes[(nodes > lo) & (nodes < hi)]}))
        for f in range(4):
            result[k, f] = np.trapezoid(np.interp(cuts, nodes, values[:, f]), cuts)
    return result


def save(s, path):
    if not valid(s) or not np.array_equal(s.z, target(s.z[:, 0])):
        raise ValueError("completed target required")
    np.savez(path, z=s.z, n=s.n, step=np.array(s.step), version=np.array(VERSION))


def load(path):
    with np.load(path, allow_pickle=False) as p:
        if set(p.files) != {"z", "n", "step", "version"}:
            raise ValueError("checkpoint keys")
        for key in ("step", "version"):
            if p[key].shape != () or p[key].dtype.kind not in "iu":
                raise ValueError("integer metadata required")
        if p["version"].item() != VERSION:
            raise ValueError("version mismatch")
        s = State(p["z"].copy(), p["n"].copy(), int(p["step"]))
    if not valid(s) or not np.array_equal(s.z, target(s.z[:, 0])):
        raise ValueError("checkpoint geometry invalid")
    return s
