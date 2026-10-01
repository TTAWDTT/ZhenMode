"""Two-column flat mixed-state dynamic step; no old solver/driver reuse."""

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "fd_static_bridge"))
from bridge import integrate, reconstruct  # noqa: E402

B = -22.5
RHO = 1025.0
G = 9.81


@dataclass
class State:
    h: np.ndarray
    n: np.ndarray  # IT,IS,Mu,Mv per horizontal area
    deep_nodes: np.ndarray
    dx: float
    source_sha: str
    step: int = 0

    def copy(self):
        return copy.deepcopy(self)

    @property
    def eta(self):
        return B + self.h[:, :3].sum(axis=1)

    @property
    def top_z(self):
        return np.c_[self.eta, self.eta[:, None] - np.cumsum(self.h[:, :3], axis=1)]


def validate(s):
    if s.h.ndim != 2 or s.h.shape[0] != 2 or s.h.shape[1] < 4 or s.n.shape != s.h.shape + (4,):
        raise ValueError("shape")
    if (
        s.h.dtype != np.float64
        or s.n.dtype != np.float64
        or s.deep_nodes.shape != (s.h.shape[1] - 3,)
    ):
        raise ValueError("dtype/deep")
    if (
        not all(np.isfinite(a).all() for a in [s.h, s.n, s.deep_nodes, s.eta])
        or np.any(s.h <= 0)
        or not np.isfinite(s.dx)
        or s.dx <= 0
    ):
        raise ValueError("finite positive geometry")
    if (
        not np.array_equal(s.h[0, 3:], s.h[1, 3:])
        or s.deep_nodes[0] <= -B
        or np.any(np.diff(s.deep_nodes) <= 0)
    ):
        raise ValueError("flat shared deep geometry")
    if (
        not isinstance(s.source_sha, str)
        or len(s.source_sha) != 40
        or any(c not in "0123456789abcdef" for c in s.source_sha)
    ):
        raise ValueError("source identity")
    if type(s.step) is not int or s.step < 0:
        raise ValueError("step identity")
    mean = s.n[:, :, :2] / s.h[:, :, None]
    u = np.finfo(float).eps / 2
    tol = 1100 * u * (abs(mean) + [45, 50])
    if np.any(mean < np.array([-5, 0]) - tol) or np.any(mean > np.array([45, 50]) + tol):
        raise ValueError("tracer bounds")


def velocity(s):
    return s.n[:, :, 2:] / (RHO * s.h[:, :, None])


def ke(s):
    return float(0.5 * RHO * np.sum(s.h[:, :, None] * velocity(s) ** 2))


def top_pressure(s, col, depth):
    eta = s.eta[col]
    q = integrate(s.top_z[col], s.n[col, :3], depth, eta)
    length = eta - depth
    return RHO * G * eta + RHO * G * (-2e-4 * (q[0] - 15 * length) + 7.6e-4 * (q[1] - 35 * length))


def pressures(s):
    out = np.zeros_like(s.h)
    for j in range(2):
        z = s.top_z[j]
        n = s.n[j, :3]
        eta = s.eta[j]

        def p(depth):
            q = integrate(z, n, depth, eta)
            length = eta - depth
            return RHO * G * eta + RHO * G * (
                -2e-4 * (q[0] - 15 * length) + 7.6e-4 * (q[1] - 35 * length)
            )

        out[j, :3] = [p(0.5 * (z[k] + z[k + 1])) for k in range(3)]
        c, m, x = reconstruct(z, n)
        temp = c[-1, 0] + m[-1, 0] * (B - x[-1])
        salt = c[-1, 1] + m[-1, 1] * (B - x[-1])
        rho_b = RHO * (-2e-4 * (temp - 15) + 7.6e-4 * (salt - 35))
        last = -B
        current = p(B)
        for k, d in enumerate(s.deep_nodes):
            cdeep = s.n[j, k + 3, :2] / s.h[j, k + 3]
            rho = RHO * (-2e-4 * (cdeep[0] - 15) + 7.6e-4 * (cdeep[1] - 35))
            current += G * 0.5 * (rho_b + rho) * (d - last)
            out[j, k + 3] = current
            rho_b = rho
            last = d
    if not np.isfinite(out).all():
        raise ValueError("nonfinite pressure")
    return out


def faces(s):
    """Q generated from actual velocities; matching force is its negative adjoint."""
    u = velocity(s)[:, :, 0]
    p = pressures(s)
    records = []
    force = np.zeros_like(s.h)
    for a in range(3):
        for b in range(3):
            lo = max(s.top_z[0, a + 1], s.top_z[1, b + 1])
            hi = min(s.top_z[0, a], s.top_z[1, b])
            if hi > lo:
                height = hi - lo
                roots, weights = np.polynomial.legendre.leggauss(2)
                depths = 0.5 * (lo + hi) + 0.5 * (hi - lo) * roots
                delta = float(
                    0.5
                    * np.sum(
                        weights
                        * np.array([top_pressure(s, 1, d) - top_pressure(s, 0, d) for d in depths])
                    )
                )
                coef = height / s.dx
                records.append((a, b, 0.5 * (u[0, a] + u[1, b]) * coef, lo, hi))
                force[0, a] -= 0.5 * coef * delta
                force[1, b] -= 0.5 * coef * delta
    for k in range(3, s.h.shape[1]):
        coef = s.h[0, k] / s.dx
        delta = p[1, k] - p[0, k]
        records.append((k, k, 0.5 * (u[0, k] + u[1, k]) * coef, None, None))
        force[:, k] -= 0.5 * coef * delta
    return records, force


def kick(s, dt):
    records, f = faces(s)
    out = s.copy()
    before = ke(s)
    out.n[:, :, 2] += dt * f
    average = 0.5 * (velocity(s)[:, :, 0] + velocity(out)[:, :, 0])
    work = float(dt * np.sum(average * f))
    residual = ke(out) - before - work
    u = np.finfo(float).eps / 2
    bound = 1024 * u * (abs(ke(out)) + abs(before) + abs(work))
    if not np.isfinite([residual, bound]).all() or abs(residual) > bound:
        raise ValueError("pressure/KE exchange")
    validate(out)
    return out, work


@np.errstate(over="raise", invalid="raise", divide="raise")
def advance(s, dt, sources):
    before = s.copy()
    try:
        validate(s)
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt")
        sources = np.asarray(sources)
        if sources.shape != s.n.shape or not np.isfinite(sources).all():
            raise ValueError("extensive source contract")
        mid, work1 = kick(s, 0.5 * dt)
        records, _ = faces(mid)
        dn = np.zeros_like(s.n)
        dh = np.zeros_like(s.h)
        outflow = np.zeros_like(s.h)
        for a, b, q, lo, hi in records:
            value = mid.n[0, a] / mid.h[0, a] if q >= 0 else mid.n[1, b] / mid.h[1, b]
            if lo is not None:
                donor = 0 if q >= 0 else 1
                value = value.copy()
                value[:2] = integrate(mid.top_z[donor], mid.n[donor, :3], lo, hi)[:2] / (hi - lo)
            volume = dt * q
            dh[0, a] -= volume
            dh[1, b] += volume
            dn[0, a] -= volume * value
            dn[1, b] += volume * value
            outflow[0 if q >= 0 else 1, a if q >= 0 else b] += abs(volume)
        # Deep FD volumes fixed: exact bottom-up continuity supplies vertical F.
        cross = []
        for j in range(2):
            vertical = -np.cumsum(dh[j, 3:][::-1])[::-1]
            cross.append(float(vertical[0] / dt))
            for local, volume in enumerate(vertical):
                upper = local + 2
                lower = upper + 1
                value = (
                    mid.n[j, upper] / mid.h[j, upper]
                    if volume >= 0
                    else mid.n[j, lower] / mid.h[j, lower]
                )
                dn[j, upper] -= volume * value
                dn[j, lower] += volume * value
                outflow[j, upper if volume >= 0 else lower] += abs(volume)
            dh[j, 2] -= vertical[0]
            dh[j, 3:] = 0
        if np.any(outflow > 0.5 * mid.h):
            raise ValueError("joint outgoing CFL")
        moved = mid.copy()
        moved.h += dh
        moved.n += dn + dt * sources
        validate(moved)
        target = (moved.eta - B)[:, None] * np.array([1 / 9, 1 / 3, 5 / 9])
        remapped = moved.copy()
        remapped.h[:, :3] = target
        for j in range(2):
            remapped.n[j, :3] = [
                integrate(
                    moved.top_z[j], moved.n[j, :3], remapped.top_z[j, k + 1], remapped.top_z[j, k]
                )
                for k in range(3)
            ]
        validate(remapped)
        out, work2 = kick(remapped, 0.5 * dt)
        out.step += 1
        impulse = (mid.n[:, :, 2:] - s.n[:, :, 2:]).sum(axis=(0, 1)) + (
            out.n[:, :, 2:] - remapped.n[:, :, 2:]
        ).sum(axis=(0, 1))
        residual = out.n.sum(axis=(0, 1)) - s.n.sum(axis=(0, 1)) - dt * sources.sum(axis=(0, 1))
        residual[2:] -= impulse
        unit = np.finfo(float).eps / 2
        budget = (
            2048
            * unit
            * (
                abs(out.n).sum(axis=(0, 1))
                + abs(s.n).sum(axis=(0, 1))
                + abs(dt * sources).sum(axis=(0, 1))
            )
        )
        if not np.isfinite([residual, budget]).all() or np.any(abs(residual) > budget):
            raise ValueError("full inventory budget")
        if not np.isfinite([ke(out), work1, work2]).all():
            raise ValueError("nonfinite diagnostic")
        return (
            out,
            True,
            dict(
                pressure_impulse_kg_per_m_s=impulse.tolist(),
                inventory_residual=residual.tolist(),
                inventory_bound=budget.tolist(),
                qualification_passed=False,
                pressure_work_J_per_m2=work1 + work2,
                band_bottom_volume_flux_m_s=cross,
                water_residual_m=float(out.h.sum() - s.h.sum()),
                source_inventory_change=(dt * sources.sum(axis=(0, 1))).tolist(),
                transport_remap_KE_change_J_per_m2=ke(remapped) - ke(mid),
            ),
        )
    except (ValueError, TypeError, IndexError, FloatingPointError) as error:
        return before, False, dict(rejection_reason=str(error), qualification_passed=False)


def save(s, path):
    validate(s)
    np.savez(
        path,
        h=s.h,
        n=s.n,
        deep_nodes=s.deep_nodes,
        dx=np.array(s.dx),
        source_sha=np.array(s.source_sha),
        step=np.array(s.step),
        version=np.array(1),
    )


def load(path):
    with np.load(path, allow_pickle=False) as p:
        if set(p.files) != {"h", "n", "deep_nodes", "dx", "source_sha", "step", "version"}:
            raise ValueError("restart fields")
        for key in ["step", "version"]:
            if p[key].shape != () or p[key].dtype.kind not in "iu":
                raise ValueError("integer restart")
        if p["version"].item() != 1 or p["dx"].shape != () or p["source_sha"].shape != ():
            raise ValueError("restart metadata")
        s = State(
            p["h"].copy(),
            p["n"].copy(),
            p["deep_nodes"].copy(),
            float(p["dx"]),
            str(p["source_sha"].item()),
            int(p["step"]),
        )
    validate(s)
    return s
