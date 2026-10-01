"""Executable inventory/GCL kernel; pressure-driven dynamics fail closed."""

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "fd_static_bridge"))
from bridge import integrate  # noqa: E402
from discrete_audit import audit  # noqa: E402

B = -22.5
FRACTION = np.array([1 / 9, 1 / 3, 5 / 9])


def finite(value):
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            finite(v)
    elif not isinstance(value, (str, type(None))):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite stage")


@dataclass
class State:
    h: np.ndarray
    inventory: np.ndarray
    eos: dict
    deep: tuple
    migration: dict
    step: int = 0
    geometry: str = "physical_top_band_v1"
    reconstruction: str = "bounded_P1_v1"

    def copy(self):
        return copy.deepcopy(self)

    @property
    def eta(self):
        return B + self.h.sum(axis=1)

    @property
    def z(self):
        return np.c_[self.eta, self.eta[:, None] - np.cumsum(self.h, axis=1)]


def sha(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_deep(record):
    required = {
        "nodes",
        "values",
        "inventory",
        "reference_weights",
        "wet_mask",
        "pressure_increment",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("deep record fields")
    if any(
        not isinstance(record[k], np.ndarray)
        or (record[k].dtype.kind not in "bif" if k == "wet_mask" else record[k].dtype != np.float64)
        for k in required
    ):
        raise ValueError("deep arrays/dtype")
    n = record["nodes"].size
    if (
        n < 1
        or any(
            record[k].shape != (n,)
            for k in ["nodes", "reference_weights", "wet_mask", "pressure_increment"]
        )
        or any(record[k].shape != (n, 4) for k in ["values", "inventory"])
    ):
        raise ValueError("deep array shape")
    finite(record)
    if (
        record["nodes"][0] <= -B
        or np.any(np.diff(record["nodes"]) <= 0)
        or np.any(record["reference_weights"] <= 0)
    ):
        raise ValueError("deep coordinates/weights")
    wet = record["wet_mask"]
    if wet[0] != 1 or not np.all((wet == 0) | (wet == 1)) or np.any(np.diff(wet) > 0):
        raise ValueError("deep wet layout")


def validate(s):
    if not isinstance(s.h, np.ndarray) or not isinstance(s.inventory, np.ndarray):
        raise ValueError("state arrays")
    if (
        s.h.shape != (2, 3)
        or s.inventory.shape != (2, 3, 4)
        or s.h.dtype != np.float64
        or s.inventory.dtype != np.float64
    ):
        raise ValueError("state shape/dtype")
    if (
        type(s.step) is not int
        or s.step < 0
        or s.geometry != "physical_top_band_v1"
        or s.reconstruction != "bounded_P1_v1"
    ):
        raise ValueError("state identity")
    if not isinstance(s.migration, dict) or not sha(s.migration.get("geometry_report_sha256"), 64):
        raise ValueError("migration geometry identity")
    sources = s.migration.get("source_sha")
    if not isinstance(sources, list) or len(sources) != 2 or not all(sha(v, 40) for v in sources):
        raise ValueError("migration source identity")
    if not isinstance(s.eos, dict) or any(
        np.asarray(v).shape != () or np.asarray(v).dtype.kind not in "fi" for v in s.eos.values()
    ):
        raise ValueError("EOS finite numeric scalar required")
    if not isinstance(s.deep, tuple) or len(s.deep) != 2:
        raise ValueError("deep identity")
    for record in s.deep:
        validate_deep(record)
    finite([s.h, s.inventory, s.eos, s.deep, s.z, s.eta])
    if (
        set(s.eos) != {"rho0", "gravity", "alpha", "beta", "Tref", "Sref"}
        or min(s.eos[k] for k in ["rho0", "gravity", "alpha", "beta"]) <= 0
    ):
        raise ValueError("EOS")
    u = np.finfo(float).eps / 2
    if np.any(s.h <= 8 * u * (abs(s.z[:, :-1]) + abs(s.z[:, 1:]))):
        raise ValueError("geometry unresolved")
    mean = s.inventory[:, :, :2] / s.h[:, :, None]
    tol = 1100 * u * (abs(mean) + [45, 50])
    if np.any(mean < np.array([-5, 0]) - tol) or np.any(mean > np.array([45, 50]) + tol):
        raise ValueError("tracer bounds")
    if len(s.deep) != 2:
        raise ValueError("deep identity")


def velocity(s):
    validate(s)
    v = s.inventory[:, :, 2:] / (s.eos["rho0"] * s.h[:, :, None])
    finite(v)
    return v


def energy(s):
    v = velocity(s)
    value = float(0.5 * s.eos["rho0"] * np.sum(s.h[:, :, None] * v * v))
    finite(value)
    return value


def target(eta):
    return (eta - B)[:, None] * FRACTION


def remap(s):
    h = target(s.eta)
    out = s.copy()
    out.h = h
    for col in range(2):
        old = s.z[col]
        new = out.z[col]
        out.inventory[col] = [
            integrate(old, s.inventory[col], new[k + 1], new[k]) for k in range(3)
        ]
    validate(out)
    u = np.finfo(float).eps / 2
    gamma = 1100 * u / (1 - 1100 * u)
    residual = out.inventory.sum(axis=1) - s.inventory.sum(axis=1)
    bound = gamma * (abs(out.inventory).sum(axis=1) + abs(s.inventory).sum(axis=1))
    finite([residual, bound])
    if np.any(abs(residual) > bound):
        raise ValueError("remap inventory budget")
    return out


@np.errstate(over="raise", invalid="raise", divide="raise")
def migrate(columns, geometry_report_sha256):
    if (
        not isinstance(geometry_report_sha256, str)
        or len(geometry_report_sha256) != 64
        or any(c not in "0123456789abcdef" for c in geometry_report_sha256)
    ):
        raise ValueError("external geometry report identity required")
    if len(columns) != 2:
        raise ValueError("two columns required")
    hs = []
    ns = []
    deep = []
    eos = None
    reference_K = 0
    for p in columns:
        finite(audit(**p))
        v = p["values"]
        w = p["reference_weights"]
        mask = p["wet_mask"]
        d = p["depth"]
        eta = float(p["eta"])
        current = {k: float(p[k]) for k in ["rho0", "gravity", "alpha", "beta", "Tref", "Sref"]}
        if eos is not None and eos != current:
            raise ValueError("EOS identity mismatch")
        eos = current
        if (
            not np.array_equal(d[:4], [0, 5, 15, 30])
            or not np.array_equal(w[:3], [2.5, 7.5, 12.5])
            or not np.all(mask[:3] == 1)
            or eta > 0
        ):
            raise ValueError("top source contract")
        h = np.array([2.5 + eta, 7.5, 12.5])
        n = h[:, None] * v[:3]
        n[:, 2:] = eos["rho0"] * w[:3, None] * v[:3, 2:]
        hs.append(h)
        ns.append(n)
        weights = w[3:] * mask[3:]
        stock = weights[:, None] * v[3:]
        stock[:, 2:] *= eos["rho0"]
        rho = (
            eos["rho0"]
            * (-eos["alpha"] * (v[:, 0] - eos["Tref"]) + eos["beta"] * (v[:, 1] - eos["Sref"]))
            * mask
        )
        pnode = (
            np.r_[0, np.cumsum(eos["gravity"] * 0.5 * (rho[:-1] + rho[1:]) * np.diff(d))]
            + eos["rho0"] * eos["gravity"] * eta
        )
        deep.append(
            dict(
                nodes=d[3:].copy(),
                values=v[3:].copy(),
                inventory=stock.copy(),
                reference_weights=w[3:].copy(),
                wet_mask=mask[3:].copy(),
                pressure_increment=(pnode[3:] - np.interp(-B, d, pnode)).copy(),
            )
        )
        reference_K += float(0.5 * eos["rho0"] * np.sum(w[:3, None] * v[:3, 2:] ** 2))
    source = State(
        np.array(hs),
        np.array(ns),
        eos,
        tuple(deep),
        dict(
            geometry_report_sha256=geometry_report_sha256,
            source_sha=[str(p["source_sha"]) for p in columns],
            reference_K_J_per_m2=reference_K,
        ),
    )
    validate(source)
    # Migration does not construct a thin-layer actual velocity or its slope.
    # T/S use bounded P1; reference momentum uses overlap FRACTIONS directly.
    out = source.copy()
    out.h = target(source.eta)
    for col in range(2):
        oldz = source.z[col]
        newz = out.z[col]
        thermal = source.inventory[col].copy()
        thermal[:, 2:] = 0.0
        for dst in range(3):
            out.inventory[col, dst] = integrate(oldz, thermal, newz[dst + 1], newz[dst])
            for src in range(3):
                overlap = max(0.0, min(newz[dst], oldz[src]) - max(newz[dst + 1], oldz[src + 1]))
                out.inventory[col, dst, 2:] += (overlap / source.h[col, src]) * source.inventory[
                    col, src, 2:
                ]
    validate(out)
    u = np.finfo(float).eps / 2
    gamma = 1100 * u / (1 - 1100 * u)
    residual = out.inventory.sum(axis=1) - source.inventory.sum(axis=1)
    bound = gamma * (abs(out.inventory).sum(axis=1) + abs(source.inventory).sum(axis=1))
    finite([residual, bound])
    if np.any(abs(residual) > bound):
        raise ValueError("migration inventory budget")
    out.migration["momentum_migration"] = "P0 overlap fractions; no old thin velocity recovery"

    out.migration["candidate_K_J_per_m2"] = energy(out)
    out.migration["velocity_max_m_s"] = float(abs(velocity(out)).max())
    finite(out.migration)
    return out


@np.errstate(over="raise", invalid="raise", divide="raise")
def advance(s, segments, sources, dt=1.0, pressure_driven=False, deep_transport=False):
    before = s.copy()
    try:
        validate(s)
        if pressure_driven or deep_transport:
            raise ValueError("unsupported dynamics")
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt")
        sources = np.asarray(sources)
        if sources.shape != s.inventory.shape:
            raise ValueError("extensive source shape")
        finite(sources)
        if len(segments) > 3:
            raise ValueError("stencil budget")
        dh = np.zeros_like(s.h)
        dn = np.zeros_like(s.inventory)
        outgoing = np.zeros_like(s.h)
        seen = []
        for lo, hi, q in segments:
            finite([lo, hi, q])
            if (
                lo < B
                or hi > min(s.eta)
                or hi <= lo
                or any(max(lo, a) < min(hi, b) for a, b in seen)
            ):
                raise ValueError("shared wet face")
            seen.append((lo, hi))
            cuts = sorted({lo, hi, *s.z[(s.z > lo) & (s.z < hi)]})
            for a, b in zip(cuts[:-1], cuts[1:]):
                mid = 0.5 * (a + b)
                cells = [
                    int(np.flatnonzero((s.z[j, :-1] >= mid) & (s.z[j, 1:] <= mid))[0])
                    for j in range(2)
                ]
                donor = 0 if q >= 0 else 1
                volume = dt * q * (b - a) / (hi - lo)
                avg = integrate(s.z[donor], s.inventory[donor], a, b) / (b - a)
                finite(avg)
                for j, sign in [(0, -1), (1, 1)]:
                    dh[j, cells[j]] += sign * volume
                    dn[j, cells[j]] += sign * volume * avg
                outgoing[donor, cells[donor]] += abs(volume)
        if np.any(outgoing > 0.5 * s.h):
            raise ValueError("outflow CFL")
        intermediate = s.copy()
        intermediate.h += dh
        intermediate.inventory += dn + dt * sources
        intermediate.step += 1
        validate(intermediate)
        out = remap(intermediate)
        residual = (
            out.inventory.sum(axis=(0, 1))
            - s.inventory.sum(axis=(0, 1))
            - dt * sources.sum(axis=(0, 1))
        )
        u = np.finfo(float).eps / 2
        gamma = 1100 * u / (1 - 1100 * u)
        bound = gamma * (
            abs(out.inventory).sum(axis=(0, 1))
            + abs(s.inventory).sum(axis=(0, 1))
            + abs(dt * sources).sum(axis=(0, 1))
        )
        finite([residual, bound])
        if np.any(abs(residual) > bound):
            raise ValueError("inventory budget")
        report = dict(
            qualification_passed=False,
            inventory_residual=residual.tolist(),
            inventory_bound=bound.tolist(),
            water_residual_m=float(out.h.sum() - s.h.sum()),
            transport_source_KE_change_J_per_m2=energy(intermediate) - energy(s),
            remap_KE_change_J_per_m2=energy(out) - energy(intermediate),
        )
        finite(report)
        return out, True, report
    except (ValueError, TypeError, IndexError, FloatingPointError) as error:
        return before, False, dict(rejection_reason=str(error), qualification_passed=False)


def pressure(s, col, depth):
    validate(s)
    if np.ndim(depth) != 0 or not np.isfinite(depth) or depth > s.eta[col]:
        raise ValueError("depth")
    eos = s.eos
    eta = s.eta[col]

    def top(d):
        q = integrate(s.z[col], s.inventory[col], d, eta)
        length = eta - d
        return eos["rho0"] * eos["gravity"] * eta + eos["rho0"] * eos["gravity"] * (
            -eos["alpha"] * (q[0] - eos["Tref"] * length)
            + eos["beta"] * (q[1] - eos["Sref"] * length)
        )

    if depth >= B:
        value = top(depth)
    else:
        record = s.deep[col]
        active = record["wet_mask"] > 0
        if not np.any(active) or -depth > record["nodes"][active][-1]:
            raise ValueError("deep pressure extrapolation")
        value = top(B) + np.interp(
            -depth,
            np.r_[-B, record["nodes"][active]],
            np.r_[0.0, record["pressure_increment"][active]],
        )
    finite(value)
    return float(value)


@np.errstate(over="raise", invalid="raise", divide="raise")
def pressure_power(s, segments):
    """Virtual pressure power for prescribed actual Q (unit column area)."""
    validate(s)
    power = 0.0
    roots, weights = np.polynomial.legendre.leggauss(2)
    for lo, hi, q in segments:
        finite([lo, hi, q])
        if lo < B or hi > min(s.eta) or hi <= lo:
            raise ValueError("shared wet face")
        cuts = sorted({lo, hi, *s.z[(s.z > lo) & (s.z < hi)]})
        for a, b in zip(cuts[:-1], cuts[1:]):
            points = 0.5 * (a + b) + 0.5 * (b - a) * roots
            delta = np.array([pressure(s, 1, d) - pressure(s, 0, d) for d in points])
            power -= q / (hi - lo) * 0.5 * (b - a) * np.sum(weights * delta)
    finite(power)
    return float(power)
