"""Production-grid inventory candidate; enabled unsupported processes reject.

Original deep FD masses are retained. Top slots are extensive FV means,
not reconstructed original point samples. Nothing here changes the driver.
"""

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from config import C_P  # noqa: E402
from grid import nodal_control_thickness  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "fd_static_bridge"))
from bridge import integrate, reconstruct  # noqa: E402

RHO, G = 1025.0, 9.81
FRACTIONS = np.array([1 / 9, 1 / 3, 5 / 9])
TIME_SCHEME = "inventory_kdk_with_linear_halves_v1"


@dataclass
class State:
    h: np.ndarray
    n: np.ndarray  # per horizontal area: hT,hS,rho*h*u,rho*h*v
    eta: np.ndarray
    ice: np.ndarray
    band_bottom: np.ndarray
    identity: dict
    step: int = 0

    def copy(self):
        return copy.deepcopy(self)


def top_edges(s, cell):
    return np.r_[s.eta[cell], s.eta[cell] - np.cumsum(s.h[cell][:3])]


def parameter_digest(p):
    """Exact consumer-local parameter snapshot, including forcing and clocks."""
    values = p._asdict() if hasattr(p, "_asdict") else vars(p)
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode() + b"\0")
        if value is None:
            digest.update(b"None")
            continue
        a = np.asarray(value)
        if a.dtype.kind not in "biufUS":
            raise ValueError("unsupported parameter identity type: " + name)
        digest.update(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes())
    return digest.hexdigest()


def code_digest():
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parents[1] / "fd_static_bridge/bridge.py",
                 Path(__file__).parents[3] / "src/config.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def migrate(fd, p, *, source_sha, forcing_sha):
    """Once-only authority conversion, including valid two-node shallow water.

    Band bottom is the ORIGINAL nodal-dual edge, never raw bathymetry.
    Top tracer stock uses original actual volume; reference momentum is
    remapped P0 before recovering velocity on actual mass. Deep stock is
    unchanged. Requires a positive accepted entry, not a negative attempt.
    """
    if p.column_geometry != "nodal_dual_v1":
        raise ValueError("nodal_dual_v1 required")
    for value, length in ((source_sha, 40), (forcing_sha, 64)):
        if not isinstance(value, str) or len(value) != length or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError("exact source/forcing identity required")
    wet = np.asarray(p.wet_mask_z)
    h0 = np.broadcast_to(np.asarray(p.dz_node), wet.shape) * wet
    if wet.shape[-1] < 4:
        raise ValueError("at least four reference slots required")
    shape = wet.shape
    values = np.stack([np.asarray(a) for a in (fd.T, fd.S, fd.u, fd.v)], axis=-1)
    eta = np.asarray(fd.eta)
    if values.shape != shape + (4,) or values.dtype != np.float64:
        raise ValueError("float64 FD state shape required")
    if eta.shape != shape[:2] or eta.dtype != np.float64:
        raise ValueError("float64 eta shape required")
    if not np.isfinite(values).all() or not np.isfinite(eta).all():
        raise ValueError("finite FD entry required")
    n = h0[..., None] * values
    n[..., 2:] *= RHO
    h = h0.copy()
    bottom = -h0[..., :3].sum(axis=-1)
    for cell in np.ndindex(shape[:2]):
        count = int(wet[cell][:3].sum())
        if not count:
            continue
        old_h = h0[cell][:count].copy()
        old_h[0] += eta[cell]
        if np.any(old_h <= 0):
            raise ValueError(f"nonpositive accepted entry at {cell}")
        old_z = np.r_[eta[cell], eta[cell] - np.cumsum(old_h)]
        old_n = old_h[:, None] * values[cell][:count]
        old_n[:, 2:] = RHO * h0[cell][:count, None] * values[cell][:count, 2:]
        new_h = (eta[cell] - bottom[cell]) * FRACTIONS
        new_z = np.r_[eta[cell], eta[cell] - np.cumsum(new_h)]
        new_z[-1] = bottom[cell]
        out = np.zeros((3, 4))
        for k in range(3):
            lo, hi = new_z[k + 1], new_z[k]
            if count == 1:
                out[k, :2] = (hi - lo) * old_n[0, :2] / old_h[0]
            else:
                out[k, :2] = integrate(old_z, old_n, lo, hi)[:2]
            for a in range(count):
                overlap = max(0., min(hi, old_z[a]) - max(lo, old_z[a + 1]))
                out[k, 2:] += overlap * old_n[a, 2:] / old_h[a]
        residual = out.sum(axis=0) - old_n.sum(axis=0)
        bound = 2048 * np.finfo(float).eps * (
            abs(out).sum(axis=0) + abs(old_n).sum(axis=0)
        )
        if np.any(abs(residual) > bound):
            raise ValueError("migration inventory closure")
        h[cell][:3], n[cell][:3] = new_h, out
    ice = np.broadcast_to(np.asarray(fd.ice), shape[:2]).astype(np.float64).copy()
    if not np.isfinite(ice).all() or np.any(ice < 0):
        raise ValueError("finite nonnegative ice required")
    return State(h, n, eta.copy(), ice, bottom, dict(
        source_sha=source_sha, forcing_sha=forcing_sha,
        parameter_sha=parameter_digest(p),
        candidate_code_sha=code_digest(),
        geometry="variable_band_bottom_original_deep_mass_v1",
    ))


def pressure(s, cell, depth, *, Tref=15., Sref=35.):
    """Physical top pressure. Depth is negative metres, not slot index."""
    z = top_edges(s, cell)
    if not z[-1] <= depth <= z[0]:
        raise ValueError("pressure outside physical top")
    q = integrate(z, s.n[cell][:3], depth, z[0])
    length = z[0] - depth
    return RHO * G * (z[0] - 2e-4 * (q[0] - Tref * length)
                      + 7.6e-4 * (q[1] - Sref * length))


def deep_pressure(s, p, grid, cell):
    """New top-to-first-node segment, then original deep FD trapezoids."""
    out = np.zeros(s.h.shape[-1])
    if not np.any(s.h[cell][:3]):
        return out
    z = top_edges(s, cell)
    c, m, x = reconstruct(z, s.n[cell][:3])
    tb, sb = (c[-1] + m[-1] * (z[-1] - x[-1]))[:2]
    last_rho = RHO * (-2e-4 * (tb - p.T_ref) + 7.6e-4 * (sb - p.S_ref))
    current = pressure(s, cell, z[-1], Tref=p.T_ref, Sref=p.S_ref)
    last_depth = -z[-1]
    nodes = -np.asarray(grid.z)
    for k in range(3, s.h.shape[-1]):
        if s.h[cell][k] == 0:
            break
        t, salt = s.n[cell][k, :2] / s.h[cell][k]
        rho = RHO * (-2e-4 * (t - p.T_ref) + 7.6e-4 * (salt - p.S_ref))
        current += G * .5 * (last_rho + rho) * (nodes[k] - last_depth)
        out[k] = current
        last_rho, last_depth = rho, nodes[k]
    return out


def top_faces(s, p):
    """Original spherical shared wet faces and area-adjoint pressure impulses.

    Returns records (left,right,slot_l,slot_r,axis,Q_m3_s,lo,hi).
    Only top faces here: callers MUST NOT treat this as whole-column transport.
    Closed latitude walls; longitude periodic; land has zero overlap.
    """
    area = np.asarray(p.dx_2d) * p.dy
    cosine = np.asarray(p.cos_lat)
    nx, ny = area.shape
    width_x = np.full((nx, ny), p.dy)
    width_y = np.asarray(p.dx_2d) / cosine[None, :] * (
        .5 * (cosine + np.roll(cosine, -1))
    )[None, :]
    u = np.zeros_like(s.n[..., 2:])
    np.divide(s.n[..., 2:], RHO * s.h[..., None], out=u, where=s.h[..., None] > 0)
    force = np.zeros_like(u)
    records = []
    roots, weights = np.polynomial.legendre.leggauss(2)
    for cell in np.ndindex((nx, ny)):
        for axis in (0, 1):
            if axis == 1 and cell[1] == ny - 1:
                continue
            other = ((cell[0] + 1) % nx, cell[1]) if axis == 0 else (cell[0], cell[1] + 1)
            if not np.any(s.h[cell][:3]) or not np.any(s.h[other][:3]):
                continue
            za, zb = top_edges(s, cell), top_edges(s, other)
            width = width_x[cell] if axis == 0 else width_y[cell]
            for a in range(3):
                for b in range(3):
                    lo, hi = max(za[a + 1], zb[b + 1]), min(za[a], zb[b])
                    if hi <= lo:
                        continue
                    coefficient = width * (hi - lo)
                    q = coefficient * .5 * (u[cell][a, axis] + u[other][b, axis])
                    depths = .5 * (lo + hi) + .5 * (hi - lo) * roots
                    delta = .5 * sum(w * (
                        pressure(s, other, d, Tref=p.T_ref, Sref=p.S_ref)
                        - pressure(s, cell, d, Tref=p.T_ref, Sref=p.S_ref)
                    ) for d, w in zip(depths, weights, strict=True))
                    force[cell][a, axis] -= .5 * coefficient * delta / area[cell]
                    force[other][b, axis] -= .5 * coefficient * delta / area[other]
                    records.append((cell, other, a, b, axis, q, lo, hi))
    return records, force


def all_faces(s, p, grid):
    records, force = top_faces(s, p)
    area = np.asarray(p.dx_2d) * p.dy
    cosine = np.asarray(p.cos_lat)
    nx, ny = area.shape
    u = np.zeros_like(force)
    np.divide(s.n[..., 2:], RHO * s.h[..., None], out=u, where=s.h[..., None] > 0)
    pp = {cell: deep_pressure(s, p, grid, cell) for cell in np.ndindex((nx, ny))}
    for cell in np.ndindex((nx, ny)):
        for axis in (0, 1):
            if axis == 1 and cell[1] == ny - 1:
                continue
            other = ((cell[0] + 1) % nx, cell[1]) if axis == 0 else (cell[0], cell[1] + 1)
            width = p.dy if axis == 0 else (
                np.asarray(p.dx_2d)[cell] / cosine[cell[1]]
                * .5 * (cosine[cell[1]] + cosine[other[1]])
            )
            for j in range(3, s.h.shape[-1]):
                if not s.h[cell][j] or not s.h[other][j]:
                    continue
                if s.h[cell][j] != s.h[other][j]:
                    raise ValueError("deep reference mass mismatch")
                coefficient = width * s.h[cell][j]
                q = coefficient * .5 * (u[cell][j, axis] + u[other][j, axis])
                delta = pp[other][j] - pp[cell][j]
                force[cell][j, axis] -= .5 * coefficient * delta / area[cell]
                force[other][j, axis] -= .5 * coefficient * delta / area[other]
                records.append((cell, other, j, j, axis, q, None, None))
    return records, force


def means(s):
    out = np.zeros_like(s.n)
    np.divide(s.n, s.h[..., None], out=out, where=s.h[..., None] > 0)
    return out


def validate(s):
    if type(s.step) is not int or s.step < 0:
        raise ValueError("strict candidate step")
    if set(s.identity) != {"source_sha", "forcing_sha", "parameter_sha", "candidate_code_sha", "geometry"}:
        raise ValueError("candidate identity schema")
    for name, length in (("source_sha", 40), ("forcing_sha", 64), ("parameter_sha", 64),
                         ("candidate_code_sha", 64)):
        value = s.identity[name]
        if not isinstance(value, str) or len(value) != length or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError("candidate identity digest")
    if s.identity["candidate_code_sha"] != code_digest():
        raise ValueError("candidate source changed since migration")
    if s.identity["geometry"] != "variable_band_bottom_original_deep_mass_v1":
        raise ValueError("candidate geometry version")
    if any(not isinstance(a, np.ndarray) or a.dtype != np.float64
           for a in (s.h, s.n, s.eta, s.ice, s.band_bottom)):
        raise ValueError("candidate float64 arrays required")
    if s.h.ndim != 3 or s.h.shape[-1] < 4 or s.n.shape != s.h.shape + (4,):
        raise ValueError("candidate shapes")
    if any(a.shape != s.h.shape[:2] for a in (s.eta, s.ice, s.band_bottom)):
        raise ValueError("candidate surface shapes")
    if any(not np.isfinite(a).all() for a in (s.h, s.n, s.eta, s.ice, s.band_bottom)):
        raise ValueError("nonfinite candidate")
    wet = s.band_bottom < 0
    if np.any(s.h < 0) or np.any(s.h[..., :3][wet] <= 0) or np.any(s.n[s.h == 0] != 0):
        raise ValueError("candidate positive geometry/inactive stock")
    if not np.allclose(s.h[..., :3].sum(axis=-1)[wet],
                       (s.eta - s.band_bottom)[wet], rtol=0, atol=2e-13):
        raise ValueError("candidate geometric conservation")
    c = means(s)
    if np.any(c[..., :2][s.h > 0] < [-5., 0.]) or np.any(c[..., :2][s.h > 0] > [45., 50.]):
        raise ValueError("candidate tracer bounds")
    if np.max(abs(c[..., 2:] / RHO)) >= 10 or (wet.any() and np.max(abs(s.eta[wet])) >= 15):
        raise ValueError("original finite velocity/eta science limits")


def unsupported(p):
    """Fail before updating; this is not permission to disable original physics."""
    zero = (
        "nu_bi", "kappa_bi", "kappa_gm", "kappa_redi", "dynamic_ice",
        "polar_cap_rows", "polar_cap_taper", "sponge_rate", "sponge_rate_2d",
        "eta_relax_rate", "coastal_kappa_h_2d", "coastal_kappa_v_2d",
        "mixed_layer_depth_m", "ice_salt_flux", "restore_coef_S",
        "coastal_restore_coef_2d", "coastal_bulk_lambda_2d",
        "fct_adv",
    )
    out = [name for name in zero if getattr(p, name, None) is not None
           and np.any(np.asarray(getattr(p, name)) != 0)]
    if getattr(p, "mixed_layer_depth_2d", None) is not None:
        out.append("mixed_layer_depth_2d")
    if getattr(p, "bottom_friction", "linear") != "linear":
        out.append("quadratic_bottom_friction")
    for name in ("adv_nsub", "conv_nsub", "nu_nsub"):
        if getattr(p, name, None) is not None and getattr(p, name) != 1:
            out.append(name + "_stage_clock")
    if getattr(p, "dealias_lon_mask", None) is not None:
        out.append("original_nonlinear_dealias_filter")
    return out


def bind_geometry(s, p, grid):
    shape = (grid.nx, grid.ny, grid.nz)
    if s.h.shape != shape:
        raise ValueError("candidate/grid shape mismatch")
    for name, expected_shape in (("dx_2d", shape[:2]), ("dy", ()),
                                 ("cos_lat", (grid.ny,))):
        if any(np.ma.getmaskarray(getattr(owner, name, None)).any()
               for owner in (p, grid)):
            raise ValueError("unknown consumed horizontal metric: " + name)
        parameter = np.asarray(getattr(p, name, None))
        metric = np.asarray(getattr(grid, name, None))
        if (parameter.shape != expected_shape or metric.shape != expected_shape
                or parameter.dtype.kind not in "iuf" or metric.dtype.kind not in "iuf"
                or not np.isfinite(parameter).all() or not np.isfinite(metric).all()
                or np.any(parameter <= 0) or np.any(metric <= 0)
                or not np.array_equal(parameter, metric)):
            raise ValueError("consumed horizontal metric mismatch: " + name)
    if not np.array_equal(np.asarray(p.dz_node).ravel(), nodal_control_thickness(grid.z)):
        raise ValueError("original node geometry mismatch")
    ref = np.broadcast_to(np.asarray(p.dz_node), shape) * np.asarray(p.wet_mask_z)
    if not np.array_equal(s.h[..., 3:], ref[..., 3:]):
        raise ValueError("deep reference mass changed")
    if not np.array_equal(s.band_bottom, -ref[..., :3].sum(axis=-1)):
        raise ValueError("candidate band-bottom changed")
    if not np.array_equal(np.asarray(p.wet_mask_z), grid.wet_mask_3d):
        raise ValueError("original wet geometry mismatch")


def validate_dissipative_controls(p):
    """Reject invalid scalar transport controls, including inactive coefficients."""
    for name in ("kappa_h", "kappa_v", "kappa_conv", "nu_h", "nu_v", "r_bot"):
        if np.ma.getmaskarray(getattr(p, name, None)).any():
            raise ValueError("unknown dissipative control: " + name)
        value = np.asarray(getattr(p, name, None))
        if (value.shape != () or value.dtype.kind not in "iuf"
                or not np.isfinite(value) or value < 0):
            raise ValueError("finite nonnegative scalar required: " + name)


def kick(s, p, grid, dt):
    _, force = all_faces(s, p, grid)
    out = s.copy()
    out.n[..., 2:] += dt * force
    out.n[:, (0, -1), :, 3] = 0.  # original closed-wall normal-velocity condition
    return out


def drift(s, p, grid, dt):
    records, _ = all_faces(s, p, grid)
    area = np.asarray(p.dx_2d) * p.dy
    dn, dh, outflow = np.zeros_like(s.n), np.zeros_like(s.h), np.zeros_like(s.h)
    c = means(s)
    for a, b, ka, kb, _, q, lo, hi in records:
        donor, kd = (a, ka) if q >= 0 else (b, kb)
        value = c[donor][kd].copy()
        if lo is not None:
            value[:2] = integrate(top_edges(s, donor), s.n[donor][:3], lo, hi)[:2] / (hi - lo)
        volume = dt * q
        for cell, slot, sign in ((a, ka, -1), (b, kb, 1)):
            dh[cell][slot] += sign * volume / area[cell]
            dn[cell][slot] += sign * volume * value / area[cell]
        outflow[donor][kd] += abs(volume) / area[donor]
    # Exact bottom-up inverse of horizontal deep divergence. This is the
    # nonzero exchange at the physical top-band bottom, not an extra source.
    cross = np.zeros(s.eta.shape)
    for cell in np.ndindex(s.eta.shape):
        vertical = -np.cumsum(dh[cell][3:][::-1])[::-1]
        cross[cell] = vertical[0] / dt
        for local, volume in enumerate(vertical):
            upper, lower = local + 2, local + 3
            donor = upper if volume >= 0 else lower
            if not s.h[cell][donor]:
                if volume != 0:
                    raise ValueError("transport across dry bottom")
                continue
            value = c[cell][donor].copy()
            if upper == 2 and donor == upper:
                z = top_edges(s, cell)
                mean, slope, centers = reconstruct(z, s.n[cell][:3])
                value[:2] = (mean[-1] + slope[-1] * (z[-1] - centers[-1]))[:2]
            dn[cell][upper] -= volume * value
            dn[cell][lower] += volume * value
            outflow[cell][donor] += abs(volume)
        dh[cell][2] -= vertical[0]
        dh[cell][3:] = 0
    if np.any(outflow > .5 * s.h):
        raise ValueError("joint outgoing CFL; no thickness clamp")
    moved = s.copy()
    moved.h += dh
    moved.n += dn
    wet = moved.band_bottom < 0
    moved.eta[wet] = moved.band_bottom[wet] + moved.h[..., :3].sum(axis=-1)[wet]
    validate(moved)
    out = moved.copy()
    for cell in np.ndindex(s.eta.shape):
        if not wet[cell]:
            continue
        old_z = top_edges(moved, cell)
        out.h[cell][:3] = (moved.eta[cell] - moved.band_bottom[cell]) * FRACTIONS
        z = top_edges(out, cell)
        out.n[cell][:3] = [integrate(old_z, moved.n[cell][:3], z[j + 1], z[j])
                             for j in range(3)]
    validate(out)
    return out, cross


def linear(s, p, grid, dt):
    """Conservative physical face diffusion, exact rotation and linear drag.

    Top thermodynamic and momentum states are cell means; deep states remain
    FD points. This is a new inventory discretization, not the old L wrapper.
    """
    c = means(s)
    rates = np.zeros_like(s.n)
    exit_rate = np.zeros_like(s.n)
    area = np.asarray(p.dx_2d) * p.dy
    records, _ = all_faces(s, p, grid)
    coefficients = np.array([p.kappa_h, p.kappa_h, p.nu_h, p.nu_h])
    for a, b, ka, kb, axis, _, lo, hi in records:
        if lo is not None:
            ca = c[a][ka].copy()
            cb = c[b][kb].copy()
            ca[:2] = integrate(top_edges(s, a), s.n[a][:3], lo, hi)[:2] / (hi - lo)
            cb[:2] = integrate(top_edges(s, b), s.n[b][:3], lo, hi)[:2] / (hi - lo)
            height = hi - lo
        else:
            ca, cb, height = c[a][ka], c[b][kb], s.h[a][ka]
        distance = np.asarray(p.dx_2d)[a] if axis == 0 else p.dy
        width = p.dy if axis == 0 else (
            np.asarray(p.dx_2d)[a] / np.asarray(p.cos_lat)[a[1]]
            * .5 * (np.asarray(p.cos_lat)[a[1]] + np.asarray(p.cos_lat)[b[1]])
        )
        conductance = coefficients * width * height / distance
        flux = conductance * (cb - ca)
        rates[a][ka] += flux / area[a]
        rates[b][kb] -= flux / area[b]
        exit_rate[a][ka] += conductance / (area[a] * s.h[a][ka])
        exit_rate[b][kb] += conductance / (area[b] * s.h[b][kb])
    for cell in np.ndindex(s.eta.shape):
        wet_slots = np.flatnonzero(s.h[cell] > 0)
        if not wet_slots.size:
            continue
        z = top_edges(s, cell)
        centers = np.r_[.5 * (z[:-1] + z[1:]), np.asarray(grid.z)[3:]]
        density = -2e-4 * (c[cell][:, 0] - p.T_ref) + 7.6e-4 * (c[cell][:, 1] - p.S_ref)
        unstable = np.any(density[wet_slots[:-1]] > density[wet_slots[1:]])
        for lower in wet_slots[1:]:
            upper = lower - 1
            distance = centers[upper] - centers[lower]
            if distance <= 0:
                raise ValueError("vertical mixing center ordering")
            gate = density[upper] > density[lower] if getattr(p, "localize_conv", False) else unstable
            conv = p.kappa_conv if gate else 0.
            coef = np.array([p.kappa_v + conv, p.kappa_v + conv, p.nu_v, p.nu_v])
            flux = coef * (c[cell][lower] - c[cell][upper]) / distance
            if upper == 2:
                mean, slope, top_centers = reconstruct(z, s.n[cell][:3])
                boundary = (mean[-1] + slope[-1] * (z[-1] - top_centers[-1]))[:2]
                boundary_distance = z[-1] - centers[lower]
                if boundary_distance <= 0:
                    raise ValueError("top/deep physical interface ordering")
                flux[:2] = coef[:2] * (c[cell][lower, :2] - boundary) / boundary_distance
                # Conservative sufficient bound uses the shorter boundary
                # segment for the thermal coupling; reject before updating.
                coef = coef.copy()
                coef[:2] *= distance / boundary_distance
            rates[cell][upper] += flux
            rates[cell][lower] -= flux
            exit_rate[cell][upper] += coef / (distance * s.h[cell][upper])
            exit_rate[cell][lower] += coef / (distance * s.h[cell][lower])
    if np.any(dt * exit_rate > .5):
        raise ValueError("joint mixing CFL; original coefficients retained")
    out = s.copy()
    out.n += dt * rates
    angle = np.asarray(p.f)[..., None] * dt
    mu, mv = out.n[..., 2].copy(), out.n[..., 3].copy()
    out.n[..., 2] = np.cos(angle) * mu + np.sin(angle) * mv
    out.n[..., 3] = -np.sin(angle) * mu + np.cos(angle) * mv
    out.n[:, (0, -1), :, 3] = 0.
    for cell in np.ndindex(s.eta.shape):
        wet_slots = np.flatnonzero(s.h[cell] > 0)
        if wet_slots.size:
            out.n[cell][wet_slots[-1], 2:] *= np.exp(-p.r_bot * dt)
    validate(out)
    return out


def surface_sources(s, p):
    out = np.zeros_like(s.n)
    for cell in np.ndindex(s.eta.shape):
        if not np.any(s.h[cell][:3]):
            continue
        z = top_edges(s, cell)
        c, m, x = reconstruct(z, s.n[cell][:3])
        sst = c[0, 0] + m[0, 0] * (z[0] - x[0])
        air = np.broadcast_to(np.asarray(p.T_atm_3d), s.h.shape)[cell][0]
        q = np.asarray(p.Q_heat_2d)[cell] + p.lambda_bulk * (air - sst)
        out[cell][0, 0] = q / (RHO * C_P)
        out[cell][0, 2] = np.asarray(p.tau_x_2d)[cell]
        out[cell][0, 3] = np.asarray(p.tau_y_2d)[cell]
    return out


def advance(s, p, grid, *, forcing_sha, max_subcycles=256):
    """Complete call for explicitly covered small-grid physics, strict rollback.

    Inventory KDK with L half stages. NOT original mode-split time equivalence,
    industrial qualification or a real failure-step repair. Original enabled
    closures listed by unsupported() block before any update.
    """
    before = s.copy()
    try:
        validate(s)
        bind_geometry(s, p, grid)
        validate_dissipative_controls(p)
        if getattr(p, "process_time_scheme", None) != TIME_SCHEME:
            raise ValueError("original stage schedule is not implemented by inventory KDK")
        missing = unsupported(p)
        if missing:
            raise ValueError("enabled processes not implemented: " + ",".join(missing))
        if forcing_sha != s.identity["forcing_sha"]:
            raise ValueError("forcing identity mismatch")
        if parameter_digest(p) != s.identity["parameter_sha"]:
            raise ValueError("parameter/forcing/clock snapshot changed")
        if type(max_subcycles) is not int or not 1 <= max_subcycles <= 256:
            raise ValueError("frozen max_subcycles bound")
        if not isinstance(p.mode_split, (bool, np.bool_)):
            raise ValueError("mode_split boolean required")
        if p.mode_split and (isinstance(p.n_subcyc, (bool, np.bool_))
                             or not isinstance(p.n_subcyc, (int, np.integer))):
            raise ValueError("fast subcycle count must be an integer")
        count = int(p.n_subcyc) if p.mode_split else 1
        if count < 1 or count > max_subcycles or not np.isfinite(p.dt) or p.dt <= 0:
            raise ValueError("bounded timestep/subcycle contract")
        if hasattr(p, "dt_bt") and abs(p.dt_bt * count - p.dt) > 8 * np.finfo(float).eps * p.dt:
            raise ValueError("fast clock does not fill original timestep")
        area = np.asarray(p.dx_2d) * p.dy
        budget = np.zeros(4)
        out = linear(s, p, grid, .5 * p.dt)
        change = np.sum(area[..., None, None] * (out.n - s.n), axis=(0, 1, 2))
        change[:2] = 0.  # diffusion has NO tracer source, independently gated
        budget += change
        sources = np.zeros(4)
        cross = np.zeros_like(s.eta)
        dt = p.dt / count
        for _ in range(count):
            prior = out
            out = kick(out, p, grid, .5 * dt)
            budget += np.sum(area[..., None, None] * (out.n - prior.n), axis=(0, 1, 2))
            out, exchange = drift(out, p, grid, dt)
            cross += exchange * dt
            rate = surface_sources(out, p)
            out.n += dt * rate
            change = np.sum(area[..., None, None] * dt * rate, axis=(0, 1, 2))
            budget += change
            sources += change
            prior = out
            out = kick(out, p, grid, .5 * dt)
            budget += np.sum(area[..., None, None] * (out.n - prior.n), axis=(0, 1, 2))
            validate(out)
        prior = out
        out = linear(out, p, grid, .5 * p.dt)
        change = np.sum(area[..., None, None] * (out.n - prior.n), axis=(0, 1, 2))
        change[:2] = 0.
        budget += change
        residual = np.sum(area[..., None, None] * (out.n - s.n), axis=(0, 1, 2)) - budget
        scale = np.sum(area[..., None, None] * (abs(out.n) + abs(s.n)), axis=(0, 1, 2))
        bound = 8192 * np.finfo(float).eps * scale
        water = np.sum(area[..., None] * (out.h - s.h))
        water_bound = 8192 * np.finfo(float).eps * np.sum(area[..., None] * (out.h + s.h))
        if not np.isfinite(residual).all() or np.any(abs(residual) > bound) or abs(water) > water_bound:
            raise ValueError("candidate inventory/GCL budget")
        out.step += 1
        return out, True, dict(
            inventory_residual=residual.tolist(), inventory_bound=bound.tolist(),
            source_inventory=sources.tolist(), water_residual_m3=float(water),
            cross_band_volume_m=cross.tolist(), subcycles=count,
            time_scheme=TIME_SCHEME, qualification_passed=False,
        )
    except (ValueError, TypeError, FloatingPointError, IndexError) as error:
        return before, False, dict(rejection_reason=str(error), qualification_passed=False)


def save(s, path):
    validate(s)
    np.savez(path, h=s.h, n=s.n, eta=s.eta, ice=s.ice, band_bottom=s.band_bottom,
             identity=np.array(json.dumps(s.identity, sort_keys=True)),
             step=np.array(s.step), version=np.array(1))


def load(path, *, expected_identity, p, grid):
    with np.load(path, allow_pickle=False) as packet:
        if set(packet.files) != {"h", "n", "eta", "ice", "band_bottom", "identity", "step", "version"}:
            raise ValueError("candidate checkpoint keys")
        for name in ("step", "version"):
            if packet[name].shape != () or packet[name].dtype.kind not in "iu":
                raise ValueError("checkpoint integer scalar metadata")
        if int(packet["version"]) != 1 or packet["identity"].shape != () or packet["identity"].dtype.kind != "U":
            raise ValueError("checkpoint version/identity")
        identity = json.loads(str(packet["identity"]))
        if identity != expected_identity:
            raise ValueError("checkpoint provenance mismatch")
        s = State(*(packet[name].copy() for name in ("h", "n", "eta", "ice", "band_bottom")),
                  identity, int(packet["step"]))
    validate(s)
    bind_geometry(s, p, grid)
    validate_dissipative_controls(p)
    if parameter_digest(p) != s.identity["parameter_sha"]:
        raise ValueError("checkpoint parameter snapshot changed")
    return s
