"""Two-column, three-slot FV controls; no ocean solver imports or caller.

N stores h times cell-average T, S, u, v (unit horizontal area).
Zero slot 0 is permitted only for the explicitly merged representation.
"""
from dataclasses import dataclass

import numpy as np

RHO0 = 1025.0
GRAVITY = 9.81
CFL_LIMIT = 0.5
CHECKPOINT_VERSION = 1


@dataclass
class State:
    h: np.ndarray
    n: np.ndarray
    scheme: str = "fixed"
    step: int = 0

    @property
    def eta(self):
        return self.h.sum(axis=1) - 20.0

    def copy(self):
        return State(self.h.copy(), self.n.copy(), self.scheme, self.step)


def valid(state):
    if (isinstance(state.step, (bool, np.bool_))
            or not isinstance(state.step, (int, np.integer)) or state.step < 0):
        return False
    if state.h.shape != (2, 3) or state.n.shape != (2, 3, 4):
        return False
    if state.h.dtype != np.float64 or state.n.dtype != np.float64:
        return False
    if not np.isfinite(state.h).all() or not np.isfinite(state.n).all():
        return False
    if not np.isfinite(state.eta).all() or not np.all(state.h[:, 2] == 10.0):
        return False
    if state.scheme == "merge":
        return (np.all(state.h[:, 0] == 0) and np.all(state.n[:, 0] == 0)
                and np.all(state.h[:, 1:] > 0))
    if not np.all(state.h > 0):
        return False
    if state.scheme == "fixed":
        return np.all(state.h[:, 1] == 7.5)
    if state.scheme == "moving":
        band = state.h[:, :2].sum(axis=1)
        return np.all(np.abs(state.h[:, 0] - .25 * band) <= 64 * np.finfo(float).eps * band)
    return state.scheme == "lagrangian"


def means(state):
    out = np.zeros_like(state.n)
    np.divide(state.n, state.h[..., None], out=out, where=state.h[..., None] > 0)
    return out


def energy(state):
    velocity = means(state)[..., 2:]
    return float(0.5 * RHO0 * np.sum(state.h[..., None] * velocity**2))


def variance(state, field):
    h, c = state.h, means(state)[..., field]
    average = state.n[..., field].sum(axis=1) / h.sum(axis=1)
    return float(np.sum(h * (c - average[:, None])**2))


def edges(state):
    return np.concatenate((state.eta[:, None],
                           state.eta[:, None] - np.cumsum(state.h, axis=1)), axis=1)


def target_h(eta, scheme):
    band = 10.0 + np.asarray(eta)
    if scheme == "merge":
        return np.stack((np.zeros(2), band, np.full(2, 10.0)), axis=1)
    if scheme == "moving":
        return np.stack((0.25 * band, 0.75 * band, np.full(2, 10.0)), axis=1)
    if scheme == "fixed":
        return np.stack((2.5 + np.asarray(eta), np.full(2, 7.5), np.full(2, 10.0)), axis=1)
    raise ValueError("unknown geometry")


def remap(state, scheme, conversion=(True, True)):
    """P0 physical overlap remap; invalid requests return original bytes.

    Synchronous conversion is a deliberate tiny-stencil limitation, not an
    implementation of asynchronous coastal or whole-model topology changes.
    """
    if (conversion != (True, True) or not valid(state)
            or not np.allclose(state.h[:, 2], 10.0, rtol=0, atol=1e-13)):
        return state.copy(), False, "unsupported_geometry_or_one_sided_conversion"
    h = target_h(state.eta, scheme)
    candidate = State(h, np.zeros_like(state.n), scheme, state.step)
    if not valid(candidate):
        return state.copy(), False, "nonpositive_target"
    old, new = edges(state), edges(candidate)
    c = means(state)
    for column in range(2):
        for dst in range(3):
            for src in range(3):
                overlap = max(0.0, min(new[column, dst], old[column, src])
                              - max(new[column, dst + 1], old[column, src + 1]))
                candidate.n[column, dst] += overlap * c[column, src]
    return candidate, True, "accepted"


def advance(state, q, sources, dt=1.0):
    """Prescribed ONE shared face, closed exterior; simultaneous donor update.

    q is the three original face transports, positive left-to-right. Merged
    transport aggregates the top two entries BEFORE donor reconstruction.
    Sources are extensive T/S/u/v increments per second, no water source.
    Moving mode first evolves layer volumes then remaps to the defined target.
    No pressure, wind or vertical mixing solver is included.
    """
    if not valid(state) or not np.isfinite(dt) or dt <= 0:
        return state.copy(), False, "invalid_input"
    q = np.asarray(q, dtype=float)
    sources = np.asarray(sources, dtype=float)
    if q.shape != (3,) or sources.shape != (2, 3, 4):
        return state.copy(), False, "invalid_shape"
    if not np.isfinite(q).all() or not np.isfinite(sources).all() or q[2] != 0:
        return state.copy(), False, "unsupported_flux"
    if state.scheme not in {"moving", "merge"}:
        return state.copy(), False, "convert_before_transport"
    q, sources = q.copy(), sources.copy()
    if state.scheme == "merge":
        q[1] += q[0]
        q[0] = 0.0
        sources[:, 1] += sources[:, 0]
        sources[:, 0] = 0.0
    outgoing = np.stack((np.maximum(q, 0), np.maximum(-q, 0))) * dt
    if np.any(outgoing > CFL_LIMIT * state.h):
        return state.copy(), False, "outflow_cfl"
    donor = np.where((q >= 0)[:, None], means(state)[0], means(state)[1])
    amount = dt * q[:, None] * donor
    h = state.h + np.stack((-dt * q, dt * q))
    n = state.n + np.stack((-amount, amount)) + dt * sources
    intermediate = State(h, n, "merge" if state.scheme == "merge" else "lagrangian", state.step + 1)
    if not valid(intermediate):
        return state.copy(), False, "nonpositive_intermediate"
    result, accepted, reason = remap(intermediate, state.scheme)
    return (result, True, reason) if accepted else (state.copy(), False, reason)


def initial(uniform=False):
    h = target_h(np.zeros(2), "fixed")
    values = np.array([[20., 34., 1., 0.], [15., 35., -1., 0.], [5., 36., 0., 0.]])
    if uniform:
        values[:] = [15., 35., 0.25, -0.1]
    both = np.broadcast_to(values, (2, 3, 4)).copy()
    if not uniform:
        both[1] = [[19., 34.2, .5, .1], [14., 35.2, -.75, .2], [5., 36., .2, 0.]]
    return State(h, h[..., None] * both)


def save(state, path):
    if not valid(state) or state.scheme == "lagrangian":
        raise ValueError("checkpoint requires valid completed target geometry")
    np.savez(path, h=state.h, n=state.n, scheme=np.array(state.scheme), step=np.array(state.step),
             schema_version=np.array(CHECKPOINT_VERSION))


def load(path):
    with np.load(path, allow_pickle=False) as packet:
        if set(packet.files) != {"h", "n", "scheme", "step", "schema_version"}:
            raise ValueError("checkpoint keys do not match schema")
        for key in ("step", "schema_version"):
            value = packet[key]
            if value.shape != () or value.dtype.kind not in "iu":
                raise ValueError(f"{key} must be an integer scalar")
        if packet["schema_version"].item() != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        scheme = packet["scheme"]
        if scheme.shape != () or scheme.dtype.kind != "U":
            raise ValueError("scheme must be a Unicode scalar")
        state = State(packet["h"].copy(), packet["n"].copy(), scheme.item(), packet["step"].item())
    if not valid(state) or state.scheme == "lagrangian":
        raise ValueError("invalid checkpoint state or completed geometry")
    return state


def stationary_profile(h):
    """Exact cell averages of one physical profile T(z)=20+0.5*z.

    The resulting linear density increases with depth. Surface is flat;
    geometry may differ by column. Momentum is zero and salinity is 35.
    """
    state = State(np.asarray(h, dtype=float).copy(), np.zeros((2, 3, 4)), "lagrangian")
    z = edges(state)
    c = np.zeros((2, 3, 4))
    c[..., 0] = 20.0 + 0.25 * (z[:, :-1] + z[:, 1:])
    c[..., 1] = 35.0
    state.n = state.h[..., None] * c
    return state


def pressure_at(state, depth):
    """P0 anomaly integral plus rho0*g*eta at a COMMON physical depth.

    A diagnostic reconstruction, not the production pressure operator.
    Uniform rho0 is cancelled analytically; no density or acceleration clamp.
    """
    z = edges(state)
    c = means(state)
    anomaly = RHO0 * (-2e-4 * (c[..., 0] - 20.0) + 8e-4 * (c[..., 1] - 35.0))
    result = RHO0 * GRAVITY * state.eta
    for column in range(2):
        for cell in range(3):
            length = max(0.0, z[column, cell] - max(depth, z[column, cell + 1]))
            result[column] += GRAVITY * anomaly[column, cell] * length
    return result


def pressure_counterexample():
    """Identical physical linear density, two distinct target grids.

    Fresh exact profile projection removes remap error from this counterexample:
    nonzero force is due to incompatible P0 pressure reconstruction alone.
    Mixed topology is ONLY for this diagnostic, rejected by component conversion.
    """
    results = {}
    for scheme in ("merge", "moving"):
        eta = np.full(2, -2.0)
        h = np.stack((target_h(eta, "fixed")[0], target_h(eta, scheme)[0]))
        profile = stationary_profile(h)
        pressure = pressure_at(profile, -4.0)
        acceleration = -(pressure[1] - pressure[0]) / (RHO0 * 1000.0)
        results[scheme] = float(acceleration)
    return results


def pressure_linear_at(state, depth):
    """Linear-profile-only reconstruction from exact physical cell means.

    Fit anomaly=a*z+b to ACTIVE cell centers and integrate analytically. Reject
    discrete means inconsistent with that fit, not all curved physical profiles.
    Any two distinct means fit a line; merge cannot detect underlying curvature.
    This is a matched control, not general well balancing.
    """
    z, c = edges(state), means(state)
    result = RHO0 * GRAVITY * state.eta
    anomaly = RHO0 * (-2e-4 * (c[..., 0] - 20) + 8e-4 * (c[..., 1] - 35))
    for column in range(2):
        active = state.h[column] > 0
        centers = .5 * (z[column, :-1] + z[column, 1:])[active]
        values = anomaly[column, active]
        if centers.size < 2:
            raise ValueError("linear control requires at least two physical means")
        design = np.column_stack((centers, np.ones_like(centers)))
        slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
        if np.any(np.abs(design @ [slope, intercept] - values) > 64 * np.finfo(float).eps * (1 + np.abs(values))):
            raise ValueError("non-affine discrete means outside linear control")
        eta = state.eta[column]
        result[column] += GRAVITY * (.5 * slope * (eta**2 - depth**2) + intercept * (eta - depth))
    return result


def pressure_scan():
    """Common depths, same analytic stationary physical density in all grids."""
    result = []
    eta = np.full(2, -2.0)
    for depth in (-2., -2.25, -2.5, -3., -4., -5., -7., -10., -15., -20.):
        exact = RHO0 * GRAVITY * -2 + GRAVITY * RHO0 * 1e-4 * (depth**2 - 4) / 2
        row = {"common_depth_m": depth, "analytic_pressure_Pa": exact,
               "p0_error_Pa": {}, "linear_error_Pa": {}, "p0_mixed_acceleration_m_per_s2": {},
               "linear_mixed_acceleration_m_per_s2": {}}
        for scheme in ("fixed", "merge", "moving"):
            h = np.stack((target_h(eta, "fixed")[0], target_h(eta, scheme)[0]))
            profile = stationary_profile(h)
            p0, linear = pressure_at(profile, depth), pressure_linear_at(profile, depth)
            row["p0_error_Pa"][scheme] = float(p0[1] - exact)
            row["linear_error_Pa"][scheme] = float(linear[1] - exact)
            row["p0_mixed_acceleration_m_per_s2"][scheme] = float(-(p0[1] - p0[0]) / (RHO0 * 1000))
            row["linear_mixed_acceleration_m_per_s2"][scheme] = float(-(linear[1] - linear[0]) / (RHO0 * 1000))
        result.append(row)
    return result


def counterflow():
    """Equal column transport does not imply equal resolved layer exchange."""
    state = initial()
    q = np.array([.1, -.1, 0.])
    donor = np.where((q >= 0)[:, None], means(state)[0], means(state)[1])
    resolved = np.sum(q[:, None] * donor, axis=0)
    merged, ok, _ = remap(state, "merge")
    if not ok:
        raise RuntimeError("counterflow conversion failed")
    next_state, ok, _ = advance(merged, q, np.zeros((2, 3, 4)))
    if not ok:
        raise RuntimeError("counterflow update failed")
    return {"layer_q_m3_per_s": q.tolist(), "column_q_m3_per_s": float(q.sum()),
            "resolved_T_S_u_v_exchange_per_s": resolved.tolist(),
            "merged_T_S_u_v_exchange_per_s": (next_state.n[1].sum(axis=0) - merged.n[1].sum(axis=0)).tolist()}


def quadratic_two_mean_control():
    """Exact means of T(z)=20+.01*z² evade a two-mean affine-fit check."""
    state = State(target_h(np.full(2, -2.), "merge"), np.zeros((2, 3, 4)), "merge")
    z = edges(state)
    c = np.zeros_like(state.n)
    c[..., 0] = 20 + .01 * (z[:, :-1]**2 + z[:, :-1] * z[:, 1:] + z[:, 1:]**2) / 3
    c[..., 1] = 35.
    state.n = state.h[..., None] * c
    depth, eta = -4., -2.
    fitted = pressure_linear_at(state, depth)
    exact = RHO0 * GRAVITY * eta - GRAVITY * RHO0 * 2e-6 * (eta**3 - depth**3) / 3
    return {"physical_temperature_profile": "20+.01*z^2", "active_means_per_column": 2,
            "fit_accepted": True, "common_depth_m": depth,
            "analytic_pressure_Pa": exact, "fit_pressure_error_Pa": (fitted - exact).tolist(),
            "qualification_passed": False,
            "limitation": "Two means cannot identify physical curvature; zero intercolumn force does not establish accurate pressure."}
