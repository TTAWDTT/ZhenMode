"""Bounded NumPy moving-stock research slice, distinct from symmetric_fast_v3.

Periodic x, closed y, all wet, flat bottom; three moving top cells and three
fixed deep cells. No production imports/default changes or restart authority.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

RHO = 1025.0
G = 9.81
CP = 3990.0
CONTRACT = "moving_stock_predict_fast_replay_v1"


def roundoff_bound(scale):
    """Fixed 256-epsilon arithmetic envelope, never a truncation allowance."""
    return 256 * np.finfo(float).eps * np.maximum(np.asarray(scale), 1.)


def _finite_scalar(value):
    # Large Python ints would promote NumPy physics arrays to object dtype.
    return (type(value) in (int, float, np.float64)
            and (type(value) is not int or abs(value) <= 2 ** 53)
            and bool(np.isfinite(value)))


@dataclass
class Geometry:
    dx: np.ndarray
    dy: float
    cosine: np.ndarray
    reference_h: np.ndarray

    def __post_init__(self):
        if any(np.ma.isMaskedArray(a) for a in (self.dx, self.cosine, self.reference_h)):
            raise ValueError("masked geometry unsupported")
        self.dx = np.array(self.dx, dtype=float, copy=True)
        self.cosine = np.array(self.cosine, dtype=float, copy=True)
        self.reference_h = np.array(self.reference_h, dtype=float, copy=True)
        if (self.dx.shape != (8, 4) or self.cosine.shape != (4,)
                or self.reference_h.shape != (6,) or not _finite_scalar(self.dy)
                or self.dy <= 0 or not all(np.isfinite(a).all() and (a > 0).all()
                                          for a in (self.dx, self.cosine, self.reference_h))):
            raise ValueError("only finite positive 8x4x6 flat all-wet geometry")
        self.area = self.dx * self.dy
        self.bottom = -float(self.reference_h.sum())
        self.band_bottom = -float(self.reference_h[:3].sum())
        self.fractions = self.reference_h[:3] / -self.band_bottom

    def thickness(self, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != self.dx.shape or not np.isfinite(eta).all() or (eta <= self.band_bottom).any():
            raise ValueError("invalid or exhausted moving band")
        h = np.broadcast_to(self.reference_h, eta.shape + (6,)).copy()
        h[..., :3] = (eta - self.band_bottom)[..., None] * self.fractions
        return h


@dataclass
class State:
    h: np.ndarray
    n: np.ndarray  # IT, IS, Mu, Mv per horizontal area
    bottom: float
    step: int = 0

    @property
    def eta(self):
        return self.bottom + self.h.sum(-1)

    def copy(self):
        return State(deepcopy(self.h), deepcopy(self.n), self.bottom, self.step)


@dataclass(frozen=True)
class Parameters:
    dt: float = 30.0
    nu_h: float = 0.0
    nu_v: float = 0.0
    kappa_h: float = 0.0
    kappa_v: float = 0.0
    kappa_conv: float = 0.0
    r_bot: float = 0.0
    lambda_bulk: float = 0.0
    air_temperature: float = 17.0
    tau_x: float = 0.0
    tau_y: float = 0.0
    coriolis: float = 0.0
    fct: bool = True
    kappa_bi: float = 0.0
    momentum_filter: bool = False


@dataclass(frozen=True)
class Face:
    left: tuple
    right: tuple
    axis: int
    conductance: float  # C row: half conductance on each incident velocity
    flux: float
    pressure_jump: float
    distance: float


def validate(state, grid, *, canonical=True):
    if not isinstance(state, State):
        raise ValueError("State required")
    _validate_grid(grid)
    if (not isinstance(state.h, np.ndarray) or not isinstance(state.n, np.ndarray)
            or np.ma.isMaskedArray(state.h) or np.ma.isMaskedArray(state.n)):
        raise ValueError("unmasked NumPy state arrays required")
    if (state.h.shape != (8, 4, 6) or state.n.shape != (8, 4, 6, 4)
            or state.h.dtype != np.float64 or state.n.dtype != np.float64
            or not np.isfinite(state.h).all() or not np.isfinite(state.n).all()
            or (state.h <= 0).any() or not _finite_scalar(state.bottom) or state.bottom != grid.bottom
            or type(state.step) is not int or state.step < 0):
        raise ValueError("invalid finite float64 full-column state")
    if canonical:
        expected = grid.thickness(state.eta)
        if np.any(abs(state.h - expected) > roundoff_bound(abs(state.h) + abs(expected))):
            raise ValueError("state is outside moving-top fixed-deep geometry")
    specific = state.n[..., :2] / state.h[..., None]
    tolerance = roundoff_bound(abs(specific) + 50)
    if np.any(specific < np.array([-5., 0.]) - tolerance) or np.any(specific > np.array([45., 50.]) + tolerance):
        raise ValueError("tracer outside declared physical bounds")


def _validate_grid(grid):
    if not isinstance(grid, Geometry):
        raise ValueError("Geometry required")
    arrays = (grid.dx, grid.cosine, grid.reference_h, grid.area, grid.fractions)
    if any(not isinstance(a, np.ndarray) or np.ma.isMaskedArray(a)
           or a.dtype != np.float64 or not np.isfinite(a).all() or (a <= 0).any()
           for a in arrays):
        raise ValueError("invalid or masked consumed geometry")
    if (grid.dx.shape != (8, 4) or grid.cosine.shape != (4,) or grid.reference_h.shape != (6,)
            or grid.area.shape != (8, 4) or grid.fractions.shape != (3,)
            or not _finite_scalar(grid.dy) or grid.dy <= 0
            or not _finite_scalar(grid.bottom) or not _finite_scalar(grid.band_bottom)
            or not np.array_equal(grid.area, grid.dx * grid.dy)
            or grid.bottom != -float(grid.reference_h.sum())
            or grid.band_bottom != -float(grid.reference_h[:3].sum())
            or not np.array_equal(grid.fractions, grid.reference_h[:3] / -grid.band_bottom)):
        raise ValueError("inconsistent consumed geometry")


def interfaces(state):
    z = np.empty(state.h.shape[:-1] + (7,))
    z[..., 0] = state.eta
    z[..., 1:] = state.eta[..., None] - np.cumsum(state.h, axis=-1)
    z[..., -1] = state.bottom
    return z


def density_reconstruction(state):
    """Mean-preserving limited P1 density in physical z, not index space."""
    z = interfaces(state)
    center = .5 * (z[..., :-1] + z[..., 1:])
    specific = state.n[..., :2] / state.h[..., None]
    anomaly = RHO * (-2e-4 * (specific[..., 0] - 10) + 8e-4 * (specific[..., 1] - 35))
    secant = np.diff(anomaly, axis=-1) / np.diff(center, axis=-1)
    slope = np.zeros_like(anomaly)
    a, b = secant[..., :-1], secant[..., 1:]
    slope[..., 1:-1] = np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0)
    slope[..., 0], slope[..., -1] = secant[..., 0], secant[..., -1]
    return z, center, anomaly, slope


def pressure_at(profile, column, depth):
    z, center, anomaly, slope = profile
    upper = z[column][:-1]
    lower = np.maximum(depth, z[column][1:])
    width = np.maximum(0., upper - lower)
    integral = width * (anomaly[column] + slope[column] * (.5 * (upper + lower) - center[column]))
    return G * (RHO * (z[column][0] - depth) + integral.sum())


def faces(state, grid):
    """One common-depth P0 C/Q and exact two-Gauss segment pressure average."""
    validate(state, grid)
    profile = density_reconstruction(state)
    z = profile[0]
    velocity = state.n[..., 2:] / (RHO * state.h[..., None])
    result = []
    for i in range(8):
        for j in range(4):
            for axis, neighbor in ((0, ((i + 1) % 8, j)), (1, (i, j + 1))):
                if neighbor[1] == 4:
                    continue
                column = (i, j)
                ceiling = min(z[column][0], z[neighbor][0])
                breaks = np.unique(np.r_[z[column], z[neighbor], ceiling])
                breaks = breaks[(breaks >= state.bottom) & (breaks <= ceiling)]
                length = grid.dy if axis == 0 else .5 * (grid.dx[column] + grid.dx[neighbor])
                distance = .5 * (grid.dx[column] + grid.dx[neighbor]) if axis == 0 else grid.dy
                for lo, hi in zip(breaks[:-1], breaks[1:]):
                    if hi - lo <= roundoff_bound(abs(hi) + abs(lo)):
                        continue
                    midpoint = .5 * (lo + hi)
                    left_layer = int(np.searchsorted(-z[column], -midpoint) - 1)
                    right_layer = int(np.searchsorted(-z[neighbor], -midpoint) - 1)
                    left, right = column + (left_layer,), neighbor + (right_layer,)
                    conductance = length * (hi - lo)
                    flux = conductance * .5 * (velocity[left][axis] + velocity[right][axis])
                    offset = (hi - lo) / (2 * np.sqrt(3.))
                    jumps = [pressure_at(profile, neighbor, midpoint + shift)
                             - pressure_at(profile, column, midpoint + shift)
                             for shift in (-offset, offset)]
                    result.append(Face(left, right, axis, conductance, float(flux),
                                       float(.5 * sum(jumps)), float(distance)))
    return result


def ale_rates(state, grid, face_list):
    divergence = np.zeros_like(state.h)
    for face in face_list:
        divergence[face.left] += face.flux / grid.area[face.left[:2]]
        divergence[face.right] -= face.flux / grid.area[face.right[:2]]
    eta_rate = -divergence.sum(-1)
    relative = np.zeros(state.h.shape[:-1] + (7,))
    for k in range(5, 2, -1):
        relative[..., k] = relative[..., k + 1] + divergence[..., k]
    top = np.zeros(state.h.shape[:-1])
    for k in range(3):
        next_top = top - divergence[..., k] - grid.fractions[k] * eta_rate
        if k < 2:
            relative[..., k + 1] = next_top
        top = next_top
    scale = abs(divergence).sum(-1) + abs(eta_rate)
    if np.any(abs(top - relative[..., 3]) > roundoff_bound(scale)):
        raise ValueError("top/deep ALE band-face mismatch")
    return divergence, eta_rate, relative


def totals(state, grid):
    area = grid.area[..., None]
    return float(np.sum(state.h * area)), np.sum(state.n * area[..., None], axis=(0, 1, 2))


def kinetic(state, grid):
    terms = np.sum(state.n[..., 2:] ** 2, axis=-1) / (2 * RHO * state.h)
    return float(np.sum(grid.area[..., None] * terms))


def potential(state, grid):
    """rho0 free-surface PE plus rho-prime gravity PE, without double counting."""
    z, center, anomaly, slope = density_reconstruction(state)
    gravity = state.h * anomaly * center + slope * state.h ** 3 / 12
    return float(G * np.sum(grid.area * (.5 * RHO * state.eta ** 2 + gravity.sum(-1))))


def _transport_edges(state, grid, face_list, relative):
    edges = [(f.left, f.right, f.flux) for f in face_list]
    for i in range(8):
        for j in range(4):
            for k in range(1, 6):
                edges.append(((i, j, k - 1), (i, j, k), relative[i, j, k] * grid.area[i, j]))
    return edges


def _euler_transport(state, grid, dt, fct):
    face_list = faces(state, grid)
    _, eta_rate, relative = ale_rates(state, grid, face_list)
    out = state.copy()
    out.h = grid.thickness(state.eta + dt * eta_rate)
    area = np.broadcast_to(grid.area[..., None], state.h.shape)
    specific = state.n / state.h[..., None]
    edges = _transport_edges(state, grid, face_list, relative)
    outgoing = np.zeros_like(state.h)
    low_stock = state.n * area[..., None]
    antidiffusion = []
    for left, right, flux in edges:
        donor = left if flux >= 0 else right
        outgoing[donor] += abs(flux) * dt
        transported = dt * flux * specific[donor]
        low_stock[left] -= transported
        low_stock[right] += transported
        correction = dt * flux * (.5 * (specific[left] + specific[right]) - specific[donor])
        antidiffusion.append(correction)
    if np.any(outgoing > .8 * state.h * area):
        raise ValueError("joint water/stock transport CFL exceeds 0.8")
    if fct:
        minimum = specific[..., :2].min(axis=(0, 1, 2))
        maximum = specific[..., :2].max(axis=(0, 1, 2))
        lower = out.h[..., None] * area[..., None] * minimum
        upper = out.h[..., None] * area[..., None] * maximum
        positive, negative = np.zeros_like(lower), np.zeros_like(lower)
        for (left, right, _), correction in zip(edges, antidiffusion):
            positive[left] += np.maximum(-correction[:2], 0)
            negative[left] += np.maximum(correction[:2], 0)
            positive[right] += np.maximum(correction[:2], 0)
            negative[right] += np.maximum(-correction[:2], 0)
        allow_positive = np.maximum(0., upper - low_stock[..., :2])
        allow_negative = np.maximum(0., low_stock[..., :2] - lower)
        ratio_positive, ratio_negative = np.ones_like(positive), np.ones_like(negative)
        np.divide(np.minimum(allow_positive, positive), positive,
                  out=ratio_positive, where=positive > 0)
        np.divide(np.minimum(allow_negative, negative), negative,
                  out=ratio_negative, where=negative > 0)
        for (left, right, _), correction in zip(edges, antidiffusion):
            left_ratio = np.where(correction[:2] >= 0, ratio_negative[left], ratio_positive[left])
            right_ratio = np.where(correction[:2] >= 0, ratio_positive[right], ratio_negative[right])
            theta = float(min(left_ratio.min(), right_ratio.min()))
            low_stock[left] -= theta * correction
            low_stock[right] += theta * correction
    out.n = low_stock / area[..., None]
    validate(out, grid)
    return out, float(np.max(abs(relative[..., 3])))


def transport(state, grid, dt, *, fct=True):
    """Joint h/IT/IS/M SSP-Heun with shared horizontal Q and relative ALE R."""
    first, cross_first = _euler_transport(state, grid, dt, fct)
    second, cross_second = _euler_transport(first, grid, dt, fct)
    out = State(.5 * (state.h + second.h), .5 * (state.n + second.n), state.bottom, state.step)
    validate(out, grid)
    before_water, before_stock = totals(state, grid)
    after_water, after_stock = totals(out, grid)
    return out, {"cross_band_max_m_s": max(cross_first, cross_second),
                 "water_roundoff_ratio": float(abs(after_water - before_water) / roundoff_bound(before_water)),
                 "stock_residual": (after_stock - before_stock).tolist()}


def pressure_kick(state, grid, dt):
    """Fixed-mass -C.T pressure kick; work measured on the true M midpoint."""
    face_list = faces(state, grid)
    out = state.copy()
    impulse = np.zeros_like(state.n[..., 2:])
    for face in face_list:
        force = -.5 * face.conductance * face.pressure_jump
        impulse[face.left][face.axis] += dt * force / grid.area[face.left[:2]]
        impulse[face.right][face.axis] += dt * force / grid.area[face.right[:2]]
    out.n[..., 2:] += impulse
    midpoint = (state.n[..., 2:] + out.n[..., 2:]) / (2 * RHO * state.h[..., None])
    work_terms = [-dt * face.pressure_jump * face.conductance
                  * .5 * (midpoint[face.left][face.axis] + midpoint[face.right][face.axis])
                  for face in face_list]
    work = float(sum(work_terms))
    before_kinetic, after_kinetic = kinetic(state, grid), kinetic(out, grid)
    change = after_kinetic - before_kinetic
    bound = float(roundoff_bound(abs(before_kinetic) + abs(after_kinetic) + sum(map(abs, work_terms))))
    expected = np.zeros(4)
    for face in face_list:
        expected[face.axis + 2] -= dt * face.conductance * face.pressure_jump
    return out, {"pressure_work_J": work, "kinetic_change_J": change,
                 "pressure_work_bound_J": bound, "pressure_work_roundoff_ratio": abs(change - work) / bound,
                 "external_stock": expected.tolist()}


def remap(state, grid):
    """P0 overlap stocks, including M_old/h_old, same eta/bottom/water domain."""
    validate(state, grid, canonical=False)
    target = grid.thickness(state.eta)
    if not np.array_equal(state.h[..., 3:], target[..., 3:]):
        raise ValueError("remap cannot change fixed deep geometry")
    out = State(target, state.n.copy(), state.bottom, state.step)
    out.n[..., :3, :] = 0.
    old_z, new_z = interfaces(state), interfaces(out)
    for k in range(3):
        for source in range(3):
            overlap = np.maximum(0., np.minimum(new_z[..., k], old_z[..., source])
                                 - np.maximum(new_z[..., k + 1], old_z[..., source + 1]))
            out.n[..., k, :] += overlap[..., None] * state.n[..., source, :] / state.h[..., source, None]
    validate(out, grid)
    before, after = kinetic(state, grid), kinetic(out, grid)
    return out, {"kinetic_change_J": after - before,
                 "kinetic_bound_J": float(roundoff_bound(abs(before) + abs(after)))}


def _diffusion_edges(state, grid, params, *, capacity=False):
    # Horizontal common-depth and vertical fluxes exchange STOCK per time.
    specific = state.n / state.h[..., None]
    edges = []
    for face in faces(state, grid):
        coefficients = np.array([params.kappa_h, params.kappa_h, params.nu_h, params.nu_h])
        edges.append((face.left, face.right, coefficients * face.conductance / face.distance))
    density = density_reconstruction(state)[2]
    for i in range(8):
        for j in range(4):
            for k in range(5):
                left, right = (i, j, k), (i, j, k + 1)
                convective = params.kappa_conv if capacity or density[left] > density[right] else 0.
                coefficients = np.array([params.kappa_v + convective] * 2 + [params.nu_v] * 2)
                distance = .5 * (state.h[left] + state.h[right])
                edges.append((left, right, coefficients * grid.area[i, j] / distance))
    return specific, edges


def mixing(state, grid, params, duration):
    """Current-mass conservative stock-flux diffusion with explicit capacity."""
    validate(state, grid)
    if not any((params.nu_h, params.nu_v, params.kappa_h, params.kappa_v, params.kappa_conv)):
        return state.copy(), {"diffusion_subcycles": 1, "external_stock": [0.] * 4}
    # A stable interface may turn convective: plan against every possible edge.
    _, edges = _diffusion_edges(state, grid, params, capacity=True)
    rates = np.zeros_like(state.n)
    for left, right, conductance in edges:
        rates[left] += conductance / (state.h[left] * grid.area[left[:2]])
        rates[right] += conductance / (state.h[right] * grid.area[right[:2]])
    required = duration * float(rates.max()) / .4
    if not np.isfinite(required) or required > 256:
        raise ValueError("current-mass diffusion subcycle capacity exceeded")
    count = max(1, int(np.ceil(required)))
    out = state.copy()
    for _ in range(count):
        # Recompute convection from the accepted tracer state; h stays fixed.
        specific, edges = _diffusion_edges(out, grid, params)
        change = np.zeros_like(out.n)
        for left, right, conductance in edges:
            flux = conductance * (specific[right] - specific[left]) * (duration / count)
            change[left] += flux / grid.area[left[:2]]
            change[right] -= flux / grid.area[right[:2]]
        out.n += change
        validate(out, grid)
    return out, {"diffusion_subcycles": count, "external_stock": [0.] * 4}


def sources(state, grid, params, duration):
    """Exact fixed-mass rotation/bottom drag/bulk heat and stock wind impulse."""
    out = state.copy()
    expected = np.zeros(4)
    angle = params.coriolis * duration
    mu, mv = out.n[..., 2].copy(), out.n[..., 3].copy()
    out.n[..., 2] = np.cos(angle) * mu + np.sin(angle) * mv
    out.n[..., 3] = -np.sin(angle) * mu + np.cos(angle) * mv
    expected[2] += np.sum(grid.area[..., None] * ((np.cos(angle) - 1) * mu + np.sin(angle) * mv))
    expected[3] += np.sum(grid.area[..., None] * (-np.sin(angle) * mu + (np.cos(angle) - 1) * mv))
    drag = np.expm1(-params.r_bot * duration)
    expected[2:] += np.sum(grid.area[..., None] * out.n[..., -1, 2:] * drag, axis=(0, 1))
    out.n[..., -1, 2:] *= 1 + drag
    wind = duration * np.array([params.tau_x, params.tau_y])
    out.n[..., 0, 2:] += wind
    expected[2:] += grid.area.sum() * wind
    relaxation = -np.expm1(-params.lambda_bulk * duration / (RHO * CP * out.h[..., 0]))
    heat = (out.h[..., 0] * params.air_temperature - out.n[..., 0, 0]) * relaxation
    out.n[..., 0, 0] += heat
    expected[0] = np.sum(grid.area * heat)
    validate(out, grid)
    return out, {"external_stock": expected.tolist(), "bulk_heat_J": float(expected[0] * RHO * CP)}


def _slow(state, grid, params, duration):
    mixed, mix_report = mixing(state, grid, params, duration)
    # Independent oracle reads the accepted pre-source state and declared laws,
    # never the source's observed change or its claimed external-stock receipt.
    mu, mv = mixed.n[..., 2], mixed.n[..., 3]
    cosine, sine = np.cos(params.coriolis * duration), np.sin(params.coriolis * duration)
    rotated = np.stack((cosine * mu + sine * mv, -sine * mu + cosine * mv), axis=-1)
    delta = rotated - mixed.n[..., 2:]
    delta[..., -1, :] += rotated[..., -1, :] * np.expm1(-params.r_bot * duration)
    expected = np.zeros(4)
    expected[2:] = np.sum(delta * grid.area[..., None, None], axis=(0, 1, 2))
    expected[2:] += grid.area.sum() * duration * np.array([params.tau_x, params.tau_y])
    temperature = mixed.n[..., 0, 0] / mixed.h[..., 0]
    heat_fraction = -np.expm1(-params.lambda_bulk * duration / (RHO * CP * mixed.h[..., 0]))
    expected[0] = np.sum(grid.area * mixed.h[..., 0] * (params.air_temperature - temperature) * heat_fraction)
    out, source_report = sources(mixed, grid, params, duration)
    claimed = _receipt_stock(source_report, ('bulk_heat_J',))
    validate(out, grid)
    _, old_stock = totals(mixed, grid)
    _, new_stock = totals(out, grid)
    scale = np.sum((abs(out.n) + abs(mixed.n)) * grid.area[..., None, None], axis=(0, 1, 2)) + abs(expected)
    if (not _finite_report(source_report) or not _finite_report(mix_report)
            or not np.array_equal(out.h, mixed.h)
            or np.any(abs(new_stock - old_stock - expected) > roundoff_bound(scale))
            or np.any(abs(claimed - expected) > roundoff_bound(scale))):
        raise ValueError("independent pre-source law/receipt budget failed")
    return out, {**mix_report, **source_report, "external_stock": expected.tolist()}


def _audited_pressure(state, grid, duration):
    # Derive pressure impulse from geometry/P1 p at entry, before the kick runs.
    expected = np.zeros(4)
    face_list = faces(state, grid)
    for face in face_list:
        expected[face.axis + 2] -= duration * face.conductance * face.pressure_jump
    out, receipt = pressure_kick(state, grid, duration)
    claimed = _receipt_stock(receipt, ('pressure_work_J', 'kinetic_change_J',
                                      'pressure_work_bound_J', 'pressure_work_roundoff_ratio'))
    validate(out, grid)
    _, old_stock = totals(state, grid)
    _, new_stock = totals(out, grid)
    scale = np.sum((abs(out.n) + abs(state.n)) * grid.area[..., None, None], axis=(0, 1, 2)) + abs(expected)
    midpoint = (out.n[..., 2:] + state.n[..., 2:]) / (2 * RHO * state.h[..., None])
    work_terms = [-duration * face.pressure_jump * face.conductance
                  * (midpoint[face.left][face.axis] + midpoint[face.right][face.axis]) / 2
                  for face in face_list]
    actual_work = float(sum(work_terms))
    before_energy, after_energy = kinetic(state, grid), kinetic(out, grid)
    actual_change = after_energy - before_energy
    work_bound = float(roundoff_bound(abs(before_energy) + abs(after_energy) + sum(map(abs, work_terms))))
    if (not _finite_report(receipt) or not np.array_equal(out.h, state.h)
            or np.any(abs(new_stock - old_stock - expected) > roundoff_bound(scale))
            or np.any(abs(claimed - expected) > roundoff_bound(scale))
            or receipt['pressure_work_bound_J'] <= 0 or receipt['pressure_work_roundoff_ratio'] < 0
            or abs(actual_change - actual_work) > work_bound
            or abs(receipt['pressure_work_J'] - actual_work) > work_bound
            or abs(receipt['kinetic_change_J'] - actual_change) > work_bound):
        raise ValueError("independent fixed-mass pressure work/impulse failed")
    return out, {**receipt, "external_stock": expected.tolist(), "pressure_work_J": actual_work,
                 "kinetic_change_J": actual_change, "pressure_work_bound_J": work_bound,
                 "pressure_work_roundoff_ratio": abs(actual_change - actual_work) / work_bound}


def _fast(state, grid, params, duration):
    first, first_report = _audited_pressure(state, grid, duration / 2)
    moved, transport_report = transport(first, grid, duration, fct=params.fct)
    if not _finite_report(transport_report):
        raise ValueError("nonfinite joint transport receipt")
    out, second_report = _audited_pressure(moved, grid, duration / 2)
    work = first_report['pressure_work_J'] + second_report['pressure_work_J']
    expected = np.array(first_report['external_stock']) + second_report['external_stock']
    return out, {**transport_report, "pressure_work_J": work,
                 "pressure_work_roundoff_ratio": max(first_report['pressure_work_roundoff_ratio'],
                                                       second_report['pressure_work_roundoff_ratio']),
                 "external_stock": expected.tolist()}


def _preflight(state, grid, params):
    if not isinstance(params, Parameters):
        raise ValueError("Parameters required")
    validate(state, grid)
    if type(params.fct) is not bool or type(params.momentum_filter) is not bool:
        raise ValueError("boolean process switches required")
    numbers = [v for key, v in vars(params).items() if key not in ('fct', 'momentum_filter')]
    if not all(_finite_scalar(v) for v in numbers) or params.dt <= 0:
        raise ValueError("nonfinite parameters or nonpositive dt")
    for name in ('nu_h', 'nu_v', 'kappa_h', 'kappa_v', 'kappa_conv', 'r_bot', 'lambda_bulk', 'kappa_bi'):
        if getattr(params, name) < 0:
            raise ValueError("negative coefficient:" + name)
    missing = []
    if params.kappa_bi:
        missing.append('biharmonic')
    if params.momentum_filter:
        missing.append('momentum_filter')
    if missing:
        raise ValueError("unadapted active processes:" + ','.join(missing))


def _finite_report(value):
    if isinstance(value, dict):
        return all(_finite_report(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_report(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in 'biuf' and bool(np.isfinite(value).all())
    return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))


def _receipt_stock(receipt, numeric_fields):
    if (not isinstance(receipt, dict) or 'external_stock' not in receipt
            or not _finite_report(receipt)
            or any(key not in receipt or not _finite_scalar(receipt[key])
                   for key in numeric_fields)):
        raise ValueError("invalid finite numeric source/pressure receipt")
    value = receipt['external_stock']
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.shape != (4,)) or len(value) != 4
            or any(not _finite_scalar(v) for v in value)):
        raise ValueError("invalid source/pressure stock receipt")
    return np.asarray(value, dtype=float)


@np.errstate(over='raise', invalid='raise', divide='raise')
def advance(state, grid, params):
    """Predict/discard, twelve all-stock commits, replay slow state once, rollback.

    The predictor supplies diagnostic anticipated slow response only. It never
    changes accepted fast stocks. Replay applies the slow second half to the
    unique accepted endpoint; it does not add predictor or fast-end momentum.
    Active diffusion uses a first-order subcycled update; order is measured
    separately for each declared configuration, never inferred from a stage name.
    """
    report = {"contract": CONTRACT, "qualification_passed": False,
              "original_configuration_covered": False,
              "missing_adapters": ['biharmonic', 'momentum_filter', 'coast', 'variable_bathymetry',
                                   'original_process_clocks', 'total_PE_ALE_compatibility'],
              "fast_steps": [], "committed_fast_steps": [], "executed_fast_steps": [],
              "stages": [], "rejection_reason": None}
    try:
        _preflight(state, grid, params)
        before_water, before_stock = totals(state, grid)
        initial_energy = kinetic(state, grid) + potential(state, grid)
        initial, initial_report = _slow(state, grid, params, params.dt / 2)
        report['stages'].append('linear_source_first')
        predictor, _ = _fast(initial, grid, params, params.dt)
        predictor, _ = _slow(predictor, grid, params, params.dt / 2)
        report['predictor_eta_change_max_m'] = float(np.max(abs(predictor.eta - initial.eta)))
        report['stages'].append('predictor_discarded')
        accepted = initial
        expected = np.array(initial_report['external_stock'])
        for index in range(12):
            candidate, fast_report = _fast(accepted, grid, params, params.dt / 12)
            validate(candidate, grid)
            if (not _finite_report(fast_report) or fast_report['water_roundoff_ratio'] > 1
                    or fast_report['pressure_work_roundoff_ratio'] > 1):
                raise ValueError("fast fixed-mass work/water identity failed")
            accepted = candidate  # Exactly one all-stock accepted commit per substep.
            expected += fast_report['external_stock']
            report['fast_steps'].append(fast_report)
            report['executed_fast_steps'].append(index + 1)
        report['stages'].append('twelve_joint_fast_steps')
        # Match is diagnostic only; no layer/column Q correction is performed.
        face_list = faces(accepted, grid)
        layer_q = {}
        segment_q = {}
        for face in face_list:
            key = (face.left[:2], face.right[:2], face.axis)
            layer_q.setdefault(key, np.zeros(6))[face.left[-1]] += face.flux
            segment_q.setdefault(key, []).append(face.flux)
        column_q = {key: float(np.sum(values)) for key, values in segment_q.items()}
        mismatches = {key: abs(float(values.sum()) - column_q[key]) for key, values in layer_q.items()}
        if any(mismatches[key] > roundoff_bound(sum(map(abs, segment_q[key]))) for key in mismatches):
            raise ValueError("diagnostic layer/column Q match failed; no repair allowed")
        report['column_Q_max_m3_s'] = max(map(abs, column_q.values()))
        report['column_Q_match_residual_m3_s'] = max(mismatches.values())
        report['stages'].append('column_Q_diagnostic')
        replayed, replay_report = _slow(accepted, grid, params, params.dt / 2)
        expected += replay_report['external_stock']
        report['stages'].append('accepted_slow_replay')
        replayed.step = state.step + 1
        validate(replayed, grid)
        after_water, after_stock = totals(replayed, grid)
        water_residual = after_water - before_water
        stock_residual = after_stock - before_stock - expected
        stock_scale = np.sum(abs(state.n) * grid.area[..., None, None], axis=(0, 1, 2))
        stock_scale += np.sum(abs(replayed.n) * grid.area[..., None, None], axis=(0, 1, 2)) + abs(expected)
        water_ratio = float(abs(water_residual) / roundoff_bound(before_water))
        stock_ratio = abs(stock_residual) / roundoff_bound(stock_scale)
        report.update(independent_water_residual_m3=float(water_residual),
                      independent_water_roundoff_ratio=water_ratio,
                      independent_stock_residual=stock_residual.tolist(),
                      independent_stock_roundoff_ratio=stock_ratio.tolist(),
                      expected_external_stock=expected.tolist(),
                      cross_band_max_m_s=max(r['cross_band_max_m_s'] for r in report['fast_steps']),
                      diffusion_subcycles=[initial_report['diffusion_subcycles'], replay_report['diffusion_subcycles']],
                      bulk_heat_J=initial_report['bulk_heat_J'] + replay_report['bulk_heat_J'],
                      total_energy_change_J=kinetic(replayed, grid) + potential(replayed, grid) - initial_energy,
                      total_PE_ALE_residual_status='unproved; energy change is not a numerical loss',
                      temporal_order_status='unmeasured; no second-order claim')
        report['pressure_work_sum_J'] = sum(r['pressure_work_J'] for r in report['fast_steps'])
        report['unclosed_total_energy_remainder_J'] = report['total_energy_change_J'] - report['pressure_work_sum_J']
        report['unclosed_remainder_includes'] = ['transport/ALE', 'mixing', 'sources; not classified as loss']
        if not _finite_report(report) or water_ratio > 1 or np.any(stock_ratio > 1):
            raise ValueError("independent water/TS/impulse budget failed")
        report['committed_fast_steps'] = list(report['executed_fast_steps'])
        return replayed, True, report
    except (ValueError, FloatingPointError) as error:
        report['rejection_reason'] = str(error)
        report['committed_fast_steps'] = []
        return deepcopy(state), False, report
