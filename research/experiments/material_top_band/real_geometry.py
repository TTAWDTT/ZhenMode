"""Read-only nodal-dual inventory bridge and candidate P0 shared faces.

This is a new stock-mean interpretation, not recovery of the historical FD
sample profile. No time step, pressure kick, terrain partial cell or filling
of the unsampled interval below the last wet node is performed here.
"""

import math
from dataclasses import dataclass

import numpy as np


def _array(value, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{name}: masked data requires explicit support provenance')
    raw = np.asarray(value)
    allowed = 'bfiu' if name == 'mask' else 'fiu'
    if raw.dtype.kind not in allowed or not np.isfinite(raw).all():
        raise ValueError(f'{name}: finite real numeric array required')
    if raw.dtype.kind in 'iu' and raw.size and (int(raw.min()) < -(2**53) or int(raw.max()) > 2**53):
        raise ValueError(f'{name}: exactly representable integer array required')
    return np.array(raw, dtype=np.float64, copy=True)


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name}: finite scalar required')
    if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:
        raise ValueError(f'{name}: exactly representable scalar required')
    scalar = float(value)
    if not math.isfinite(scalar) or (positive and scalar <= 0):
        raise ValueError(f'{name}: finite positive scalar required')
    return scalar


def _readonly(value):
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class ColumnStocks:
    """Per-area water and IT/IS/Mu/Mv on all original vertical slots."""

    active_layers: np.ndarray
    wet_mask: np.ndarray
    eta: np.ndarray
    bottom: np.ndarray
    band_bottom: np.ndarray
    interfaces: np.ndarray
    h: np.ndarray
    stocks: np.ndarray


@dataclass(frozen=True)
class InventoryBridge:
    original: ColumnStocks
    candidate: ColumnStocks
    rho0: float
    sample_lower_z: np.ndarray
    sample_upper_z: np.ndarray
    node0_above_surface: np.ndarray
    unsampled_bottom_m: np.ndarray
    reference_kinetic_J_m2: np.ndarray
    original_actual_kinetic_J_m2: np.ndarray
    candidate_kinetic_J_m2: np.ndarray
    reference_to_actual_kinetic_J_m2: np.ndarray
    remap_kinetic_change_J_m2: np.ndarray
    stock_roundoff_ratio: np.ndarray
    contract: str = 'nodal_dual_stock_mean_bridge_v1'
    original_physical_profile_recovered: bool = False


@dataclass(frozen=True)
class FaceSegment:
    upper_z_m: float
    lower_z_m: float
    left_layer: int
    right_layer: int
    Q_m3_s: float
    original_profile_supported: bool


@dataclass(frozen=True)
class SharedFace:
    common_upper_z_m: float | None
    common_lower_z_m: float | None
    segments: tuple[FaceSegment, ...]
    total_Q_m3_s: float
    interpretation: str = 'candidate_stock_mean_p0_v1'


def _kinetic(stocks, h, rho0):
    energy = np.zeros_like(h)
    np.divide(np.sum(stocks[..., 2:]**2, axis=-1), 2. * rho0 * h,
              out=energy, where=h > 0)
    return energy.sum(axis=-1)


def bridge_nodal_dual(depth, reference_weights, wet_mask, eta, values, *,
                      column_geometry, rho0=1025., band_layers=3):
    """Preserve every accepted inventory and remap only the complete top band.

    Original IT/IS use actual h; original M uses reference nodal-dual h.
    P0 overlap acts on stocks/actual h, so converting M to actual velocity
    exposes a kinetic cost before any remap. Deep stocks and widths are copied
    exactly. Dry source slots are not interpreted as water or inventories.
    """
    if column_geometry != 'nodal_dual_v1':
        raise ValueError('only nodal_dual_v1 geometry is admissible')
    if isinstance(band_layers, (bool, np.bool_)) or not isinstance(band_layers, (int, np.integer)) or not 1 <= band_layers <= 3:
        raise ValueError('band_layers must be one, two or three complete cells')
    rho0 = _scalar(rho0, 'rho0', positive=True)
    depth, weights, mask, eta, values = (
        _array(value, name) for value, name in
        [(depth, 'depth'), (reference_weights, 'weights'), (wet_mask, 'mask'),
         (eta, 'eta'), (values, 'values')])
    if depth.ndim != 1 or depth.size < 2 or depth[0] != 0 or np.any(np.diff(depth) <= 0):
        raise ValueError('finite ordered depth nodes beginning at zero required')
    if weights.shape != depth.shape or mask.ndim < 2 or mask.shape[-1] != depth.size or eta.shape != mask.shape[:-1] or values.shape != mask.shape + (4,):
        raise ValueError('nodal inventory shape mismatch')
    band_layers = min(band_layers, depth.size)
    edges = np.r_[0., .5 * (depth[:-1] + depth[1:]), depth[-1]]
    if not np.array_equal(weights, np.diff(edges)):
        raise ValueError('reference weights must match depth-derived nodal dual edges exactly')
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('binary wet mask required')
    if np.any(np.diff(mask, axis=-1) > 0):
        raise ValueError('contiguous wet prefix required')
    wet = mask == 1
    # Source dry slots stay untouched; they have no candidate stock meaning.
    values = np.where(wet[..., None], values, 0.)
    counts = wet.sum(axis=-1).astype(np.int64)
    live = counts > 0
    if np.any(eta[~live] != 0):
        raise ValueError('nonzero dry eta has no authorized material inventory interpretation')
    reference_h = mask * weights
    old_h = reference_h.copy()
    old_h[..., 0] += eta
    if np.any(old_h[wet] <= 0):
        raise ValueError('accepted wet material thickness must be positive')
    bottom = -edges[counts]
    band_count = np.minimum(counts, band_layers)
    band_depth = edges[band_count]
    band_bottom = -band_depth
    old_z = np.maximum(np.broadcast_to(-edges, eta.shape + (depth.size + 1,)), bottom[..., None]).copy()
    old_z[..., 0] = eta
    old_stocks = old_h[..., None] * values
    old_stocks[..., 2:] = rho0 * reference_h[..., None] * values[..., 2:]
    new_h, new_z, new_stocks = old_h.copy(), old_z.copy(), old_stocks.copy()
    band_height = eta + band_depth
    for layer in range(band_layers):
        selected = band_count > layer
        fraction = np.divide(weights[layer], band_depth, out=np.zeros_like(eta), where=selected)
        new_h[..., layer] = np.where(selected, fraction * band_height, new_h[..., layer])
        # The band endpoint stays the depth-derived dual edge exactly.
        interior = band_count > layer + 1
        new_z[..., layer + 1] = np.where(interior, new_z[..., layer] - new_h[..., layer], new_z[..., layer + 1])
    for target in range(band_layers):
        stock = np.zeros(eta.shape + (4,))
        for source in range(band_layers):
            overlap = np.maximum(0., np.minimum(old_z[..., source], new_z[..., target])
                                 - np.maximum(old_z[..., source + 1], new_z[..., target + 1]))
            fraction = np.divide(overlap, old_h[..., source], out=np.zeros_like(eta), where=wet[..., source])
            stock += fraction[..., None] * old_stocks[..., source, :]
        new_stocks[..., target, :] = np.where((band_count > target)[..., None], stock,
                                              new_stocks[..., target, :])
    residual = new_stocks.sum(axis=-2) - old_stocks.sum(axis=-2)
    bound = 256. * np.finfo(float).eps * np.maximum(1., np.abs(old_stocks).sum(axis=-2))
    ratio = np.abs(residual) / bound
    reference_kinetic = (.5 * rho0 * reference_h * np.sum(values[..., 2:]**2, axis=-1)).sum(axis=-1)
    actual_kinetic = _kinetic(old_stocks, old_h, rho0)
    candidate_kinetic = _kinetic(new_stocks, new_h, rho0)
    if not all(np.isfinite(item).all() for item in [old_stocks, new_stocks, new_h, ratio,
                                                  reference_kinetic, actual_kinetic, candidate_kinetic]) or np.any(ratio > 1):
        raise ValueError('finite conservative inventory bridge required')
    if np.any(new_h[wet] <= 0):
        raise ValueError('candidate wet thickness must stay positive')
    energy_bound = 256. * np.finfo(float).eps * np.maximum(1., actual_kinetic)
    if np.any(candidate_kinetic - actual_kinetic > energy_bound):
        raise ValueError('P0 overlap remap cannot gain fixed-domain kinetic energy')
    physical = wet & (-depth <= eta[..., None])
    has_sample = physical.any(axis=-1)
    sample_upper = np.max(np.where(physical, -depth, -np.inf), axis=-1)
    sample_lower = np.min(np.where(physical, -depth, np.inf), axis=-1)
    sample_upper = np.where(has_sample, sample_upper, np.nan)
    sample_lower = np.where(has_sample, sample_lower, np.nan)
    unsampled_bottom = np.where(live, edges[counts] - depth[np.maximum(counts - 1, 0)], 0.)

    def state(h, stocks, interfaces):
        return ColumnStocks(*[_readonly(item) for item in
                            [counts.copy(), wet.copy(), eta.copy(), bottom.copy(),
                             band_bottom.copy(), interfaces, h, stocks]])

    return InventoryBridge(state(old_h, old_stocks, old_z), state(new_h, new_stocks, new_z), rho0,
                           *[_readonly(item) for item in
                             [sample_lower, sample_upper, live & (eta < 0), unsampled_bottom,
                              reference_kinetic, actual_kinetic, candidate_kinetic,
                              actual_kinetic - reference_kinetic, candidate_kinetic - actual_kinetic, ratio]])


def _column(state, index):
    shape = state.eta.shape
    if not isinstance(index, tuple) or len(index) != len(shape) or any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 0 <= i < size
        for i, size in zip(index, shape)
    ):
        raise ValueError('explicit in-range column tuple required')
    return index


def require_original_profile(bridge, column, upper_z, lower_z):
    """Reject intervals needing missing surface or bottom sample support.

    Passing only confirms bracketing physical wet samples. It does not choose
    a reconstruction/EOS or establish equivalence of the candidate P0 means.
    """
    column = _column(bridge.candidate, column)
    upper, lower = _scalar(upper_z, 'upper_z'), _scalar(lower_z, 'lower_z')
    if lower >= upper or not bridge.candidate.active_layers[column]:
        raise ValueError('ordered wet physical sample interval required')
    if not math.isfinite(float(bridge.sample_upper_z[column])) or upper > bridge.sample_upper_z[column]:
        raise ValueError('surface boundary profile / physical sample support is missing')
    if lower < bridge.sample_lower_z[column]:
        raise ValueError('bottom profile / physical sample support is missing')


def shared_face(bridge, left, right, *, length, normal=(1., 0.)):
    """One P0 mean-stock Q per shared physical-depth segment, dry Q exactly 0.

    Face domain is [max(bottom_left,bottom_right), min(eta_left,eta_right)].
    This explicit candidate interpretation does not use raw terrain depths or
    claim a recovered historical pressure/velocity sample profile.
    """
    state = bridge.candidate
    left, right = _column(state, left), _column(state, right)
    length = _scalar(length, 'length', positive=True)
    normal = _array(normal, 'normal')
    if normal.shape != (2,) or abs(float(np.dot(normal, normal)) - 1.) > 8 * np.finfo(float).eps:
        raise ValueError('unit two-component face normal required')
    nl, nr = int(state.active_layers[left]), int(state.active_layers[right])
    if not nl or not nr:
        return SharedFace(None, None, (), 0.)
    upper = float(min(state.eta[left], state.eta[right]))
    lower = float(max(state.bottom[left], state.bottom[right]))
    if lower >= upper:
        return SharedFace(None, None, (), 0.)
    zl, zr = state.interfaces[left][:nl + 1], state.interfaces[right][:nr + 1]
    cuts = np.unique(np.r_[upper, lower, zl[(zl < upper) & (zl > lower)], zr[(zr < upper) & (zr > lower)]])[::-1]
    segments = []
    for top, base in zip(cuts[:-1], cuts[1:]):
        middle = .5 * (top + base)
        il = int(np.searchsorted(-zl, -middle, side='right') - 1)
        ir = int(np.searchsorted(-zr, -middle, side='right') - 1)
        vl = state.stocks[left][il, 2:] / (bridge.rho0 * state.h[left][il])
        vr = state.stocks[right][ir, 2:] / (bridge.rho0 * state.h[right][ir])
        Q = length * (top - base) * .5 * float(np.dot(vl + vr, normal))
        if not math.isfinite(Q):
            raise ValueError('finite shared Q required')
        supported = all(math.isfinite(float(bridge.sample_upper_z[index]))
                        and top <= bridge.sample_upper_z[index]
                        and base >= bridge.sample_lower_z[index] for index in [left, right])
        segments.append(FaceSegment(float(top), float(base), il, ir, Q, bool(supported)))
    total = math.fsum(segment.Q_m3_s for segment in segments)
    if not math.isfinite(total):
        raise ValueError('finite column Q required')
    return SharedFace(upper, lower, tuple(segments), total)
