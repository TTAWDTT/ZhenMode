"""A real raw-stock pressure step on an explicitly manufactured fixed-eta patch.

Uniform CV means coincide with endpoint coefficients only on this subspace.
No moving geometry, original global CV or general transport adapter is supplied.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from . import inventory_pressure as inventory
from .affine_physical import EPS, bound, total
from .fixed_eta_raw_oracle import verify_actual_faces
from .slope_dual_stock import FrozenPhysicalProjection, readonly


def require_close(value, reference, scale, message):
    value, reference, scale = map(np.asarray, (value, reference, scale))
    if not all(np.isfinite(a).all() for a in (value, reference, scale)) or np.any(abs(value - reference) > 512. * EPS * np.maximum(1., scale)):
        raise ValueError(message)


def variable_mass_identity(mass_before, mass_after, u_before, u_after):
    """True before/after momentum and the symmetric changing-mass KE chain."""
    if any(np.ma.isMaskedArray(v) or np.iscomplexobj(v) for v in (mass_before, mass_after, u_before, u_after)):
        raise ValueError('real unmasked actual mass and velocity required')
    W0, W1, u0, u1 = map(readonly, (mass_before, mass_after, u_before, u_after))
    if W0.ndim != 2 or W0.shape != W1.shape or not W0.shape[0] or W0.shape[0] != W0.shape[1] or u0.ndim not in (1, 2) or u0.shape != u1.shape or u0.shape[0] != W0.shape[0]:
        raise ValueError('mass and actual velocity shapes disagree')
    for mass in (W0, W1):
        require_close(mass, mass.T, abs(mass) + abs(mass.T), 'symmetric mass required')
        if np.linalg.eigvalsh(mass)[0] <= 0.:
            raise ValueError('positive actual mass required')
    momentum0, momentum1 = W0 @ u0, W1 @ u1
    kinetic0, kinetic1 = .5 * total(u0 * momentum0), .5 * total(u1 * momentum1)
    impulse = total(.5 * (u0 + u1) * (momentum1 - momentum0))
    mass_work = .5 * total(u1 * ((W1 - W0) @ u0))
    scale = (.5 * total(abs(u0) * (abs(W0) @ abs(u0))) + .5 * total(abs(u1) * (abs(W1) @ abs(u1)))
             + .5 * total((abs(u0) + abs(u1)) * (abs(momentum0) + abs(momentum1)))
             + .5 * total(abs(u1) * ((abs(W0) + abs(W1)) @ abs(u0))))
    require_close(kinetic1 - kinetic0, impulse - mass_work, scale, 'actual changing-mass KE chain failed')
    return dict(kinetic_change=kinetic1 - kinetic0, impulse_work=impulse, mass_work=mass_work, scale=scale)


@dataclass(frozen=True)
class RawPressureStep:
    state: inventory.ColumnStocks
    profile: inventory.Profile
    duration_s: float
    time_s: float
    accepted_raw_steps: int
    raw_force: np.ndarray
    raw_force_scale: np.ndarray
    net_water: np.ndarray
    net_stocks: np.ndarray
    ledger_scale: np.ndarray
    net_energy: np.ndarray
    energy_flux_scale: np.ndarray
    faces: tuple
    raw_KE_change_J: float
    physical_KE_change_J: float
    PE_change_J: float
    pressure_work_J: float
    boundary_pressure_work_J: float
    external_pressure_work_J: float
    energy_scale: float
    maximum_inverse_residual: float
    production_qualified: bool = False
    original_global_CV_identified: bool = False
    accepted_moving_geometry: bool = False


class FixedEtaRawPressureSlice:
    """Atomic raw ColumnStocks updates on two new manufactured half-prisms."""

    def __init__(self, profile, *, distance_m, length_m):
        self.distance, self.length = float(distance_m), float(length_m)
        binding = self._binding(profile)
        self.profile = binding['projection'].profile
        self.time_s, self.accepted_raw_steps, self.last_receipt = 0., 0, None

    def _binding(self, profile):
        projection = FrozenPhysicalProjection(profile, distance_m=self.distance, length_m=self.length)
        op = projection.base
        if op.state.eta[0] != op.state.eta[1]:
            raise ValueError('common flat eta required; sloped inverse remains unsupported')
        a0, slope = op.anomaly(op.distance / 2.), op.slope
        expected = a0 + slope * op.centers
        require_close(expected, op.density_mean, op.eos_scale + abs(expected), 'common a1=0 actual density required')
        if op.profile.slope_limited_count or op.profile.density_slope_limited_count:
            raise ValueError('inactive actual TS/density physical-bound limiter required')
        for sign in (-1., 1.):
            z = op.centers + sign * .5 * op.state.h
            actual_T = op.profile.temperature_mean + sign * .5 * op.state.h * op.profile.temperature_slope
            actual_S = op.profile.salinity_mean + sign * .5 * op.state.h * op.profile.salinity_slope
            expected_S = op.eos.Sref + (a0 + slope * z) / (op.eos.rho0 * op.eos.beta)
            require_close(actual_T, op.eos.Tref, abs(actual_T) + op.eos.Tref, 'actual P1 T differs from common Tref')
            require_close(actual_S, expected_S, abs(actual_S) + abs(expected_S) + op.eos_scale / (op.eos.rho0 * op.eos.beta),
                          'actual P1 S differs from common affine EOS field')
        actual = op.state.stocks[:, :, 2:] / (op.eos.rho0 * op.state.h[:, :, None])
        U = total(op.state.stocks[:, :, 2]) / (op.eos.rho0 * total(op.state.h))
        V = total(op.state.stocks[:, :, 3]) / (op.eos.rho0 * total(op.state.h))
        require_close(actual, np.broadcast_to([U, V], actual.shape), abs(actual) + np.abs([U, V]), 'uniform actual CV velocity required; moving alpha/shear refuses')
        P = float(np.diff(op.profile.external_pressure_Pa)[0] / op.distance)
        total([U, V, P, a0, slope])
        column_slope_scale = (op.eos_scale[:, 0] + op.eos_scale[:, -1]) / abs(op.centers[:, -1] - op.centers[:, 0])
        fit_slope_scale = .5 * total(column_slope_scale)
        fit_intercept_scale = .5 * total(op.eos_scale[:, 0] + column_slope_scale * abs(op.centers[:, 0]))
        return dict(projection=projection, op=op, U=U, V=V, P=P, a0=a0, slope=slope,
                    fit_slope_scale=fit_slope_scale, fit_intercept_scale=fit_intercept_scale)

    def _faces(self, binding, dt):
        op, U0, V, P = (binding[key] for key in ('op', 'U', 'V', 'P'))
        eos, eta = op.eos, float(op.state.eta[0])
        U1 = U0 - P * dt / eos.rho0
        J1 = dt * (U0 + U1) / 2.
        J2 = dt * (U0**2 + U0 * U1 + U1**2) / 3.
        J3 = dt * (U0**3 + U0**2 * U1 + U0 * U1**2 + U1**3) / 4.
        J1scale = dt * (abs(U0) + abs(U1)) / 2.
        J2scale = dt * (abs(U0**2) + abs(U0 * U1) + abs(U1**2)) / 3.
        J3scale = dt * total([abs(U0**3), abs(U0**2 * U1), abs(U0 * U1**2), abs(U1**3)]) / 4.
        rows = []
        cuts = np.unique(op.state.interfaces.ravel())
        for lower, upper in zip(cuts[:-1], cuts[1:]):
            middle, width = .5 * (lower + upper), upper - lower
            owners = [int(np.flatnonzero((op.state.interfaces[side, 1:] < middle) & (middle < op.state.interfaces[side, :-1]))[0]) for side in range(2)]
            rho = binding['a0'] + binding['slope'] * middle
            rho_scale = binding['fit_intercept_scale'] + binding['fit_slope_scale'] * abs(middle)
            S = eos.Sref + rho / (eos.rho0 * eos.beta)
            water = op.length * width * J1
            transport = np.array([water, eos.Tref * water, S * water, eos.rho0 * op.length * width * J2, eos.rho0 * V * water])
            water_scale = op.length * width * J1scale
            transport_scale = np.array([water_scale, abs(eos.Tref) * water_scale,
                                        (abs(eos.Sref) + rho_scale / (eos.rho0 * eos.beta)) * water_scale,
                                        eos.rho0 * op.length * width * J2scale, eos.rho0 * abs(V) * water_scale])
            pressure, pressure_scale = [], []
            for x in (0., op.distance / 2., op.distance):
                values, operations = [], []
                for z in (middle - width / (2. * math.sqrt(3.)), middle + width / (2. * math.sqrt(3.))):
                    left = inventory.pressure(op.profile, (0,), z)
                    right = inventory.pressure(op.profile, (1,), z)
                    value = left if x == 0. else right if x == op.distance else .5 * (left + right)
                    operation = (abs(op.profile.external_pressure_Pa[0]) + abs(P * x) + eos.gravity *
                                 (eos.rho0 * (abs(eta) + abs(z)) + binding['fit_intercept_scale'] * (abs(eta) + abs(z))
                                  + binding['fit_slope_scale'] * (eta**2 + z**2) / 2.))
                    values.append(op.length * width * value / 2.)
                    operations.append(op.length * width * operation / 2.)
                pressure.append(total(values))
                pressure_scale.append(total(operations))
            KE = .5 * eos.rho0 * op.length * width * (J3 + V**2 * J1)
            KEscale = .5 * eos.rho0 * op.length * width * (J3scale + V**2 * J1scale)
            PE = op.length * eos.gravity * J1 * total([
                .5 * width * z * (eos.rho0 + binding['a0'] + binding['slope'] * z)
                for z in (middle - width / (2. * math.sqrt(3.)), middle + width / (2. * math.sqrt(3.)))])
            PEscale = op.length * eos.gravity * J1scale * total([
                .5 * width * abs(z) * (eos.rho0 + binding['fit_intercept_scale'] + binding['fit_slope_scale'] * abs(z))
                for z in (middle - width / (2. * math.sqrt(3.)), middle + width / (2. * math.sqrt(3.)))])
            rows.append(dict(lower=float(lower), upper=float(upper), left_layer=owners[0], right_layer=owners[1],
                             transport=readonly(transport), transport_scale=readonly(transport_scale),
                             outer_transport=readonly(np.array([transport.copy(), transport.copy()])),
                             outer_transport_scale=readonly(np.array([transport_scale.copy(), transport_scale.copy()])),
                             pressure=readonly(pressure), pressure_scale=readonly(pressure_scale),
                             KE_transport=KE, KE_scale=KEscale, PE_transport=PE, PE_scale=PEscale,
                             outer_energy=readonly([[KE, PE], [KE, PE]]),
                             outer_energy_scale=readonly([[KEscale, PEscale], [KEscale, PEscale]])))
        return rows

    @staticmethod
    def _consume(binding, faces):
        state = binding['op'].state
        transport_terms = [[[[] for _ in range(5)] for _ in range(14)] for _ in range(2)]
        force_terms = [[[] for _ in range(14)] for _ in range(2)]
        energy_terms = [[[[] for _ in range(2)] for _ in range(14)] for _ in range(2)]
        energy_scale = np.zeros((2, 14, 2))
        scale, force_scale = np.zeros((2, 14, 5)), np.zeros((2, 14))
        previous = float(state.bottom[0])
        coverage = np.zeros_like(state.h)
        for row in faces:
            lower, upper = row['lower'], row['upper']
            require_close(lower, previous, abs(lower) + abs(previous), 'physical face partition gap/overlap')
            if not math.isfinite(upper) or upper <= lower:
                raise ValueError('positive finite physical segment required')
            for side, layer in enumerate((row['left_layer'], row['right_layer'])):
                middle = .5 * (lower + upper)
                owners = np.flatnonzero((state.interfaces[side, 1:] < middle) & (middle < state.interfaces[side, :-1]))
                if len(owners) != 1 or int(owners[0]) != layer:
                    raise ValueError('actual raw face owner mismatch')
                if lower < state.interfaces[side, layer + 1] - bound(abs(lower)) or upper > state.interfaces[side, layer] + bound(abs(upper)):
                    raise ValueError('segment outside actual raw CV')
                coverage[side, layer] += upper - lower
                for slot, shared in enumerate(row['transport']):
                    outer = row['outer_transport'][side, slot]
                    total([shared, outer, row['transport_scale'][slot], row['outer_transport_scale'][side, slot]])
                    # Signed shared and distinct outer-face consumptions. The
                    # uniform physical field gives equal values, not a patched
                    # row residual or a discarded transport stage.
                    sign = 1. if side == 0 else -1.
                    transport_terms[side][layer][slot].extend([sign * shared, -sign * outer])
                scale[side, layer] += row['transport_scale'] + row['outer_transport_scale'][side]
                shared_energy = [row['KE_transport'], row['PE_transport']]
                for slot in range(2):
                    outer = row['outer_energy'][side, slot]
                    total([shared_energy[slot], outer, row['outer_energy_scale'][side, slot]])
                    sign = 1. if side == 0 else -1.
                    energy_terms[side][layer][slot].extend([sign * shared_energy[slot], -sign * outer])
                energy_scale[side, layer] += np.array([row['KE_scale'], row['PE_scale']]) + row['outer_energy_scale'][side]
                pressure = row['pressure']
                total(pressure)
                force_terms[side][layer].extend([pressure[side], -pressure[side + 1]])
                force_scale[side, layer] += row['pressure_scale'][side] + row['pressure_scale'][side + 1]
            previous = upper
        require_close(previous, state.eta[0], abs(previous) + abs(state.eta[0]), 'physical face domain incomplete')
        require_close(coverage, state.h, abs(coverage) + abs(state.h), 'actual raw face coverage incomplete')
        net = np.array([[[total(transport_terms[side][layer][slot]) for slot in range(5)] for layer in range(14)] for side in range(2)])
        force = np.array([[total(force_terms[side][layer]) for layer in range(14)] for side in range(2)])
        net_energy = np.array([[[total(energy_terms[side][layer][slot]) for slot in range(2)] for layer in range(14)] for side in range(2)])
        return net, scale, force, force_scale, net_energy, energy_scale

    def _audit(self, binding, candidate, faces, net, scale, force, force_scale, net_energy, energy_flux_scale, dt):
        op, before = binding['op'], binding['op'].state
        after_binding = self._binding(candidate)
        after = candidate.state
        area, rho0 = op.node_area_m2[0], op.eos.rho0
        require_close(net, 0., scale, 'raw CV face transport divergence refuses')
        require_close(net_energy, 0., energy_flux_scale, 'raw CV KE/PE face transport divergence refuses')
        require_close(force, -binding['P'] * area * before.h, force_scale + abs(binding['P'] * area * before.h), 'true raw face pressure force disagrees')
        require_close(area * (after.h - before.h), -net[:, :, 0], area * (abs(after.h) + abs(before.h)) + scale[:, :, 0], 'raw water ledger failed')
        impulse = np.zeros_like(before.stocks)
        impulse[:, :, 2] = dt * force
        require_close(area * (after.stocks - before.stocks), -net[:, :, 1:] + impulse,
                      area * (abs(after.stocks) + abs(before.stocks)) + scale[:, :, 1:] + dt * force_scale[:, :, None], 'all raw stock ledgers failed')
        mapped0 = binding['projection'].evaluate()
        mapped1 = binding['projection'].evaluate(candidate)
        mass_diag = rho0 * area * before.h.ravel()
        R = mapped0.mass / mass_diag[None, :]
        require_close(R @ force.ravel(), op.force_N.ravel(), abs(R) @ force_scale.ravel() + op.force_scale_N.ravel(), 'full raw/physical force duality failed')
        solved = np.linalg.solve(mapped1.mass, mapped1.values[:, :, 3].ravel())
        recovered = mass_diag * solved
        raw_after = area * after.stocks[:, :, 2].ravel()
        inverse_scale = abs(mapped1.mass) @ abs(solved) + abs(mapped1.values[:, :, 3].ravel())
        require_close(mapped1.mass @ solved, mapped1.values[:, :, 3].ravel(), inverse_scale, 'physical inverse backward error failed')
        require_close(recovered, raw_after, abs(recovered) + abs(raw_after), 'inverse did not return actual raw momentum')
        require_close(total(recovered), total(mapped1.values[:, :, 3]), total(abs(recovered)) + total(mapped1.scale[:, :, 3]), 'flat global inverse conservation failed')
        u0, u1 = mapped0.velocity, mapped1.velocity
        D = np.diag(mass_diag)
        raw = variable_mass_identity(D, D, u0, u1)
        physical = variable_mass_identity(mapped0.mass, mapped1.mass, u0, u1)
        midpoint = .5 * (u0[:, 0] + u1[:, 0])
        work = dt * total(force.ravel() * midpoint)
        J1 = dt * (binding['U'] - binding['P'] * dt / (2. * rho0))
        boundary = J1 * total([row['pressure'][0] - row['pressure'][2] for row in faces])
        external = J1 * op.length * (before.eta[0] - before.bottom[0]) * (op.profile.external_pressure_Pa[0] - op.profile.external_pressure_Pa[1])
        energy_scale = (raw['scale'] + physical['scale']
                        + dt * total(force_scale.ravel() * abs(midpoint))
                        + abs(J1) * total([total(row['pressure_scale']) for row in faces]))
        for value in (raw['kinetic_change'], physical['kinetic_change'], boundary, external):
            require_close(value, work, energy_scale, 'finite raw/physical/boundary pressure-work ledger failed')
        require_close(mapped1.PE, mapped0.PE, mapped0.PE_scale + mapped1.PE_scale, 'actual PE changed on fixed-eta slice')
        require_close(raw['kinetic_change'] + mapped1.PE - mapped0.PE + total(net_energy), boundary,
                      energy_scale + mapped0.PE_scale + mapped1.PE_scale + total(energy_flux_scale),
                      'finite total energy and consumed advective flux ledger failed')
        if not np.array_equal(after.h, before.h) or not np.array_equal(after.interfaces, before.interfaces):
            raise ValueError('accepted fixed geometry changed')
        total([self.time_s + dt, after_binding['U']])
        return RawPressureStep(after, candidate, dt, self.time_s + dt, self.accepted_raw_steps + 1,
                               readonly(force), readonly(force_scale), readonly(net[:, :, 0]), readonly(net[:, :, 1:]), readonly(scale),
                               readonly(net_energy), readonly(energy_flux_scale), tuple(faces),
                               raw['kinetic_change'], physical['kinetic_change'], mapped1.PE - mapped0.PE, work, boundary, external, energy_scale,
                               float(np.max(abs(recovered - raw_after))))

    def advance(self, duration_s):
        if isinstance(duration_s, (bool, np.bool_)) or not isinstance(duration_s, (int, float, np.integer, np.floating)) or not math.isfinite(duration_s) or not 0. < duration_s <= .1:
            raise ValueError('finite positive research duration at most0.1s required')
        dt = float(duration_s)
        binding = self._binding(self.profile)  # actual previous accepted stocks
        faces = self._faces(binding, dt)
        # Independent seven-point actual-profile quadrature binds absolute
        # face values before cancellation and before any accepted mutation.
        verify_actual_faces(self.profile, dt, self.distance, self.length, faces)
        net, scale, force, force_scale, net_energy, energy_flux_scale = self._consume(binding, faces)
        before, area = self.profile.state, binding['op'].node_area_m2[0]
        h = before.h - net[:, :, 0] / area
        stocks = before.stocks - net[:, :, 1:] / area
        stocks[:, :, 2] += dt * force / area
        candidate = inventory.reconstruct(replace(before, h=h, stocks=stocks), eos=self.profile.eos,
                                          external_pressure_Pa=self.profile.external_pressure_Pa)
        receipt = self._audit(binding, candidate, faces, net, scale, force, force_scale, net_energy, energy_flux_scale, dt)
        self.profile, self.time_s, self.accepted_raw_steps, self.last_receipt = candidate, receipt.time_s, receipt.accepted_raw_steps, receipt
        return receipt
