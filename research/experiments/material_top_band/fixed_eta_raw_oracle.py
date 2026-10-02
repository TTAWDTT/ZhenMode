"""Independent raw half-prism quadrature and unaccepted moving obstructions.

No new step, force consumer, time moments or mass-chain helper is imported.
"""
import math
from dataclasses import replace

import numpy as np

from . import affine_physical_oracle as physical
from . import inventory_pressure as inventory
from . import slope_dual_stock_oracle as projection_oracle
from .affine_physical import AffinePhysicalDual
from .affine_physical_cases import manufactured_case

RHO, G, BETA, TREF, SREF = 1025., 9.81, .00076, 15., 35.


def _close(actual, expected, scale, message):
    actual, expected, scale = map(np.asarray, (actual, expected, scale))
    if actual.shape != expected.shape or not all(np.isfinite(v).all() for v in (actual, expected, scale)):
        raise ValueError(message)
    if np.any(abs(actual - expected) > 512. * np.finfo(float).eps * np.maximum(1., scale)):
        raise ValueError(message)


def actual_face_reference(profile, dt, distance, length):
    """Integrate actual raw P1 fields and actual pressure independently.

    Seven-point time/depth quadrature, including independent hydrostatic
    integrals over all overlying raw layers. No candidate moments or pressure
    primitive is called. Absolute operation scales are derived here.
    """
    state, eos = profile.state, profile.eos
    center = .5 * (state.interfaces[:, :-1] + state.interfaces[:, 1:])
    specific = state.stocks / state.h[:, :, None]
    U = math.fsum(float(v) for v in state.stocks[:, :, 2].ravel()) / (eos.rho0 * math.fsum(float(v) for v in state.h.ravel()))
    V = math.fsum(float(v) for v in state.stocks[:, :, 3].ravel()) / (eos.rho0 * math.fsum(float(v) for v in state.h.ravel()))
    _close(specific[:, :, 2:] / eos.rho0, np.broadcast_to([U, V], (2, 14, 2)),
           abs(specific[:, :, 2:] / eos.rho0) + np.abs([U, V]), 'independent actual uniform velocity binding failed')
    P = (profile.external_pressure_Pa[1] - profile.external_pressure_Pa[0]) / distance
    if profile.slope_limited_count or profile.density_slope_limited_count:
        raise ValueError('independent inactive TS/density limiter required')
    for sign in (-1., 1.):
        T = profile.temperature_mean + sign * .5 * state.h * profile.temperature_slope
        S = profile.salinity_mean + sign * .5 * state.h * profile.salinity_slope
        rho = profile.density_mean + sign * .5 * state.h * profile.density_slope
        actual_eos = eos.rho0 * (-eos.alpha * (T - eos.Tref) + eos.beta * (S - eos.Sref))
        _close(rho, actual_eos, abs(rho) + eos.rho0 * (eos.alpha * (abs(T) + eos.Tref) + eos.beta * (abs(S) + eos.Sref)),
               'independent actual P1 TS/density conformity failed')

    def density(side, layer, z, absolute=False):
        deviation = profile.density_slope[side, layer] * (z - center[side, layer])
        if absolute:
            T, S = specific[side, layer, :2]
            return (abs(profile.density_mean[side, layer]) + abs(deviation)
                    + eos.rho0 * (eos.alpha * (abs(T) + abs(eos.Tref)) + eos.beta * (abs(S) + abs(eos.Sref))))
        return profile.density_mean[side, layer] + deviation

    def hydrostatic(side, z, absolute=False):
        terms = []
        for layer in range(14):
            low, high = max(z, state.interfaces[side, layer + 1]), state.interfaces[side, layer]
            if high > low:
                terms.append(physical.gauss(lambda depth: density(side, layer, depth, absolute), low, high))
        baseline = eos.rho0 * ((abs(state.eta[side]) + abs(z)) if absolute else state.eta[side] - z)
        external = abs(profile.external_pressure_Pa[side]) if absolute else profile.external_pressure_Pa[side]
        return external + eos.gravity * (baseline + math.fsum(terms))

    rows = []
    levels = sorted(set(float(z) for z in state.interfaces.ravel()))
    for lower, upper in zip(levels[:-1], levels[1:]):
        middle = .5 * (lower + upper)
        owners = [next(k for k in range(14) if state.interfaces[j, k + 1] < middle < state.interfaces[j, k]) for j in range(2)]

        def fields(side, z, t):
            layer = owners[side]
            rho = density(side, layer, z)
            u = U - P * t / eos.rho0
            T = profile.temperature_mean[side, layer] + profile.temperature_slope[side, layer] * (z - center[side, layer])
            S = profile.salinity_mean[side, layer] + profile.salinity_slope[side, layer] * (z - center[side, layer])
            return [1., T, S, eos.rho0 * u, eos.rho0 * V,
                    .5 * eos.rho0 * (u*u + V*V), eos.gravity * (eos.rho0 + rho) * z]

        values, scales = [], []
        for side in range(2):
            side_values, side_scales = [], []
            for slot in range(7):
                def integral(absolute=False):
                    def time(t):
                        u = U - P * t / eos.rho0
                        def depth(z):
                            q = fields(side, z, t)[slot]
                            if absolute:
                                q = abs(q)
                                if slot == 2:
                                    q += abs(eos.Sref) + density(side, owners[side], z, True) / (eos.rho0 * eos.beta)
                                if slot == 6:
                                    q += eos.gravity * abs(z) * density(side, owners[side], z, True)
                            return length * (abs(u) if absolute else u) * q
                        return physical.gauss(depth, lower, upper)
                    return physical.gauss(time, 0., dt)
                side_values.append(integral())
                side_scales.append(integral(True))
            values.append(side_values)
            scales.append(side_scales)
        values, scales = np.array(values), np.array(scales)
        outer_pressure = [length * physical.gauss(lambda z: hydrostatic(side, z), lower, upper) for side in range(2)]
        pressure_ops = [length * physical.gauss(lambda z: hydrostatic(side, z, True), lower, upper) for side in range(2)]
        rows.append(dict(lower=lower, upper=upper, left_layer=owners[0], right_layer=owners[1],
                         transport=.5 * (values[0, :5] + values[1, :5]), transport_scale=.5 * (scales[0, :5] + scales[1, :5]),
                         outer_transport=values[:, :5], outer_transport_scale=scales[:, :5],
                         KE_transport=.5 * (values[0, 5] + values[1, 5]), KE_scale=.5 * (scales[0, 5] + scales[1, 5]),
                         PE_transport=.5 * (values[0, 6] + values[1, 6]), PE_scale=.5 * (scales[0, 6] + scales[1, 6]),
                         outer_energy=values[:, 5:], outer_energy_scale=scales[:, 5:],
                         pressure=np.array([outer_pressure[0], .5 * math.fsum(outer_pressure), outer_pressure[1]]),
                         pressure_scale=np.array([pressure_ops[0], .5 * math.fsum(pressure_ops), pressure_ops[1]])))
    return rows


def verify_actual_faces(profile, dt, distance, length, faces):
    references = actual_face_reference(profile, dt, distance, length)
    if len(faces) != len(references):
        raise ValueError('independent physical face partition count failed')
    fields = ('transport', 'outer_transport', 'pressure', 'KE_transport', 'PE_transport', 'outer_energy')
    scales = ('transport_scale', 'outer_transport_scale', 'pressure_scale', 'KE_scale', 'PE_scale', 'outer_energy_scale')
    for face, reference in zip(faces, references, strict=True):
        for key in ('lower', 'upper', 'left_layer', 'right_layer'):
            if face[key] != reference[key]:
                raise ValueError('independent physical face cut/owner binding failed')
        for key, scale in zip(fields, scales, strict=True):
            _close(face[key], reference[key], reference[scale] + abs(reference[key]), 'independent actual face ' + key + ' binding failed')
            if not np.isfinite(face[scale]).all() or np.any(np.asarray(face[scale]) < 0.):
                raise ValueError('finite nonnegative actual face operation scale required')
    return references


def audit(before, result, spec):
    state, dt = before.state, result.duration_s
    area = spec.distance * spec.length / 2.
    U = math.fsum(float(v) for v in state.stocks[:, :, 2].ravel()) / (RHO * math.fsum(float(v) for v in state.h.ravel()))
    V = math.fsum(float(v) for v in state.stocks[:, :, 3].ravel()) / (RHO * math.fsum(float(v) for v in state.h.ravel()))
    P = (spec.external[1] - spec.external[0]) / spec.distance
    if spec.eta[0] != spec.eta[1] or spec.density_gradient_x != 0. or not np.array_equal(state.eta, spec.eta) or np.any(state.bottom != spec.bottom):
        raise ValueError('independent fixed-eta declared raw domain failed')
    if not np.array_equal(before.external_pressure_Pa, spec.external):
        raise ValueError('independent declared external pressure binding failed')
    projection_oracle.validate_state(spec, state, material=(U, 0.))
    projection_oracle.validate_state(spec, result.state, material=(U - P * dt / RHO, 0.))
    if not all(np.array_equal(getattr(state, key), getattr(result.state, key)) for key in ('h', 'interfaces', 'eta', 'bottom')):
        raise ValueError('independent actual after geometry changed')
    if not np.array_equal(state.stocks[:, :, :2], result.state.stocks[:, :, :2]):
        raise ValueError('independent actual after thermodynamics changed')
    force, force_scale = np.zeros((2, 14)), np.zeros((2, 14))
    net, transport_scale = np.zeros((2, 14, 5)), np.zeros((2, 14, 5))
    net_energy, energy_flux_scale = np.zeros((2, 14, 2)), np.zeros((2, 14, 2))
    faces = []
    levels = sorted(set(float(z) for z in state.interfaces.ravel()))
    for lower, upper in zip(levels[:-1], levels[1:]):
        middle = (lower + upper) / 2.
        owners = [next(k for k in range(14) if state.interfaces[j, k + 1] < middle < state.interfaces[j, k]) for j in range(2)]
        def specific(z, t):
            velocity = U - P * t / RHO
            return np.array([1., TREF, SREF + physical.anomaly(spec, 0., z) / (RHO * BETA), RHO * velocity, RHO * V])
        def flux(slot, absolute=False):
            def time(t):
                velocity = U - P * t / RHO
                def depth(z):
                    value = spec.length * velocity * specific(z, t)[slot]
                    return abs(value) if absolute else value
                return physical.gauss(depth, lower, upper)
            return physical.gauss(time, 0., dt)
        transported = np.array([flux(slot) for slot in range(5)])
        flux_scale = np.array([flux(slot, True) for slot in range(5)])
        # EOS subtraction is an actual input operation even for affine S.
        flux_scale[2] += physical.gauss(lambda t: abs(U - P * t / RHO) * spec.length * physical.gauss(
            lambda z: SREF + RHO * (2e-4 * 30. + BETA * (abs(specific(z, t)[2]) + SREF)) / (RHO * BETA), lower, upper), 0., dt)
        KE = physical.gauss(lambda t: .5 * RHO * spec.length * (U - P * t / RHO) * ((U - P * t / RHO)**2 + V**2) * (upper - lower), 0., dt)
        KEscale = physical.gauss(lambda t: abs(.5 * RHO * spec.length * (U - P * t / RHO) * ((U - P * t / RHO)**2 + V**2) * (upper - lower)), 0., dt)
        PE = physical.gauss(lambda t: spec.length * (U - P * t / RHO) * physical.gauss(
            lambda z: G * (RHO + physical.anomaly(spec, 0., z)) * z, lower, upper), 0., dt)
        PEscale = physical.gauss(lambda t: spec.length * abs(U - P * t / RHO) * physical.gauss(
            lambda z: G * (RHO + abs(physical.anomaly(spec, 0., z))) * abs(z), lower, upper), 0., dt)
        pressure_integrals = [spec.length * physical.gauss(lambda z: physical.pressure(spec, x, z), lower, upper) for x in (0., spec.distance / 2., spec.distance)]
        for side, layer in enumerate(owners):
            transport_scale[side, layer] += 2. * flux_scale
            force_scale[side, layer] += abs(pressure_integrals[side]) + abs(pressure_integrals[side + 1])
        faces.append(dict(left_layer=owners[0], right_layer=owners[1], lower=lower, upper=upper,
                          transport=transported, transport_scale=flux_scale, KE_transport=KE, KE_scale=KEscale,
                          PE_transport=PE, PE_scale=PEscale, pressure=np.array(pressure_integrals)))
    # Independent whole-row CV traction, not the candidate segment sum.
    for side in range(2):
        xleft, xright = side * spec.distance / 2., (side + 1) * spec.distance / 2.
        for layer in range(14):
            lower, upper = state.interfaces[side, layer + 1], state.interfaces[side, layer]
            left = spec.length * physical.gauss(lambda z: physical.pressure(spec, xleft, z), lower, upper)
            right = spec.length * physical.gauss(lambda z: physical.pressure(spec, xright, z), lower, upper)
            force[side, layer] = left - right
            force_scale[side, layer] += abs(left) + abs(right)
            for slot in range(5):
                def whole_face(x, absolute=False):
                    def time(t):
                        u = U - P * t / RHO
                        def depth(z):
                            q = (1., TREF, SREF + physical.anomaly(spec, x, z) / (RHO * BETA), RHO * u, RHO * V)[slot]
                            value = spec.length * u * q
                            return abs(value) if absolute else value
                        return physical.gauss(depth, lower, upper)
                    return physical.gauss(time, 0., dt)
                net[side, layer, slot] = whole_face(xright) - whole_face(xleft)
                transport_scale[side, layer, slot] += whole_face(xright, True) + whole_face(xleft, True)
            for slot in range(2):
                def energy_face(x, absolute=False):
                    def time(t):
                        u = U - P * t / RHO
                        def depth(z):
                            content = .5 * RHO * (u*u + V*V) if slot == 0 else G * (RHO + physical.anomaly(spec, x, z)) * z
                            value = spec.length * u * content
                            return abs(value) if absolute else value
                        return physical.gauss(depth, lower, upper)
                    return physical.gauss(time, 0., dt)
                net_energy[side, layer, slot] = energy_face(xright) - energy_face(xleft)
                energy_flux_scale[side, layer, slot] = energy_face(xright, True) + energy_face(xleft, True)
    outer_pressure = [spec.length * physical.gauss(lambda z: physical.pressure(spec, x, z), spec.bottom, spec.eta[0]) for x in (0., spec.distance)]
    boundary = physical.gauss(lambda t: (U - P * t / RHO) * (outer_pressure[0] - outer_pressure[1]), 0., dt)
    expected_u = U - P * dt / RHO
    volume = area * math.fsum(float(h) for h in state.h.ravel())
    K0, K1 = .5 * RHO * volume * (U**2 + V**2), .5 * RHO * volume * (expected_u**2 + V**2)
    energy_scale = abs(K0) + abs(K1) + physical.gauss(lambda t: abs(U - P * t / RHO) * (abs(outer_pressure[0]) + abs(outer_pressure[1])), 0., dt)
    increment = np.zeros_like(state.stocks)
    increment[:, :, 2] = -P * area * state.h * dt
    return dict(force=force, force_scale=force_scale, net_water=net[:, :, 0], net_stocks=net[:, :, 1:],
                transport_scale=transport_scale, faces=faces, KE_change=K1 - K0,
                net_energy=net_energy, energy_flux_scale=energy_flux_scale,
                boundary_pressure_work=boundary, energy_scale=energy_scale, stock_increment=increment)


def moving_snapshots():
    """Lagrangian-generated actual raw states; never an accepted moving step."""
    U0, alpha, dt = .03, .008, .01
    profile, spec = manufactured_case(density_gradient_x=0., u0=U0, strain=alpha)
    state = profile.state
    P = (spec.external[1] - spec.external[0]) / spec.distance
    stretch = 1. + alpha * dt
    displacement = U0 * dt - .5 * P / RHO * dt**2
    interfaces = state.interfaces.copy()
    interfaces[:, 0] = spec.bottom + (spec.eta[0] - spec.bottom) / stretch
    h = -np.diff(interfaces, axis=-1)
    stocks = np.empty_like(state.stocks)
    for side, x in enumerate((0., spec.distance)):
        label_x = (x - displacement) / stretch
        velocity = U0 + alpha * label_x - P * dt / RHO
        for layer in range(14):
            lower, upper = interfaces[side, layer + 1], interfaces[side, layer]
            def salinity(z):
                label_z = spec.bottom + stretch * (z - spec.bottom)
                return SREF + physical.anomaly(spec, 0., label_z) / (RHO * BETA)
            stocks[side, layer] = [TREF * h[side, layer], physical.gauss(salinity, lower, upper),
                                  RHO * velocity * h[side, layer], RHO * spec.velocity_y * h[side, layer]]
    changed = inventory.reconstruct(replace(state, interfaces=interfaces, h=h, eta=interfaces[:, 0], stocks=stocks), external_pressure_Pa=spec.external)
    def new_density(z):
        return physical.anomaly(spec, 0., spec.bottom + stretch * (z - spec.bottom))
    changed_spec = replace(spec, eta=tuple(interfaces[:, 0]), density_intercept=new_density(0.), density_slope=new_density(1.) - new_density(0.))
    op = AffinePhysicalDual(profile, distance_m=spec.distance, length_m=spec.length)
    chart0, chart1 = op.segments, projection_oracle.fixed_chart(op.segments, changed_spec.eta)
    physical.validate_strip_ownership(spec, state, chart0)
    physical.validate_strip_ownership(changed_spec, changed.state, chart1)
    B0, B1 = projection_oracle.mapped(spec, chart0, state), projection_oracle.mapped(changed_spec, chart1, changed.state)
    area = spec.distance * spec.length / 2.
    D0, D1 = np.diag(RHO * area * state.h.ravel()), np.diag(RHO * area * changed.state.h.ravel())
    u0, u1 = state.stocks[:, :, 2:].reshape(28, 2) / (RHO * state.h.ravel()[:, None]), changed.state.stocks[:, :, 2:].reshape(28, 2) / (RHO * changed.state.h.ravel()[:, None])
    raw0, raw1 = .5 * np.sum(u0 * (D0 @ u0)), .5 * np.sum(u1 * (D1 @ u1))
    K0, K1 = projection_oracle.kinetic(spec, chart0, u0), projection_oracle.kinetic(changed_spec, chart1, u1)
    g0, g1 = area * state.stocks[:, :, 2].ravel() - B0['values'][:, :, 3].ravel(), area * changed.state.stocks[:, :, 2].ravel() - B1['values'][:, :, 3].ravel()
    eta_dot = -alpha * (spec.eta[0] - spec.bottom)
    hdot = np.zeros_like(state.h)
    hdot[:, 0] = eta_dot
    Ddot = np.diag(RHO * area * hdot.ravel())
    Wdot = np.zeros((28, 28))
    for side in range(2):
        for other in range(2):
            Wdot[side * 14, other * 14] = RHO * spec.length * eta_dot * physical.gauss(lambda x: projection_oracle.phi(spec, side, x) * projection_oracle.phi(spec, other, x), 0., spec.distance)
    udot = -alpha * u0[:, 0] - P / RHO
    raw_dot = area * (RHO * state.h.ravel() * udot + RHO * u0[:, 0] * hdot.ravel())
    mu_dot = Wdot @ u0[:, 0] + B0['mass'] @ udot
    omit_Wdot = Ddot @ u0[:, 0] + D0 @ np.linalg.solve(B0['mass'], mu_dot)
    omit_Rdot = D0 @ np.linalg.solve(B0['mass'], mu_dot)
    return dict(raw=dict(mass_before=D0, mass_after=D1, u_before=u0, u_after=u1, KE_before=raw0, KE_after=raw1),
                physical=dict(mass_before=B0['mass'], mass_after=B1['mass'], u_before=u0, u_after=u1, KE_before=K0['value'], KE_after=K1['value']),
                omitted_Wdot_max_residual=float(np.max(abs(raw_dot - omit_Wdot))),
                omitted_Rdot_max_residual=float(np.max(abs(raw_dot - omit_Rdot))),
                moving_local_storage_change=float(np.max(abs(g1 - g0))), moving_KE_gap_change=float((raw1 - K1['value']) - (raw0 - K0['value'])), moving_accepted_steps=0)


def obstructions():
    profile, spec = manufactured_case(eta=(-.2, .3))
    op = AffinePhysicalDual(profile, distance_m=spec.distance, length_m=spec.length)
    mapped = projection_oracle.mapped(spec, op.segments, profile.state)
    raw = projection_oracle.raw_inventory(profile.state)
    difference = raw - np.sum(mapped['values'][:, :, :5], axis=(0, 1))
    raw_PE = inventory.potential_energy(profile, np.full(2, spec.distance * spec.length / 2.))
    moving = moving_snapshots()
    return dict(sloped_Mu_gap=float(difference[3]), sloped_IS_gap=float(difference[2]), sloped_PE_gap=raw_PE - mapped['PE'],
                sloped_scale=float(np.sum(abs(raw)) + np.sum(mapped['scale'])),
                moving_local_storage_change=moving['moving_local_storage_change'], moving_KE_gap_change=moving['moving_KE_gap_change'], moving_accepted_steps=0)
