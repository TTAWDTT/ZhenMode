"""Independent Eulerian PDE, volume and absolute finite-face oracle.

No characteristic, candidate polynomial, mean metric or consumer is imported.
The physical solution is derived from Eulerian coefficient equations; actual
raw end stocks are separately bound by volume integration.
"""
import math

import numpy as np

from .moving_raw_cases import MeanFlowSpecification

EPS = np.finfo(float).eps
RHO, GRAVITY, TREF, SREF, BETA = 1025., 9.81, 15., 35., 7.6e-4


def assert_bound(value, expected, scale, *, extra=0., label='oracle'):
    if any(np.ma.isMaskedArray(item) for item in (value, expected, scale, extra)) or any(np.iscomplexobj(item) for item in (value, expected, scale, extra)):
        raise ValueError(f'{label}: real unmasked ledger required')
    value, expected, scale, extra = map(np.asarray, (value, expected, scale, extra))
    if any(item.dtype.kind not in 'fiu' for item in (value,expected,scale,extra)) or not all(np.isfinite(item).all() for item in (value, expected, scale, extra)) or np.any(scale < 0.) or np.any(extra < 0.):
        raise ValueError(f'{label}: finite nonnegative scale required')
    for item in (value,expected,scale,extra):
        if not item.size or (item.dtype.kind in 'iu' and (int(item.min()) < -(2**53) or int(item.max()) > 2**53)):
            raise ValueError(f'{label}: nonempty exactly representable numeric inputs required')
    value,expected,scale,extra = np.broadcast_arrays(*(item.astype(np.float64) for item in (value,expected,scale,extra)))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        budget = 512. * EPS * np.maximum(1., scale) + extra
        difference = abs(value-expected)
        ratios = difference/budget
    if not all(np.isfinite(item).all() for item in (budget,difference,ratios)) or np.any(budget <= 0.) or np.any(difference < 0.) or np.any(ratios < 0.):
        raise ValueError(f'{label}: finite positive bound and residual required')
    if np.any(difference > budget):
        ratio = float(np.max(ratios))
        raise ValueError(f'{label}: predetermined bound failed (ratio {ratio:.6g})')
    return float(np.max(ratios))


def spec_from_profile(profile, distance, length):
    """Own actual-CV fit; no Q/inverse-label or candidate binding helper."""
    state = profile.state
    means = np.sum(state.stocks[..., 2:], axis=1) / (RHO * np.sum(state.h, axis=1)[:, None])
    centers = .5 * (state.interfaces[:, :-1] + state.interfaces[:, 1:])
    rho = RHO * (-2e-4 * (state.stocks[..., 0] / state.h - TREF) + BETA * (state.stocks[..., 1] / state.h - SREF))
    slope = math.fsum((rho[:, -1] - rho[:, 0]) / (centers[:, -1] - centers[:, 0])) / 2.
    intercept = math.fsum(rho[:, 0] - slope * centers[:, 0]) / 2.
    return MeanFlowSpecification(distance=distance, length=length, eta=float(state.eta[0]), bottom=float(state.bottom[0]),
                                 U=float((3. * means[0, 0] - means[1, 0]) / 2.), alpha=float(2. * (means[1, 0] - means[0, 0]) / distance),
                                 V=float(np.mean(means[:, 1])), density_intercept=intercept, density_slope=slope,
                                 external=tuple(profile.external_pressure_Pa), interior_interfaces=tuple(tuple(z) for z in state.interfaces[:, 1:]))


def solution(spec, t, x, z):
    """Eulerian solution of alpha', H', U', s'; deliberately no particle labels."""
    acceleration = (spec.external[1] - spec.external[0]) / (RHO * spec.distance)
    lam = 1. + spec.alpha * t
    alpha = spec.alpha / lam
    U = (spec.U - acceleration * t - .5 * acceleration * spec.alpha * t**2) / lam
    H = (spec.eta - spec.bottom) / lam
    u, w = U + alpha * x, -alpha * (z - spec.bottom)
    qb = spec.density_intercept + spec.density_slope * spec.bottom
    rho = qb + spec.density_slope * lam * (z - spec.bottom)
    p = spec.external[0] + (spec.external[1]-spec.external[0]) * x / spec.distance
    p = p + GRAVITY * ((RHO + qb) * (H - (z-spec.bottom)) + .5 * spec.density_slope * lam * (H**2 - (z-spec.bottom)**2))
    S = SREF + rho / (RHO * BETA)
    KE = .5 * RHO * (u**2 + spec.V**2)
    PE = GRAVITY * (RHO + rho) * z
    return u, w, rho, p, np.stack(np.broadcast_arrays(np.ones_like(u), TREF + np.zeros_like(u), S, RHO*u, RHO*spec.V + np.zeros_like(u), KE, PE), axis=-1)


def evolved(spec, time):
    a = (spec.external[1] - spec.external[0]) / (RHO * spec.distance)
    lam = 1. + spec.alpha * time
    return MeanFlowSpecification(distance=spec.distance, length=spec.length, eta=spec.eta-(spec.eta-spec.bottom)*(spec.alpha*time)/lam,
                                 bottom=spec.bottom, U=(spec.U-a*time-.5*a*spec.alpha*time**2)/lam,
                                 alpha=spec.alpha/lam, V=spec.V, density_intercept=(spec.density_intercept+spec.density_slope*spec.bottom)-spec.density_slope*lam*spec.bottom,
                                 density_slope=spec.density_slope*lam, external=spec.external, interior_interfaces=spec.interior_interfaces)


def direct_volumes(spec, time):
    """Seven-point raw half-CV integrals from declared initial physical inputs."""
    eta = spec.eta-(spec.eta-spec.bottom)*(spec.alpha*time)/(1.+spec.alpha*time)
    zcuts = np.array([[eta, *column] for column in spec.interior_interfaces])
    nodes, weights = np.polynomial.legendre.leggauss(7)
    h = -np.diff(zcuts, axis=1)
    stocks = np.zeros((2, 14, 4))
    energy = np.zeros((2, 14, 2))
    for side in range(2):
        x = spec.distance * (side / 2. + (nodes + 1.) / 4.)
        for layer in range(14):
            z = .5 * (zcuts[side, layer] + zcuts[side, layer+1]) + .5 * h[side, layer] * nodes
            values = solution(spec, time, x[:, None], z[None, :])[-1]
            averaged = np.einsum('i,j,ijc->c', weights/2., weights/2., values)
            stocks[side, layer] = h[side, layer] * averaged[1:5]
            energy[side, layer] = spec.distance * spec.length / 2. * h[side, layer] * averaged[5:7]
    return h, stocks, energy


def actual_energies(profile, distance, length):
    """Actual mean reconstruction via independent volume Q, then actual P1 PE."""
    state = profile.state
    nodes, weights = np.polynomial.legendre.leggauss(7)
    Q = np.zeros((28, 28))
    cuts = np.unique(state.interfaces)
    samples = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        middle = (lo + hi) / 2.
        owners = [int(np.flatnonzero((state.interfaces[s, 1:] < middle) & (middle < state.interfaces[s, :-1]))[0]) for s in range(2)]
        for side in range(2):
            for node, weight in zip(nodes, weights):
                x = distance * (side / 2. + (node+1.) / 4.)
                basis = np.zeros(28)
                basis[owners[0]], basis[14+owners[1]] = 1.-x/distance, x/distance
                volume = (hi-lo) * length * distance * weight / 4.
                row = side * 14 + owners[side]
                Q[row] += volume * basis
                samples.append((row, basis, volume))
    Q /= (distance*length/2. * state.h.ravel())[:, None]
    means = (state.stocks[..., 2:] / (RHO * state.h[..., None])).reshape(28, 2)
    endpoint = np.linalg.solve(Q, means)
    kinetic = np.zeros(28)
    for row, basis, volume in samples:
        kinetic[row] += .5 * RHO * volume * np.sum((basis @ endpoint)**2)
    potential = np.zeros((2, 14))
    for side in range(2):
        for layer in range(14):
            center = .5 * (state.interfaces[side, layer] + state.interfaces[side, layer+1])
            z = center + .5 * state.h[side, layer] * nodes
            rho = profile.density_mean[side, layer] + profile.density_slope[side, layer] * (z-center)
            potential[side, layer] = distance*length/2. * state.h[side, layer] * np.sum(weights/2. * GRAVITY * (RHO+rho) * z)
    return np.stack((kinetic.reshape(2, 14), potential), axis=-1)


def absolute_faces(spec, dt):
    """Independent 32-time, seven-space rules and dimensional tail majorants."""
    times, tw = np.polynomial.legendre.leggauss(32)
    t, tw = dt * (times[:, None]+1.) / 2., dt * tw / 2.
    nodes, weights = np.polynomial.legendre.leggauss(7)
    H = spec.eta-spec.bottom
    alpha, q = spec.alpha, spec.alpha*dt
    lam = 1. + alpha*t
    eta = spec.bottom + H/lam
    rows, vertical = [], []
    cuts = np.unique([spec.eta, *np.asarray(spec.interior_interfaces).ravel()])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        moving = hi == spec.eta
        upper = eta if moving else hi + np.zeros_like(t)
        width = upper-lo
        z = lo + width*(nodes[None, :]+1.)/2.
        transport, pressure, pwork = [], [], []
        for x in (0., spec.distance/2., spec.distance):
            u, _, _, p, stock = solution(spec, t, x, z)
            measure = tw[:, None] * weights[None, :]/2. * width * spec.length
            transport.append(np.einsum('ij,ij,ijc->c', measure, u, stock))
            pressure.append(float(np.sum(measure*p)))
            pwork.append(float(np.sum(measure*p*u)))
        rows.append(dict(lower=float(lo), upper=float(hi), transport=np.array(transport), pressure=np.array(pressure), pressure_energy=np.array(pwork)))
    for side in range(2):
        x = spec.distance*(side/2. + (nodes[None, :]+1.)/4.)
        measure = tw[:, None] * weights[None, :]/2. * spec.distance*spec.length/2.
        for k in range(15):
            z = eta if k == 0 else spec.interior_interfaces[side][k-1]
            _, w, _, p, stock = solution(spec, t, x, z)
            if 0 < k < 14:
                transport = np.einsum('ij,ij,ijc->c', measure, -w, stock)
                pwork = float(np.sum(measure*p*(-w)))
            else:
                transport = np.zeros(7)
                pwork = float(np.sum(measure*p*(alpha*H/lam**2))) if k == 0 else 0.
            vertical.append(dict(side=side, interface=k, transport=transport, pressure_energy=pwork))
    # Coefficient norms of Eulerian rational forms, independent of candidate polynomial.
    norm_lambda = 1.+abs(q)
    velocity = abs(spec.U)+abs(alpha)*spec.distance+abs((spec.external[1]-spec.external[0])/(RHO*spec.distance))*dt*(1.+.5*abs(q))
    zheight = H*(1.+norm_lambda)
    zabsolute = abs(spec.bottom)*norm_lambda+zheight
    qb = spec.density_intercept+spec.density_slope*spec.bottom
    density = abs(qb)*norm_lambda+abs(spec.density_slope)*norm_lambda*zheight
    tracer = SREF*norm_lambda+density/(RHO*BETA)
    potential = GRAVITY*(RHO*norm_lambda+density)*zabsolute
    kinetic = .5*RHO*(velocity**2+spec.V**2*norm_lambda**2)
    pressure = max(abs(v) for v in spec.external)*norm_lambda**2 + GRAVITY*((RHO+abs(qb))*zheight*norm_lambda+.5*abs(spec.density_slope)*norm_lambda*(H**2+zheight**2))
    coeff = np.array([1., TREF, tracer, RHO*velocity, RHO*abs(spec.V), kinetic, potential])
    # Promotion to a common denominator only multiplies norms by norm_lambda^8.
    horizontal_norm = spec.length*zheight*velocity*coeff*norm_lambda**8
    vertical_norm = spec.distance*spec.length/2.*abs(alpha)*zheight*coeff*norm_lambda**8
    pa_norm = spec.length*zheight*pressure*norm_lambda**8
    pw_norm = pa_norm*velocity
    vertical_pw_norm = spec.distance*spec.length/2.*pressure*abs(alpha)*zheight*norm_lambda**8
    tail = math.comb(63, 7)*abs(q)**56/(1.-64./57.*abs(q))
    return dict(horizontal=rows, vertical=vertical, horizontal_tail=2.*dt*horizontal_norm*tail,
                vertical_tail=2.*dt*vertical_norm*tail, pressure_tail=2.*dt*pa_norm*tail,
                pressure_energy_tail=2.*dt*pw_norm*tail, vertical_pressure_energy_tail=2.*dt*vertical_pw_norm*tail)


def check_faces(spec, faces, dt):
    independent = absolute_faces(spec, dt)
    if len(faces['horizontal']) != len(independent['horizontal']) or len(faces['vertical']) != 30:
        raise ValueError('absolute face partition count failed')
    maximum = 0.
    interfaces = np.array([[spec.eta,*column] for column in spec.interior_interfaces])
    for actual, expected in zip(faces['horizontal'], independent['horizontal']):
        if actual['lower'] != expected['lower'] or (expected['upper'] != spec.eta and actual['upper'] != expected['upper']):
            raise ValueError('absolute physical partition changed')
        if expected['upper'] == spec.eta:
            maximum = max(maximum,assert_bound(actual['upper'],spec.eta,abs(spec.eta)+abs(spec.bottom)+(spec.eta-spec.bottom),label='absolute moving-top geometry'))
        middle = (expected['lower']+expected['upper'])/2.
        owners = tuple(int(np.flatnonzero((interfaces[side,1:] < middle)&(middle < interfaces[side,:-1]))[0]) for side in range(2))
        if actual['owners'] != owners:
            raise ValueError('absolute raw face ownership changed')
        for key, tail_key in [('transport','horizontal_tail'), ('pressure','pressure_tail'), ('pressure_energy','pressure_energy_tail')]:
            maximum = max(maximum, assert_bound(actual[key], expected[key], actual[key+'_scale']+abs(expected[key]),
                                               extra=actual[key+'_truncation']+independent[tail_key]+actual[key+'_input'], label='absolute horizontal '+key))
    for actual, expected in zip(faces['vertical'], independent['vertical']):
        if (actual['side'], actual['interface']) != (expected['side'], expected['interface']):
            raise ValueError('absolute vertical owner changed')
        for key, tail_key in [('transport','vertical_tail'), ('pressure_energy','vertical_pressure_energy_tail')]:
            maximum = max(maximum, assert_bound(actual[key], expected[key], actual[key+'_scale']+abs(expected[key]),
                                               extra=actual[key+'_truncation']+independent[tail_key]+actual[key+'_input'], label='absolute vertical '+key))
    return maximum


def independent_ledger(spec, faces):
    """Signed fsum ledger, deriving CV ownership independently of consumer rows."""
    flow = [[[[] for _ in range(7)] for _ in range(14)] for _ in range(2)]
    force = [[[] for _ in range(14)] for _ in range(2)]
    pressure_work = [[[] for _ in range(14)] for _ in range(2)]
    interfaces = np.array([[spec.eta, *column] for column in spec.interior_interfaces])
    for row in faces['horizontal']:
        middle = (row['lower']+row['upper'])/2.
        for side in range(2):
            layer = int(np.flatnonzero((interfaces[side, 1:] < middle) & (middle < interfaces[side, :-1]))[0])
            for channel in range(7):
                flow[side][layer][channel].extend([float(row['transport'][side+1,channel]),-float(row['transport'][side,channel])])
            force[side][layer].extend([float(row['pressure'][side]),-float(row['pressure'][side+1])])
            pressure_work[side][layer].extend([float(row['pressure_energy'][side]),-float(row['pressure_energy'][side+1])])
    for row in faces['vertical']:
        side,k = row['side'],row['interface']
        if k < 14:
            for channel in range(7):
                flow[side][k][channel].append(-float(row['transport'][channel]))
            pressure_work[side][k].append(float(row['pressure_energy']))
        if k > 0:
            for channel in range(7):
                flow[side][k-1][channel].append(float(row['transport'][channel]))
            pressure_work[side][k-1].append(-float(row['pressure_energy']))
    flow = np.array([[[math.fsum(values) for values in row] for row in column] for column in flow])
    force = np.array([[math.fsum(values) for values in column] for column in force])
    pressure_work = np.array([[math.fsum(values) for values in column] for column in pressure_work])
    return flow,force,pressure_work


def true_pressure_work(spec, initial_time, dt):
    nodes, weights = np.polynomial.legendre.leggauss(32)
    times = initial_time + dt*(nodes+1.)/2.
    acceleration = (spec.external[1]-spec.external[0])/(RHO*spec.distance)
    u = solution(spec,times,spec.distance/2.,spec.bottom)[0]
    H = (spec.eta-spec.bottom)/(1.+spec.alpha*times)
    terms = -RHO*acceleration*spec.distance*spec.length*H*u
    value = dt*math.fsum(weights*terms)/2.
    # Independent Eulerian coefficient norm: H*u has degree2 / lambda².
    norm = RHO*abs(acceleration)*spec.distance*spec.length*(spec.eta-spec.bottom)*(abs(spec.U)+abs(spec.alpha)*spec.distance/2.+abs(acceleration)*(initial_time+dt)*(1.+.5*abs(spec.alpha)*(initial_time+dt)))
    q = abs(spec.alpha*(initial_time+dt))
    tail = math.comb(63,7)*q**56/(1.-64./57.*q)
    error = 512.*EPS*max(1.,dt*norm/(1.-q)**2)+2.*dt*norm*tail
    return value,error


def audit_receipt(spec, receipt):
    initial_time = receipt.time_s-receipt.duration_s
    expected_before = direct_volumes(spec, initial_time)
    expected_after = direct_volumes(spec, receipt.time_s)
    maximum = 0.
    for profile, expected in ((receipt.before, expected_before), (receipt.profile, expected_after)):
        maximum = max(maximum, assert_bound(profile.state.h, expected[0], abs(expected[0])+abs(profile.state.h), extra=receipt.volume_error_bound, label='declared actual end-CV water'))
        maximum = max(maximum, assert_bound(profile.state.stocks, expected[1], abs(expected[1])+abs(profile.state.stocks), extra=receipt.stock_error_bound, label='declared actual end-CV stocks'))
    maximum = max(maximum, check_faces(evolved(spec, initial_time), receipt.faces, receipt.duration_s))
    flow,force,pressure_work = independent_ledger(evolved(spec,initial_time),receipt.faces)
    area = spec.distance*spec.length/2.
    state0,state1 = receipt.before.state,receipt.profile.state
    maximum = max(maximum, assert_bound(area*(state1.h-state0.h)+flow[...,0],0.,area*(abs(state0.h)+abs(state1.h))+abs(flow[...,0]),
                                        extra=area*receipt.volume_error_bound,label='independent actual water ledger'))
    change = area*(state1.stocks-state0.stocks)+flow[...,1:5]
    change[...,2] -= force
    stock_scale = area*(abs(state0.stocks)+abs(state1.stocks))+abs(flow[...,1:5])
    stock_scale[...,2] += abs(force)
    maximum = max(maximum, assert_bound(change,0.,stock_scale,
                                        label='independent actual TS impulse ledger'))
    maximum = max(maximum, assert_bound(np.sum(change,axis=(0,1)),0.,np.sum(stock_scale,axis=(0,1)),label='global actual TS impulse ledger'))
    actual0, actual1 = actual_energies(receipt.before, spec.distance, spec.length), actual_energies(receipt.profile, spec.distance, spec.length)
    maximum = max(maximum, assert_bound(np.sum(actual1-actual0+flow[...,5:7], axis=-1), pressure_work,
                                        np.sum(abs(actual0)+abs(actual1), axis=-1)+receipt.energy_scale,
                                        extra=receipt.energy_error_bound_J, label='independent local physical KE PE'))
    work,error = true_pressure_work(spec,initial_time,receipt.duration_s)
    maximum = max(maximum,assert_bound(receipt.true_pressure_work_J,work,abs(work)+abs(receipt.true_pressure_work_J),extra=error+receipt.work_error_bound_J,label='independent true pressure time work'))
    maximum = max(maximum,assert_bound(np.sum(actual1[...,0]-actual0[...,0]+flow[...,5]),work,np.sum(abs(actual0[...,0])+abs(actual1[...,0])+abs(flow[...,5])),
                                       extra=error+float(np.sum(receipt.energy_error_bound_J)),label='independent finite KE pressure work ledger'))
    # Independent Eulerian coefficient/PDE residuals at nontrivial interior points.
    t = receipt.time_s*.37
    lam, a = 1.+spec.alpha*t, (spec.external[1]-spec.external[0])/(RHO*spec.distance)
    alpha = spec.alpha/lam
    numerator = spec.U-a*t-.5*a*spec.alpha*t*t
    U, Ud = numerator/lam, ((-a-a*spec.alpha*t)*lam-numerator*spec.alpha)/lam**2
    assert_bound(Ud+alpha*U, -a, abs(Ud)+abs(alpha*U)+abs(a), label='Eulerian horizontal PDE')
    H = (spec.eta-spec.bottom)/lam
    assert_bound(-spec.alpha*(spec.eta-spec.bottom)/lam**2+alpha*H, 0., 2.*abs(alpha*H), label='Eulerian continuity')
    return dict(passed=True, maximum_bound_ratio=maximum, accepted_raw_steps=receipt.accepted_raw_steps,
                moving=receipt.accepted_moving_geometry, independent_time_points=32, independent_space_points=7,
                maximum_independent_stock_ledger_residuals={name:float(np.max(abs(change[...,channel]))) for channel,name in enumerate(('IT_temperature_m3','IS_salinity_m3','Mu_kg_m_s','Mv_kg_m_s'))},
                maximum_independent_local_energy_residual_J=float(np.max(abs(np.sum(actual1-actual0+flow[...,5:7],axis=-1)-pressure_work))),
                independent_global_energy_residual_J=math.fsum((np.sum(actual1-actual0+flow[...,5:7],axis=-1)-pressure_work).ravel()))
