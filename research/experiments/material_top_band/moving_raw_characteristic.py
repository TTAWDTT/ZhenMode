"""Flux-only accepted raw ALE step for one explicit manufactured mean family.

Only the first raw slot moves. This does not adapt the production integrator.
Canonical reconstruction is qualified against actual stocks/P1 endpoints;
analytic endpoint stocks are never assigned to the accepted state.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from . import inventory_pressure as inventory
from .fixed_eta_raw import variable_mass_identity
from .moving_raw_cases import MeanFlowSpecification
from .moving_raw_oracle import assert_bound, check_faces, spec_from_profile
from .rational_time_integral import RationalTimePolynomial as TimePolynomial
from .raw_mean_geometry import RawMeanGeometry
from .real_geometry import _scalar
from .slope_dual_stock import readonly

EPS = np.finfo(float).eps


def upper_round(value):
    """Outward upper rounding of one finite primitive operation."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite input extent')
    result = math.nextafter(value,math.inf)
    if not math.isfinite(result):
        raise ValueError('nonfinite input extent')
    return result


def lower_positive(value):
    """Outward lower rounding for a strictly positive denominator."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite input extent denominator')
    result = math.nextafter(value,-math.inf)
    if not math.isfinite(result) or result <= 0.:
        raise ValueError('input extent leaves positive denominator domain')
    return result


def upper_product(*factors):
    """Upper product of nonnegative factors, rounded at every operation."""
    result = 1.
    for factor in factors:
        if not math.isfinite(float(factor)) or factor < 0.:
            raise ValueError('nonfinite or negative input extent factor')
        result = 0. if result == 0. or factor == 0. else upper_round(result*factor)
    return result


def canonical_cap_extent(spec,dt):
    """Nominal monotone cap expansion; no endpoint-subtraction cancellation."""
    if spec.alpha >= 0.:
        return 0.,spec.eta
    height = upper_round(spec.eta-spec.bottom)
    ratio = upper_product(-spec.alpha,dt)
    denominator = lower_positive(1.-ratio)
    expansion = upper_round(upper_product(height,ratio)/denominator)
    return expansion,upper_round(spec.eta+expansion)


def local_energy(geometry):
    """Own exact half-prism moments; independent oracle uses seven-point volume rules."""
    state, eos = geometry.profile.state, geometry.profile.eos
    endpoint = geometry.inverse @ geometry.means
    energy = np.zeros((2, 14, 2))
    cuts = np.unique(state.interfaces)
    moments = (np.array([[7./24., 1./12.], [1./12., 1./24.]]), np.array([[1./24., 1./12.], [1./12., 7./24.]]))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        middle = .5*(lo+hi)
        owners = [int(np.flatnonzero((state.interfaces[s, 1:] < middle) & (middle < state.interfaces[s, :-1]))[0]) for s in range(2)]
        c = endpoint[[owners[0], 14+owners[1]]]
        for side in range(2):
            W = eos.rho0*geometry.length*geometry.distance*(hi-lo)*moments[side]
            energy[side, owners[side], 0] += .5*math.fsum((c*(W@c)).ravel())
    for side in range(2):
        for k in range(14):
            lo, hi = state.interfaces[side, k+1], state.interfaces[side, k]
            anomaly = inventory.density_integral(geometry.profile, (side,), lo, hi, first_moment=True)
            energy[side, k, 1] = geometry.area*eos.gravity*(.5*eos.rho0*(hi-lo)*(hi+lo)+anomaly)
    return energy


@dataclass(frozen=True)
class MovingRawReceipt:
    before: inventory.Profile
    profile: inventory.Profile
    time_s: float
    duration_s: float
    accepted_raw_steps: int
    faces: dict
    net_water: np.ndarray
    net_stocks: np.ndarray
    pressure_impulse: np.ndarray
    net_energy: np.ndarray
    net_pressure_energy: np.ndarray
    energy_scale: np.ndarray
    energy_error_bound_J: np.ndarray
    volume_error_bound: np.ndarray
    stock_error_bound: np.ndarray
    gcl_residual: np.ndarray
    cumulative_gcl: np.ndarray
    cumulative_gcl_bound: np.ndarray
    M_before: np.ndarray
    M_after: np.ndarray
    R_before: np.ndarray
    R_after: np.ndarray
    physical_KE_change_J: float
    raw_KE_change_J: float
    PE_change_J: float
    covariance_change_J: float
    true_pressure_work_J: float
    midpoint_pressure_work_J: float
    work_error_bound_J: float
    raw_mass_chain: dict
    physical_mass_chain: dict
    maximum_face_bound_ratio: float
    accepted_moving_geometry: bool
    production_qualified: bool = False
    original_global_CV_identified: bool = False


class MovingRawSlice:
    def __init__(self, profile, *, authority, distance_m=2., length_m=1.5):
        geometry = RawMeanGeometry(profile, authority=authority, distance_m=distance_m, length_m=length_m)
        self.authority, self.distance, self.length = authority, geometry.distance, geometry.length
        self.profile = geometry.profile
        self._binding(self.profile)
        self.time_s, self.accepted_raw_steps, self.last_receipt = 0., 0, None
        self.cumulative_gcl = readonly(np.zeros((2, 14)))
        self.cumulative_gcl_bound = readonly(np.zeros((2, 14)))

    def _binding(self, profile):
        geometry = RawMeanGeometry(profile, authority=self.authority, distance_m=self.distance, length_m=self.length)
        profile, state, eos = geometry.profile, geometry.profile.state, geometry.profile.eos
        if profile.representation != 'p1' or profile.slope_limited_count or profile.density_slope_limited_count:
            raise ValueError('inactive actual P1 TS/density limiter required')
        endpoint = geometry.inverse @ geometry.means
        U = math.fsum(endpoint[:14, 0])/14.
        right = math.fsum(endpoint[14:, 0])/14.
        alpha, V = (right-U)/self.distance, math.fsum(endpoint[:, 1])/28.
        expected = np.tile([[U,V]], (28,1))
        expected[14:, 0] = right
        velocity_scale = abs(geometry.inverse) @ abs(geometry.means) + abs(expected)
        assert_bound(endpoint, expected, velocity_scale, label='depth-uniform actual mean affine velocity')
        centers = .5*(state.interfaces[:, :-1]+state.interfaces[:, 1:])
        slope = math.fsum((profile.density_mean[:, -1]-profile.density_mean[:, 0])/(centers[:, -1]-centers[:, 0]))/2.
        intercept = math.fsum(profile.density_mean[:, 0]-slope*centers[:, 0])/2.
        eos_scale = eos.rho0*(eos.alpha*(abs(profile.temperature_mean)+eos.Tref)+eos.beta*(abs(profile.salinity_mean)+eos.Sref))
        assert_bound(profile.density_mean, intercept+slope*centers, eos_scale+abs(intercept)+abs(slope*centers), label='common a1=0 canonical density')
        if slope > 0.:
            raise ValueError('stable nonpositive density slope required')
        delta_T, delta_S, delta_rho = 0., 0., 0.
        for sign in (-1.,1.):
            z = centers+sign*.5*state.h
            T = profile.temperature_mean+sign*.5*state.h*profile.temperature_slope
            S = profile.salinity_mean+sign*.5*state.h*profile.salinity_slope
            canonical_rho = intercept+slope*z
            expected_S = eos.Sref+canonical_rho/(eos.rho0*eos.beta)
            actual_rho = profile.density_mean+sign*.5*state.h*profile.density_slope
            assert_bound(T, eos.Tref, abs(T)+eos.Tref, label='actual P1 temperature endpoints')
            assert_bound(S, expected_S, abs(S)+abs(expected_S)+eos_scale/(eos.rho0*eos.beta), label='actual P1 salinity endpoints')
            assert_bound(actual_rho, canonical_rho, eos_scale+abs(canonical_rho)+abs(actual_rho), label='actual P1 density endpoints')
            delta_T = max(delta_T, float(np.max(abs(T-eos.Tref)+512.*EPS*(abs(T)+eos.Tref))))
            delta_S = max(delta_S, float(np.max(abs(S-expected_S)+512.*EPS*(abs(S)+abs(expected_S)+eos_scale/(eos.rho0*eos.beta)))))
            delta_rho = max(delta_rho, float(np.max(abs(actual_rho-canonical_rho)+512.*EPS*(eos_scale+abs(canonical_rho)+abs(actual_rho)))))
        delta_velocity = np.max(abs(endpoint-expected)+512.*EPS*np.maximum(1.,velocity_scale), axis=0)
        spec = MeanFlowSpecification(distance=self.distance, length=self.length, eta=float(state.eta[0]), bottom=float(state.bottom[0]),
                                     U=U, alpha=alpha, V=V, density_intercept=intercept, density_slope=slope,
                                     external=tuple(profile.external_pressure_Pa), interior_interfaces=tuple(tuple(z) for z in state.interfaces[:,1:]))
        return dict(geometry=geometry, spec=spec, delta_T=delta_T, delta_S=delta_S, delta_rho=delta_rho, delta_velocity=delta_velocity)

    @staticmethod
    def _input_envelope(binding, dt):
        """A priori product/quotient bounds from pre-step actual P1 discrepancies."""
        spec, eos = binding['spec'], binding['geometry'].profile.eos
        H = upper_round(spec.eta-spec.bottom)
        acceleration = abs((spec.external[1]-spec.external[0])/(eos.rho0*spec.distance))
        du0, dv = binding['delta_velocity']
        da = upper_round(upper_product(4.,du0)/spec.distance)
        alpha_abs = upper_round(abs(spec.alpha)+da)
        q = upper_product(dt,alpha_abs)
        lower, upper = lower_positive(1.-q),upper_round(1.+q)
        lower_squared = lower_positive(lower*lower)
        lower_cubed = lower_positive(lower_squared*lower)
        expansion,eta_max = canonical_cap_extent(spec,dt)
        U_abs = upper_round(abs(spec.U)+du0)
        # Outside [0,d], the initial affine extension uses absolute endpoint weights.
        excursion = (U_abs*dt+.5*acceleration*dt*dt+alpha_abs*dt*spec.distance)/lower
        extrapolation = 1.+2.*excursion/spec.distance
        du = du0*extrapolation/lower+da*(spec.distance+U_abs*dt+.5*acceleration*dt*dt)/lower_squared
        dv *= extrapolation
        dw = upper_round(upper_product(da,H)/lower_squared)
        deta = upper_round(upper_product(H,dt,da)/lower_squared)
        # Fixed-cut w has a different alpha derivative from material cap eta_dot.
        cap_rate_error = upper_round(upper_product(H,da,upper)/lower_cubed)
        drho = binding['delta_rho']*extrapolation + abs(spec.density_slope)*upper*deta + 2.*binding['delta_rho']*dt*da
        umax = (abs(spec.U)+abs(spec.alpha)*spec.distance+acceleration*dt*(1.+.5*q))/lower
        wmax = abs(spec.alpha)*H/lower_squared
        zmax = upper_round(max(abs(spec.bottom),abs(spec.eta),abs(eta_max))+deta)
        rhomax = abs(spec.density_intercept+spec.density_slope*spec.bottom)+abs(spec.density_slope)*upper*H/lower
        dS = binding['delta_S']*extrapolation+drho/(eos.rho0*eos.beta)
        content = np.array([1.,eos.Tref,eos.Sref+rhomax/(eos.rho0*eos.beta),eos.rho0*umax,eos.rho0*abs(spec.V),
                            .5*eos.rho0*(umax**2+spec.V**2),eos.gravity*(eos.rho0+rhomax)*zmax])
        change = np.array([0.,binding['delta_T']*extrapolation,dS,eos.rho0*du,eos.rho0*dv,
                           eos.rho0*(umax*du+.5*du**2+abs(spec.V)*dv+.5*dv**2),eos.gravity*(drho*zmax+(eos.rho0+rhomax)*deta)])
        pressure = max(abs(v) for v in spec.external)+eos.gravity*(eos.rho0+rhomax)*H/lower
        dp = eos.gravity*((eos.rho0+rhomax)*deta+drho*H/lower)
        return dict(content=content,change=change,du=du,dw=dw,deta=deta,drho=drho,umax=umax,wmax=wmax,pressure=pressure,dp=dp,
                    cap_rate_error=cap_rate_error,alpha_interval_q_max=q,canonical_cap_expansion_m=expansion,zmax=zmax)

    def _faces(self, binding, dt):
        spec, geometry = binding['spec'], binding['geometry']
        eos, state = geometry.profile.eos, geometry.profile.state
        q = spec.alpha*dt
        def polynomial(value):
            return TimePolynomial(np.atleast_1d(value), q)
        time, lam = polynomial([0.,dt]), polynomial([1.,q])
        H, bottom = spec.eta-spec.bottom, spec.bottom
        eta = bottom+polynomial(H).over_lambda()
        acceleration = (spec.external[1]-spec.external[0])/(eos.rho0*spec.distance)
        qb = spec.density_intercept+spec.density_slope*bottom
        envelope = self._input_envelope(binding, dt)

        def field(x, z):
            label_x = (x-spec.U*time+.5*acceleration*time*time).over_lambda()
            u = spec.U+spec.alpha*label_x-acceleration*time
            label_z = bottom+lam*(z-bottom)
            rho = qb+spec.density_slope*(label_z-bottom)
            pressure = spec.external[0]+(spec.external[1]-spec.external[0])*x/spec.distance
            pressure = pressure+eos.gravity*((eos.rho0+qb)*(eta-z)+.5*spec.density_slope*lam*((eta-bottom)*(eta-bottom)-(z-bottom)*(z-bottom)))
            content = [polynomial(1.),polynomial(eos.Tref),eos.Sref+rho*(1./(eos.rho0*eos.beta)),eos.rho0*u,polynomial(eos.rho0*spec.V),
                       .5*eos.rho0*(u*u+spec.V**2),eos.gravity*(eos.rho0+rho)*z]
            return u, pressure, content

        def integrate(functions, limits):
            results = [f.integrate(dt, degree_limit=limit[0], power_limit=limit[1]) for f,limit in zip(functions,limits)]
            return {key: np.array([result[key] for result in results]) for key in ('value','scale','truncation','degree','power')}

        nodes = [-1./math.sqrt(3.),1./math.sqrt(3.)]
        horizontal, vertical = [], []
        cuts = np.unique(state.interfaces)
        hlimits = [(3,2),(3,2),(5,3),(5,3),(3,2),(7,4),(6,4)]
        for lo, hi in zip(cuts[:-1],cuts[1:]):
            width = eta-lo if hi == spec.eta else polynomial(hi-lo)
            middle = .5*(lo+hi)
            owners = tuple(int(np.flatnonzero((state.interfaces[s,1:] < middle)&(middle < state.interfaces[s,:-1]))[0]) for s in range(2))
            row = dict(lower=float(lo),upper=float(hi),owners=owners)
            transports, pressures, works = [], [], []
            for x in (0.,spec.distance/2.,spec.distance):
                terms, pa, pw = [polynomial(0.) for _ in range(7)],polynomial(0.),polynomial(0.)
                for node in nodes:
                    z = lo+.5*(node+1.)*width
                    u,p,content = field(x,z)
                    measure = .5*spec.length*width
                    terms = [old+measure*u*c for old,c in zip(terms,content)]
                    pa, pw = pa+measure*p,pw+measure*p*u
                transports.append(integrate(terms,hlimits))
                pressures.append(integrate([pa],[(4,3)]))
                works.append(integrate([pw],[(6,4)]))
            for key, values in [('transport',transports),('pressure',pressures),('pressure_energy',works)]:
                for suffix in ('value','scale','truncation','degree','power'):
                    array = np.array([value[suffix] for value in values])
                    row[key if suffix == 'value' else key+'_'+suffix] = array if key == 'transport' else array[:,0]
            maximum_width = upper_round(hi-lo)
            if hi == spec.eta:
                maximum_width = upper_round(maximum_width+envelope['canonical_cap_expansion_m'])
            measure_max = upper_product(spec.length,dt,upper_round(maximum_width+envelope['deta']))
            row['canonical_max_width_m'] = maximum_width
            row['input_area_time_bound_m2_s'] = measure_max
            flux_input = measure_max*(envelope['du']*envelope['content']+envelope['umax']*envelope['change']+envelope['du']*envelope['change'])
            flux_input += spec.length*dt*envelope['deta']*(envelope['umax']+envelope['du'])*(envelope['content']+envelope['change'])
            row['transport_input'] = np.tile(flux_input,(3,1))
            row['pressure_input'] = np.full(3,measure_max*envelope['dp']+spec.length*dt*envelope['deta']*envelope['pressure'])
            row['pressure_energy_input'] = row['pressure_input']*(envelope['umax']+envelope['du'])+measure_max*envelope['pressure']*envelope['du']
            horizontal.append(row)
        vlimits = [(0,1),(0,1),(1,1),(2,2),(0,1),(4,3),(1,1)]
        for side in range(2):
            for k in range(15):
                row = dict(side=side,interface=k)
                terms, pw = [polynomial(0.) for _ in range(7)],polynomial(0.)
                for node in nodes:
                    x = spec.distance*(side/2.+(node+1.)/4.)
                    z = eta if k == 0 else polynomial(state.interfaces[side,k])
                    _,p,content = field(x,z)
                    measure = geometry.area/2.
                    relative = spec.alpha*(z-bottom).over_lambda()
                    if 0 < k < 14:
                        terms = [old+measure*relative*c for old,c in zip(terms,content)]
                        pw = pw+measure*p*relative
                    elif k == 0:
                        external = spec.external[0]+(spec.external[1]-spec.external[0])*x/spec.distance
                        pw = pw+measure*external*polynomial(spec.alpha*H).over_lambda(2)
                for key, functions, limits in [('transport',terms,vlimits),('pressure_energy',[pw],[(3,3) if k else (0,2)])]:
                    result = integrate(functions,limits)
                    for suffix in ('value','scale','truncation','degree','power'):
                        row[key if suffix == 'value' else key+'_'+suffix] = result[suffix] if key == 'transport' else float(result[suffix][0])
                row['transport_input'] = geometry.area*dt*(envelope['dw']*envelope['content']+envelope['wmax']*envelope['change']+envelope['dw']*envelope['change']) if 0 < k < 14 else np.zeros(7)
                rate_error = envelope['cap_rate_error'] if k == 0 else envelope['dw']
                row['pressure_energy_input'] = geometry.area*dt*(envelope['dp']*(envelope['wmax']+rate_error)+envelope['pressure']*rate_error) if k < 14 else 0.
                vertical.append(row)
        force_work = []
        for side in range(2):
            x = spec.distance*(side/2.+.25)
            u,_,_ = field(x,polynomial(bottom))
            for k in range(14):
                h = eta-state.interfaces[side,1] if k == 0 else polynomial(state.h[side,k])
                force_work.append(integrate([-eos.rho0*acceleration*geometry.area*h*u],[(3,2)]))
        height_max = upper_round(upper_round(H+envelope['canonical_cap_expansion_m'])+envelope['deta'])
        work_product = upper_round(upper_product(height_max,envelope['du'])+upper_product(envelope['deta'],envelope['umax']))
        body_work_input = upper_product(abs((spec.external[1]-spec.external[0])/spec.distance),spec.distance,spec.length,dt,work_product)
        return dict(horizontal=horizontal,vertical=vertical,body_work=force_work,body_work_input_J=body_work_input)

    @staticmethod
    def _consume(faces):
        net, scale, error = np.zeros((2,14,7)),np.zeros((2,14,7)),np.zeros((2,14,7))
        impulse, impulse_scale, impulse_error = np.zeros((2,14)),np.zeros((2,14)),np.zeros((2,14))
        pressure_energy, pressure_scale, pressure_error = np.zeros((2,14)),np.zeros((2,14)),np.zeros((2,14))
        for row in faces['horizontal']:
            for side,k in enumerate(row['owners']):
                net[side,k] += row['transport'][side+1]-row['transport'][side]
                scale[side,k] += row['transport_scale'][side+1]+row['transport_scale'][side]
                error[side,k] += row['transport_truncation'][side+1]+row['transport_truncation'][side]+row['transport_input'][side+1]+row['transport_input'][side]
                impulse[side,k] += row['pressure'][side]-row['pressure'][side+1]
                impulse_scale[side,k] += row['pressure_scale'][side]+row['pressure_scale'][side+1]
                impulse_error[side,k] += row['pressure_truncation'][side]+row['pressure_truncation'][side+1]+row['pressure_input'][side]+row['pressure_input'][side+1]
                pressure_energy[side,k] += row['pressure_energy'][side]-row['pressure_energy'][side+1]
                pressure_scale[side,k] += row['pressure_energy_scale'][side]+row['pressure_energy_scale'][side+1]
                pressure_error[side,k] += row['pressure_energy_truncation'][side]+row['pressure_energy_truncation'][side+1]+row['pressure_energy_input'][side]+row['pressure_energy_input'][side+1]
        for row in faces['vertical']:
            side,k = row['side'],row['interface']
            for layer,sign in ((k-1,1.),(k,-1.)):
                if 0 <= layer < 14:
                    net[side,layer] += sign*row['transport']
                    scale[side,layer] += row['transport_scale']
                    error[side,layer] += row['transport_truncation']+row['transport_input']
                    pressure_energy[side,layer] -= sign*row['pressure_energy']
                    pressure_scale[side,layer] += row['pressure_energy_scale']
                    pressure_error[side,layer] += row['pressure_energy_truncation']+row['pressure_energy_input']
        return dict(net=net,scale=scale,error=error,impulse=impulse,impulse_scale=impulse_scale,impulse_error=impulse_error,
                    pressure_energy=pressure_energy,pressure_scale=pressure_scale,pressure_error=pressure_error)

    def _audit(self, receipt, binding, consumed):
        geometry0 = binding['geometry']
        geometry1 = RawMeanGeometry(receipt.profile, authority=self.authority,distance_m=self.distance,length_m=self.length)
        energy0,energy1 = local_energy(geometry0),local_energy(geometry1)
        assert_bound(np.sum(energy1-energy0+receipt.net_energy,axis=-1),receipt.net_pressure_energy,
                     receipt.energy_scale+np.sum(abs(energy0)+abs(energy1),axis=-1),extra=receipt.energy_error_bound_J,label='actual local physical KE PE pressure balance')
        assert_bound(np.sum(energy1[...,0]-energy0[...,0]+receipt.net_energy[...,0]),receipt.true_pressure_work_J,
                     np.sum(abs(energy0[...,0])+abs(energy1[...,0])+consumed['scale'][...,5]),extra=float(np.sum(receipt.energy_error_bound_J))+receipt.work_error_bound_J,label='true finite pressure work KE transport')
        self._binding(receipt.profile)

    def step(self, duration_s):
        dt = _scalar(duration_s,'duration',positive=True)
        if dt > .05:
            raise ValueError('duration exceeds frozen maximum')
        binding = self._binding(self.profile)
        geometry0,spec = binding['geometry'],binding['spec']
        state = self.profile.state
        q = spec.alpha*dt
        if abs(q) > .02:
            raise ValueError('alpha dt exceeds frozen maximum')
        z = state.interfaces.copy()
        z[:,0] = spec.eta-(spec.eta-spec.bottom)*q/(1.+q)
        h = -np.diff(z,axis=1)
        if np.any(h[:,0] < .5*state.h[:,0]) or np.any(h <= 0.):
            raise ValueError('top thickness/topology crossing refuses')
        faces = self._faces(binding,dt)
        ratio = check_faces(spec_from_profile(self.profile,self.distance,self.length),faces,dt)
        consumed = self._consume(faces)
        net, scale, error, area = consumed['net'],consumed['scale'],consumed['error'],geometry0.area
        gcl = area*(h-state.h)+net[...,0]
        gcl_scale = area*(abs(h)+abs(state.h))+scale[...,0]
        assert_bound(gcl,0.,gcl_scale,extra=error[...,0],label='actual local ALE GCL')
        cumulative = self.cumulative_gcl+gcl
        step_gcl_bound = 512.*EPS*np.maximum(1.,gcl_scale)+error[...,0]
        cumulative_bound = np.nextafter(self.cumulative_gcl_bound+step_gcl_bound+512.*EPS*(abs(self.cumulative_gcl)+abs(gcl)),np.inf)
        if not np.isfinite(cumulative_bound).all() or np.any(abs(cumulative) > cumulative_bound):
            raise ValueError('actual cumulative ALE GCL bound failed')
        stocks = state.stocks-net[...,1:5]/area
        stocks[...,2] += consumed['impulse']/area
        after = inventory.reconstruct(replace(state,eta=z[:,0].copy(),interfaces=z,h=h,stocks=stocks),external_pressure_Pa=self.profile.external_pressure_Pa)
        geometry1 = RawMeanGeometry(after,authority=self.authority,distance_m=self.distance,length_m=self.length)
        energy0,energy1 = local_energy(geometry0),local_energy(geometry1)
        raw_chain = variable_mass_identity(geometry0.D,geometry1.D,geometry0.means,geometry1.means)
        physical_chain = variable_mass_identity(geometry0.M,geometry1.M,geometry0.means,geometry1.means)
        body = faces['body_work']
        work = math.fsum(float(value['value'][0]) for value in body)
        work_scale = math.fsum(float(value['scale'][0]) for value in body)
        envelope = self._input_envelope(binding,dt)
        work_error = math.fsum(float(value['truncation'][0]) for value in body)+512.*EPS*max(1.,work_scale)
        work_error += faces['body_work_input_J']
        midpoint = math.fsum((.5*(geometry0.means[:,0]+geometry1.means[:,0])*consumed['impulse'].ravel()).ravel())
        stock_error = (error[...,1:5]+512.*EPS*(area*abs(state.stocks)+scale[...,1:5]))/area
        stock_error[...,2] += (consumed['impulse_error']+512.*EPS*consumed['impulse_scale'])/area
        # Initial actual/canonical input discrepancies are retained, never projected.
        stock_error += state.h[...,None]*np.array([binding['delta_T'],binding['delta_S'],geometry0.profile.eos.rho0*binding['delta_velocity'][0],geometry0.profile.eos.rho0*binding['delta_velocity'][1]])
        volume_error = (error[...,0]+512.*EPS*gcl_scale)/area
        if np.any(volume_error >= h):
            raise ValueError('input volume envelope leaves positive quotient domain')
        quotient_error = (stock_error+abs(stocks/h[...,None])*volume_error[...,None])/(h-volume_error)[...,None]
        rho_error = geometry0.profile.eos.rho0*(geometry0.profile.eos.alpha*quotient_error[...,0]+geometry0.profile.eos.beta*quotient_error[...,1])
        # Inactive P1 endpoint reconstruction has Lipschitz factor <=3 on ordered adjacent cells.
        global_rho_error = float(np.max(rho_error))
        zmax = envelope['zmax']
        PE_error = area*geometry0.profile.eos.gravity*zmax*((h+state.h)*3.*(global_rho_error+binding['delta_rho'])+
                                                          envelope['content'][6]/(geometry0.profile.eos.gravity*zmax)*volume_error)
        # J is an inverse strictly diagonally dominant mean map, ||J||infinity<=2.
        # A global endpoint perturbation controls every local physical KE integral.
        u_error = 2.*float(np.max(quotient_error[...,2]))/geometry0.profile.eos.rho0+envelope['du']
        v_error = 2.*float(np.max(quotient_error[...,3]))/geometry0.profile.eos.rho0+envelope['change'][4]/geometry0.profile.eos.rho0
        # Shape perturbation of Q^-1: ||J||infinity<=2 and a quotient-row
        # difference bound 2*delta_h/(minimum_h-delta_h), without linearization.
        shape_error = 8.*float(np.max(volume_error/(h-volume_error)))
        u_error += shape_error*(envelope['umax']+u_error)
        v_error += shape_error*(abs(spec.V)+v_error)
        K_error = area*geometry0.profile.eos.rho0*((h+state.h)*((envelope['umax']+.5*u_error)*u_error+(abs(spec.V)+.5*v_error)*v_error)+
                                                .5*volume_error*((envelope['umax']+u_error)**2+(abs(spec.V)+v_error)**2))
        energy_error = np.sum(error[...,5:7],axis=-1)+consumed['pressure_error']+PE_error+K_error
        receipt = MovingRawReceipt(self.profile,after,self.time_s+dt,dt,self.accepted_raw_steps+1,faces,
                                   readonly(net[...,0]),readonly(net[...,1:5]),readonly(consumed['impulse']),readonly(net[...,5:7]),readonly(consumed['pressure_energy']),
                                   readonly(np.sum(scale[...,5:7],axis=-1)+consumed['pressure_scale']),readonly(energy_error),readonly(volume_error),readonly(stock_error),readonly(gcl),readonly(cumulative),readonly(cumulative_bound),
                                   geometry0.M,geometry1.M,geometry0.R,geometry1.R,float(np.sum(energy1[...,0]-energy0[...,0])),raw_chain['kinetic_change'],float(np.sum(energy1[...,1]-energy0[...,1])),
                                   physical_chain['kinetic_change']-raw_chain['kinetic_change'],work,midpoint,work_error,raw_chain,physical_chain,ratio,bool(np.any(h != state.h)))
        self._audit(receipt,binding,consumed)
        # Unique accepted full state commit; every preceding check sees only a candidate.
        self.profile,self.time_s,self.accepted_raw_steps,self.last_receipt,self.cumulative_gcl,self.cumulative_gcl_bound = after,receipt.time_s,receipt.accepted_raw_steps,receipt,receipt.cumulative_gcl,receipt.cumulative_gcl_bound
        return receipt
