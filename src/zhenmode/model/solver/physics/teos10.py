"""JAX evaluation of the Roquet 75-term specific-volume polynomial.

Inputs are Absolute Salinity [g/kg], Conservative Temperature [ITS-90 degC]
and sea pressure [dbar, absolute pressure minus atmospheric pressure]. These
are NOT the legacy solver's T/S variables. An explicit CT/SR reference variant
uses these components in the FD factory and open-water coupling. The production
default and CLI remain linear; the complete benchmark case is not ready.

The mathematical coefficients are inventoried in GSW-Fortran commit
29e64d652786e1d076a05128c920f394202bfe10. This generic nested Horner evaluator
does not import GSW or duplicate its Fortran implementation. Validation uses
the unmodified, pinned Fortran routines and their analytic derivatives.
See docs/thermodynamics_zh.md for scope, units, sources and fitting limits.
"""

from zhenmode.model.config import CP0_TEOS10 as CP0
from zhenmode.model.solver.numerics.backend import jax, jnp, np

_SFAC = 0.0248826675584615

# Coefficient axes: pressure power, temperature power, transformed salinity power.
# Within each row coefficients are in increasing power order.
_V = (
    (
        (0.0010769995862, -0.00031038981976, 0.00066928067038, -0.00085047933937, 0.00058086069943, -0.00021092370507, 3.1932457305e-05),
        (-1.5649734675e-05, 3.5009599764e-05, -4.3592678561e-05, 3.4532461828e-05, -1.1959409788e-05, 1.3864594581e-06),
        (2.7762106484e-05, -3.7435842344e-05, 3.590782276e-05, -1.8698584187e-05, 3.8595339244e-06),
        (-1.6521159259e-05, 2.4141479483e-05, -1.4353633048e-05, 2.2863324556e-06),
        (6.9111322702e-06, -8.7595873154e-06, 4.3703680598e-06),
        (-8.053961554e-07, -3.30527589e-07),
        (2.0543094268e-07,),
    ),
    (
        (-6.0799143809e-05, 2.4262468747e-05, -3.4792460974e-05, 3.7470777305e-05, -1.7322218612e-05, 3.0927427253e-06),
        (1.8505765429e-05, -9.5677088156e-06, 1.1100834765e-05, -9.8447117844e-06, 2.590922526e-06),
        (-1.1716606853e-05, -2.3678308361e-07, 2.9283346295e-06, -4.88261392e-07),
        (7.9279656173e-06, -3.4558773655e-06, 3.1655306078e-07),
        (-3.4102187482e-06, 1.2956717783e-06),
        (5.0736766814e-07,),
    ),
    (
        (9.9856169219e-06, -5.8484432984e-07, -4.8122251597e-06, 4.9263106998e-06, -1.7811974727e-06),
        (-1.1736386731e-06, -5.5699154557e-06, 5.4620748834e-06, -1.3544185627e-06),
        (2.130502874e-06, 3.913738708e-07, -6.5731104067e-07),
        (-4.6132540037e-07, 7.7618888092e-09),
        (-6.3352916514e-08,),
    ),
    (
        (-1.1309361437e-06, 3.6310188515e-07, 1.674630378e-08),
        (-3.6527006553e-07, -2.7295696237e-07),
        (2.8695905159e-07,),
    ),
    (
        (1.053115308e-07, -1.1147125423e-07),
        (3.1454099902e-07,),
    ),
    ((-1.2647261286e-08,),),
    ((1.961350393e-09,),),
)


def _inputs(sa, ct, pressure):
    dtype = jnp.result_type(sa, ct, pressure, 1.0)
    return jnp.broadcast_arrays(*(jnp.asarray(value, dtype=dtype)
                                  for value in (sa, ct, pressure)))


def _horner(coefficients, x):
    value = jnp.asarray(coefficients[-1], dtype=x.dtype)
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def specific_volume(sa_g_kg, ct_deg_c, sea_pressure_dbar):
    """Specific volume [m³/kg]; pure broadcastable JAX kernel, no clipping.

    Host callers must validate inputs before tracing; this kernel does not
    certify the oceanographic funnel or silently convert SP/temperature/Pa.
    """
    sa, ct, pressure = _inputs(sa_g_kg, ct_deg_c, sea_pressure_dbar)
    x = jnp.sqrt(_SFAC * sa + 0.5971840214030754)
    y, z = 0.025 * ct, 1e-4 * pressure
    pressure_terms = tuple(_horner(tuple(_horner(row, x) for row in block), y)
                           for block in _V)
    return _horner(pressure_terms, z)


def density(sa_g_kg, ct_deg_c, sea_pressure_dbar):
    """In-situ density [kg/m³]; at a common reference p this is potential density."""
    return 1.0 / specific_volume(sa_g_kg, ct_deg_c, sea_pressure_dbar)


def density_derivatives(sa_g_kg, ct_deg_c, sea_pressure_dbar):
    """Elementwise (dρ/dSA, dρ/dCT, dρ/dp); the last derivative is PER DBAR.

    Broadcast first so a scalar input's derivative is not an unintended sum
    over the other inputs. Analytic GSW dρ/dP is per Pa: multiply it by 10⁴
    when comparing with this API. AD differentiates only the JAX polynomial.
    """
    inputs = _inputs(sa_g_kg, ct_deg_c, sea_pressure_dbar)
    return jax.grad(lambda s, t, p: jnp.sum(density(s, t, p)), argnums=(0, 1, 2))(*inputs)


def parcel_density_contrast(sa_above, ct_above, sa_below, ct_below, common_pressure_dbar):
    """ρ(above)-ρ(below) at ONE pressure [kg/m³]; positive means unstable.

    Comparing densities at two different in-situ pressures would include
    compressibility and can hide an unstable pair. Caller selects the actual
    interface pressure; no layer-depth interpretation is guessed here.
    """
    return (density(sa_above, ct_above, common_pressure_dbar)
            - density(sa_below, ct_below, common_pressure_dbar))


def neutral_density_gradient(sa_g_kg, ct_deg_c, sea_pressure_dbar, grad_sa, grad_ct):
    """Thermohaline density gradient at fixed local pressure.

    grad_sa and grad_ct use the same spatial derivative convention. Omitting
    dρ/dp * grad(p) is deliberate: compressibility is not a neutral slope's
    stratification. This component does not implement a mixing discretization.
    """
    rho_sa, rho_ct, _ = density_derivatives(sa_g_kg, ct_deg_c, sea_pressure_dbar)
    return rho_sa * grad_sa + rho_ct * grad_ct


def reference_salinity(sp_pss78):
    """SR [g/kg] from SP (PSS-78); NOT geographic Absolute Salinity SA.

    The pinned MOM6 initialization uses this factor as an approximation to SA.
    A caller doing the same must explicitly record that approximation.
    """
    return jnp.asarray(sp_pss78) * (35.16504 / 35.0)


# Potential enthalpy: temperature-polynomial coefficients for x^0,x^2,...,x^7,
# x² = sfac*SA. The x^1 term is absent. Fractional powers of x² keep the SA=0
# derivative finite instead of AD differentiating sqrt(0) inside a product.
_ENTHALPY = (
    (61.01362420681071, 168776.46138048015, -2735.2785605119625,
     2574.2164453821433, -1536.6644434977543, 545.7340497931629,
     -50.91091728474331, -18.30489878927802),
    (268.5520265845071, -12019.028203559312, 3734.858026725145,
     -2046.7671145057618, 465.28655623826234, -.6370820302376359, -10.650848542359153),
    (937.2099110620707, 588.1802812170108, 248.39476522971285,
     -3.871557904936333, -2.6268019854268356),
    (-1687.914374187449, 936.3206544460336, -942.7827304544439,
     369.4389437509002, -33.83664947895248, -9.987880382780322),
    (246.9598888781377,), (123.59576582457964,), (-48.5891069025409,),
)
# Entropy minus SA-only terms: each block is temperature power then pressure
# power; blocks multiply x^0,x^2,x^3,x^4. SA-only terms cancel at fixed SA.
_ENTROPY = (
    (
        (0., -270.983805184062, 776.153611613101, -196.51255088122, 28.9796526294175, -2.13290083518327),
        (-24715.571866078, 2910.0729080936, -1513.116771538718, 546.959324647056, -111.1208127634436, 8.68841343834394),
        (2210.2236124548363, -2017.52334943521, 1498.081172457456, -718.6359919632359, 146.4037555781616, -4.9892131862671505),
        (-592.743745734632, 1591.873781627888, -1207.261522487504, 608.785486935364, -105.4993508931208),
        (290.12956292128547, -973.091553087975, 602.603274510125, -276.361526170076, 32.40953340386105),
        (-113.90630790850321, 381.06836198507096, -133.7383902842754, 49.023632509086724),
        (21.35571525415769, -67.41756835751434),
    ),
    (
        (0., 729.116529735046, -343.956902961561, 124.687671116248, -31.656964386073, 7.04658803315449),
        (1760.062705994408, -1721.528607567954, 674.819060538734, -356.629112415276, 88.4080716616, -15.84003094423364),
        (-675.802947790203, 2082.7344423998043, -614.668925894709, 340.685093521782, -33.3848202979239),
        (365.7041791005036, -1190.914967948748, 298.904564555024, -145.9491676006352),
        (-108.30162043765552,), (12.78101825083098,),
    ),
    (
        (0., -175.292041186547, 83.1923927801819, -29.483064349429),
        (-86.1329351956084, 766.116132004952, -108.3834525034224, 51.2796974779828),
        (-30.0682112585625, -1380.9597954037708), (3.50240264723578, 938.26075044542),
    ),
    ((0., -22.6683558512829), (-137.1145018408982,), (148.10030845687618,),
     (-68.5590309679152,), (12.4848504784754,)),
)


def conservative_from_potential(sa_g_kg, pt_deg_c):
    """CT [degC] from surface-referenced potential temperature, via h⁰/CP0.

    Pure JAX kernel. SA=0 has a finite first derivative; no geographic salinity
    conversion, clipping, host callback or legacy heat-capacity substitution.
    """
    sa, pt, _ = _inputs(sa_g_kg, pt_deg_c, 0.)
    x2, y = _SFAC*sa, .025*pt
    enthalpy = _horner(_ENTHALPY[0], y)
    for power, coefficients in enumerate(_ENTHALPY[1:], start=2):
        enthalpy += x2**(power/2) * _horner(coefficients, y)
    return enthalpy / CP0


def _entropy_part(sa, temperature, pressure):
    x2, y, z = _SFAC*sa, .025*temperature, 1e-4*pressure
    terms = tuple(_horner(tuple(_horner(row, z) for row in block), y)
                  for block in _ENTROPY)
    value = terms[0]
    for power, term in enumerate(terms[1:], start=2):
        value += x2**(power/2) * term
    return -.025 * value


def _newton_temperature(initial, target, evaluate):
    """Five fixed Newton steps, using the JAX polynomial's temperature derivative.

    Input range is preflighted by the caller. Independent Fortran values and
    entropy/enthalpy residuals validate this iteration, not round-trip alone.
    """
    def iteration(_, temperature):
        result = evaluate(temperature)
        derivative = jax.grad(lambda t: jnp.sum(evaluate(t)))(temperature)
        return temperature - (result-target)/derivative
    return jax.lax.fori_loop(0, 5, iteration, initial)


def potential_from_conservative(sa_g_kg, ct_deg_c):
    """Surface potential temperature [degC]; this is in-situ SST at p=0 dbar."""
    sa, ct, _ = _inputs(sa_g_kg, ct_deg_c, 0.)
    return _newton_temperature(ct, ct, lambda pt: conservative_from_potential(sa, pt))


def _temperature_at_pressure(sa, temperature, source_pressure, target_pressure):
    target = _entropy_part(sa, temperature, source_pressure)
    return _newton_temperature(temperature, target, lambda t: _entropy_part(sa, t, target_pressure))


def potential_from_in_situ(sa_g_kg, t_deg_c, sea_pressure_dbar):
    """Surface-referenced potential temperature; conserve entropy at fixed SA."""
    sa, temperature, pressure = _inputs(sa_g_kg, t_deg_c, sea_pressure_dbar)
    return _temperature_at_pressure(sa, temperature, pressure, jnp.zeros_like(pressure))


def conservative_from_in_situ(sa_g_kg, t_deg_c, sea_pressure_dbar):
    """CT [degC] from in-situ temperature and actual sea pressure [dbar]."""
    return conservative_from_potential(sa_g_kg, potential_from_in_situ(sa_g_kg, t_deg_c, sea_pressure_dbar))


def in_situ_from_conservative(sa_g_kg, ct_deg_c, sea_pressure_dbar):
    """In-situ temperature [degC] at the requested sea pressure [dbar]."""
    sa, ct, pressure = _inputs(sa_g_kg, ct_deg_c, sea_pressure_dbar)
    pt = potential_from_conservative(sa, ct)
    return _temperature_at_pressure(sa, pt, jnp.zeros_like(pressure), pressure)


def surface_freezing_ct(sa_g_kg):
    """GSW polynomial freezing CT [degC], surface p=0 and dissolved-air fraction=0.

    This gives a salinity-dependent component rejection gate, not an ice model.
    Reference-salinity users must retain their SR≈SA approximation declaration.
    """
    sa=jnp.asarray(sa_g_kg)
    scaled=.01*sa
    coefficients=(-6.076099099929818,4.883198653547851,-11.88081601230542,
                  13.34658511480257,-8.722761043208607,2.082038908808201)
    result=jnp.full_like(scaled,.017947064327968736)
    for power,coefficient in enumerate(coefficients,start=2):
        result += scaled**(power/2)*coefficient
    return result


def validate_state(sa_g_kg, ct_deg_c, sea_pressure_dbar):
    """Host preflight for the checked component's numeric evaluation range.

    This rectangular range is not the oceanographic funnel. Agreement with
    the GSW polynomial in this range proves evaluation, not exact Gibbs-EOS
    accuracy or eligibility of a global state. No fill, clip or unit guessing.
    """
    values = (sa_g_kg, ct_deg_c, sea_pressure_dbar)
    if any(np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any() for value in values):
        raise ValueError('TEOS inputs contain masked values')
    arrays = np.broadcast_arrays(*(np.asarray(value) for value in values))
    if arrays[0].size == 0:
        raise ValueError('TEOS inputs must not be empty')
    for array, name, limits in zip(arrays, ('SA [g/kg]', 'CT [degC]', 'sea pressure [dbar]'),
                                   ((0, 42), (-3, 40), (0, 8000)), strict=True):
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError(f'{name} must be finite real values')
        if not ((array >= limits[0]) & (array <= limits[1])).all():
            raise ValueError(f'{name} outside checked numeric range {limits}')
