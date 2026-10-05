"""JAX evaluation of the Roquet 75-term specific-volume polynomial.

Inputs are Absolute Salinity [g/kg], Conservative Temperature [ITS-90 degC]
and sea pressure [dbar, absolute pressure minus atmospheric pressure]. These
are NOT the legacy solver's T/S variables. No production switch uses this
component yet: initialization, surface temperature, heat inventories and all
density-dependent processes must be wired together before enabling it.

The mathematical coefficients are inventoried in GSW-Fortran commit
29e64d652786e1d076a05128c920f394202bfe10. This generic nested Horner evaluator
does not import GSW or duplicate its Fortran implementation. Validation uses
the unmodified, pinned Fortran routines and their analytic derivatives.
See docs/thermodynamics_zh.md for scope, units, sources and fitting limits.
"""

from zhenmode.model.solver.numerics.backend import jax, jnp, np

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
    x = jnp.sqrt(0.0248826675584615 * sa + 0.5971840214030754)
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
