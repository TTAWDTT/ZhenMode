"""Original GSW conversions, derivative controls and explicit native-field semantics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from zhenmode.model.inputs.initial_conditions import convert_teos_reference_fields
from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.physics import teos10


def reference():
    directory=Path(__file__).resolve().parents[1]/'support'
    payload=(directory/'temperature_reference.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest()=='e1218f8e581928004c96f2a4f124684e1a81c3f490db12cb28c927e59f337f36'
    value=json.loads(payload)
    assert hashlib.sha256((directory/'temperature_reference.f90').read_bytes()).hexdigest()==value['driver_sha256']
    return np.asarray(value['rows'],dtype=float)


@pytest.mark.parametrize('precision,atol',[('float64',1e-10),('float32',1e-4)])
def test_original_fortran_temperature_values_and_analytic_derivatives(precision,atol):
    expected=reference()
    s,t,p=jnp.asarray(expected[:,:3],dtype=getattr(jnp,precision)).T
    conversions=[jax.jit(function)(s,t,*args) for function,args in (
        (teos10.conservative_from_potential,()),(teos10.potential_from_conservative,()),
        (teos10.conservative_from_in_situ,(p,)),(teos10.potential_from_in_situ,(p,)),
        (teos10.in_situ_from_conservative,(p,)))]
    derivatives=jax.jit(jax.grad(lambda s,t:jnp.sum(teos10.conservative_from_potential(s,t)),
                                 argnums=(0,1)))(s,t)
    actual=np.asarray(jnp.stack((*conversions,*derivatives,teos10.surface_freezing_ct(s)),axis=1))
    np.testing.assert_allclose(actual,expected[:,3:],rtol=0,atol=atol)
    damaged=expected[:,3:].copy()
    damaged[0,0]+=.01
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual,damaged,rtol=0,atol=atol)
    assert np.isfinite(actual[s==0]).all()
    assert conversions[0].dtype==getattr(jnp,precision)


def test_pressure_conversion_is_not_a_surface_temperature_alias():
    expected=reference()[6]  # SA=35.16504, in-situ t=0, p=4000 dbar
    actual=teos10.conservative_from_in_situ(*expected[:3])
    assert actual==pytest.approx(expected[5],abs=1e-10)
    wrong=teos10.conservative_from_potential(*expected[:2])
    assert abs(float(actual-wrong))>.1


def test_native_field_conversion_matches_gsw_without_unit_guessing():
    expected=reference()[[3,7,9]]
    # The helper takes SP. Its declared reference approximation converts SP
    # back to the SA-number used by this fixed Fortran reference.
    sp=expected[:,0]/(35.16504/35)
    ct,sr=convert_teos_reference_fields(expected[:,1],sp,expected[:,2])
    np.testing.assert_allclose(sr,expected[:,0],rtol=0,atol=1e-13)
    np.testing.assert_allclose(ct,expected[:,5],rtol=0,atol=1e-10)


@pytest.mark.parametrize('t,s,p,reason',[
    (290.,35.,0.,'CT'),(10.,True,0.,'real numeric'),(10.,-1.,0.,'SA'),
    (10.,35.,1e6,'dbar'),(np.ma.array([10.],mask=True),35.,0.,'masked'),
])
def test_native_field_conversion_refuses_invalid_units_and_masks(t,s,p,reason):
    with pytest.raises(ValueError,match=reason):
        convert_teos_reference_fields(t,s,p)
