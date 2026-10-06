"""Actual FD wiring; external GSW values and independently stated heat inventory."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.diagnostics.budgets import _StageRecorder
from zhenmode.model.diagnostics.mixed_layer import mixed_layer_depth
from zhenmode.model.diagnostics.snapshot import compute_budget_diagnostics
from zhenmode.model.diagnostics.surface_budget import surface_budget
from zhenmode.model.inputs.forcing.online import bind_open_water_step
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.physics.air_sea import (
    AirState,
    SurfaceFluxes,
    apply_open_water_exchange,
    open_water_fluxes,
)
from zhenmode.model.solver.physics.isopycnal import _isopycnal_slope
from zhenmode.model.solver.physics.teos10 import conservative_from_potential
from zhenmode.model.solver.physics.vertical import _convective_mask


def setup(*,deep=False,**options):
    z=np.array([0.,-500.,-1000.,-4000.]) if deep else np.array([0.,-5.,-20.,-50.])
    grid=replace(all_wet_grid(nx=8,ny=8,nz=4),z=z,dz=-np.diff(z))
    physics=replace(PhysicsConfig(),thermodynamics='teos10_reference',nu_h=0.,nu_v=0.,nu_bi=0.,
                    kappa_h=0.,kappa_v=0.,kappa_bi=0.,kappa_conv=0.,kappa_gm=0.,kappa_redi=0.)
    pressure=np.broadcast_to(-z,(8,8,4)).copy()
    result=make_solver_global(grid,physics,60.,return_params=True,polar_cap_rows=0,polar_cap_taper=0,
                               eos_pressure_dbar=pressure,**options)
    return grid,result


def test_actual_fd_density_pressure_and_common_pressure_convection():
    grid,(step,initial,diagnostics,params,_)=setup(deep=True)
    t=np.broadcast_to([10.,10.,11.,0.],(8,8,4))
    state=initial(t,np.full((8,8,4),35.16504))
    support=Path(__file__).resolve().parents[1]/'support'
    original=np.asarray(json.loads((support/'teos10_reference.json').read_text())['rows'])
    expected=original[[9,7,10,6],3]
    rho,pressure,_=diagnostics(state)
    np.testing.assert_allclose(rho,np.broadcast_to(expected,(8,8,4)),rtol=0,atol=2e-10)
    increments=9.81*.5*((expected-1025)[:-1]+(expected-1025)[1:])*(-np.diff(grid.z))
    np.testing.assert_allclose(pressure,np.broadcast_to(np.r_[0,np.cumsum(increments)],(8,8,4)),rtol=0,atol=2e-5)
    _,mask=_convective_mask(state,params)
    np.testing.assert_array_equal(mask,np.broadcast_to([False,True,False],(8,8,3)))
    assert expected[1]<expected[2]  # wrong different-pressure comparison would miss this instability
    assert np.isfinite(np.asarray(step(state).T)).all()


def test_neutral_slopes_and_potential_density_mld_do_not_mix_compressibility():
    _,(_,initial,_,params,_)=setup()
    state=initial(np.full((8,8,4),10.),np.full((8,8,4),35.16504))
    for slope in _isopycnal_slope(state,params):
        np.testing.assert_array_equal(slope,0.)
    t=np.broadcast_to([10.,10.,-2.,-2.],(2,2,4))
    np.testing.assert_array_equal(mixed_layer_depth(t,np.full(t.shape,35.16504),
        np.array([0.,-5.,-20.,-50.]),thermodynamics='teos10_reference'),20.)
    with pytest.raises(ValueError,match='CT'):
        mixed_layer_depth(t+273.15,np.full(t.shape,35.16504),np.array([0.,-5.,-20.,-50.]),
                           thermodynamics='teos10_reference')


def test_ct_heat_inventory_uses_the_declared_gsw_cp0_with_negative_control():
    grid,(_,initial,_,params,_)=setup()
    before=initial(np.full((8,8,4),10.),np.full((8,8,4),35.16504))
    field=jnp.ones((8,8))
    air=AirState(*[field*v for v in (290.,.005,101325.,0.,0.,0.,0.,0.,0.,0.,0.)])
    flux=SurfaceFluxes(*[field*v for v in (0.,0.,0.,0.,0.,100.,0.,.001,.001,.001)])
    weights=jnp.broadcast_to(jnp.array([1/5,0.,0.,0.]),before.T.shape)
    cp0=3991.86795711963  # independently fixed GSW constant, not model selector
    after,heat,salt,*_=apply_open_water_exchange(before,flux,air,field*35.16504,weights,60.,
                                                heat_capacity_j_kg_k=cp0)
    area=np.asarray(params.dx_2d)*params.dy
    volume=area[...,None]*np.asarray(params.dz_node)*np.asarray(params.wet_mask_z)
    expected=area.sum()*100*60
    recorder=_StageRecorder(params)
    change,_=recorder.difference(before,after)
    assert float(change[0])==pytest.approx(expected,rel=1e-10)
    before_snapshot=compute_budget_diagnostics(before,grid,thermodynamics='teos10_reference')
    after_snapshot=compute_budget_diagnostics(after,grid,thermodynamics='teos10_reference')
    assert after_snapshot.heat_content_J-before_snapshot.heat_content_J==pytest.approx(expected,rel=1e-9)
    with pytest.raises(ValueError,match='requires CP0'):
        compute_budget_diagnostics(after,grid,cp=3992.,thermodynamics='teos10_reference')
    report=surface_budget(before,after,wet_volume=volume,wet_area=area,heat_flux=heat,salt_flux=salt,
                          dt_seconds=60.,heat_capacity_j_kg_k=cp0)
    assert abs(report['residual'][0])/expected<1e-10
    damaged=surface_budget(before,after,wet_volume=volume,wet_area=area,heat_flux=heat*2,salt_flux=salt,
                           dt_seconds=60.,heat_capacity_j_kg_k=cp0)
    assert damaged['residual'][0]==pytest.approx(-expected,rel=1e-10)


@pytest.mark.parametrize('precision',['float64','float32'])
def test_actual_online_step_uses_independent_surface_temperature(precision):
    _,(_,initial,_,params,_,dynamic)=setup(dynamic_forcing=True,dtype=precision)
    state=initial(np.full((8,8,4),10.),np.full((8,8,4),35.16504))
    air=AirState(*[jnp.full((8,8),v) for v in (290.,.005,101325.,6.,2.,200.,320.,1e-5,0.,0.,0.)])
    class Weather:
        data_kind='manufactured'
        def sample(self,start,*,interval_end_seconds):
            assert interval_end_seconds-start==60
            return air
    advance=bind_open_water_step(Weather(),dynamic,params,sss_reference=np.full((8,8),35.16504))
    after,report=advance(state,0.)
    reference=np.asarray(json.loads((Path(__file__).resolve().parents[1]/'support/temperature_reference.json').read_text())['rows'])
    # GSW pt_from_CT at SA=35.16504, CT=10; same bulk was separately checked
    # against the author oracle. This comparison isolates the SST coupling.
    expected_sst=reference[7,4]
    expected=open_water_fluxes(air,expected_sst,0.,0.)
    area=np.asarray(params.dx_2d)*params.dy
    assert report['surface_integrals']['latent']==pytest.approx(float(np.sum(area*np.asarray(expected.latent))*60),
                                                               rel=1e-5 if precision=='float32' else 1e-10)
    assert report['thermodynamics']=='teos10_reference' and report['temperature_definition']=='CT'
    assert report['salinity_definition']=='SR_approximates_SA'
    assert after.T.dtype==after.S.dtype==getattr(jnp,precision)
    assert not report['execution_ready_for_full_protocol']
    # Fresh water at -1 C is frozen; the legacy fixed -1.8 C test would accept it.
    cold=state._replace(S=jnp.zeros_like(state.S),T=jnp.full_like(state.T,
        conservative_from_potential(0.,-1.)))
    with pytest.raises(ValueError,match='unfrozen'):
        advance(cold,0.)


@pytest.mark.parametrize('corruption',['missing','kelvin','pa','shape','ice'])
def test_thermodynamic_factory_rejects_incomplete_or_incompatible_definitions(corruption):
    grid,result=setup()
    physics=replace(PhysicsConfig(),thermodynamics='teos10_reference')
    if corruption=='missing':
        with pytest.raises(ValueError,match='explicit sea pressure'):
            make_solver_global(grid,physics,60.)
    elif corruption=='kelvin':
        with pytest.raises(ValueError,match='CT'):
            result[1](np.full((8,8,4),290.),np.full((8,8,4),35.))
    elif corruption in {'pa','shape'}:
        p=np.full((8,8,4),1e6) if corruption=='pa' else np.zeros((4,))
        with pytest.raises(ValueError,match='full grid'):
            make_solver_global(grid,physics,60.,eos_pressure_dbar=p)
    else:
        with pytest.raises(ValueError,match='legacy ice'):
            make_solver_global(grid,physics,60.,eos_pressure_dbar=np.broadcast_to(-grid.z,(8,8,4)),dynamic_ice=True)
