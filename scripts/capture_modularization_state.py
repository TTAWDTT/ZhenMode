"""Capture synthetic full-state behavior independently of code-source identities.

Run with the bounded test supervisor. Outputs are local regression artifacts,
not physical qualification, performance measurements or observational data.
"""
import argparse
import hashlib
import inspect
import json
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--scheme', choices=('legacy', 'symmetric_fast_v3'), default='legacy')
parser.add_argument('--dtype', choices=('float64', 'float32'), default='float64')
arguments = parser.parse_args()
source_root = arguments.source_root.resolve()
sys.path.insert(0, str(source_root))

import jax
import jax.numpy as jnp
import numpy as np

import jax_solver_global as solver
from config import PhysicsConfig
from grid import GlobalOceanGrid
from integration_monitor import make_monitored_advance
from restart_contract import load_restart, make_restart_contract, save_restart
from stage_budgets import empty_budget, make_budget_step

nx, ny, nz = 8, 4, 6
latitude = np.linspace(-30., 30., ny)
cosine = np.cos(np.radians(latitude))
z = np.array([0., -5., -15., -30., -50., -80.])
grid = GlobalOceanGrid(
    lon=np.linspace(.5, 359.5, nx), lat=latitude,
    dx_2d=np.broadcast_to(6.371e6 * cosine * np.radians(360. / nx), (nx, ny)).copy(),
    dy=float(6.371e6 * np.radians(latitude[1] - latitude[0])), cos_lat=cosine,
    f=np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(latitude)), (nx, ny)).copy(),
    z=z, dz=-np.diff(z), nz=nz, depth=np.full((nx, ny), 80.),
    wet_mask=np.ones((nx, ny)), ocean_mask=np.ones((nx, ny), bool),
    land_mask=np.zeros((nx, ny)), wet_mask_3d=np.ones((nx, ny, nz)), nx=nx, ny=ny)
physics = PhysicsConfig(nu_h=2., nu_v=.0003, kappa_h=.4, kappa_v=.0002,
                        kappa_conv=.01, nu_bi=100., kappa_bi=20., kappa_gm=.1,
                        kappa_redi=.1, r_bot=.0001, tau_x=.02, tau_y=.003, Q_heat=35.)
options = dict(dtype=arguments.dtype, mode_split=True, dt_bt=5., nu_nsub=1,
               use_scan=True, conservative_kv=True, localize_conv=True,
               monotone_adv=True, fct_adv=True, column_geometry='nodal_dual_v1',
               process_time_scheme=arguments.scheme,
               match_barotropic_transport=arguments.scheme == 'symmetric_fast_v3',
               polar_cap_rows=0, polar_cap_taper=0, return_params=True,
               lambda_bulk=50., T_atm=np.full((nx, ny), 16.))
step, initialize, diagnostics, params, terms = solver.make_solver_global(grid, physics, 10., **options)
state = initialize()
x, y, level = np.indices((nx, ny, nz))
state = state._replace(
    u=jnp.asarray(.002 * np.sin(2 * np.pi * x / nx) * (1 + level / 10), dtype=state.u.dtype),
    v=jnp.asarray(.001 * np.sin(np.pi * y / (ny - 1)) * (1 + level / 20), dtype=state.v.dtype),
    T=jnp.asarray(15. + .2 * level + .03 * np.cos(2 * np.pi * x / nx), dtype=state.T.dtype),
    S=jnp.asarray(35. + .005 * level + .004 * np.sin(np.pi * y / (ny - 1)), dtype=state.S.dtype),
    eta=jnp.asarray(.0001 * np.cos(2 * np.pi * np.arange(nx)[:, None] / nx)
                    * np.ones((1, ny)), dtype=state.eta.dtype))
payload = {}


def record(prefix, values):
    if isinstance(values, dict):
        for name, value in values.items():
            record(prefix + '__' + name, value)
    elif hasattr(values, '_asdict'):
        record(prefix, values._asdict())
    elif isinstance(values, (tuple, list)):
        for index, value in enumerate(values):
            record(prefix + '__' + str(index), value)
    else:
        array = np.asarray('__none__' if values is None else values)
        if array.dtype.kind == 'O':
            raise TypeError('object payload is not a reproducible array: ' + prefix)
        payload[prefix] = array


record('input', state)
record('params', params)
record('density', solver._density_anomaly(state.T, state.S, params))
record('pressure_gradient', solver._compute_pressure_gradient(state, params))
record('face_transport', solver._layer_face_transports(state.u, state.v, params))
record('relative_vertical', solver._vertical_transport_iface(state.u, state.v, params))
record('momentum_tendency', solver._compute_momentum_tendency(state, params))
record('tracer_tendency', solver._compute_tracer_tendency(state, params))
audited_step = make_budget_step(params)
for index in range(2):
    before = state
    state = step(before)
    audited, ledger = audited_step(before)
    record('accepted_' + str(index), state)
    record('shadow_' + str(index), audited)
    record('ledger_' + str(index), ledger)
    record('diagnostics_' + str(index), diagnostics(state))
    record('terms_' + str(index), terms(state))
    assert all(np.isfinite(np.asarray(field)).all() for field in state)

# A deliberately faulty step verifies the complete monitored rejection tuple.
zero_budget = empty_budget()


def faulty_step(current):
    return current._replace(T=current.T + .01, ice=current.ice + 1.,
                            eta=jnp.where(jnp.max(current.ice) >= 2., 16., current.eta))


monitored = make_monitored_advance(faulty_step, zero_budget)
record('rejection', monitored(state._replace(ice=jnp.zeros_like(state.ice)), 5))

contract = make_restart_contract(grid, params, dtype=arguments.dtype,
                                 forcing={'test': 'nonzero_synthetic'}, controls={'scheme': arguments.scheme},
                                 code_paths={path.relative_to(source_root).as_posix(): path
                                             for path in sorted(source_root.rglob('*.py'))},
                                 execution={'backend': 'cpu'})
checkpoint = arguments.output.with_suffix('.restart.npz')
save_restart(checkpoint, state, contract, step=2, counters={'accepted': 2},
             cumulative={}, history={})
restored = load_restart(checkpoint, contract)
record('restart', restored.state)
changed = dict(contract)
changed['controls'] = {'scheme': 'tampered'}
try:
    load_restart(checkpoint, changed)
except ValueError as exception:
    record('restart_rejection', str(exception))
else:
    raise AssertionError('changed restart contract was accepted')

toy = solver.JaxStateG(*(np.arange(2, dtype=np.float64) for _ in range(5)))
record('legacy_pickle', np.frombuffer(pickle.dumps(toy, protocol=4), dtype=np.uint8))
record('state_defaults', np.asarray(solver.JaxStateG.__new__.__defaults__))
record('tree_identity', str(jax.tree_util.tree_structure(state)))
record('pytree_roundtrip', jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(state))))
record('factory_signature', str(inspect.signature(solver.make_solver_global)))
record('factory_arities', [len(solver.make_solver_global(
    grid, physics, 10., **{**options, 'return_params': with_params, 'dynamic_forcing': dynamic}))
    for with_params, dynamic in ((False, False), (False, True), (True, False), (True, True))])
with arguments.output.open('xb') as stream:
    np.savez_compressed(stream, **payload)
source_identity = {path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(source_root.rglob('*.py'))}
metadata = {'python': sys.version.split()[0], 'jax': jax.__version__, 'numpy': np.__version__,
            'backend': jax.default_backend(), 'scheme': arguments.scheme, 'dtype': arguments.dtype,
            'array_count': len(payload), 'source_identity': source_identity}
arguments.output.with_suffix('.sources.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in metadata.items() if key != 'source_identity'}))
