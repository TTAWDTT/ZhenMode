"""Compatibility and dependency boundaries of the production module migration."""

import json
import os
import pickle
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.paths import REPOSITORY_ROOT


def test_legacy_exports_use_one_canonical_state_and_parameter_type():
    import jax_solver_global as legacy
    from ocean_solver.fd.types import FDParams, FDPhysParams, JaxStateG

    assert legacy.JaxStateG is JaxStateG
    assert legacy.FDParams is FDParams
    assert legacy.FDPhysParams is FDPhysParams
    assert JaxStateG._fields == ('u', 'v', 'T', 'S', 'eta', 'ice')
    assert JaxStateG.__new__.__defaults__ == (0.0,)
    assert JaxStateG._field_defaults == {}
    assert JaxStateG.__module__ == 'jax_solver_global'
    for kind in (FDParams, FDPhysParams):
        assert kind.__module__ == 'jax_solver_global'
        assert kind._field_defaults == {}


def test_pickle_and_jax_pytree_keep_legacy_type_and_field_order():
    import jax_solver_global as legacy
    from ocean_solver.fd.types import JaxStateG

    state = JaxStateG(*(jnp.arange(3, dtype=jnp.float64) + index for index in range(5)))
    restored = pickle.loads(pickle.dumps(state, protocol=4))
    assert type(restored) is legacy.JaxStateG
    assert restored.ice == 0.0
    leaves, definition = jax.tree_util.tree_flatten(state)
    reconstructed = jax.tree_util.tree_unflatten(definition, leaves)
    assert type(reconstructed) is JaxStateG
    for original, actual in zip(state, reconstructed, strict=True):
        np.testing.assert_array_equal(original, actual)
    transformed = jax.jit(lambda current: current._replace(T=current.T + 1.))(state)
    assert type(transformed) is JaxStateG
    np.testing.assert_array_equal(transformed.T, state.T + 1.)
    shapes = jax.eval_shape(lambda current: current, state)
    assert type(shapes) is JaxStateG
    assert shapes.T.dtype == np.dtype('float64')


@pytest.mark.parametrize("module", ["ocean_solver/fd/horizontal.py", "ocean_solver/fd/integration.py", "ocean_solver/runtime/forcing.py"])
def test_new_execution_module_change_rejects_restart_and_missing_source_fails(tmp_path, module):
    from restart_contract import load_restart, make_restart_contract, save_restart
    from source_identity import solver_source_modules, source_paths
    from tests.support.material.reference_geometry import _fixture

    source = REPOSITORY_ROOT / 'src'
    copied = tmp_path / 'source'
    modules = solver_source_modules()
    for name, path in source_paths(source, modules).items():
        target = copied / (name + '.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    grid, (_, initialize, _, params, _) = _fixture()

    def contract():
        return make_restart_contract(grid, params, dtype='float64', forcing={}, controls={},
                                     code_paths=source_paths(copied, modules), execution={})

    original = contract()
    checkpoint = tmp_path / 'restart.npz'
    save_restart(checkpoint, initialize(), original, step=1, counters={}, cumulative={}, history={})
    modified = copied / module
    modified.write_bytes(modified.read_bytes() + b'\n# changed source identity\n')
    changed = contract()
    assert original['sources'] != changed['sources']
    with pytest.raises(ValueError, match='contract'):
        load_restart(checkpoint, changed)
    modified.unlink()
    with pytest.raises(ValueError, match='missing required source'):
        contract()


def test_source_registry_covers_every_installed_execution_module():
    from source_identity import PACKAGE_SOURCE_MODULES, production_source_modules, source_paths

    source = REPOSITORY_ROOT / 'src'
    actual = {path.relative_to(source).with_suffix('').as_posix()
              for path in (source / 'ocean_solver').rglob('*.py')}
    assert set(PACKAGE_SOURCE_MODULES) == actual
    assert len(production_source_modules()) == len(set(production_source_modules()))
    source_paths(source, production_source_modules())


@pytest.mark.parametrize('override', [None, 'true'])
def test_monitor_import_preserves_runtime_default_without_solver_assembly(override):
    environment = dict(os.environ)
    source = REPOSITORY_ROOT / "src"
    environment["PYTHONPATH"] = os.pathsep.join((str(source / "compat"), str(source), environment.get("PYTHONPATH", "")))
    if override is None:
        environment.pop('XLA_PYTHON_CLIENT_PREALLOCATE', None)
    else:
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = override
    probe = subprocess.run([sys.executable, '-c',
        "import integration_monitor, os, sys, json, jax; "
        "print(json.dumps({'allocation': os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], "
        "'precision': jax.config.jax_enable_x64, "
        "'solver_loaded': 'jax_solver_global' in sys.modules, "
        "'audit_loaded': 'ocean_solver.audit.stages' in sys.modules}))"],
        env=environment, check=True, capture_output=True, text=True, timeout=20)
    result = json.loads(probe.stdout)
    assert result == {'allocation': override or 'false', 'precision': True,
                      'solver_loaded': False, 'audit_loaded': False}
