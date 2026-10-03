"""Verify a real installed wheel outside the checkout, without path insertion."""
import argparse
import ast
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed-root', type=Path)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    import ocean_solver

    installed = (args.installed_root or Path(ocean_solver.__file__).parent.parent).resolve()
    source = args.source_root.resolve()
    expected = {path.name: path for path in (source / 'compat').glob('*.py')}
    expected.update({path.relative_to(source).as_posix(): path
                     for path in (source / 'ocean_solver').rglob('*.py')})
    installed_package = {path.relative_to(installed).as_posix()
                         for path in (installed / 'ocean_solver').rglob('*.py')}
    expected_package = {name for name in expected if name.startswith('ocean_solver/')}
    assert installed_package == expected_package, {
        'missing': sorted(expected_package - installed_package),
        'unexpected': sorted(installed_package - expected_package),
    }
    research_bridges = ('finite_volume', 'bounded_transport', 'cgrid_momentum', 'wet_fluxes',
                        'physical_velocity', 'paired_dynamics', 'nonlinear_dynamics',
                        'barotropic_transport', 'material_top')
    for name in ('zhenmode_research', *research_bridges):
        assert importlib.util.find_spec(name) is None, name
    for relative, original in expected.items():
        assert (installed / relative).read_bytes() == original.read_bytes(), relative
    assert source not in [Path(value).resolve() for value in sys.path if value]
    aliases = 0
    for bridge in (source / 'compat').glob('*.py'):
        target = next(node.args[0].value for node in ast.walk(ast.parse(bridge.read_text()))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == 'import_module')
        legacy = importlib.import_module(bridge.stem)
        canonical = importlib.import_module(target)
        assert legacy is canonical, bridge.stem
        assert Path(canonical.__file__).resolve().is_relative_to(installed), target
        aliases += 1

    import jax
    import jax.numpy as jnp
    import numpy as np

    import jax_solver_global as legacy
    import run_long_integration_global as driver
    from ocean_solver.fd.factory import make_solver_global
    from ocean_solver.fd.types import JaxStateG
    from ocean_solver.provenance.sources import source_paths

    assert legacy.make_solver_global is make_solver_global
    assert legacy.JaxStateG is JaxStateG
    state = JaxStateG(*(jnp.arange(3, dtype=jnp.float64) + index for index in range(5)))
    restored = pickle.loads(pickle.dumps(state, protocol=4))
    assert type(restored) is JaxStateG
    leaves, structure = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    assert type(rebuilt) is JaxStateG
    for before, after in zip(state, rebuilt, strict=True):
        np.testing.assert_array_equal(before, after)
    for name in driver.SOURCE_MODULES:
        loaded = importlib.import_module(name.removesuffix('/__init__').replace('/', '.'))
        assert Path(loaded.__file__).resolve().is_relative_to(installed), name
    identity = driver._source_identity()['source_sha256']
    for name, path in source_paths(installed, driver.SOURCE_MODULES).items():
        assert identity[name + '.py'] == hashlib.sha256(path.read_bytes()).hexdigest()
    previous = sys.argv
    sys.argv = ['ocean-solver', '--help']
    try:
        try:
            driver.main()
        except SystemExit as exit_status:
            assert exit_status.code == 0
        else:
            raise AssertionError('CLI --help did not stop before loading')
    finally:
        sys.argv = previous
    print(json.dumps({'installed_modules': len(expected), 'source_identity_files': len(identity),
                      'legacy_aliases': aliases, 'pickle_and_pytree': True,
                      'all_imports_inside_wheel': True, 'cli_help': True,
                      'exact_package_payload': True, 'research_absent': True,
                      'checkout_inserted_in_sys_path': False}))


if __name__ == '__main__':
    main()
