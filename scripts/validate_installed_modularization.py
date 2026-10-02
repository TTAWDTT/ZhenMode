"""Verify an installed wheel retains legacy entry points and actual source identities."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    installed = args.installed_root.resolve()
    source = args.source_root.resolve()
    expected = sorted(path.relative_to(source) for path in source.glob('*.py'))
    expected += sorted(path.relative_to(source) for path in (source / 'ocean_solver').rglob('*.py'))
    for relative in expected:
        assert (installed / relative).read_bytes() == (source / relative).read_bytes(), relative.as_posix()
    sys.path.insert(0, str(installed))
    import jax_solver_global as legacy
    import run_long_integration_global as driver
    from ocean_solver.fd.factory import make_solver_global
    from ocean_solver.fd.types import JaxStateG
    from source_identity import source_paths

    assert Path(legacy.__file__).parent == installed
    assert Path(driver.__file__).parent == installed
    assert legacy.make_solver_global is make_solver_global
    assert legacy.JaxStateG is JaxStateG
    for name in driver.SOURCE_MODULES:
        module_name = name.removesuffix('/__init__').replace('/', '.')
        loaded = importlib.import_module(module_name)
        assert Path(loaded.__file__).resolve().is_relative_to(installed), module_name
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
                      'imported_required_modules': len(driver.SOURCE_MODULES),
                      'legacy_factory_and_state_aliases': True, 'cli_help': True}))


if __name__ == '__main__':
    main()
