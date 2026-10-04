"""Required current source identities; historical reports remain commit-bound."""

import hashlib
import re
from pathlib import Path


def source_root(location):
    """Resolve a module/file/directory to its actual distribution source root.

    Checkout implementations are below src/zhenmode. Installed modules
    share a site-packages source root.
    Explicit external source directories remain valid for strict restart tests.
    """
    path = Path(location).resolve()
    directory = path.parent if path.suffix == ".py" else path
    if (directory / "zhenmode").is_dir():
        return directory
    for ancestor in (directory, *directory.parents):
        if ancestor.name == "zhenmode" and (ancestor / "__init__.py").is_file():
            return ancestor.parent
    return directory

PACKAGE_SOURCE_MODULES = (
    'zhenmode/__init__',
    'zhenmode/__main__',
    'zhenmode/baselines/__init__',
    'zhenmode/baselines/cli',
    'zhenmode/baselines/mom6/__init__',
    'zhenmode/baselines/mom6/adapter',
    'zhenmode/baselines/mom6/forcing',
    'zhenmode/cli',
    'zhenmode/evaluation/__init__',
    'zhenmode/evaluation/cli',
    'zhenmode/evaluation/external',
    'zhenmode/evaluation/gate',
    'zhenmode/evaluation/manifest',
    'zhenmode/evaluation/metrics',
    'zhenmode/evaluation/pipeline',
    'zhenmode/evaluation/protocols',
    'zhenmode/evaluation/table',
    'zhenmode/execution/__init__',
    'zhenmode/execution/cli',
    'zhenmode/execution/options',
    'zhenmode/execution/resolve',
    'zhenmode/execution/runs',
    'zhenmode/execution/schema',
    'zhenmode/execution/worker',
    'zhenmode/model/__init__',
    'zhenmode/model/audit/__init__',
    'zhenmode/model/audit/monitor',
    'zhenmode/model/audit/schema',
    'zhenmode/model/audit/stages',
    'zhenmode/model/audit/validation',
    'zhenmode/model/config/__init__',
    'zhenmode/model/config/definitions',
    'zhenmode/model/diagnostics/__init__',
    'zhenmode/evaluation/ice',
    'zhenmode/model/diagnostics/mixed_layer',
    'zhenmode/model/diagnostics/state',
    'zhenmode/model/dynamics/__init__',
    'zhenmode/model/dynamics/barotropic',
    'zhenmode/model/dynamics/pressure',
    'zhenmode/model/dynamics/processes',
    'zhenmode/model/dynamics/projection',
    'zhenmode/model/dynamics/transport',
    'zhenmode/model/factory',
    'zhenmode/model/forcing/__init__',
    'zhenmode/model/forcing/air',
    'zhenmode/model/forcing/fields',
    'zhenmode/model/forcing/seasonal',
    'zhenmode/model/forcing/wind',
    'zhenmode/model/geometry/__init__',
    'zhenmode/model/geometry/fd',
    'zhenmode/model/geometry/mesh',
    'zhenmode/model/geometry/types',
    'zhenmode/model/io/__init__',
    'zhenmode/model/io/bathymetry',
    'zhenmode/model/io/climatology',
    'zhenmode/model/io/data_quality',
    'zhenmode/model/io/grid',
    'zhenmode/model/io/input_sources',
    'zhenmode/model/io/output',
    'zhenmode/model/io/paths',
    'zhenmode/model/io/records',
    'zhenmode/model/io/recovery',
    'zhenmode/model/io/restart',
    'zhenmode/model/numerics/__init__',
    'zhenmode/model/numerics/backend',
    'zhenmode/model/numerics/horizontal',
    'zhenmode/model/numerics/vertical',
    'zhenmode/model/physics/__init__',
    'zhenmode/model/physics/eos',
    'zhenmode/model/physics/ice',
    'zhenmode/model/physics/isopycnal',
    'zhenmode/model/physics/surface',
    'zhenmode/model/physics/vertical',
    'zhenmode/model/runtime/__init__',
    'zhenmode/model/runtime/application',
    'zhenmode/model/runtime/cli',
    'zhenmode/model/runtime/context',
    'zhenmode/model/runtime/entry',
    'zhenmode/model/runtime/forcing',
    'zhenmode/model/runtime/identity',
    'zhenmode/model/runtime/inputs',
    'zhenmode/model/runtime/integration',
    'zhenmode/model/runtime/reporting',
    'zhenmode/model/state/__init__',
    'zhenmode/model/state/types',
    'zhenmode/model/timestepping/__init__',
    'zhenmode/model/timestepping/integration',
    'zhenmode/model/timestepping/subcycles',
    'zhenmode/model/validation/__init__',
    'zhenmode/model/validation/mms',
    'zhenmode/provenance/__init__',
    'zhenmode/provenance/sources',
)

def production_source_modules():
    return PACKAGE_SOURCE_MODULES


def solver_source_modules():
    """FD source envelope. Optional methods own their separate envelopes."""
    return production_source_modules()


def source_paths(source_directory, modules):
    """Hash actual files in a checkout or wheel; never synthesize legacy hashes."""
    directory = source_root(source_directory)
    result = {}
    for name in modules:
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)*', name):
            raise ValueError('invalid source name: ' + str(name))
        path = directory / (name + '.py')
        if not path.is_file():
            raise ValueError("missing required source: " + name)
        if not path.resolve().is_relative_to(directory):
            raise ValueError('outside required source root: ' + name)
        result[name] = path
    return result


def _repository_file(root, name):
    if not isinstance(name, str):
        raise ValueError("source label must be a relative string")
    requested = Path(name)
    path = (root / requested).resolve()
    if requested.is_absolute() or ".." in requested.parts or not path.is_relative_to(root):
        raise ValueError("outside-workspace source: " + name)
    if not path.is_file():
        raise ValueError("missing declared current source: " + name)
    return path


def current_source_files(repository, selected=()):
    """Collect real current package files, test helpers and selected producers.

    Labels are repository-relative paths. No historical aliases or local research
    are included. Every required implementation must exist; missing files fail.
    """
    root = Path(repository).resolve()
    files = {path.relative_to(root).as_posix(): path
             for path in source_paths(root / "src", production_source_modules()).values()}
    for path in sorted((root / "tests").rglob("*.py")):
        if "support" in path.relative_to(root / "tests").parts or path.name == "__init__.py":
            name = path.relative_to(root).as_posix()
            files[name] = _repository_file(root, name)
    for item in selected:
        if isinstance(item, Path):
            try:
                name = item.resolve().relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError("outside-workspace source: " + str(item)) from error
        else:
            name = item
        files[name] = _repository_file(root, name)
    return files


def verify_current_source_hashes(repository, manifest):
    """Reject incomplete, missing, escaped or changed current source manifests."""
    root = Path(repository).resolve()
    files = current_source_files(root, manifest)
    if not set(files).issubset(manifest):
        raise ValueError("incomplete current source manifest")
    for name, expected in manifest.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid source hash: " + name)
        if hashlib.sha256(files[name].read_bytes()).hexdigest() != expected:
            raise ValueError("runtime source mismatch: " + name)
