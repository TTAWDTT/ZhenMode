"""Required current source identities; historical reports remain commit-bound."""

import hashlib
import re
from pathlib import Path


def source_root(location):
    """Resolve a module/file/directory to its actual distribution source root.

    Checkout implementations are below src/ocean_solver. Installed modules
    share a site-packages source root.
    Explicit external source directories remain valid for strict restart tests.
    """
    path = Path(location).resolve()
    directory = path.parent if path.suffix == ".py" else path
    if (directory / "ocean_solver").is_dir():
        return directory
    for ancestor in (directory, *directory.parents):
        if ancestor.name == "ocean_solver" and (ancestor / "__init__.py").is_file():
            return ancestor.parent
    return directory

PACKAGE_SOURCE_MODULES = (
    'ocean_solver/__init__',
    'ocean_solver/__main__',
    'ocean_solver/audit/__init__',
    'ocean_solver/audit/monitor',
    'ocean_solver/audit/schema',
    'ocean_solver/audit/stages',
    'ocean_solver/audit/validation',
    'ocean_solver/baselines/__init__',
    'ocean_solver/baselines/cli',
    'ocean_solver/baselines/forcing',
    'ocean_solver/baselines/mom6',
    'ocean_solver/cli',
    'ocean_solver/config/__init__',
    'ocean_solver/config/definitions',
    'ocean_solver/diagnostics/__init__',
    'ocean_solver/diagnostics/ice',
    'ocean_solver/diagnostics/state',
    'ocean_solver/dynamics/__init__',
    'ocean_solver/dynamics/barotropic',
    'ocean_solver/dynamics/pressure',
    'ocean_solver/dynamics/processes',
    'ocean_solver/dynamics/projection',
    'ocean_solver/dynamics/transport',
    'ocean_solver/evaluation/__init__',
    'ocean_solver/evaluation/cli',
    'ocean_solver/evaluation/external',
    'ocean_solver/evaluation/gate',
    'ocean_solver/evaluation/manifest',
    'ocean_solver/evaluation/metrics',
    'ocean_solver/evaluation/pipeline',
    'ocean_solver/evaluation/protocols',
    'ocean_solver/evaluation/table',
    'ocean_solver/experiments/__init__',
    'ocean_solver/experiments/cli',
    'ocean_solver/experiments/options',
    'ocean_solver/experiments/resolve',
    'ocean_solver/experiments/runs',
    'ocean_solver/experiments/schema',
    'ocean_solver/experiments/worker',
    'ocean_solver/forcing/__init__',
    'ocean_solver/forcing/air',
    'ocean_solver/forcing/fields',
    'ocean_solver/forcing/seasonal',
    'ocean_solver/forcing/wind',
    'ocean_solver/geometry/__init__',
    'ocean_solver/geometry/fd',
    'ocean_solver/geometry/mesh',
    'ocean_solver/geometry/types',
    'ocean_solver/io/__init__',
    'ocean_solver/io/bathymetry',
    'ocean_solver/io/climatology',
    'ocean_solver/io/data_quality',
    'ocean_solver/io/grid',
    'ocean_solver/io/input_sources',
    'ocean_solver/io/output',
    'ocean_solver/io/paths',
    'ocean_solver/io/records',
    'ocean_solver/io/recovery',
    'ocean_solver/io/restart',
    'ocean_solver/model/__init__',
    'ocean_solver/model/factory',
    'ocean_solver/numerics/__init__',
    'ocean_solver/numerics/backend',
    'ocean_solver/numerics/horizontal',
    'ocean_solver/numerics/vertical',
    'ocean_solver/physics/__init__',
    'ocean_solver/physics/eos',
    'ocean_solver/physics/ice',
    'ocean_solver/physics/isopycnal',
    'ocean_solver/physics/surface',
    'ocean_solver/physics/vertical',
    'ocean_solver/provenance/__init__',
    'ocean_solver/provenance/sources',
    'ocean_solver/runtime/__init__',
    'ocean_solver/runtime/application',
    'ocean_solver/runtime/cli',
    'ocean_solver/runtime/context',
    'ocean_solver/runtime/entry',
    'ocean_solver/runtime/forcing',
    'ocean_solver/runtime/identity',
    'ocean_solver/runtime/inputs',
    'ocean_solver/runtime/integration',
    'ocean_solver/runtime/reporting',
    'ocean_solver/state/__init__',
    'ocean_solver/state/types',
    'ocean_solver/timestepping/__init__',
    'ocean_solver/timestepping/integration',
    'ocean_solver/timestepping/subcycles',
    'ocean_solver/validation/__init__',
    'ocean_solver/validation/mms',
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
