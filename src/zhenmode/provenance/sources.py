"""Required current source identities; historical reports remain commit-bound."""

import hashlib
import re
import subprocess
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
    'zhenmode/evaluation/ice',
    'zhenmode/evaluation/manifest',
    'zhenmode/evaluation/metrics',
    'zhenmode/evaluation/pipeline',
    'zhenmode/evaluation/protocols',
    'zhenmode/evaluation/table',
    'zhenmode/execution/__init__',
    'zhenmode/execution/cli',
    'zhenmode/execution/options',
    'zhenmode/execution/profiling',
    'zhenmode/execution/resolve',
    'zhenmode/execution/resources',
    'zhenmode/execution/runs',
    'zhenmode/execution/schema',
    'zhenmode/execution/worker',
    'zhenmode/model/__init__',
    'zhenmode/model/config',
    'zhenmode/model/diagnostics/__init__',
    'zhenmode/model/diagnostics/budgets',
    'zhenmode/model/diagnostics/mixed_layer',
    'zhenmode/model/diagnostics/snapshot',
    'zhenmode/model/inputs/__init__',
    'zhenmode/model/inputs/bathymetry',
    'zhenmode/model/inputs/forcing/__init__',
    'zhenmode/model/inputs/forcing/bundle',
    'zhenmode/model/inputs/forcing/idealized',
    'zhenmode/model/inputs/forcing/reanalysis',
    'zhenmode/model/inputs/forcing/seasonal',
    'zhenmode/model/inputs/initial_conditions',
    'zhenmode/model/inputs/prepare',
    'zhenmode/model/inputs/quality',
    'zhenmode/model/inputs/sources',
    'zhenmode/model/io/__init__',
    'zhenmode/model/io/output',
    'zhenmode/model/io/records',
    'zhenmode/model/io/restart',
    'zhenmode/model/runtime/__init__',
    'zhenmode/model/runtime/cli',
    'zhenmode/model/runtime/monitor',
    'zhenmode/model/runtime/reporting',
    'zhenmode/model/runtime/run',
    'zhenmode/model/runtime/run_loop',
    'zhenmode/model/solver/__init__',
    'zhenmode/model/solver/dynamics/__init__',
    'zhenmode/model/solver/dynamics/barotropic',
    'zhenmode/model/solver/dynamics/pressure',
    'zhenmode/model/solver/dynamics/projection',
    'zhenmode/model/solver/dynamics/tendencies',
    'zhenmode/model/solver/dynamics/transport',
    'zhenmode/model/solver/factory',
    'zhenmode/model/solver/geometry/__init__',
    'zhenmode/model/solver/geometry/fd_metrics',
    'zhenmode/model/solver/geometry/grid',
    'zhenmode/model/solver/numerics/__init__',
    'zhenmode/model/solver/numerics/backend',
    'zhenmode/model/solver/numerics/horizontal',
    'zhenmode/model/solver/numerics/vertical',
    'zhenmode/model/solver/physics/__init__',
    'zhenmode/model/solver/physics/eos',
    'zhenmode/model/solver/physics/isopycnal',
    'zhenmode/model/solver/physics/surface',
    'zhenmode/model/solver/physics/vertical',
    'zhenmode/model/solver/state',
    'zhenmode/model/solver/timestepping/__init__',
    'zhenmode/model/solver/timestepping/step',
    'zhenmode/model/solver/timestepping/subcycles',
    'zhenmode/model/verification',
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



def _source_identity(source_directory=None):
    source_dir = (
        Path(source_directory)
        if source_directory is not None
        else Path(__file__)
    )
    source_dir = source_root(source_dir)
    root = source_dir.parent
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable_installed_distribution"
    return {
        "git_head": head,
        "source_sha256": {
            name + ".py": sha256_file(path) for name, path in source_paths(source_dir, production_source_modules()).items()
        },
    }


def sha256_file(path):
    """SHA256 of file bytes, streamed in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
