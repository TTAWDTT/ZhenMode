"""Required current source identities; historical reports remain commit-bound."""

import re

from ocean_solver.provenance.locations import source_root

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
    'ocean_solver/baselines/mom6',
    'ocean_solver/cli',
    'ocean_solver/config/__init__',
    'ocean_solver/config/definitions',
    'ocean_solver/diagnostics/__init__',
    'ocean_solver/diagnostics/ice',
    'ocean_solver/diagnostics/runtime',
    'ocean_solver/diagnostics/state',
    'ocean_solver/dynamics/__init__',
    'ocean_solver/dynamics/barotropic',
    'ocean_solver/dynamics/pressure',
    'ocean_solver/dynamics/processes',
    'ocean_solver/dynamics/projection',
    'ocean_solver/dynamics/transport',
    'ocean_solver/evaluation/__init__',
    'ocean_solver/evaluation/cli',
    'ocean_solver/evaluation/pipeline',
    'ocean_solver/evaluation/protocols',
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
    'ocean_solver/geometry/columns',
    'ocean_solver/geometry/fd',
    'ocean_solver/geometry/mesh',
    'ocean_solver/geometry/types',
    'ocean_solver/interop/__init__',
    'ocean_solver/interop/mom6/__init__',
    'ocean_solver/interop/mom6/air_temperature',
    'ocean_solver/interop/mom6/sensible_heat',
    'ocean_solver/interop/mom6/wind',
    'ocean_solver/io/__init__',
    'ocean_solver/io/bathymetry',
    'ocean_solver/io/climatology',
    'ocean_solver/io/data_quality',
    'ocean_solver/io/erddap',
    'ocean_solver/io/grid',
    'ocean_solver/io/input_sources',
    'ocean_solver/io/output',
    'ocean_solver/io/paths',
    'ocean_solver/io/records',
    'ocean_solver/io/recovery',
    'ocean_solver/io/restart',
    'ocean_solver/io/sla',
    'ocean_solver/io/ssh',
    'ocean_solver/model/__init__',
    'ocean_solver/model/factory',
    'ocean_solver/numerics/__init__',
    'ocean_solver/numerics/backend',
    'ocean_solver/numerics/horizontal',
    'ocean_solver/numerics/stability',
    'ocean_solver/numerics/vertical',
    'ocean_solver/physics/__init__',
    'ocean_solver/physics/eos',
    'ocean_solver/physics/ice',
    'ocean_solver/physics/isopycnal',
    'ocean_solver/physics/surface',
    'ocean_solver/physics/vertical',
    'ocean_solver/provenance/__init__',
    'ocean_solver/provenance/archives',
    'ocean_solver/provenance/locations',
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
    'ocean_solver/runtime/services',
    'ocean_solver/state/__init__',
    'ocean_solver/state/types',
    'ocean_solver/timestepping/__init__',
    'ocean_solver/timestepping/integration',
    'ocean_solver/timestepping/subcycles',
    'ocean_solver/validation/__init__',
    'ocean_solver/validation/benchmarks/__init__',
    'ocean_solver/validation/benchmarks/climatology',
    'ocean_solver/validation/benchmarks/external',
    'ocean_solver/validation/benchmarks/gate',
    'ocean_solver/validation/benchmarks/manifest',
    'ocean_solver/validation/benchmarks/metrics',
    'ocean_solver/validation/benchmarks/mom6',
    'ocean_solver/validation/benchmarks/table',
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
        # An explicit research envelope may resolve a separately installed package
        # or checkout research sources. The production registry never requests it.
        research_root = directory.parent / 'research/src'
        if (not path.is_file() and research_root.is_dir()
                and name.startswith('zhenmode_research/')):
            path = research_root / (name + '.py')
        if not path.is_file():
            raise ValueError("missing required source: " + name)
        if not (path.resolve().is_relative_to(directory.resolve())
                or (research_root.is_dir() and path.resolve().is_relative_to(research_root.resolve()))):
            raise ValueError('outside required source roots: ' + name)
        result[name] = path
    return result
