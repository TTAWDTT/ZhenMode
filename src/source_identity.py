"""Required current source identities; historical reports remain commit-bound."""
from pathlib import Path

LEGACY_PRODUCTION_MODULES = ('run_long_integration_global', 'jax_solver_global', 'restart_contract', 'config', 'grid', 'diagnostics', 'forcing', 'wind_reanalysis', 'air_reanalysis', 'woa_data', 'benchmark_metrics', 'mixed_layer_ice', 'integration_monitor', 'runtime_validation', 'stage_budgets', 'finite_volume', 'bounded_transport', 'cgrid_momentum', 'wet_fluxes', 'physical_velocity', 'paired_dynamics', 'nonlinear_dynamics', 'barotropic_transport', 'material_top')
PACKAGE_SOURCE_MODULES = ('ocean_solver/__init__', 'ocean_solver/audit/__init__', 'ocean_solver/audit/schema', 'ocean_solver/fd/__init__', 'ocean_solver/fd/backend', 'ocean_solver/fd/eos', 'ocean_solver/fd/geometry', 'ocean_solver/fd/horizontal', 'ocean_solver/fd/pressure', 'ocean_solver/fd/projection', 'ocean_solver/fd/stability', 'ocean_solver/fd/transport', 'ocean_solver/fd/types', 'ocean_solver/fd/vertical', 'ocean_solver/geometry/__init__', 'ocean_solver/geometry/columns', 'ocean_solver/geometry/types')


def production_source_modules():
    return (*LEGACY_PRODUCTION_MODULES, 'source_identity', *PACKAGE_SOURCE_MODULES)


def solver_source_modules():
    return ('jax_solver_global', 'material_top', 'restart_contract', 'config', 'grid',
            'runtime_validation', 'source_identity', *PACKAGE_SOURCE_MODULES)


def source_paths(source_directory, modules):
    directory = Path(source_directory)
    result = {name: directory / (name + '.py') for name in modules}
    for name, path in result.items():
        if not path.is_file():
            raise ValueError('missing required source: ' + name)
    return result
