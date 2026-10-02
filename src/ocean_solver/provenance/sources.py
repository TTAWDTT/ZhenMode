"""Required current source identities; historical reports remain commit-bound."""


from ocean_solver._compat import preserve_legacy_names
from ocean_solver.provenance.locations import source_root

LEGACY_PRODUCTION_MODULES = (
    "run_long_integration_global",
    "jax_solver_global",
    "restart_contract",
    "config",
    "grid",
    "diagnostics",
    "forcing",
    "wind_reanalysis",
    "air_reanalysis",
    "woa_data",
    "benchmark_metrics",
    "mixed_layer_ice",
    "integration_monitor",
    "runtime_validation",
    "stage_budgets",
    "finite_volume",
    "bounded_transport",
    "cgrid_momentum",
    "wet_fluxes",
    "physical_velocity",
    "paired_dynamics",
    "nonlinear_dynamics",
    "barotropic_transport",
    "material_top",
)
PACKAGE_SOURCE_MODULES = (
    'ocean_solver/__init__',
    'ocean_solver/_compat',
    'ocean_solver/audit/__init__',
    'ocean_solver/audit/legacy',
    'ocean_solver/audit/monitor',
    'ocean_solver/audit/schema',
    'ocean_solver/audit/stages',
    'ocean_solver/audit/validation',
    'ocean_solver/candidates/__init__',
    'ocean_solver/candidates/fv/__init__',
    'ocean_solver/candidates/fv/barotropic',
    'ocean_solver/candidates/fv/fluxes',
    'ocean_solver/candidates/fv/geometry',
    'ocean_solver/candidates/fv/momentum',
    'ocean_solver/candidates/fv/nonlinear',
    'ocean_solver/candidates/fv/paired',
    'ocean_solver/candidates/fv/transport',
    'ocean_solver/candidates/fv/velocity',
    'ocean_solver/candidates/material/__init__',
    'ocean_solver/candidates/material/solver',
    'ocean_solver/configuration',
    'ocean_solver/data/__init__',
    'ocean_solver/data/air',
    'ocean_solver/data/climatology',
    'ocean_solver/data/erddap',
    'ocean_solver/data/forcing',
    'ocean_solver/data/quality',
    'ocean_solver/data/sla',
    'ocean_solver/data/sources',
    'ocean_solver/data/ssh',
    'ocean_solver/data/wind',
    'ocean_solver/diagnostics/__init__',
    'ocean_solver/diagnostics/ice',
    'ocean_solver/diagnostics/state',
    'ocean_solver/fd/__init__',
    'ocean_solver/fd/backend',
    'ocean_solver/fd/barotropic',
    'ocean_solver/fd/closures',
    'ocean_solver/fd/eos',
    'ocean_solver/fd/factory',
    'ocean_solver/fd/geometry',
    'ocean_solver/fd/horizontal',
    'ocean_solver/fd/integration',
    'ocean_solver/fd/legacy',
    'ocean_solver/fd/pressure',
    'ocean_solver/fd/processes',
    'ocean_solver/fd/projection',
    'ocean_solver/fd/sources',
    'ocean_solver/fd/stability',
    'ocean_solver/fd/subcycles',
    'ocean_solver/fd/transport',
    'ocean_solver/fd/types',
    'ocean_solver/fd/vertical',
    'ocean_solver/geometry/__init__',
    'ocean_solver/geometry/columns',
    'ocean_solver/geometry/grid',
    'ocean_solver/geometry/types',
    'ocean_solver/interop/__init__',
    'ocean_solver/interop/mom6/__init__',
    'ocean_solver/interop/mom6/air_temperature',
    'ocean_solver/interop/mom6/sensible_heat',
    'ocean_solver/interop/mom6/wind',
    'ocean_solver/physics/__init__',
    'ocean_solver/physics/ice',
    'ocean_solver/provenance/__init__',
    'ocean_solver/provenance/archives',
    'ocean_solver/provenance/locations',
    'ocean_solver/provenance/restart',
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
    'ocean_solver/runtime/metrics',
    'ocean_solver/runtime/output',
    'ocean_solver/runtime/paths',
    'ocean_solver/runtime/records',
    'ocean_solver/runtime/recovery',
    'ocean_solver/runtime/reporting',
    'ocean_solver/runtime/seasonal',
    'ocean_solver/runtime/services',
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
    return (*LEGACY_PRODUCTION_MODULES, "source_identity", *PACKAGE_SOURCE_MODULES)


def solver_source_modules():
    return (
        "jax_solver_global",
        "material_top",
        "restart_contract",
        "config",
        "grid",
        "runtime_validation",
        "source_identity",
        *PACKAGE_SOURCE_MODULES,
    )


def source_paths(source_directory, modules):
    """Hash actual files in a checkout or wheel; never synthesize legacy hashes."""
    directory = source_root(source_directory)
    legacy_directory = directory / "compat" if (directory / "compat").is_dir() else directory
    result = {
        name: (directory if name.startswith("ocean_solver/") else legacy_directory) / (name + ".py")
        for name in modules
    }
    for name, path in result.items():
        if not path.is_file():
            raise ValueError("missing required source: " + name)
    return result


preserve_legacy_names(globals(), 'source_identity')
