"""Complete source envelope for opt-in candidates, separate from production."""
from ocean_solver.provenance.sources import production_source_modules, source_paths

RESEARCH_LEGACY_MODULES = (
    'finite_volume', 'bounded_transport', 'cgrid_momentum', 'wet_fluxes',
    'physical_velocity', 'paired_dynamics', 'nonlinear_dynamics',
    'barotropic_transport', 'material_top',
)
RESEARCH_SOURCE_MODULES = (
    'zhenmode_research/__init__', 'zhenmode_research/provenance',
    'zhenmode_research/candidates/__init__',
    'zhenmode_research/candidates/fv/__init__',
    'zhenmode_research/candidates/fv/barotropic', 'zhenmode_research/candidates/fv/fluxes',
    'zhenmode_research/candidates/fv/geometry', 'zhenmode_research/candidates/fv/momentum',
    'zhenmode_research/candidates/fv/nonlinear', 'zhenmode_research/candidates/fv/paired',
    'zhenmode_research/candidates/fv/transport', 'zhenmode_research/candidates/fv/velocity',
    'zhenmode_research/candidates/material/__init__', 'zhenmode_research/candidates/material/solver',
)


def solver_source_modules():
    """Require real research bytes as well as the production dependencies."""
    return (*production_source_modules(), *RESEARCH_LEGACY_MODULES, *RESEARCH_SOURCE_MODULES)


__all__ = ['solver_source_modules', 'source_paths']
