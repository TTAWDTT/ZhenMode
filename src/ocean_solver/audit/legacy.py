"""Read-only actual-stage audit, not a complete physical moving-volume budget.

Heat is fixed-node water sensible heat minus ice latent heat. Salt is nominal
water salt mass; eta displacement is separate from the fixed reference volume.
Decomposition closure does not imply that the independent source budget closes.
Bottom-drag reference kinetic loss is a separate audit, not a heat source or
whole momentum/energy closure. A zero audited-half count means uninstrumented,
not necessarily zero physical drag in older schemes.
"""

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.audit.schema import MAXIMUM_BUDGET_FIELDS as MAXIMUM_BUDGET_FIELDS
from ocean_solver.audit.schema import METRIC_NAMES as METRIC_NAMES
from ocean_solver.audit.schema import NONLINEAR_PROCESS_NAMES as NONLINEAR_PROCESS_NAMES
from ocean_solver.audit.schema import SOURCE_NAMES as SOURCE_NAMES
from ocean_solver.audit.schema import STAGE_NAMES as STAGE_NAMES
from ocean_solver.audit.schema import TRANSPORT_METRIC_NAMES as TRANSPORT_METRIC_NAMES
from ocean_solver.audit.schema import accumulate_budget as accumulate_budget
from ocean_solver.audit.schema import empty_budget as empty_budget
from ocean_solver.audit.stages import _StageRecorder as _StageRecorder
from ocean_solver.audit.stages import make_budget_step as make_budget_step

preserve_legacy_names(globals(), 'stage_budgets')
