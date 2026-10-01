"""Execute only the unchanged production final-mask AST on small arrays.

This is a pure boundary-operator witness, not a model step or wave simulation.
No copied boundary formula or modified production source is executed.
"""
import ast
import hashlib
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def final_mask_witness(source):
    content = source.read_bytes()
    module = ast.parse(content)
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_step_impl')
    start = next(index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == 'wmask')
    end = next(index for index in range(start, len(function.body))
               if isinstance(function.body[index], ast.Assign)
               and isinstance(function.body[index].targets[0], ast.Name)
               and function.body[index].targets[0].id == 'masked')
    statements = function.body[start:end + 1]
    nx, ny, nz = 8, 8, 4
    field = np.broadcast_to((1.25 + np.sin(np.arange(nx)))[:, None, None],
                            (nx, ny, nz)).copy()
    normal = np.ones((nx, ny, 1))
    normal[:, (0, -1)] = 0.
    state_type = namedtuple('State', 'u v T S eta ice')
    state = state_type(field.copy(), np.full_like(field, 2.),
                       np.full_like(field, 15.), np.full_like(field, 35.),
                       np.zeros((nx, ny)), np.zeros((nx, ny)))
    namespace = dict(p=SimpleNamespace(wet_mask_z=np.ones_like(field),
                                       interior_mask_z=normal),
                     state=state, JaxStateG=state_type,
                     u=state.u.copy(), v=state.v.copy(), T=state.T.copy(), S=state.S.copy(),
                     land_u=state.u.copy(), land_v=state.v.copy(),
                     land_T=state.T.copy(), land_S=state.S.copy())
    selected = ast.Module(body=statements, type_ignores=[])
    exec(compile(selected, str(source), 'exec'), namespace)
    result = namespace['masked']
    evidence = dict(scope='pure final boundary mask only; no time step',
                    source_sha256=hashlib.sha256(content).hexdigest(),
                    source_function='_step_impl',
                    line_start=statements[0].lineno, line_end=statements[-1].end_lineno,
                    u_byte_identical=result.u.tobytes() == state.u.tobytes(),
                    u_wall_max_abs_difference=float(np.max(np.abs(
                        result.u[:, (0, -1)] - state.u[:, (0, -1)]))),
                    v_wall_max_abs=float(np.max(np.abs(result.v[:, (0, -1)]))),
                    v_interior_byte_identical=result.v[:, 1:-1].tobytes()
                        == state.v[:, 1:-1].tobytes(),
                    temperature_byte_identical=result.T.tobytes() == state.T.tobytes(),
                    salinity_byte_identical=result.S.tobytes() == state.S.tobytes(),
                    limitation='Does not verify full-step y invariance, filters or other stages.')
    if not (evidence['u_byte_identical'] and evidence['v_interior_byte_identical']
            and evidence['v_wall_max_abs'] == 0.):
        raise ValueError('production final mask violates expected normal-only wall contract')
    return evidence


if __name__ == '__main__':
    source = Path(__file__).resolve().parents[3] / 'src/jax_solver_global.py'
    print(json.dumps(final_mask_witness(source), indent=2, allow_nan=False))
