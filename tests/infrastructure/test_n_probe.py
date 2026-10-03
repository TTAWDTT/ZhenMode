"""Independent algebra witnesses for actual two-RHS N work accounting."""

import importlib.util
import sys

import numpy as np

from tests.support.paths import REPOSITORY_ROOT

folder = REPOSITORY_ROOT / 'archive/tools/controlled_window'
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('n_summary', folder / 'summarize_n_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_global_endpoint_work_not_euler_stage_work():
    u0 = np.array([1.])
    u1 = np.array([4.])
    zero = np.zeros(1)
    mass = np.array([2.])
    first = {'a': (np.array([1.]), zero), 'b': (np.array([0.]), zero)}
    second = {'a': (np.array([0.]), zero), 'b': (np.array([2.]), zero)}
    result = module.component_work(first, second, u0, zero, u1, zero, mass, 2.)
    assert result['a']['work_J'] == 5.
    assert result['b']['work_J'] == 10.
    assert sum(item['work_J'] for item in result.values()) == 15.


def test_auxiliary_ast_preserves_tuple_and_single_return():
    import ast

    from prepare_n_source import NProjection
    original = 'def f(x):\n a,b=g(x)\n y=h(a)\n return y\n'
    observed = 'def f(x):\n a,b,_probe_meta=g(x)\n _probe_raw=a\n y,_probe_n=h(a)\n return y,_probe_n\n'
    assert ast.dump(NProjection().visit(ast.parse(observed))) == ast.dump(ast.parse(original))
    changed = observed.replace('h(a)', 'h(a+1)')
    assert ast.dump(NProjection().visit(ast.parse(changed))) != ast.dump(ast.parse(original))


def test_rejected_original_gate_blocks_interpretation():
    import pytest
    for record in [dict(valid=False,checks={'finite':True},equivalence={'passed':True}),
                   dict(valid=True,checks={'finite':False},equivalence={'passed':True})]:
        with pytest.raises(ValueError):
            module.require_accepted_record(record)
    module.require_accepted_record(dict(valid=True,checks={'finite':True},equivalence={'passed':True}))


def test_frozen_bound_uses_serializable_boolean():
    import json
    assert json.loads(json.dumps({'passed':1. <= module.FACTOR*1e15}))['passed'] is True
