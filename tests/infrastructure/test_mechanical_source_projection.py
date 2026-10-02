"""Reject numerical edits while allowing explicitly separate observation outputs."""

import ast
import importlib.util

from tests.support.paths import REPOSITORY_ROOT

spec = importlib.util.spec_from_file_location(
    'mechanical_source', REPOSITORY_ROOT / 'scripts/controlled_window/prepare_mechanical_source.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def projected(source):
    return ast.dump(module.NumericalProjection().visit(ast.parse(source)), include_attributes=False)


def test_observation_return_preserves_original_update():
    original = 'def step(x):\n y=x+1\n return y,None\n'
    observed = 'def step(x):\n _probe_before=x\n y=x+1\n _probe_data=y\n return y,_probe_data\n'
    assert projected(observed) == ast.dump(ast.parse(original), include_attributes=False)


def test_changed_numerical_expression_is_not_hidden():
    original = 'def step(x):\n y=x+1\n return y,None\n'
    changed = 'def step(x):\n _probe_before=x\n y=x+2\n _probe_data=y\n return y,_probe_data\n'
    assert projected(changed) != ast.dump(ast.parse(original), include_attributes=False)


def test_scan_observation_retains_original_carry_assignment():
    original = 'def step(fn,c,n,p):\n final,_=_subcycle(fn,c,n,p)\n return final\n'
    observed = 'def step(fn,c,n,p):\n final,_probe_records=_probe_scan(fn,c,n,p)\n return final\n'
    assert projected(observed) == ast.dump(ast.parse(original), include_attributes=False)
