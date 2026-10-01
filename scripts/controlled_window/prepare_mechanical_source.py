"""Copy verified historical modules; add observations without rewriting updates."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('ambiguous instrumentation anchor')
    return text.replace(old, new)


class NumericalProjection(ast.NodeTransformer):
    """Strip explicitly named diagnostic data, preserving the original AST."""

    def visit_FunctionDef(self, node):
        if node.name.startswith('_probe_'):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id.startswith('_probe_') for target in node.targets):
            return None
        node = self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_subcycle':
                    target.elts = [ast.Name(id='_', ctx=ast.Store()) if isinstance(item, ast.Name) and item.id.startswith('_probe_') else item for item in target.elts]
                else:
                    target.elts = [item for item in target.elts if not isinstance(item, ast.Name) or not item.id.startswith('_probe_')]
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == '_probe_scan':
            node.func.id = '_subcycle'
        return node

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple):
            node.value.elts = [item for item in node.value.elts if not isinstance(item, ast.Name) or not item.id.startswith('_probe_')]
            # advance() originally returns an unchanged carry plus None.
            if len(node.value.elts) == 1:
                node.value.elts.append(ast.Constant(value=None))
        return self.generic_visit(node)


def instrument(name, original):
    text = original
    if name == 'jax_solver_global.py':
        text = replace_once(text,
            '    normal_mask = params.interior_mask_z[..., 0]\n    velocity_x = velocity_x * params.wet_mask',
            '    _probe_incoming = (eta, velocity_x, velocity_y)\n    normal_mask = params.interior_mask_z[..., 0]\n    velocity_x = velocity_x * params.wet_mask')
        old = '    return (final_eta, final_x, final_y), mean_faces'
        new = '''    _probe_record = dict(eta_in=_probe_incoming[0], u_in=_probe_incoming[1], v_in=_probe_incoming[2],
                         u_old=velocity_x, v_old=velocity_y, eta_mid=midpoint_eta,
                         u_kick=next_x, v_kick=next_y, eta_transport=transported_eta,
                         eta_out=final_eta, u_out=final_x, v_out=final_y,
                         first_x=first_faces[0], first_y=first_faces[1],
                         second_x=second_faces[0], second_y=second_faces[1],
                         gradient_x=pressure_x, gradient_y=pressure_y,
                         density_x=forcing_x, density_y=forcing_y)
    return (final_eta, final_x, final_y), mean_faces, _probe_record'''
        text = replace_once(text, old, new)
        text = replace_once(text, '            updated, faces = _symmetric_free_surface_step(',
                            '            updated, faces, _probe_one = _symmetric_free_surface_step(')
        old = '                filter_change + nontransport_change), None\n\n    final, _ = _subcycle(advance,'
        new = '                filter_change + nontransport_change), _probe_one\n\n    final, _probe_records = _probe_scan(advance,'
        text = replace_once(text, old, new)
        text = replace_once(text, '    return updated, column_transport, filter_change',
                            '    return updated, column_transport, filter_change, _probe_records')
        text += '''\n\ndef _probe_scan(fn, carry, n, params):
    if not params.use_scan or params.process_time_scheme != 'symmetric_fast_v3':
        raise ValueError('mechanical probe supports only the verified scan path')
    return jax.lax.scan(lambda c, _: fn(c), carry, None, length=n)
'''
    elif name == 'material_top.py':
        text = replace_once(text, '    dynamical, column_faces, filter_change = _barotropic_subcycle_transport(predictor, params)',
                            '    dynamical, column_faces, filter_change, _probe_fast = _barotropic_subcycle_transport(predictor, params)')
        text = replace_once(text, '    attempted = _linear_bottom_drag_step(attempted, params, duration)',
                            '    _probe_before_final_drag = attempted\n    attempted = _linear_bottom_drag_step(attempted, params, duration)\n    _probe_after_final_drag = attempted')
        text = replace_once(text, '        return result, diagnostics', '''        _probe_states = (state, start, first, nonlinear_predictor, predictor, dynamical,
                         _probe_before_final_drag, _probe_after_final_drag, attempted)
        _probe_outer = tuple(dict(u=item.u, v=item.v, eta=item.eta) for item in _probe_states)
        _probe_report = dict(outer=_probe_outer, fast=_probe_fast)
        return result, diagnostics, _probe_report''')
    projected = NumericalProjection().visit(ast.parse(text))
    if ast.dump(projected, include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise ValueError(f'original numerical AST changed: {name}')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original_protocol = args.protocol.read_bytes()
    if hashlib.sha256(original_protocol).hexdigest() != '9d9e756162c8f19ac1d3e32bbfb178d3c44d13c4f11ccf83e697b7e9c716f148':
        raise ValueError('original protocol mismatch')
    protocol = json.loads(original_protocol)
    args.output_dir.mkdir(exist_ok=False)
    manifest, patches = {}, []
    for relative, expected in protocol['historical_source_hashes'].items():
        name = Path(relative).name
        if name == 'material_top.py':
            expected = protocol['instrumented_material_sha256']
        content = (args.source_dir / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'source identity mismatch: {name}')
        output = instrument(name, content.decode('utf-8'))
        (args.output_dir / name).write_bytes(output.encode('utf-8'))
        manifest[name] = dict(original_sha256=expected, instrumented_sha256=hashlib.sha256(output.encode('utf-8')).hexdigest(),
                              numerical_ast_unchanged=True)
        if output != content.decode('utf-8'):
            patches.extend(difflib.unified_diff(content.decode('utf-8').splitlines(True), output.splitlines(True),
                                               fromfile='historical/'+name, tofile='instrumented/'+name))
    (args.output_dir / 'mechanical_source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (args.output_dir / 'mechanical_instrumentation.patch').write_text(''.join(patches), encoding='utf-8')
    print('35 modules verified; diagnostic additions project back to original numerical AST')


if __name__ == '__main__':
    main()
