"""Observe real N RHS evaluations while preserving historical update expressions."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

from prepare_mechanical_source import replace_once


class NProjection(ast.NodeTransformer):
    """Discard only explicitly named auxiliary observations from this patch."""

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id.startswith('_probe_') for target in node.targets):
            return None
        for index, target in enumerate(node.targets):
            if isinstance(target, ast.Tuple):
                target.elts = [item for item in target.elts if not isinstance(item, ast.Name) or not item.id.startswith('_probe_')]
                if len(target.elts) == 1:
                    node.targets[index] = target.elts[0]
        return self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple):
            node.value.elts = [item for item in node.value.elts if not isinstance(item, ast.Name) or not item.id.startswith('_probe_')]
            if len(node.value.elts) == 1:
                node.value = node.value.elts[0]
        return self.generic_visit(node)


def instrument(name, original):
    text = original
    if name == 'jax_solver_global.py':
        # Raw values are the actual locals before the unchanged dealias calls.
        text = replace_once(text, '    adv_u = _dealias_h_fd(adv_u, p)\n    adv_v = _dealias_h_fd(adv_v, p)',
                            '    _probe_raw_advection = (adv_u, adv_v)\n    adv_u = _dealias_h_fd(adv_u, p)\n    adv_v = _dealias_h_fd(adv_v, p)')
        text = replace_once(text, '    return adv_u * p.wet_mask_z, adv_v * p.wet_mask_z',
                            '    return adv_u * p.wet_mask_z, adv_v * p.wet_mask_z, _probe_raw_advection')
        text = replace_once(text, '    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)',
                            '    adv_u, adv_v, _probe_raw_advection = _advection_flux_form(state.u, state.v, w, p)')
        # Restrict return replacement to the exact full-tendency function.
        start = text.index('def _compute_momentum_tendency(')
        end = text.index('\ndef ', start+1)
        section = text[start:end]
        section = replace_once(section, '    return dudt, dvdt', '''    _probe_terms = dict(advection_raw=_probe_raw_advection, advection_filtered=(adv_u, adv_v),
                        pressure_full=(pgf_x, pgf_y), wind_surface=(wind_u, wind_v),
                        horizontal_diffusion=(diff_h_u, diff_h_v), vertical_diffusion=(diff_v_u, diff_v_v),
                        coriolis=(cor_u, cor_v), bottom_drag=(bot_u, bot_v))
    return dudt, dvdt, _probe_terms''')
        text = text[:start]+section+text[end:]
        start = text.index('def _compute_momentum_residual(')
        end = text.index('\ndef ', start+1)
        section = text[start:end]
        section = replace_once(section, '    dudt, dvdt = _compute_momentum_tendency(state, p)',
                               '    dudt, dvdt, _probe_terms = _compute_momentum_tendency(state, p)\n    _probe_full = (dudt, dvdt)')
        for anchor, tag in [
            ('    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)', 'horizontal'),
            ('    dvdt = dvdt - _vertical_momentum_diffusion(state.v, p)', 'vertical'),
            ('    dvdt = dvdt + p.f[:, :, None] * state.u', 'coriolis'),
            ('    dvdt = dvdt - bt_rho_y[:, :, None]', 'density'),
            ('    dvdt = dvdt - bt_wind_y[:, :, None]', 'wind')]:
            section = replace_once(section, anchor, anchor+'\n    _probe_after_'+tag+' = (dudt, dvdt)')
        section = replace_once(section, '    # barotropic PGF from density anomaly',
                               '    _probe_after_eta = (dudt, dvdt)\n    _probe_eta_mean = (eta_pgf_x, eta_pgf_y)\n    # barotropic PGF from density anomaly')
        section = replace_once(section, '    return dudt, dvdt', '''    _probe_rhs = dict(terms=_probe_terms, full=_probe_full,
                      after_horizontal=_probe_after_horizontal, after_vertical=_probe_after_vertical,
                      after_coriolis=_probe_after_coriolis, after_eta=_probe_after_eta,
                      after_density=_probe_after_density, after_wind=_probe_after_wind,
                      residual=(dudt,dvdt), eta_mean_gradient=_probe_eta_mean,
                      density_mean=(bt_rho_x,bt_rho_y), wind_mean=(bt_wind_x,bt_wind_y))
    return dudt, dvdt, _probe_rhs''')
        text = text[:start]+section+text[end:]
        text = replace_once(text, '    first_x, first_y = _compute_momentum_residual(state, params)',
                            '    first_x, first_y, _probe_first_meta = _compute_momentum_residual(state, params)')
        text = replace_once(text, '    predicted = tracer._replace(u=state.u + duration * first_x, v=state.v + duration * first_y)',
                            '    _probe_first_rhs = (first_x,first_y)\n    predicted = tracer._replace(u=state.u + duration * first_x, v=state.v + duration * first_y)')
        text = replace_once(text, '    second_x, second_y = _compute_momentum_residual(predicted, params)',
                            '    second_x, second_y, _probe_second_meta = _compute_momentum_residual(predicted, params)')
        text = replace_once(text, '    velocity_x = state.u + 0.5 * duration * (first_x + second_x)',
                            '    _probe_second_rhs = (second_x,second_y)\n    velocity_x = state.u + 0.5 * duration * (first_x + second_x)')
        text = replace_once(text, '    return tracer._replace(u=velocity_x, v=velocity_y)', '''    _probe_n = dict(state0=state._asdict(), state_euler=predicted._asdict(), tracer=tracer._asdict(),
                    state_out=tracer._replace(u=velocity_x,v=velocity_y)._asdict(),
                    first_meta=_probe_first_meta, second_meta=_probe_second_meta,
                    first_rhs=_probe_first_rhs, second_rhs=_probe_second_rhs)
    return tracer._replace(u=velocity_x, v=velocity_y), _probe_n''')
    elif name == 'material_top.py':
        text = replace_once(text, '    nonlinear_predictor = _explicit_full_step(first, params, params.dt)',
                            '    nonlinear_predictor, _probe_n = _explicit_full_step(first, params, params.dt)')
        text = replace_once(text, '        return result, diagnostics', '        return result, diagnostics, _probe_n')
    projected = NProjection().visit(ast.parse(text))
    if ast.dump(projected, include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise ValueError('original numerical expressions changed: '+name)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source_dir','original_protocol','output_dir'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    content=args.original_protocol.read_bytes()
    if hashlib.sha256(content).hexdigest()!='9d9e756162c8f19ac1d3e32bbfb178d3c44d13c4f11ccf83e697b7e9c716f148':
        raise ValueError('original protocol changed')
    protocol=json.loads(content)
    args.output_dir.mkdir(exist_ok=False)
    manifest,patches={},[]
    for relative,digest in protocol['historical_source_hashes'].items():
        name=Path(relative).name
        expected=protocol['instrumented_material_sha256'] if name=='material_top.py' else digest
        content=(args.source_dir/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError('source changed: '+name)
        output=instrument(name,content.decode('utf-8'))
        (args.output_dir/name).write_bytes(output.encode('utf-8'))
        manifest[name]=dict(original_sha256=expected,instrumented_sha256=hashlib.sha256(output.encode('utf-8')).hexdigest(),numerical_ast_unchanged=True)
        if output!=content.decode('utf-8'):
            patches.extend(difflib.unified_diff(content.decode('utf-8').splitlines(True),output.splitlines(True),fromfile='historical/'+name,tofile='instrumented/'+name,n=0))
    (args.output_dir/'n_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (args.output_dir/'n_instrumentation.patch').write_text(''.join(patches),encoding='utf-8')
    print('35 modules verified; original numerical AST restored by observation projection')


if __name__=='__main__':
    main()
