import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

root = Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0'


def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_boundary_mask_preserves_tangential_velocity():
    evidence = load('boundary_witness').final_mask_witness(
        Path(__file__).parents[1] / 'src/jax_solver_global.py')
    assert evidence['u_byte_identical']
    assert evidence['v_wall_max_abs'] == 0.
    assert evidence['v_interior_byte_identical']


def test_native_output_is_instantaneous_and_labels_static_geometry(tmp_path):
    arrays = load('prepare_inputs').native_arrays(0.)
    state = SimpleNamespace(**{name: arrays[key] for name, key in
        [('u', 'u_m_s'), ('v', 'v_m_s'), ('T', 'T_degC'), ('S', 'S_psu'),
         ('eta', 'eta_m')]}, ice=np.zeros((64, 8)))
    writer = load('native_output').NativeSnapshots(tmp_path / 'run', arrays,
                                                  {'model': 'test_only'})
    writer.write(0, state, accepted=True, gate_evidence={'passed': True})
    with np.load(tmp_path / 'run/instant_00000s.npz') as saved:
        np.testing.assert_array_equal(saved['eta'], state.eta)
        np.testing.assert_array_equal(saved['u'], state.u)
        np.testing.assert_allclose(saved['h_geometric_m'].sum(axis=-1), 100. + state.eta)
        np.testing.assert_array_equal(saved['h0_m'], arrays['h0_m'])
    with pytest.raises(ValueError, match='successive'):
        writer.write(0, state, accepted=True, gate_evidence={'passed': True})
    with pytest.raises(ValueError, match='rejected'):
        writer.write(1000, state, accepted=False, gate_evidence={'passed': False})


def test_mom_request_preserves_native_fields_and_disables_averaging(tmp_path):
    path = tmp_path / 'diag_table'
    load('mom_diagnostics').write_native_diag_table(path)
    lines = path.read_text().splitlines()
    assert lines[2] == '"native",1000,"seconds",1,"seconds","time",'
    assert len(lines[3:]) == 7
    assert all('"all",.false.' in line for line in lines[3:])
    with pytest.raises(FileExistsError):
        load('mom_diagnostics').write_native_diag_table(path)
