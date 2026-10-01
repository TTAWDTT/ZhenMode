"""Independent NumPy/JSON witness verification; never imports solver/restart code."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
case = root / 'cross_process_reviewed'
report = json.loads((case / 'report.json').read_text())
def sha(data):
    return hashlib.sha256(data).hexdigest()
def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
for name, expected in report['artifact_sha256'].items():
    assert sha((case / name).read_bytes()) == expected, name
with np.load(case / 'continuous/continuous_endpoint.npz') as first, np.load(case / 'resumed/last_endpoint.npz') as last:
    assert set(first.files) == set(last.files) == {'u', 'v', 'T', 'S', 'eta', 'ice'}
    with np.load(case / 'continuous/global_controlled.npz') as output:
        identities = json.loads(str(output['final_state_identity_json']))
        assert str(output['verdict']) == 'PASS'
        assert output['accepted_steps'] == output['attempted_steps'] == 8
        assert output['elapsed_seconds'] == 80
        assert not output['physical_budget_closed']
        for name in first.files:
            a, b = first[name], last[name]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
            assert identities[name]['sha256'] == sha(a.tobytes()), name
    states = {name: sha(first[name].tobytes()) for name in first.files}
identities = [json.loads(path.read_text()) for path in case.rglob('*_identity.json')]
assert len(identities) == 4
assert len({item['stablehlo_sha256'] for item in identities}) == 1
assert len({canonical(item['contract']) for item in identities}) == 1
contract = identities[0]['contract']
commit = 'c4df6039e94dd3b3efcae02eaa86f86f0559dcbd'
for name, expected in contract['sources'].items():
    content = subprocess.check_output(['git', 'show', f'{commit}:src/{name}.py'])
    assert sha(content) == expected, name
summary = {'verified': True, 'implementation_commit': commit,
           'scope': 'synthetic CPU 80s; raw six-field identity and saved evidence only',
           'artifact_count': len(report['artifact_sha256']),
           'source_files_verified_against_commit': len(contract['sources']),
           'raw_endpoint_sha256': states,
           'effective_config_sha256': canonical(contract['effective_params']),
           'forcing_data_sha256': canonical(contract['forcing']),
           'grid_sha256': canonical(contract['grid']),
           'restart_contract_sha256': canonical(contract),
           'stablehlo_sha256': identities[0]['stablehlo_sha256'],
           'cross_process_wall_seconds': sum(report['process_wall_seconds'].values()),
           'historical_30_day_gate': 'FAIL unchanged', 'industrial_qualification': False}
(root / 'independent_verification.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
