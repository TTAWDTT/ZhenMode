"""Verify original Git evidence bytes without reinterpreting its metrics."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index_path = root / 'archive/evidence/index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    verified = []
    payloads = []
    for item in index['evidence']:
        selector = item['commit'] + ':' + item['path']
        result = subprocess.run(['git', 'show', selector], cwd=root, capture_output=True)
        if result.returncode:
            raise SystemExit('Missing Git evidence object: ' + selector + '; fetch its historical commit.')
        digest = hashlib.sha256(result.stdout).hexdigest()
        if digest != item['sha256']:
            raise SystemExit('Historical byte mismatch: ' + selector)
        payloads.append((item, result.stdout))
        verified.append(selector)
    lineage = index['lineage']
    subprocess.run(['git', 'merge-base', '--is-ancestor', lineage['ancestor'], lineage['descendant']], cwd=root, check=True)
    if args.export:
        destination = args.export.resolve()
        destination.mkdir(parents=True, exist_ok=False)
        for item, payload in payloads:
            path = destination / item['id'] / Path(item['path']).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (destination / 'index.json').write_bytes(index_path.read_bytes())
    print(json.dumps({'status': 'verified_original_git_bytes', 'evidence': verified,
                      'lineage_verified': True, 'numerical_rerun': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
