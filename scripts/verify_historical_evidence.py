"""Verify original Git evidence bytes without reinterpreting its metrics."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def verify_cleanup(root):
    """Check retired paths and byte-preserved archives, without executing models."""
    declaration = json.loads((root / 'docs/cleanup_manifest.json').read_text(encoding='utf-8'))

    def path(name):
        selected = (root / name).resolve()
        if not selected.is_relative_to(root) or Path(name).is_absolute():
            raise ValueError('outside cleanup workspace: ' + name)
        return selected

    def git_hash(commit, name):
        path(name)
        content = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
        return hashlib.sha256(content).hexdigest()

    for item in declaration['deletions']:
        if path(item['path']).exists():
            raise ValueError('retired source still exists: ' + item['path'])
        if git_hash(declaration['baseline_commit'], item['path']) != item['git_blob_sha256']:
            raise ValueError('retired source identity mismatch: ' + item['path'])
    for item in declaration['moves']:
        if path(item['from']).exists():
            raise ValueError('duplicate pre-migration file: ' + item['from'])
        if hashlib.sha256(path(item['to']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('moved file mismatch: ' + item['to'])
        if item.get('commit') and git_hash(item['commit'], item['from']) != item['sha256']:
            raise ValueError('historical tool identity mismatch: ' + item['from'])
    return {'deleted_files': len(declaration['deletions']), 'moved_files': len(declaration['moves']),
            'models_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--cleanup', action='store_true', help='also verify retirement and archive bytes')
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
    report = {'status': 'verified_original_git_bytes', 'evidence': verified,
              'lineage_verified': True, 'numerical_rerun': False}
    if args.cleanup:
        report['cleanup'] = verify_cleanup(root)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
