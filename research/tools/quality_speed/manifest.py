"""Content identities and portable, explicitly selected evidence attachments."""
from pathlib import Path

from contract import file_digest, sha


def attach_files(run, root, files):
    """Hash selected relative files. Do not infer absent metrics from old payloads."""
    root = Path(root).resolve()
    result = dict(run)
    result['artifacts'] = {}
    for name in files:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'not a regular artifact within root: {name}')
        result['artifacts'][name] = file_digest(path)
    return result


def verify_files(run, root):
    errors = []
    root = Path(root).resolve()
    artifacts = run.get('artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        return ['artifacts: missing']
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            errors.append('artifacts: paths must be relative')
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'artifacts: unavailable/outside root: {name}')
        elif not sha(expected) or file_digest(path) != expected:
            errors.append(f'artifacts: hash mismatch: {name}')
    # Every provenance identity must refer to attached bytes, not just a label.
    referenced = [v for k, v in run.items() if k.endswith('_sha256')]
    quality = run.get('quality')
    if isinstance(quality, dict):
        referenced += [q.get('evidence_sha256') for q in quality.values() if isinstance(q, dict)]
        referenced += [q.get('definition_sha256') for q in quality.values() if isinstance(q, dict)]
    for value in referenced:
        if value not in artifacts.values():
            errors.append(f'artifacts: unattached identity: {value}')
    return errors
