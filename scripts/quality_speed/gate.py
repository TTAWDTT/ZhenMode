"""CLI: exit 0 PASS, 1 quality/speed failure, 2 incomplete, 3 incomparable."""
import argparse
import json
from pathlib import Path

from contract import evaluate
from manifest import verify_files


def load(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'duplicate JSON key: {key}')
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        control, candidate, policy = map(load, (args.control, args.candidate, args.policy))
        result = evaluate(control, candidate, policy, args.policy_sha256)
        errors = []
        for run, path in ((control, args.control), (candidate, args.candidate)):
            if isinstance(run, dict):
                errors.extend(verify_files(run, Path(path).parent))
        if errors:
            result.update(status='INCOMPLETE', **{'pass': False, 'industrial_qualified': False,
                                                'speed': None})
            result['reasons'].extend(errors)
    except (OSError, ValueError, TypeError) as error:
        result = {'status': 'INCOMPLETE', 'pass': False, 'industrial_qualified': False,
                  'speed': None, 'reasons': [str(error)]}
    with Path(args.out).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, allow_nan=False))
    return {'PASS': 0, 'QUALITY_FAIL': 1, 'SPEED_FAIL': 1,
            'INCOMPLETE': 2, 'NOT_COMPARABLE': 3}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
