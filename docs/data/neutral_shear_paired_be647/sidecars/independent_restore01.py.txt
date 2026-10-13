"""One finite independent byte restore audit; stdlib only, no scientific imports."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
BASE = STAGE.parent
PACKET = HERE / 'packet01'
OUTPUT = HERE / 'independent-restored01'
JOINED = HERE / 'independent-restored01-joined.zip'
PREFIX = 'neutral_shear_paired_be647_attempt02'
MANIFEST = PACKET / (PREFIX + '_manifest.json')
SOURCE = 'be647d70fc7394f3014dca8bc776e207d4f02ae3'
EXPECTED_MANIFEST = '476ddf2848d6a6ab341d6b7f40223f287a3ad19f3a1ba4533c9fe96bcca48796'
EXPECTED_SELECTION = '57c8517de678ae4921490af23fa76464f0165a574a83127d601b38065181b603'
EXPECTED_READER = '37c6dac738db527495f389708dc906de5a3d47a71d653fe5b82c87f384e9b4ac'
EXPECTED_PACKER = '6122c29af2004d62ba95ade12cde3dd43d0d74e9f1016cfb26ef9465da314809'
EXPECTED_ARCHIVE = '79747029cdf37d73f79f0263f6faa5b38a9e586fa7c866140656e4f7cbcab8ab'
CASES = ('00-main', '01-main', '02-main', '03-zero-spice', '04-zero-shear-null')
ROOTS = {STAGE.name, 'neutral-shear-paired-be647', 'horizontal-paired-product-v1',
         'independent-neutral-shear-native'}
OMIT_DIRS = {'__pycache__', 'publication-preparation', 'report-frame', 'packet01',
             'independent-restored01'}
LATE_PATTERNS = ('results-draft*.md', 'review/RESULTS_DRAFT*.json',
                 'review/MAIN3_REPORT_ADDITION01.md')
CORE = ('oracle.py', 'analytic_source_controls.py', 'audit_initial.py',
        'audit_arrays.py', 'prepare_audit_input.py', 'audit_summary.py')
TOOLS = ('EXPORT_SPEC01.md', 'pack_original01.py', 'restore_original01.py.txt',
         'ROOT_EXPORT_GATE_TEMPLATE01.json')
CHECKS = []


def require(condition, message):
    CHECKS.append({'name': message, 'pass': bool(condition)})
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name):
    require(isinstance(name, str) and '\\' not in name and ':' not in name,
            'member syntax: ' + str(name))
    parts = name.split('/')
    require(all(part not in ('', '.', '..') for part in parts)
            and not PurePosixPath(name).is_absolute() and parts[0] in ROOTS,
            'member confined to fixed original roots: ' + name)
    return parts


def original_catalog():
    """Independent inventory construction; never import the packer's catalog."""
    files = set()
    roots = (STAGE, BASE / 'neutral-shear-paired-be647',
             BASE / 'horizontal-paired-product-v1' / 'installed')
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), 'catalog root regular: ' + str(root))
        for current, dirs, names in os.walk(root, followlinks=False):
            current = Path(current)
            dirs[:] = [name for name in dirs if name not in OMIT_DIRS]
            for name in dirs:
                require(not (current / name).is_symlink(), 'no source directory symlink')
            for name in names:
                item = current / name
                rel = item.relative_to(root).as_posix()
                if item.suffix in ('.pyc', '.pyo') or any(fnmatch.fnmatchcase(rel, p) for p in LATE_PATTERNS):
                    continue
                require(item.is_file() and not item.is_symlink(), 'regular original: ' + rel)
                files.add(item.relative_to(BASE).as_posix())
    hp = 'horizontal-paired-product-v1/'
    files.update(hp + name for name in ('wheel-identity.json', 'build-input.json', 'build_install.py',
                 'SOURCE_READY-identity01.json', 'SOURCE_READY-qualified02.json', 'review/source-git.tar',
                 'wheel/zhenmode-0.1.0-py3-none-any.whl'))
    old = 'independent-neutral-shear-native/'
    files.update(old + name for name in ('declared-plan.json', 'pre-execution-definitions01.json'))
    files.update(old + 'review/' + name for name in CORE)
    files.update(old + case + '/science-verdict01.json' for case in CASES)
    files.update(STAGE.name + '/publication-preparation/' + name for name in TOOLS)
    return files


def main():
    started = time.monotonic()
    write_new(HERE / 'independent-restore01-worker-start.json', {
        'status': 'STARTED_NOT_COMPLETED', 'script_sha256': sha(__file__),
        'declaration_sha256': sha(HERE / 'INDEPENDENT_RESTORE_DECLARATION01.json'),
        'pid': os.getpid(), 'model_calls': 0})
    result = {'status': 'FAIL_PRESERVED_NO_RETRY', 'model_calls': 0, 'npz_decoded': 0,
              'oracle_calls': 0, 'scientific_metrics_recomputed': 0, 'reader_invocations': 0,
              'scope': 'Selected original bytes only; not new scientific acceptance.'}
    try:
        require(not OUTPUT.exists() and not JOINED.exists(), 'fresh restore and joined ZIP paths')
        require(OUTPUT.parent == HERE and JOINED.parent == HERE, 'all outputs inside excluded preparation area')
        require(sha(MANIFEST) == EXPECTED_MANIFEST, 'root-bound manifest SHA')
        require(sha(HERE / 'selection01.json') == EXPECTED_SELECTION
                and sha(PACKET / 'selection01.json') == EXPECTED_SELECTION, 'both exact selection copies')
        require(sha(PACKET / 'restore_original01.py.txt') == EXPECTED_READER
                and sha(HERE / 'restore_original01.py.txt') == EXPECTED_READER, 'both reviewed reader copies')
        require(sha(HERE / 'pack_original01.py') == EXPECTED_PACKER, 'reviewed packer bytes, not executed')
        manifest, selection = read(MANIFEST), read(HERE / 'selection01.json')
        require(manifest['source_revision'] == SOURCE and manifest['selection_sha256'] == EXPECTED_SELECTION
                and manifest['restore_reader_sha256'] == EXPECTED_READER
                and manifest['packer_sha256'] == EXPECTED_PACKER, 'manifest source/reader/packer binding')
        require(all(manifest[key] == value for key, value in selection.items()), 'manifest retains every selection field')
        require(manifest['files_count'] == len(manifest['files']) == 841, 'actual reviewed 841-file count')
        require(manifest['compression'] == 'ZIP_STORED' and manifest['part_bytes'] == 20971520
                and manifest['archive_bytes'] == 673508371 and manifest['archive_sha256'] == EXPECTED_ARCHIVE,
                'archive method/size/hash declaration')
        members = {}
        for row in manifest['files']:
            parts = safe_member(row['member'])
            require(row['member'] not in members, 'unique exact member')
            path = BASE.joinpath(*parts)
            require(str(path) == row['path'] and path.resolve().is_relative_to(BASE.resolve()), 'original path exact binding')
            members[row['member']] = row
        require(len({x.casefold() for x in members}) == len(members), 'no case-fold collision')
        require(original_catalog() == set(members), 'complete original scope, exact catalog without generic exclusions')
        require(sum(x['bytes'] for x in members.values()) == selection['original_bytes'] == 673309247,
                'original selected total bytes')
        before = {}
        for name, row in members.items():
            path = Path(row['path'])
            stat0 = path.stat()
            require(path.is_file() and not path.is_symlink()
                    and (stat0.st_size, stat0.st_mtime_ns) == (row['bytes'], row['mtime_ns']),
                    'original size/mtime before: ' + name)
            digest = sha(path)
            stat1 = path.stat()
            require(digest == row['sha256'] and (stat1.st_size, stat1.st_mtime_ns)
                    == (stat0.st_size, stat0.st_mtime_ns), 'original exact SHA before: ' + name)
            before[name] = digest
        print('Original 841-file scope and original SHA identities verified; invoking reader once after parts.', flush=True)
        hp = BASE / 'horizontal-paired-product-v1'
        wheel = read(hp / 'wheel-identity.json')
        installed = {hp.name + '/installed/' + key + '.py': digest
                     for key, digest in wheel['package_source_sha256'].items()}
        actual_py = {name for name in members if name.startswith(hp.name + '/installed/') and name.endswith('.py')}
        require(len(installed) == 133 and actual_py == set(installed)
                and all(members[name]['sha256'] == digest for name, digest in installed.items()), 'complete installed133 bytes')
        for case in CASES:
            prefix = STAGE.name + '/' + case + '/'
            required = {'native.npz', 'initial-readback.npz', 'constant-source-readback.npz',
                        'audit-input01/audit-native.npz', 'audit-input01/audit-initial-diagnostics.npz'}
            require(all(prefix + name in members for name in required), 'native/initial/audit original group: ' + case)
            expected_outputs = {prefix + f'outputs/output-{i:04d}.npz' for i in range(41)}
            actual_outputs = {name for name in members if name.startswith(prefix + 'outputs/')}
            require(actual_outputs == expected_outputs, 'exact 41 original output archives: ' + case)
        bridge = read(STAGE / 'review/analytic01-binding.json')
        require(bridge['reuse'] is True and bridge['new_observer_calls'] == 0
                and len(bridge['origin_artifact_sha256']) == 23, 'bridge remains reused23, zero new analytic calls')
        for name, digest in bridge['origin_artifact_sha256'].items():
            require(members['neutral-shear-paired-be647/' + name]['sha256'] == digest, 'original bridge artifact: ' + name)
        require('neutral-shear-paired-be647/INITIALIZE_FAILURE_DISPOSITION01.json' in members
                and 'neutral-shear-paired-be647/00-main/initial-readback.npz' in members,
                'failed initialization originals retained')
        require(len([x for x in members if x.startswith('independent-neutral-shear-native/')]) == 13,
                'only original five verdicts plus plan/definitions/six cores from old70')
        whole, total, checked_parts = hashlib.sha256(), 0, []
        require(len(manifest['parts']) == 33, 'actual 33-part distribution')
        for index, item in enumerate(manifest['parts'], 1):
            require(item['name'] == f'{PREFIX}_archive.part{index:03d}', 'safe sequential part name')
            require(type(item['bytes']) is int and 0 < item['bytes'] <= 20971520
                    and (index == 33 or item['bytes'] == 20971520), 'part byte declaration')
            digest, size = hashlib.sha256(), 0
            with (PACKET / item['name']).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block); whole.update(block); size += len(block); total += len(block)
            require(size == item['bytes'] and digest.hexdigest() == item['sha256'], 'independent part SHA: ' + item['name'])
            checked_parts.append(dict(item))
        require(total == 673508371 and whole.hexdigest() == EXPECTED_ARCHIVE, 'independent concatenated archive hash')
        command = [sys.executable, str(PACKET / 'restore_original01.py.txt'), '--manifest', str(MANIFEST), '--output', str(OUTPUT)]
        result['reader_invocations'] = 1
        restored = subprocess.run(command, check=False, capture_output=True, text=True)
        result['reader_returncode'] = restored.returncode
        result['reader_stdout'] = restored.stdout
        result['reader_stderr'] = restored.stderr
        require(restored.returncode == 0, 'single reader exited0')
        print('Single reader completed; independently comparing all original/restored bytes.', flush=True)
        with zipfile.ZipFile(JOINED) as archive:
            entries = archive.infolist()
            require(len(entries) == len(members) and {entry.filename for entry in entries} == set(members), 'exact joined ZIP inventory')
            for entry in entries:
                mode = entry.external_attr >> 16
                require(entry.compress_type == zipfile.ZIP_STORED and not entry.flag_bits & 1
                        and not entry.is_dir() and not stat.S_ISLNK(mode)
                        and stat.S_IFMT(mode) in (0, stat.S_IFREG)
                        and entry.file_size == members[entry.filename]['bytes'], 'safe ZIP entry metadata: ' + entry.filename)
        restored_names = {path.relative_to(OUTPUT).as_posix() for path in OUTPUT.rglob('*') if path.is_file()}
        require(restored_names == set(members) | {'RESTORE_RECEIPT.json'}, 'exact restored inventory plus sole receipt')
        comparisons = []
        for name, row in members.items():
            original = Path(row['path'])
            restored_path = OUTPUT.joinpath(*PurePosixPath(name).parts)
            require(restored_path.resolve().is_relative_to(OUTPUT.resolve()) and not restored_path.is_symlink(), 'restored path confinement')
            original_sha, restored_sha = sha(original), sha(restored_path)
            original_stat, restored_stat = original.stat(), restored_path.stat()
            require(original_sha == restored_sha == before[name] == row['sha256']
                    and original_stat.st_size == restored_stat.st_size == row['bytes']
                    and original_stat.st_mtime_ns == row['mtime_ns'], 'original vs restored exact bytes: ' + name)
            comparisons.append({'member': name, 'bytes': row['bytes'], 'expected_sha256': row['sha256'],
                                'original_before_sha256': before[name], 'original_after_sha256': original_sha,
                                'restored_sha256': restored_sha})
        require(original_catalog() == set(members), 'original scope unchanged after restoration')
        receipt = read(OUTPUT / 'RESTORE_RECEIPT.json')
        require(receipt['status'] == 'RESTORED_ALL_SELECTED_ORIGINAL_BYTES' and receipt['files'] == 841
                and receipt['parts'] == 33 and receipt['archive_sha256'] == EXPECTED_ARCHIVE
                and receipt['manifest_sha256'] == EXPECTED_MANIFEST and receipt['model_calls'] == 0
                and receipt['numerical_decoding'] is False and receipt['scientific_recalculation'] is False,
                'authentic byte-only reader receipt')
        result.update(status='PASS_INDEPENDENT_SELECTED_ORIGINAL_BYTE_RESTORE', files=841, parts=33,
                      manifest_sha256=EXPECTED_MANIFEST, selection_sha256=EXPECTED_SELECTION,
                      archive_sha256=EXPECTED_ARCHIVE, reader_sha256=EXPECTED_READER,
                      output=str(OUTPUT), joined_archive=str(JOINED), checked_parts=checked_parts,
                      file_comparisons=comparisons, restored_receipt_sha256=sha(OUTPUT / 'RESTORE_RECEIPT.json'),
                      coverage={'installed_python': 133, 'native_case_groups': 5, 'output_archives': 205,
                                'failed_predecessor': True, 'original_bridge_artifacts': 23,
                                'old70_metadata_and_core_only_files': 13},
                      late_report_metadata='Excluded by the declared selection patterns; not part of archive identity.')
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started, checks=CHECKS,
                      script_sha256=sha(__file__), declaration_sha256=sha(HERE / 'INDEPENDENT_RESTORE_DECLARATION01.json'))
        write_new(HERE / 'INDEPENDENT_RESTORE_VERDICT01.json', result)


if __name__ == '__main__':
    main()
