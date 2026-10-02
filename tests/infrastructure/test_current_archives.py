"""Current snapshots contain real implementations and moved independent helpers."""
import hashlib
import shutil

import pytest

from ocean_solver.provenance.archives import current_source_files
from ocean_solver.provenance.sources import production_source_modules, source_paths
from tests.support.paths import REPOSITORY_ROOT


def test_current_archive_resolves_legacy_keys_and_contains_complete_source_closure():
    files = current_source_files(REPOSITORY_ROOT, [
        'src/config.py', 'tests/test_rstar_weak_time.py', 'tests/_helpers.py',
    ])
    assert files['src/config.py'] == REPOSITORY_ROOT / 'src/compat/config.py'
    assert files['tests/test_rstar_weak_time.py'] == REPOSITORY_ROOT / 'tests/research/rstar/test_rstar_weak_time.py'
    assert files['tests/_helpers.py'] == REPOSITORY_ROOT / 'tests/support/grid.py'
    assert files['tests/support/rstar/weak_time.py'].is_file()
    required = source_paths(REPOSITORY_ROOT / 'src', production_source_modules())
    assert set(required.values()) <= set(files.values())
    assert len({key for key in files if key.startswith('src/ocean_solver/')}) == 96
    # Every key hashes actual current bytes; aliases and implementations differ.
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    assert hashes['src/config.py'] == hashes['src/compat/config.py']
    assert hashes['src/config.py'] != hashes['src/ocean_solver/configuration.py']


def test_current_archive_rejects_missing_implementation_instead_of_hash_fallback(tmp_path):
    shutil.copytree(REPOSITORY_ROOT / 'src', tmp_path / 'src')
    (tmp_path / 'docs').mkdir()
    shutil.copyfile(REPOSITORY_ROOT / 'docs/source_test_layout.json', tmp_path / 'docs/source_test_layout.json')
    (tmp_path / 'src/ocean_solver/data/quality.py').unlink()
    with pytest.raises(ValueError, match='missing required source: ocean_solver/data/quality'):
        current_source_files(tmp_path, ['src/config.py'])


def test_current_archive_rejects_missing_declared_test():
    with pytest.raises(ValueError, match='missing declared current source'):
        current_source_files(REPOSITORY_ROOT, ['tests/missing_test.py'])
