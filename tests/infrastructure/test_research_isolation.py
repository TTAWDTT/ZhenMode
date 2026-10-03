"""The production distribution and its source envelope do not require research."""
import ast
import json
import tomllib

from ocean_solver.provenance.sources import production_source_modules
from tests.support.paths import REPOSITORY_ROOT


def test_production_package_has_no_research_implementations_or_imports():
    root = REPOSITORY_ROOT / 'src/ocean_solver'
    assert not (root / 'candidates').exists()
    scanned = list(root.rglob('*.py'))
    assert scanned
    for path in scanned:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
            assert not any(x.startswith(('research', 'zhenmode_research', 'ocean_solver.candidates'))
                           for x in modules), path


def test_production_distribution_and_identity_exclude_research_bridges():
    layout = json.loads((REPOSITORY_ROOT / 'docs/research_engineering_layout.json').read_text(encoding='utf-8'))
    research = set(layout['research_legacy_modules'])
    config = tomllib.loads((REPOSITORY_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert not set(config['tool']['setuptools']['py-modules']) & research
    assert not set(production_source_modules()) & research
    assert not any(x.startswith('zhenmode_research/') for x in production_source_modules())
    for name in research:
        assert (REPOSITORY_ROOT / 'research/src/compat' / (name + '.py')).is_file()


def test_historical_layout_declaration_remains_original_git_bytes():
    import subprocess
    original = subprocess.check_output(['git', 'show', '25258950905f9d1aa84509c4c99ebad9ef33ba2b:docs/source_test_layout.json'], cwd=REPOSITORY_ROOT)
    assert (REPOSITORY_ROOT / 'docs/source_test_layout.json').read_bytes() == original
