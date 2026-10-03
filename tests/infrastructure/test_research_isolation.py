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
    assert not set(config['tool']['setuptools'].get('py-modules', [])) & research
    assert not set(production_source_modules()) & research
    assert not any(x.startswith('zhenmode_research/') for x in production_source_modules())
    for name in research:
        assert not (REPOSITORY_ROOT / 'research/src/compat' / (name + '.py')).exists()
