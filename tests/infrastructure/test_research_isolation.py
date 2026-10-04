"""The production distribution and its source envelope do not require research."""
import ast
import tomllib

from tests.support.paths import REPOSITORY_ROOT
from zhenmode.provenance.sources import production_source_modules


def test_production_package_has_no_research_implementations_or_imports():
    root = REPOSITORY_ROOT / 'src/zhenmode'
    assert not (root / 'candidates').exists()
    scanned = list(root.rglob('*.py'))
    assert scanned
    for path in scanned:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
            assert not any(x.startswith(('research', 'zhenmode_research', 'zhenmode.candidates'))
                           for x in modules), path


def test_production_distribution_and_identity_exclude_research_bridges():
    config = tomllib.loads((REPOSITORY_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert config['tool']['setuptools']['packages']['find']['include'] == ['zhenmode*']
    assert not config['tool']['setuptools'].get('py-modules', [])
    assert all(name.startswith('zhenmode/') for name in production_source_modules())
