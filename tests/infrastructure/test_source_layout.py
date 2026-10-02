"""Architecture contracts for canonical sources, bridges and test support."""
import ast
import importlib
import json
from pathlib import Path

import pytest


def repository():
    from tests.support.paths import REPOSITORY_ROOT
    return REPOSITORY_ROOT


def test_canonical_modules_do_not_import_legacy_facades():
    # This also fails before the migration without depending on its manifest.
    root = Path(__file__).resolve().parents[2]
    legacy = {path.stem for path in (root / "src/compat").glob("*.py")}
    if not legacy:
        legacy = {path.stem for path in (root / "src").glob("*.py")}
    for path in (root / "src/ocean_solver").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not {alias.name for alias in node.names} & legacy, path
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in legacy, path
                assert node.module not in {"ocean_solver.fd.legacy", "ocean_solver.audit.legacy"}, path


def test_legacy_imports_alias_the_same_canonical_module_and_pickle_definitions():
    manifest = json.loads((repository() / "docs/source_test_layout.json").read_text(encoding="utf-8"))
    for legacy, canonical in manifest["legacy_modules"].items():
        old = importlib.import_module(legacy)
        current = importlib.import_module(canonical)
        assert old is current, legacy
        for definition in vars(current).values():
            if isinstance(definition, type) and definition.__module__ == legacy:
                assert getattr(old, definition.__name__) is definition


def test_tests_use_support_instead_of_importing_other_test_files():
    def is_test_owner(module):
        return any(part.startswith("test_") for part in module.replace("\\", "/").replace("/", ".").split("."))

    for directory in ("tests", "research/reviews", "scripts", "src"):
        for path in (repository() / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not is_test_owner(node.module or ""), path
                elif isinstance(node, ast.Import):
                    assert not any(is_test_owner(alias.name) for alias in node.names), path
                elif isinstance(node, ast.Call):
                    function = node.func
                    name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
                    if name in {"run_path", "run_module", "spec_from_file_location", "import_module", "__import__"}:
                        strings = [part.value for arg in node.args for part in ast.walk(arg)
                                   if isinstance(part, ast.Constant) and isinstance(part.value, str)]
                        assert not any(is_test_owner(value) for value in strings), path


def test_checkout_data_roots_stay_at_the_repository():
    from ocean_solver import configuration
    from ocean_solver.data import air, climatology, wind
    from ocean_solver.provenance.locations import source_root

    root = repository()
    assert Path(configuration._REPO_ROOT) == root
    assert Path(climatology._REPO_ROOT) == root
    for module in (air, wind):
        assert Path(module.CACHE_DIR).is_relative_to(root / "data")
    assert source_root(configuration.__file__) == root / "src"


@pytest.mark.parametrize("directory_name", ["installed", "compat", "ocean_solver"])
def test_source_root_supports_installed_paths_and_external_source_directory(tmp_path, directory_name):
    from ocean_solver.provenance.locations import source_root

    installed = tmp_path / directory_name
    package = installed / "ocean_solver"
    module = package / "data/air.py"
    module.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("", encoding="utf-8")
    assert source_root(module) == installed
    assert source_root(installed) == installed
