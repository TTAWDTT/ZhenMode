"""Architecture contracts for canonical sources, bridges and test support."""
import ast
import importlib
import json
from pathlib import Path

import pytest


def repository():
    from tests.support.paths import REPOSITORY_ROOT
    return REPOSITORY_ROOT


def assert_canonical_import_contract(root):
    modules = sorted((root / "src/ocean_solver").rglob("*.py"))
    assert modules, "No canonical implementation files were found"
    forbidden = {"config", "grid", "jax_solver_global", "stage_budgets", "run_long_integration_global"}
    manifest = root / "docs/research_engineering_layout.json"
    if manifest.is_file():
        forbidden.update(json.loads(manifest.read_text(encoding="utf-8"))["retired_modules"])
    forbidden |= {"ocean_solver.fd", "ocean_solver.data", "ocean_solver.fd.legacy", "ocean_solver.audit.legacy"}
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not {alias.name for alias in node.names} & forbidden, path
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden, path
                assert not {f"{node.module}.{alias.name}" for alias in node.names} & forbidden, path


def test_canonical_modules_do_not_import_legacy_facades():
    assert_canonical_import_contract(repository())


@pytest.mark.parametrize("forbidden_import", [
    "import config",
    "import ocean_solver.fd.legacy",
    "import ocean_solver.audit.legacy",
    "from config import PhysicsConfig",
    "from ocean_solver.fd.legacy import FDState",
    "from ocean_solver.audit.legacy import budget",
])
def test_canonical_import_contract_rejects_forbidden_imports(tmp_path, forbidden_import):
    canonical = tmp_path / "src/ocean_solver"
    facades = tmp_path / "src/compat"
    canonical.mkdir(parents=True)
    facades.mkdir(parents=True)
    (facades / "config.py").write_text("", encoding="utf-8")
    (canonical / "consumer.py").write_text(forbidden_import + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="consumer.py"):
        assert_canonical_import_contract(tmp_path)


def test_canonical_import_contract_rejects_empty_scan(tmp_path):
    facades = tmp_path / "src/compat"
    facades.mkdir(parents=True)
    (facades / "config.py").write_text("", encoding="utf-8")
    with pytest.raises(AssertionError, match="No canonical implementation files"):
        assert_canonical_import_contract(tmp_path)


def test_legacy_imports_alias_the_same_canonical_module_and_pickle_definitions():
    # Historical test ID retained; the current contract explicitly retires aliases.
    manifest = json.loads((repository() / "docs/research_engineering_layout.json").read_text(encoding="utf-8"))
    for legacy, canonical in manifest["retired_modules"].items():
        try:
            spec = importlib.util.find_spec(legacy)
        except ModuleNotFoundError:
            spec = None
        assert spec is None, legacy
        current = importlib.import_module(canonical)
        assert Path(current.__file__).is_relative_to(repository()), canonical
        for definition in vars(current).values():
            if isinstance(definition, type):
                assert definition.__module__ != legacy


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
    import ocean_solver.config.definitions as configuration
    import ocean_solver.forcing.air as air
    import ocean_solver.forcing.wind as wind
    import ocean_solver.io.climatology as climatology
    from ocean_solver.provenance.sources import source_root

    root = repository()
    assert Path(configuration._REPO_ROOT) == root
    assert Path(climatology._REPO_ROOT) == root
    for module in (air, wind):
        assert Path(module.CACHE_DIR).is_relative_to(root / "data")
    assert source_root(configuration.__file__) == root / "src"


@pytest.mark.parametrize("directory_name", ["installed", "compat", "ocean_solver"])
def test_source_root_supports_installed_paths_and_external_source_directory(tmp_path, directory_name):
    from ocean_solver.provenance.sources import source_root

    installed = tmp_path / directory_name
    package = installed / "ocean_solver"
    module = package / "data/air.py"
    module.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("", encoding="utf-8")
    assert source_root(module) == installed
    assert source_root(installed) == installed
