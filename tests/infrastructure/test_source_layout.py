"""Architecture contracts for canonical sources, bridges and test support."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


def repository():
    from tests.support.paths import REPOSITORY_ROOT
    return REPOSITORY_ROOT


def assert_canonical_import_contract(root):
    modules = sorted((root / "src/zhenmode").rglob("*.py"))
    assert modules, "No canonical implementation files were found"
    forbidden = {"config", "grid", "jax_solver_global", "stage_budgets", "run_long_integration_global"}
    forbidden |= {"zhenmode.fd", "zhenmode.data", "zhenmode.fd.legacy", "zhenmode.model.audit.legacy"}
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
    "import zhenmode.fd.legacy",
    "import zhenmode.model.audit.legacy",
    "from config import PhysicsConfig",
    "from zhenmode.fd.legacy import FDState",
    "from zhenmode.model.audit.legacy import budget",
])
def test_canonical_import_contract_rejects_forbidden_imports(tmp_path, forbidden_import):
    canonical = tmp_path / "src/zhenmode"
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


def test_tests_use_support_instead_of_importing_other_test_files():
    def is_test_owner(module):
        return any(part.startswith("test_") for part in module.replace("\\", "/").replace("/", ".").split("."))

    for directory in ("tests", "scripts", "src"):
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
    import zhenmode.model.config.definitions as configuration
    import zhenmode.model.forcing.air as air
    import zhenmode.model.forcing.wind as wind
    import zhenmode.model.io.climatology as climatology
    from zhenmode.provenance.sources import source_root

    root = repository()
    assert Path(configuration._REPO_ROOT) == root
    assert Path(climatology._REPO_ROOT) == root
    for module in (air, wind):
        assert Path(module.CACHE_DIR).is_relative_to(root / "data")
    assert source_root(configuration.__file__) == root / "src"


@pytest.mark.parametrize("directory_name", ["installed", "compat", "zhenmode"])
def test_source_root_supports_installed_paths_and_external_source_directory(tmp_path, directory_name):
    from zhenmode.provenance.sources import source_root

    installed = tmp_path / directory_name
    package = installed / "zhenmode"
    module = package / "data/air.py"
    module.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("", encoding="utf-8")
    assert source_root(module) == installed
    assert source_root(installed) == installed


def assert_model_dependencies(text, package):
    allowed = ("zhenmode.model", "zhenmode.provenance")
    for node in ast.walk(ast.parse(text)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Call) and ast.unparse(node.func) in {
            "importlib.import_module", "__import__", "runpy.run_module",
        }:
            modules = [arg.value for arg in node.args[:1]
                       if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        for module in modules:
            if module and module.startswith("zhenmode"):
                assert any(module == prefix or module.startswith(prefix + ".") for prefix in allowed), module


def test_model_does_not_depend_on_product_workflow():
    paths = list((repository() / "src/zhenmode/model").rglob("*.py"))
    assert paths, "an empty dependency scan is not a pass"
    for path in paths:
        package = ".".join(path.parent.relative_to(repository() / "src").parts)
        assert_model_dependencies(path.read_text(encoding="utf-8"), package)


@pytest.mark.parametrize("source", [
    "from zhenmode.evaluation.metrics import mixed_layer_depth",
    "import zhenmode.baselines.mom6.adapter",
    "from zhenmode import execution",
    "from ...evaluation import metrics",
    "importlib.import_module('zhenmode.execution.runs')",
    "from zhenmode.cli import main",
])
def test_model_boundary_has_independent_negative_controls(source):
    with pytest.raises(AssertionError):
        assert_model_dependencies(source, "zhenmode.model.runtime")
