"""The installed CLI must contain all its local module dependencies."""
import ast
import tomllib
from pathlib import Path


def test_distribution_includes_local_import_dependencies():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    modules = set(config["tool"]["setuptools"]["py-modules"])
    local_modules = {path.stem for path in (root / "src").glob("*.py")}
    for module in modules:
        tree = ast.parse((root / "src" / f"{module}.py").read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert imports & local_modules <= modules, module
