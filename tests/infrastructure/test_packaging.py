"""The distribution contains canonical imports without forwarding modules."""
import ast
import tomllib

from tests.support.paths import REPOSITORY_ROOT


def test_distribution_includes_local_import_dependencies():
    root = REPOSITORY_ROOT
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert not config["tool"]["setuptools"].get("py-modules")
    assert config["tool"]["setuptools"]["package-dir"][""] == "src"
    assert not list((root / "src/compat").glob("*.py"))
    assert "ocean_solver*" in config["tool"]["setuptools"]["packages"]["find"]["include"]
    package_modules = {".".join(path.relative_to(root / "src").with_suffix("").parts).removesuffix(".__init__")
                       for path in (root / "src/ocean_solver").rglob("*.py")}
    for path in (root / "src/ocean_solver").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ocean_solver"):
                assert node.module in package_modules, (path, node.module)
            elif isinstance(node, ast.Import):
                for name in (alias.name for alias in node.names if alias.name.startswith("ocean_solver")):
                    assert name in package_modules, (path, name)
