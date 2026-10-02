"""Distribution sources cover canonical imports and historical aliases."""
import ast
import tomllib

from tests.support.paths import REPOSITORY_ROOT


def test_distribution_includes_local_import_dependencies():
    root = REPOSITORY_ROOT
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    modules = set(config["tool"]["setuptools"]["py-modules"])
    assert modules == {path.stem for path in (root / "src/compat").glob("*.py")}
    assert config["tool"]["setuptools"]["package-dir"]["ocean_solver"] == "src/ocean_solver"
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
