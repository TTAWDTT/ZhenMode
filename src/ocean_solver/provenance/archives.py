"""Collect complete current sources without rewriting historical receipts."""
import hashlib
import json
from pathlib import Path

from ocean_solver.provenance.sources import production_source_modules, source_paths


def current_source_files(repository, selected):
    """Resolve moved files and retain logical archive names for old replay readers.

    Current canonical implementations, bridges, support/oracle helpers and the
    layout declaration are all included. The reserved checkout/src labels bind
    the two real direct-file launchers without replacing legacy alias labels. Archived historical manifests stay
    bound to their historical commit; this helper never invents their hashes.
    """
    root = Path(repository)
    declaration = root / "docs/source_test_layout.json"
    layout = json.loads(declaration.read_text(encoding="utf-8"))
    engineering_declaration = root / 'docs/research_engineering_layout.json'
    engineering = (json.loads(engineering_declaration.read_text(encoding='utf-8'))
                   if engineering_declaration.is_file() else {})
    moves = engineering.get('source_moves', {})
    entrypoints = {"checkout/src/" + name + ".py": root / "src" / (name + ".py")
                   for name in ("jax_solver_global", "run_long_integration_global")}
    result = {}
    for item in selected:
        name = Path(item).relative_to(root).as_posix() if isinstance(item, Path) else str(item)
        if name in moves:
            path = root / moves[name]
        elif name.startswith("src/") and name.count("/") == 1:
            module = Path(name).stem
            path = source_paths(root / "src", (module,))[module]
        elif name in entrypoints:
            path = entrypoints[name]
        elif name in layout["test_moves"]:
            path = root / layout["test_moves"][name]
        elif name == "tests/_helpers.py":
            path = root / "tests/support/grid.py"
        elif name == "tests/_driver_helpers.py":
            path = root / "tests/support/driver.py"
        else:
            path = root / name
        if not path.is_file():
            raise ValueError("missing declared current source: " + name)
        result[name] = path
    for name, path in entrypoints.items():
        if not path.is_file():
            raise ValueError("missing required checkout entrypoint: " + name)
        result[name] = path
    required = tuple(dict.fromkeys((*production_source_modules(), *layout["legacy_modules"])))
    for name, path in source_paths(root / "src", required).items():
        result["src/" + name + ".py"] = path
        result[path.relative_to(root).as_posix()] = path
    research_root = root / 'research/src'
    if research_root.is_dir():
        for path in sorted(research_root.rglob('*.py')):
            result[path.relative_to(root).as_posix()] = path
    for path in sorted((root / "tests").rglob("*.py")):
        if "support" in path.parts or path.name == "__init__.py":
            result[path.relative_to(root).as_posix()] = path
    result["docs/source_test_layout.json"] = declaration
    if engineering_declaration.is_file():
        result['docs/research_engineering_layout.json'] = engineering_declaration
    return result


def _checked_source_name(root, name):
    if not isinstance(name, str):
        raise ValueError("source label must be a relative string")
    name = name.replace("\\", "/")
    requested = Path(name)
    if requested.is_absolute() or ".." in requested.parts or not (root / requested).resolve().is_relative_to(root):
        raise ValueError("outside-workspace source: " + name)
    return name


def _checked_source_file(root, path):
    actual = path.resolve()
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ValueError("missing or outside-workspace source: " + str(path))
    return actual


def current_source_path(repository, name):
    """Resolve a logical current source label with containment before and after mapping."""
    root = Path(repository).resolve()
    name = _checked_source_name(root, name)
    return _checked_source_file(root, current_source_files(root, [name])[name])


def verify_current_source_hashes(repository, manifest):
    """Verify a complete current closure; historical receipts require their old commit."""
    root = Path(repository).resolve()
    names = {name: _checked_source_name(root, name) for name in manifest}
    files = current_source_files(root, list(names.values()))
    if not set(current_source_files(root, [])).issubset(names.values()):
        raise ValueError("incomplete current source manifest")
    for name, expected in manifest.items():
        path = _checked_source_file(root, files[names[name]])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("runtime source mismatch: " + name)
