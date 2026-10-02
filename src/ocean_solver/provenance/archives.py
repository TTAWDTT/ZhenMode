"""Collect complete current sources without rewriting historical receipts."""
import json
from pathlib import Path

from ocean_solver.provenance.sources import production_source_modules, source_paths


def current_source_files(repository, selected):
    """Resolve moved files and retain logical archive names for old replay readers.

    Current canonical implementations, bridges, support/oracle helpers and the
    layout declaration are all included. Archived historical manifests stay
    bound to their historical commit; this helper never invents their hashes.
    """
    root = Path(repository)
    declaration = root / "docs/source_test_layout.json"
    layout = json.loads(declaration.read_text(encoding="utf-8"))
    result = {}
    for item in selected:
        name = Path(item).relative_to(root).as_posix() if isinstance(item, Path) else str(item)
        if name.startswith("src/") and name.count("/") == 1:
            module = Path(name).stem
            path = source_paths(root / "src", (module,))[module]
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
    required = tuple(dict.fromkeys((*production_source_modules(), *layout["legacy_modules"])))
    for name, path in source_paths(root / "src", required).items():
        result["src/" + name + ".py"] = path
        result[path.relative_to(root).as_posix()] = path
    for path in sorted((root / "tests").rglob("*.py")):
        if "support" in path.parts or path.name == "__init__.py":
            result[path.relative_to(root).as_posix()] = path
    result["docs/source_test_layout.json"] = declaration
    return result
