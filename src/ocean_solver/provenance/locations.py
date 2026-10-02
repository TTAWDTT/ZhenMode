"""Locate source and data roots consistently in checkouts and installed wheels."""
from pathlib import Path


def source_root(location):
    """Resolve a module/file/directory to its actual distribution source root.

    Checkout implementations are below src/ocean_solver and legacy bridges
    below src/compat. Installed modules share a site-packages source root.
    Explicit external source directories remain valid for strict restart tests.
    """
    path = Path(location).resolve()
    directory = path.parent if path.suffix == ".py" else path
    if (directory / "ocean_solver").is_dir():
        return directory
    for ancestor in (directory, *directory.parents):
        if ancestor.name == "ocean_solver" and (ancestor / "__init__.py").is_file():
            return ancestor.parent
        if ancestor.name == "compat" and (ancestor.parent / "ocean_solver").is_dir():
            return ancestor.parent
    return directory
