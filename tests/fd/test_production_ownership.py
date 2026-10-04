"""Keep research and file loading out of the formal array operators."""
import ast

import pytest

from tests.support.paths import REPOSITORY_ROOT

PACKAGE = REPOSITORY_ROOT / "src" / "zhenmode" / "model"
PURE_OWNERS = (
    "solver/numerics", "solver/geometry", "solver/dynamics", "solver/physics",
    "solver/timestepping", "solver/state.py",
)
OLD_OWNER_PREFIXES = (
    "zhenmode.fd.", "zhenmode.data.", "zhenmode.configuration",
    "zhenmode.model.geometry.grid", "zhenmode.provenance.restart",
    "zhenmode.model.runtime.seasonal", "zhenmode.model.runtime.records",
    "zhenmode.model.runtime.output", "zhenmode.model.runtime.paths",
    "zhenmode.model.runtime.recovery", "zhenmode.model.runtime.metrics",
)
INPUT_LIBRARIES = {"netCDF4", "h5py", "xarray", "pandas", "requests", "urllib"}


def forbidden_dependencies(text, *, array_owner=False):
    """Inspect imports and actual reader calls, including dynamic import strings."""
    failures = []
    tree = ast.parse(text)
    for node in ast.walk(tree):
        imported = []
        if isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        elif isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.Call):
            callable_name = ast.unparse(node.func)
            if callable_name in {"importlib.import_module", "__import__", "runpy.run_module"}:
                imported = [argument.value for argument in node.args[:1]
                            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)]
            if array_owner and callable_name in {
                "open", "np.load", "numpy.load", "np.loadtxt", "numpy.loadtxt",
                "np.save", "np.savez", "np.savez_compressed", "Path.read_bytes",
            }:
                failures.append(callable_name)
        for module in imported:
            if module.split(".")[0] in {"research", "zhenmode_research"}:
                failures.append(module)
            if module.startswith("zhenmode.candidates") or module.startswith(OLD_OWNER_PREFIXES):
                failures.append(module)
            if array_owner and (
                module.split(".")[0] in INPUT_LIBRARIES
                or module.startswith(("zhenmode.model.io", "zhenmode.model.inputs"))
            ):
                failures.append(module)
    return failures


def canonical_sources():
    yield from PACKAGE.rglob("*.py")


def test_formal_owners_do_not_import_research_or_old_implementation_paths():
    paths = list(canonical_sources())
    assert paths, "an empty architecture scan is not a pass"
    failures = {str(path.relative_to(PACKAGE)): forbidden_dependencies(path.read_text(encoding="utf-8"))
                for path in paths}
    assert not {path: problems for path, problems in failures.items() if problems}


def test_array_owners_do_not_load_external_inputs():
    owners = [PACKAGE / owner for owner in PURE_OWNERS]
    assert all(owner.is_dir() or owner.is_file() for owner in owners)
    paths = [path for owner in owners
             for path in ([owner] if owner.is_file() else owner.rglob("*.py"))]
    assert paths
    failures = {str(path.relative_to(PACKAGE)): forbidden_dependencies(
        path.read_text(encoding="utf-8"), array_owner=True) for path in paths}
    assert not {path: problems for path, problems in failures.items() if problems}


@pytest.mark.parametrize("source", [
    "from zhenmode_research.candidates.material import solver",
    "import research.experiments.example",
    "from zhenmode.fd.horizontal import _d_dx",
    "importlib.import_module('zhenmode.candidates.material.solver')",
    "__import__('zhenmode_research.candidates.fv.geometry')",
])
def test_dependency_negative_controls(source):
    assert forbidden_dependencies(source)


@pytest.mark.parametrize("source", [
    "import netCDF4", "from zhenmode.model.inputs.initial_conditions import get_initial_fields",
    "np.load('input.npz')", "open('input.nc')",
])
def test_input_negative_controls(source):
    assert forbidden_dependencies(source, array_owner=True)


def test_allowed_array_dependency():
    assert forbidden_dependencies("from zhenmode.model.solver.numerics.vertical import _d_dz",
                                  array_owner=True) == []
