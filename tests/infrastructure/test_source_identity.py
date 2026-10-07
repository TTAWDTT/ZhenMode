"""Actual-file source closure and negative controls, independent of migration maps."""
import hashlib
import importlib
import shutil
from pathlib import Path

import pytest

from tests.support.paths import REPOSITORY_ROOT
from zhenmode.provenance.sources import (
    current_source_files,
    production_source_modules,
    source_paths,
    verify_current_source_hashes,
)


def hashes(files):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def test_identity_covers_every_actual_package_file_without_local_materials():
    files = current_source_files(REPOSITORY_ROOT)
    actual = set((REPOSITORY_ROOT / "src/zhenmode").rglob("*.py"))
    registered = set(source_paths(REPOSITORY_ROOT / "src", production_source_modules()).values())
    assert actual == registered
    assert actual <= set(files.values())
    assert "tests/support/driver.py" in files
    assert "tests/support/grid.py" in files
    assert all(name == path.relative_to(REPOSITORY_ROOT).as_posix() for name, path in files.items())
    assert not any(name.startswith(("research/", "dashboard/", "docs/")) for name in files)


@pytest.mark.parametrize("producer", ["verify_production_restart_cpu", "verify_debug_integration"])
def test_producer_includes_itself_and_all_actual_implementations(producer):
    owner = importlib.import_module("scripts." + producer)
    files = current_source_files(REPOSITORY_ROOT, [Path(owner.__file__)])
    recorded = owner.source_hashes()
    assert recorded == hashes(files)
    assert f"scripts/{producer}.py" in recorded
    assert set((REPOSITORY_ROOT / "src/zhenmode").rglob("*.py")) <= set(files.values())


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "tests/missing.py", "src/config.py"])
def test_selected_sources_reject_escape_missing_files_and_retired_aliases(name):
    with pytest.raises(ValueError, match="outside-workspace|missing declared"):
        current_source_files(REPOSITORY_ROOT, [name])


def test_selected_absolute_path_object_must_be_inside_repository(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="outside-workspace"):
        current_source_files(REPOSITORY_ROOT, [outside])


def test_missing_implementation_fails_even_with_valid_other_files(tmp_path):
    shutil.copytree(REPOSITORY_ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "src/zhenmode/model/inputs/quality.py").unlink()
    with pytest.raises(ValueError, match="missing required source: zhenmode/model/inputs/quality"):
        current_source_files(tmp_path)


@pytest.mark.parametrize("corruption", [None, "bytes", "implementation", "helper", "invalid_hash"])
def test_manifest_requires_complete_real_hashes(corruption):
    manifest = hashes(current_source_files(REPOSITORY_ROOT))
    if corruption == "bytes":
        manifest["src/zhenmode/model/config.py"] = "0" * 64
    elif corruption == "implementation":
        manifest.pop("src/zhenmode/model/config.py")
    elif corruption == "helper":
        manifest.pop("tests/support/grid.py")
    elif corruption == "invalid_hash":
        manifest["src/zhenmode/model/config.py"] = "invalid"
    if corruption:
        with pytest.raises(ValueError, match="source mismatch|incomplete|invalid source hash"):
            verify_current_source_hashes(REPOSITORY_ROOT, manifest)
    else:
        verify_current_source_hashes(REPOSITORY_ROOT, manifest)


@pytest.mark.parametrize("module", ["model/runtime/run", "model/solver/dynamics/transport", "model/solver/timestepping/step", "model/solver/numerics/contacts"])
def test_changes_to_launcher_and_actual_operators_are_rejected(tmp_path, module):
    shutil.copytree(REPOSITORY_ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = hashes(current_source_files(tmp_path))
    path = tmp_path / f"src/zhenmode/{module}.py"
    path.write_bytes(path.read_bytes() + b"\n# independent tamper control\n")
    with pytest.raises(ValueError, match=f"runtime source mismatch: src/zhenmode/{module}"):
        verify_current_source_hashes(tmp_path, manifest)


def test_production_source_lookup_does_not_fall_back_to_local_research(tmp_path):
    local = tmp_path / "research/src/zhenmode_research/example.py"
    local.parent.mkdir(parents=True)
    local.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required source"):
        source_paths(tmp_path / "src", ["zhenmode_research/example"])
