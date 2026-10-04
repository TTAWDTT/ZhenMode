"""A cached pre-migration build must not reintroduce retired Python modules."""
from pathlib import Path

import pytest
from setuptools import Distribution

from scripts import build_distribution
from scripts.build_distribution import CleanBuildPy


def test_cached_build_excludes_retired_modules_and_keeps_non_python_cache(tmp_path, monkeypatch):
    source = tmp_path / "src"
    package = source / "zhenmode"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (source / "kept.py").write_text("VALUE = 2\n")
    monkeypatch.setattr(build_distribution, "BUILD_STAGING_ROOT", tmp_path / "build")
    staging = tmp_path / "build" / "lib"
    stale_package = staging / "zhenmode" / "candidates"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text("retired = True\n")
    (staging / "material_top.py").write_text("retired = True\n")
    cache = staging / "unrelated.cache"
    cache.write_bytes(b"retain this cache")
    distribution = Distribution({"packages": ["zhenmode"], "py_modules": ["kept"],
                                 "package_dir": {"": str(source)}})
    distribution.script_name = str(tmp_path / "setup.py")
    command = CleanBuildPy(distribution)
    command.ensure_finalized()
    command.build_lib = str(staging)
    command.run()
    assert {p.relative_to(staging).as_posix() for p in staging.rglob("*.py")} == {
        "zhenmode/__init__.py", "kept.py"}
    assert (staging / "kept.py").read_bytes() == (source / "kept.py").read_bytes()
    assert (package / "__init__.py").read_text() == "VALUE = 1\n"
    assert cache.read_bytes() == b"retain this cache"
    assert all(p.is_relative_to(tmp_path) for p in map(Path, command.get_outputs(False)))


@pytest.mark.parametrize("destination", ("", "research/src", "archive/regional", "tests"))
def test_build_staging_cannot_overlap_source(tmp_path, monkeypatch, destination):
    monkeypatch.setattr(build_distribution, "BUILD_STAGING_ROOT", tmp_path / "build")
    package = tmp_path / "src" / "zhenmode"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_bytes(b"retain original source\n")
    distribution = Distribution({"packages": ["zhenmode"],
                                 "package_dir": {"": str(tmp_path / "src")}})
    command = CleanBuildPy(distribution)
    command.ensure_finalized()
    command.build_lib = str(tmp_path / destination)
    with pytest.raises(RuntimeError, match="overlaps source|managed build directory"):
        command.run()
    assert source.read_bytes() == b"retain original source\n"
