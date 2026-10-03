"""Exclude retired modules left in setuptools' generated staging directory."""
from pathlib import Path

from setuptools.command.build_py import build_py

BUILD_STAGING_ROOT = Path(__file__).resolve().parents[1] / "build"


class CleanBuildPy(build_py):
    def run(self):
        staging = Path(self.build_lib).resolve()
        for source in (self.package_dir or {}).values():
            source_root = Path(source).resolve()
            if staging.is_relative_to(source_root) or source_root.is_relative_to(staging):
                raise RuntimeError("build staging overlaps source: " + str(staging))
        managed_root = BUILD_STAGING_ROOT.resolve()
        if staging == managed_root or not staging.is_relative_to(managed_root):
            raise RuntimeError("build staging must be inside the managed build directory: " + str(staging))
        super().run()
        if self.editable_mode:
            return
        expected = {Path(path).resolve() for path in self.get_outputs(include_bytecode=False)}
        # Only generated Python payloads in this command's staging tree are touched.
        # Source, research, data and other cached artifacts are never traversed.
        for path in staging.rglob("*.py"):
            resolved = path.resolve()
            if not resolved.is_relative_to(staging):
                raise RuntimeError("build staging path escapes its root: " + str(path))
            if resolved not in expected:
                self.announce("excluding stale staged module: " + str(path), level=2)
                path.unlink()
