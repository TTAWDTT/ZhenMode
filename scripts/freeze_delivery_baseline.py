"""Freeze source/input identities without fetching data or integrating the model."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def describe(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("baseline destination already exists")
    sources = sorted({*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py"),
                      ROOT / "pyproject.toml", Path(__file__).resolve(),
                      ROOT / "docs/production_delivery_protocol.json",
                      ROOT / "docs/production_delivery_plan_zh.md",
                      ROOT / "scripts/verify_debug_integration.py",
                      ROOT / "research/reviews/test_driver_contract_review.py"})
    input_dirs = [ROOT / 'data']
    if os.environ.get('OCEAN_SOLVER_WOA_DIR'):
        input_dirs.append(Path(os.environ['OCEAN_SOLVER_WOA_DIR']).resolve())
    inputs = sorted({path.resolve() for directory in input_dirs
                     for pattern in ('*.nc', '*.npz') for path in directory.rglob(pattern)})
    record = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "jax": jax.__version__,
                        "numpy": np.__version__, "devices": [str(device) for device in jax.devices()]},
        "sources": {str(path.relative_to(ROOT)): describe(path) for path in sources},
        "inputs": {str(path): describe(path) for path in inputs},
        "qualification": "identity_freeze_only_no_integration",
        "data_gaps": ["input identities alone do not prove selected source, valid contents or remapping",
                      *[f'input directory missing: {directory}' for directory in input_dirs if not directory.is_dir()],
                      "historical century command/inputs remain unverified"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(f"FROZEN {record['commit']} ({len(sources)} sources, {len(inputs)} inputs): {args.out}")


if __name__ == "__main__":
    main()
