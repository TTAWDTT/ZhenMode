"""Create a reproducibility manifest for a standardized benchmark run."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from ocean_solver.evaluation.metrics import score_npz


def _load_config(npz_path: str | Path) -> dict:
    """Read the saved solver config without requiring a live run."""
    z = np.load(npz_path, allow_pickle=True)
    if "config" not in z:
        return {}
    raw = z["config"]
    text = str(raw.item() if raw.shape == () else raw)
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {"raw": text}
    except (SyntaxError, ValueError):
        return {"raw": text}


def make_manifest(npz_path: str | Path,
                  metrics_path: str | Path | None = None,
                  commit: str | None = None,
                  model: str = "ocean_solver",
                  config_json: str | Path | None = None) -> dict:
    """Build one portable manifest for benchmark reporting."""
    npz_path = Path(npz_path)
    if metrics_path is not None:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    else:
        metrics = score_npz(npz_path)
    if config_json is not None:
        config = json.loads(Path(config_json).read_text(encoding="utf-8"))
    else:
        config = _load_config(npz_path)
    return {
        "model": model,
        "npz": str(npz_path),
        "commit": commit,
        "config": config,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reproducible benchmark manifest.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--metrics", default=None,
                        help="benchmark JSON; if omitted, score the NPZ")
    parser.add_argument("--commit", default=None,
                        help="git commit used for the run")
    parser.add_argument("--model", default="ocean_solver",
                        help="model label for external or internal comparisons")
    parser.add_argument("--config-json", default=None,
                        help="JSON config for an external model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = make_manifest(args.npz, args.metrics, args.commit,
                             model=args.model, config_json=args.config_json)
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))




if __name__ == "__main__":
    main()
