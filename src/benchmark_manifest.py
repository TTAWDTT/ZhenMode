"""Create a reproducibility manifest for a standardized benchmark run."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from benchmark_metrics import score_npz


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


def make_manifest(npz_path: str | Path | None,
                  metrics_path: str | Path | None = None,
                  commit: str | None = None,
                  model: str = "ocean_solver",
                  config_json: str | Path | None = None,
                  run_id: str | None = None,
                  status: str | None = None) -> dict:
    """Build one portable manifest for benchmark reporting.

    For a pre-registered run, npz_path may be None; then the caller must
    supply config_json, run_id, and status.
    """
    if npz_path is None:
        if metrics_path is not None:
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        else:
            metrics = None
        if config_json is None:
            raise ValueError("pre-registered manifest requires --config-json")
        if run_id is None:
            raise ValueError("pre-registered manifest requires --run-id")
        config = json.loads(Path(config_json).read_text(encoding="utf-8"))
        npz_text = None
    else:
        npz_path = Path(npz_path)
        if metrics_path is not None:
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        else:
            metrics = score_npz(npz_path)
        if config_json is not None:
            config = json.loads(Path(config_json).read_text(encoding="utf-8"))
        else:
            config = _load_config(npz_path)
        if run_id is None:
            run_id = npz_path.stem
        npz_text = str(npz_path)

    if commit is None and isinstance(config, dict):
        commit = config.get('git_commit') or None
    if status is None:
        verdict = metrics.get("verdict") if isinstance(metrics, dict) else None
        status = "completed" if verdict == "PASS" else "not_comparable"
    return {
        "model": model,
        "run_id": run_id,
        "status": status,
        "npz": npz_text,
        "commit": commit,
        "config": config,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reproducible benchmark manifest.")
    parser.add_argument("--npz", default=None,
                        help="run NPZ; omit with --pre-registered")
    parser.add_argument("--metrics", default=None,
                        help="benchmark JSON; if omitted, score the NPZ")
    parser.add_argument("--commit", default=None,
                        help="git commit used for the run")
    parser.add_argument("--model", default="ocean_solver",
                        help="model label for external or internal comparisons")
    parser.add_argument("--config-json", default=None,
                        help="JSON config for an external model")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--pre-registered", action="store_true",
                        help="build a pre-run manifest without an NPZ")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.pre_registered:
        if args.npz is not None:
            raise SystemExit("--pre-registered must not be combined with --npz")
        if args.config_json is None or args.run_id is None or args.status is None:
            raise SystemExit("--pre-registered requires --config-json, --run-id and --status")
    else:
        if args.npz is None:
            raise SystemExit("--npz is required unless --pre-registered is used")
    manifest = make_manifest(args.npz, args.metrics, args.commit,
                             model=args.model, config_json=args.config_json,
                             run_id=args.run_id, status=args.status)
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
