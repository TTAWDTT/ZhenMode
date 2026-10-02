"""Capture full synthetic CLI payloads, retaining semantic checkpoint metadata.

Actual source identities and their dependent metadata checksums are recorded in
a sidecar. Only output-directory text is normalized while collecting records.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def assert_partner_records(payload):
    continuous = {
        name.removeprefix("continuous/") for name in payload if name.startswith("continuous/")
    }
    resumed = {name.removeprefix("resumed/") for name in payload if name.startswith("resumed/")}
    if continuous != resumed:
        raise ValueError("continuous and resumed record names differ")
    for name in continuous:
        left, right = payload["continuous/" + name], payload["resumed/" + name]
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes() != right.tobytes()
        ):
            raise ValueError("continuous and resumed record differs: " + name)


def collect_records(directory):
    import numpy as np

    payload = {}
    identities = {}
    for label in ("continuous", "resumed"):
        output = directory / label
        with np.load(output / "global_controlled.npz", allow_pickle=False) as saved:
            count = int(saved["n_3d_snaps"])
            assert count == 5 and int(saved["accepted_steps"]) == 8
            for name in saved.files:
                value = saved[name]
                if name == "source_identity_json":
                    identities[label] = json.loads(str(value))
                elif name in {"three_d_dir", "rejected_state_path"}:
                    payload[label + "/" + name] = np.asarray(
                        str(value).replace(str(output), "<output>")
                    )
                else:
                    payload[label + "/" + name] = value
        with np.load(output / "ckpt_controlled.npz", allow_pickle=False) as saved:
            encoded = str(saved["metadata_json"])
            checksum = str(saved["metadata_sha256"])
            assert hashlib.sha256(encoded.encode()).hexdigest() == checksum
            metadata = json.loads(encoded)
            identities[label]["checkpoint_contract_sources"] = metadata["contract"].pop("sources")
            identities[label]["checkpoint_metadata_sha256"] = checksum
            payload[label + "/checkpoint/metadata_semantics_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
            for name in saved.files:
                if name not in {"metadata_json", "metadata_sha256"}:
                    payload[label + "/checkpoint/" + name] = saved[name]
        for folder in ("global_controlled_3d", "global_controlled_terms"):
            files = sorted((output / folder).glob("*.npy"))
            assert len(files) == count, folder
            for path in files:
                payload[label + "/" + folder + "/" + path.name] = np.load(path, allow_pickle=False)
    assert_partner_records(payload)
    return payload, identities


def run_synthetic_driver(source_root, directory, wind_jit):
    import pytest

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(source_root.resolve()))
    from tests.support.driver import run_controlled_driver

    directory.mkdir(parents=True, exist_ok=False)
    options = ("--budget-audit",) + (("--wind-jit",) if wind_jit else ())
    with pytest.MonkeyPatch.context() as patch:
        run_controlled_driver(patch, directory / "continuous", save_3d=True, options=options)
    resumed = directory / "resumed"
    checkpoint = resumed / "ckpt_controlled.npz"
    for restart in (None, checkpoint):
        with pytest.MonkeyPatch.context() as patch:
            run_controlled_driver(
                patch, resumed, restart=restart, crash_after=3, save_3d=True, options=options
            )
    with pytest.MonkeyPatch.context() as patch:
        run_controlled_driver(patch, resumed, restart=checkpoint, save_3d=True, options=options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wind-jit", action="store_true")
    parser.add_argument(
        "--records-directory",
        type=Path,
        help="Collect existing validated run files without executing a trajectory again.",
    )
    args = parser.parse_args()
    import numpy as np

    directory = args.records_directory or args.output.with_suffix("")
    if args.records_directory is None:
        run_synthetic_driver(args.source_root, directory, args.wind_jit)
    payload, identities = collect_records(directory)
    np.savez_compressed(args.output, **payload)
    args.output.with_suffix(".sources.json").write_text(
        json.dumps(identities, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "payload_records": len(payload),
                "wind_jit": args.wind_jit,
                "accepted_steps": 8,
                "interruptions": 2,
                "continuous_restart_bytes_equal": True,
                "reused_run_files": args.records_directory is not None,
                "source_sidecar": "actual source identity, contract.sources, verified metadata_sha256",
                "metadata_payload": "all metadata fields except contract.sources retained",
                "normalization": "output directory text only; input provenance paths normalized separately by the comparator",
            }
        )
    )


if __name__ == "__main__":
    main()
