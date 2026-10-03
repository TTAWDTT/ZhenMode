"""Materialize verified historical modules in a new private directory, never checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(exist_ok=False)
    source = args.output / "src"
    source.mkdir()
    for name, expected in protocol["historical_source_hashes"].items():
        if Path(name).parts[0] != "src" or len(Path(name).parts) != 2:
            raise ValueError("Expected a single historical src module")
        content = subprocess.check_output(
            ["git", "-C", str(args.repository), "show", protocol["source_commit"] + ":" + name]
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Historical hash mismatch: " + name)
        (args.output / name).write_bytes(content)
    for patch in args.patch:
        subprocess.run(
            ["git", "apply", "--no-index", "--directory=src", "-p1", "-"],
            input=patch.read_bytes().replace(b"\r\n", b"\n"),
            cwd=args.output,
            check=True,
        )
    actual = hashlib.sha256((source / "material_top.py").read_bytes()).hexdigest()
    if actual != protocol["instrumented_material_sha256"]:
        raise ValueError("Instrumented module does not match protocol")
    print(source)


if __name__ == "__main__":
    main()
