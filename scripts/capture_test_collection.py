"""Capture or compare pytest collection through an explicit file migration map."""
import argparse
import hashlib
import json
from pathlib import Path


def compare_collections(reference, candidate, moves):
    """Require every original ID, preserving its full function/parameter suffix."""
    old = reference["nodeids"]
    new = candidate["nodeids"]
    if len(old) != len(set(old)) or len(new) != len(set(new)):
        raise ValueError("duplicate test node identifiers")
    mapped = []
    for node in old:
        filename, separator, suffix = node.partition("::")
        if not separator:
            raise ValueError("test node has no function suffix: " + node)
        if filename not in moves:
            raise ValueError("test file has no declared move: " + filename)
        mapped.append(moves[filename] + separator + suffix)
    if len(mapped) != len(set(mapped)):
        raise ValueError("test mapping collides")
    missing = sorted(set(mapped) - set(new))
    if missing:
        raise ValueError("missing original test nodes: " + repr(missing))
    return {
        "reference_nodes": len(old),
        "candidate_nodes": len(new),
        "preserved_nodes": len(mapped),
        "new_nodes": sorted(set(new) - set(mapped)),
        "preserved_mapping_sha256": hashlib.sha256("\n".join(sorted(mapped)).encode()).hexdigest(),
    }


class CollectionInventory:
    def __init__(self):
        self.nodeids = []

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("capture")
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("targets", nargs="*", default=["tests"])
    compare = actions.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--layout", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        import pytest

        inventory = CollectionInventory()
        code = pytest.main(["--collect-only", "-q", *args.targets], plugins=[inventory])
        if code != 0:
            raise SystemExit(code)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({"nodeids": inventory.nodeids, "pytest": pytest.__version__}, stream, indent=2)
        print(json.dumps({"collected_nodes": len(inventory.nodeids)}))
    else:
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        result = compare_collections(
            json.loads(args.reference.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
            layout["test_moves"],
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
