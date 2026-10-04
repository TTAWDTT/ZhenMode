"""Prepare output paths and the production stdout log."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass


class _Tee:
    """Duplicate writes to the original stdout and a log file.

    Long runs are typically launched detached (nohup / taskset), where the
    console scrollback is lost; out_log preserves the progress table and the
    VERDICT block for later inspection.
    """

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@dataclass
class RunPaths:
    tag: str
    out_npz: str
    out_log: str
    three_d_dir: str | None
    three_d_terms_dir: str | None


def prepare_output_paths(args):
    tag = args.tag or f"g{int(args.days)}d"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"global_{tag}.npz")
    out_log = os.path.join(args.log_dir, f"global_{tag}.log")
    sys.stdout = _Tee(out_log)
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {sys.executable}")
    print(f"# {' '.join(sys.argv)}")
    three_d_dir = None
    if args.save_3d:
        three_d_dir = os.path.join(args.out_dir, f"global_{tag}_3d")
        os.makedirs(three_d_dir, exist_ok=True)
    three_d_terms_dir = None
    if args.save_3d_terms:
        assert args.save_3d, "--save-3d-terms requires --save-3d"
        three_d_terms_dir = os.path.join(args.out_dir, f"global_{tag}_terms")
        os.makedirs(three_d_terms_dir, exist_ok=True)
    return RunPaths(tag, out_npz, out_log, three_d_dir, three_d_terms_dir)
