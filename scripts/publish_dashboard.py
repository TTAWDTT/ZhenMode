# -*- coding: utf-8 -*-
"""Publish a static dashboard snapshot to TTAWDTT.github.io/ocean_solver/.

Pulls live status + analysis series + log tail from the cluster (same gateway
PTY approach as dashboard/cluster.py), assembles one self-contained
data.json, and commits+pushes it into the github.io repo under public/
ocean_solver/ (Next.js copies public/ verbatim into the Pages artifact).

Run manually or from a Windows scheduled task for push-mode observability:
  python scripts/publish_dashboard.py --once        # single snapshot
  python scripts/publish_dashboard.py               # loop, 30-min cadence
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "dashboard"))

import numpy as np  # noqa: E402
import paramiko  # noqa: E402  (same env as the dashboard bridge)

IO_REPO = r"C:\Users\zhen.luo\.research\oio.io"   # TTAWDTT.github.io clone
PUB_DIR = os.path.join(IO_REPO, "public", "ocean_solver")
SECRET = json.load(io.open(os.path.join(ROOT, "dashboard", "secret.json"),
                           encoding="utf-8"))

_log = lambda m: print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def drain(sh, timeout=4.0):
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        if sh.recv_ready():
            buf += sh.recv(65536)
            end = time.time() + 1.0
        else:
            time.sleep(0.15)
    return buf


def to_ascii(s):
    return re.sub(r"[^\x20-\x7e\r\n]", ".", s.decode("utf-8", errors="replace"))


def connect_node(idx):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(SECRET["host"], port=SECRET["port"], username=SECRET["user"],
                password=SECRET["password"], look_for_keys=False,
                allow_agent=False, timeout=40, banner_timeout=40,
                auth_timeout=40)
    sh = cli.invoke_shell(width=220, height=50)
    sh.settimeout(10)
    ok = False
    for _ in range(int(90 / 4)):
        d = drain(sh, timeout=4)
        if b"Quit" in d:
            ok = True
            break
        time.sleep(0.3)
    if not ok:
        raise RuntimeError("gateway menu never appeared")
    time.sleep(1.0)
    _ = drain(sh, timeout=2)
    sh.send(":")
    time.sleep(0.5)
    for ch in idx:
        sh.send(ch)
        time.sleep(0.3)
    sh.send("\n")
    time.sleep(1.5)
    sh.send("\n")
    ok = False
    for _ in range(15):
        d = drain(sh, timeout=2)
        if b"root@" in d:
            ok = True
            break
        time.sleep(0.5)
    if not ok:
        raise RuntimeError("no root shell after node select")
    _ = drain(sh, timeout=3)
    sh.send("stty -echo\n")
    time.sleep(2)
    _ = drain(sh, timeout=4)
    return cli, sh


def run(sh, cmd, wait=10):
    sh.send(cmd + "\n")
    return to_ascii(drain(sh, timeout=wait))


def strip_ansi(s):
    # eslint-disable-next-line no-control-regex
    return re.sub(r"\x1b\[[0-9;]*m", "", s).replace("\x1b[0m", "")


def status_for(entry, sh):
    """Parse the runner log for verdict / latest progress row / liveness."""
    log = entry.get("log", "")
    q = (f"grep VERDICT {log} 2>/dev/null | tail -1; "
         f"grep -E '^\\s*[0-9]+\\.[0-9]' {log} | tail -1; "
         f"pgrep -fc 'tag {entry['tag']}' || echo 0")
    out = strip_ansi(run(sh, q, 12))
    lines = [l.strip() for l in out.split("\n")
             if l.strip() and not l.endswith("~]") and "root@" not in l]
    verdict, row, alive = None, "", False
    for l in lines:
        if "VERDICT" in l:
            verdict = "PASS" if "PASS" in l else "FAIL"
        elif re.match(r"^\d+(\.\d+)?\s+\d+\s", l):
            row = l
        elif re.match(r"^\d+$", l):
            alive = int(l) > 0
    f = row.split()
    warn = []
    if f and len(f) > 2 and f[2] and float(f[2]) > 5:
        warn.append(f"max|u|={f[2]}")
    if f and len(f) > 4 and f[4] and float(f[4]) > 10:
        warn.append(f"max|eta|={f[4]}m")
    return {
        "tag": entry["tag"], "node": entry.get("node"),
        "gpu": entry.get("gpu"), "total_days": entry.get("total_days"),
        "started": entry.get("started"), "note": entry.get("note", ""),
        "alive": alive, "verdict": verdict,
        "day": float(f[0]) if f and f[0] else None,
        "max_u": float(f[2]) if len(f) > 2 and f[2] else None,
        "max_T": float(f[3]) if len(f) > 3 and f[3] else None,
        "max_eta": float(f[4]) if len(f) > 4 and f[4] else None,
        "nan": int(f[7]) if len(f) > 7 and f[7].isdigit() else None,
        "warn": warn,
    }


def curves_for(tag, sh):
    """Fetch the analysis npz and keep all 1-D non-placeholder series."""
    b64raw = run(sh, f"base64 -w0 /data/tmp/ocean/results/{tag}_analysis.npz",
                 40)
    m = re.search(r"[A-Za-z0-9+/=]{100,}", b64raw)
    if not m:
        raise RuntimeError("no base64 payload")
    d = np.load(io.BytesIO(base64.b64decode(m[0])))
    out = {}
    for k in d.files:
        if d[k].ndim != 1 or d[k].size > 5000:
            continue
        vals = d[k].tolist()
        # drop placeholder series (all-zero: sshstd/maxeta before the run
        # completes) but never crit_* flags — a FAIL flag is exactly 0.0 —
        # and map NaN to null (illegal in strict JSON)
        if not k.startswith("crit_") and vals and all(v == 0 for v in vals):
            continue
        out[k] = [None if (isinstance(v, float) and v != v) else round(v, 6)
                  for v in vals]
    out["ts"] = time.time() * 1000
    return out


def build_snapshot(sh, entries):
    runs, curves, logs = [], {}, {}
    for entry in entries:
        tag = entry["tag"]
        try:
            runs.append(status_for(entry, sh))
        except Exception as e:
            _log(f"status {tag}: {e}")
            runs.append({"tag": tag, "error": str(e)[:80]})
        try:
            curves[tag] = curves_for(tag, sh)
        except Exception as e:
            _log(f"curves {tag}: {e}")
        try:
            logs[tag] = run(sh, f"tail -c 3000 {entry.get('log','')}", 10)
        except Exception as e:
            _log(f"log {tag}: {e}")
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "runs": runs, "curves": curves, "logs": logs,
    }


def publish(snap):
    os.makedirs(PUB_DIR, exist_ok=True)
    with io.open(os.path.join(PUB_DIR, "data.json"), "w",
                 encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, separators=(",", ":"))
    # index.html comes from the ocean_solver repo (single source of truth)
    src = os.path.join(ROOT, "dashboard", "public", "index.html")
    dst = os.path.join(PUB_DIR, "index.html")
    if not (os.path.exists(dst) and
            open(src, "rb").read() == open(dst, "rb").read()):
        import shutil
        shutil.copyfile(src, dst)
        _log("index.html copied")
    return subprocess.run(["git", "status", "--porcelain", "public/ocean_solver"],
                          cwd=IO_REPO, capture_output=True, text=True).stdout


def git_commit_push(snap, status_out):
    if not status_out.strip():
        return False
    msg = ("ocean: auto snapshot " + snap["generated"][:16] + " — " +
           ", ".join(f"{r['tag']} {('%.0f%%' % (100 * (r.get('day') or 0) / r['total_days'])) if r.get('total_days') else '?'}"
                     for r in snap["runs"]))
    subprocess.run(["git", "add", "public/ocean_solver"], cwd=IO_REPO,
                   check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=IO_REPO, check=True)
    r = subprocess.run(["git", "push", "origin", "main"], cwd=IO_REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        _log("push failed: " + (r.stderr or r.stdout)[-300:])
        return False
    _log("pushed: " + msg)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="one snapshot and exit")
    ap.add_argument("--interval", type=int, default=1800,
                    help="loop cadence in seconds (default 1800)")
    ap.add_argument("--no-push", action="store_true",
                    help="build data.json locally but skip git push")
    args = ap.parse_args()
    while True:
        try:
            cli, sh = connect_node(SECRET["node"])
            entries = json.load(io.open(
                os.path.join(ROOT, "dashboard", "runs.json"),
                encoding="utf-8"))["runs"]
            snap = build_snapshot(sh, entries)
            cli.close()
            status_out = publish(snap)
            if args.no_push:
                _log("no-push: snapshot written locally")
            elif status_out.strip():
                git_commit_push(snap, status_out)
            else:
                _log("no changes")
        except Exception as e:
            _log(f"publish failed: {e!r}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
