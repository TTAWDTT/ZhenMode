# -*- coding: utf-8 -*-
"""Publish a static dashboard snapshot to TTAWDTT.github.io/ocean_solver/.

!! MACHINE-SPECIFIC / NOT PORTABLE !!
Needs three things that exist only on the author's machine:
  - ``paramiko`` (not a project dependency; install it yourself),
  - ``dashboard/secret.json`` (gitignored) holding the gateway host/user/
    password for the private GPU cluster,
  - a local clone of the TTAWDTT.github.io repo, at $OCEAN_SOLVER_IO_REPO
    (defaults to the historical path).
Without them this script cannot run. The model does not depend on it.

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
from channel_io import drain as drain  # noqa: E402

try:
    import paramiko  # noqa: E402  (private-gateway dependency, not a project dep)
except ImportError as _e:  # pragma: no cover - machine-specific
    raise SystemExit(
        "publish_dashboard.py needs paramiko (private-gateway bridge): "
        "pip install paramiko  [%s]" % _e)

# Local clone of the TTAWDTT.github.io repo. Override with
# OCEAN_SOLVER_IO_REPO; the default is the historical author-machine path.
IO_REPO = os.environ.get("OCEAN_SOLVER_IO_REPO",
                         r"C:\Users\zhen.luo\.research\oio.io")
PUB_DIR = os.path.join(IO_REPO, "public", "ocean_solver")

SECRET_PATH = os.path.join(ROOT, "dashboard", "secret.json")


def _load_secret():
    """Private gateway credentials. Empty dict when the (gitignored) file is
    absent, so the module still imports on a machine that has none."""
    if not os.path.exists(SECRET_PATH):
        return {}
    return json.load(io.open(SECRET_PATH, encoding="utf-8"))


SECRET = _load_secret()

def _log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def to_ascii(s):
    return re.sub(r"[^\x20-\x7e\r\n]", ".", s.decode("utf-8", errors="replace"))


def _expected_host(idx):
    # gateway menu index N maps to host k8s-sh-azn-gpu-{N-2:03d}
    # (012→gpu-010, 014→gpu-012, 015→gpu-013)
    return "k8s-sh-azn-gpu-%03d" % (int(idx) - 2)


def connect_node(idx):
    """Open the gateway PTY on node idx, verifying the landed hostname.

    The gateway menu can race and swallow the selection, in which case the
    shell silently lands on a different (default) node — that once published
    a snapshot with every run on the wrong host shown as 未运行.  Retry the
    whole handshake until the hostname matches.
    """
    last = None
    for attempt in range(3):
        cli = None
        try:
            cli, sh = _connect_node_once(idx)
            # verify through the PTY — exec_command would run on the gateway,
            # not on the node the menu dropped us into
            host = run(sh, "hostname", 8)
            if _expected_host(idx) in host:
                return cli, sh
            last = RuntimeError(
                f"landed on {host.strip()!r}, expected {_expected_host(idx)!r}")
            _log(f"connect {idx}: {last}; retry {attempt + 1}")
        except Exception as exc:
            last = exc
            _log(f"connect {idx}: {exc!r}; retry {attempt + 1}")
        if cli is not None:
            try:
                cli.close()
            except Exception:
                pass
        time.sleep(5)
    raise RuntimeError(f"connect {idx} failed after 3 attempts: {last!r}")


def _connect_node_once(idx):
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
    lines = [ln.strip() for ln in out.split("\n")
             if ln.strip() and not ln.endswith("~]") and "root@" not in ln]
    verdict, row, alive = None, "", False
    for ln in lines:
        if "VERDICT" in ln:
            verdict = "PASS" if "PASS" in ln else "FAIL"
        elif re.match(r"^\d+(\.\d+)?\s+\d+\s", ln):
            row = ln
        elif re.match(r"^\d+$", ln):
            alive = int(ln) > 0
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
        # 20000 cap: psi_flat (nlat*nz = 120*45 = 5400) must survive the
        # filter for the frontend AMOC ψ(y,z) heatmap
        if d[k].ndim != 1 or d[k].size > 20000:
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


def refresh_drift_curves(sh, tags):
    """Rebuild <tag>_analysis.npz from the drift CSV for every tag.

    Runs launched without --save-3d never produce 3-D snaps, so the full
    analysis (which needs them) is impossible and their charts would stay
    blank forever. dashboard/curves_from_drift.py rebuilds the 1-D payload
    from the checkpoint drift log instead; re-running it each cycle is what
    keeps the in-flight runs' curves advancing. Cheap (~2 s for 12 tags) and
    harmless for --save-3d runs, which keep their own richer npz when the
    analysis has already written one newer than the CSV.
    """
    gen = ("/data/tmp/ocean/dashboard/curves_from_drift.py")
    cmd = ("test -f %s && docker exec -w /data/tmp/ocean jaxtest2 "
           "/opt/conda/envs/py/bin/python %s %s || true"
           % (gen, gen, " ".join(tags)))
    try:
        out = strip_ansi(run(sh, cmd, 90))
        for line in out.split("\n"):
            if ":" in line and ("pts ->" in line or "no data" in line
                                or "FAILED" in line):
                _log("drift " + line.strip()[:110])
    except Exception as e:
        _log(f"drift curves: {e}")


def build_snapshot(sh, entries):
    runs, curves, logs = [], {}, {}
    refresh_drift_curves(sh, [e["tag"] for e in entries])
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
    return runs, curves, logs


def _denan(obj):
    """Map NaN/Inf floats to None so data.json stays strict JSON.

    Python's json module happily emits bare NaN — one blown-up run (e.g.
    max_u=NaN on a FAILED spinB) then makes the whole frontend unparseable.
    allow_nan=False below trips instead of ever publishing bare NaN again.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) \
            else obj
    if isinstance(obj, dict):
        return {k: _denan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_denan(v) for v in obj]
    return obj


def publish(snap):
    os.makedirs(PUB_DIR, exist_ok=True)
    with io.open(os.path.join(PUB_DIR, "data.json"), "w",
                 encoding="utf-8") as f:
        json.dump(_denan(snap), f, ensure_ascii=False, separators=(",", ":"),
                  allow_nan=False)
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
    # the repo has a stale http.proxy configured (127.0.0.1:9910, usually not
    # running) which silently turns every push into a connection failure and
    # lets the published snapshot fall hours behind. GitHub is reachable
    # directly here, so push with the proxy disabled.
    r = subprocess.run(["git", "-c", "http.proxy=", "-c", "https.proxy=",
                        "push", "origin", "main"], cwd=IO_REPO,
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
    if not SECRET:
        raise SystemExit(
            "dashboard/secret.json is missing -- it holds the private gateway "
            "credentials this publisher needs (see the module docstring). "
            "Nothing to publish without it.")
    if not os.path.isdir(IO_REPO):
        raise SystemExit(
            "github.io clone not found at %r; set OCEAN_SOLVER_IO_REPO"
            % IO_REPO)
    while True:
        try:
            entries = json.load(io.open(
                os.path.join(ROOT, "dashboard", "runs.json"),
                encoding="utf-8"))["runs"]
            # group entries by node and open one gateway PTY per node that
            # has at least one run (014, 012, ...)
            by_node = {}
            for ent in entries:
                by_node.setdefault(ent.get("node") or SECRET["node"],
                                   []).append(ent)
            runs, curves, logs = [], {}, {}
            for node, ents in by_node.items():
                try:
                    cli, sh = connect_node(node)
                except Exception as e:
                    _log(f"connect {node}: {e!r}")
                    for ent in ents:
                        runs.append({"tag": ent["tag"], "node": node,
                                     "error": f"connect: {e!r}"[:80]})
                    continue
                try:
                    r, c, lg = build_snapshot(sh, ents)
                    runs.extend(r)
                    curves.update(c)
                    logs.update(lg)
                finally:
                    cli.close()
            snap = {
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                "runs": runs, "curves": curves, "logs": logs,
            }
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
