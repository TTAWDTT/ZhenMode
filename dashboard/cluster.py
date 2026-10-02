# -*- coding: utf-8 -*-
"""Cluster bridge for the dashboard: JSON-lines protocol over stdio.

server.js spawns this once per node. Protocol (one JSON per line):
  -> {"cmd": "shell command", "id": N}
  <- {"id": N, "ok": true, "out": "..."}  or  {"id": N, "ok": false, "err": "..."}

Keeps one persistent gateway PTY shell per node (connect ~5-10 s, then
~1 s/command), serializing commands under a lock. Reconnects on failure.

Secrets come from dashboard/secret.json (gitignored).
"""
import io
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paramiko  # from the same Python env as ~/.research helper
from channel_io import drain as drain

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = json.load(io.open(os.path.join(HERE, "secret.json"), encoding="utf-8"))

sys.stdout.reconfigure(encoding="utf-8")

_lock = threading.Lock()
_cli = None
_sh = None


def to_ascii(s):
    import re
    return re.sub(r"[^\x20-\x7e\r\n]", "", s.decode("utf-8", errors="replace"))


def connect(node):
    """Open gateway PTY, select node, return shell (same protocol as
    ~/.research/_gs_yd_session.py)."""
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(SECRET["host"], port=SECRET["port"], username=SECRET["user"],
                password=SECRET["password"], look_for_keys=False,
                allow_agent=False, timeout=40, banner_timeout=40,
                auth_timeout=40)
    sh = cli.invoke_shell(width=220, height=50)
    sh.settimeout(10)
    ok = False
    for _ in range(22):
        d = drain(sh, timeout=4)
        if b"Quit" in d:
            ok = True
            break
        time.sleep(0.3)
    if not ok:
        cli.close()
        raise RuntimeError("gateway menu never appeared")
    time.sleep(1.0)
    drain(sh, timeout=2)
    sh.send(":")
    time.sleep(0.5)
    for ch in node:
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
        cli.close()
        raise RuntimeError("no root shell after node select")
    drain(sh, timeout=3)
    sh.send("stty -echo\n")
    time.sleep(2)
    drain(sh, timeout=4)
    return cli, sh


def run_cmd(cmd, wait=10, marker=None):
    global _cli, _sh
    with _lock:
        for attempt in (1, 2):
            try:
                if _sh is None:
                    _cli, _sh = connect(SECRET["node"])
                if marker is None:
                    _sh.send(cmd + "\n")
                    out = drain(_sh, timeout=wait)
                    return to_ascii(out)
                # marker mode: append a unique echo so we can slice the
                # output between markers (strips command echo + prompt)
                _sh.send(f"{cmd}; echo {marker}\n")
                out = drain(_sh, timeout=wait)
                text = to_ascii(out)
                parts = text.split(marker)
                return parts[1] if len(parts) > 1 else text
            except Exception as e:
                err = f"{e}"
                try:
                    if _cli:
                        _cli.close()
                except Exception:
                    pass
                _cli = _sh = None
                if attempt == 2:
                    raise RuntimeError(f"cluster cmd failed: {err}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        try:
            out = run_cmd(req["cmd"], wait=req.get("wait", 10),
                          marker=req.get("marker"))
            resp = {"id": rid, "ok": True, "out": out}
        except Exception as e:
            resp = {"id": rid, "ok": False, "err": str(e)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
