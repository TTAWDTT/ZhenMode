# -*- coding: utf-8 -*-
"""Background watcher for the spinA300 run on node 014 (24 h horizon).

Same protocol as _probe_watch.py: 10-min polls, heartbeat line per round,
exit-on-VERDICT (wake session for final analysis) or death-without-verdict.
"""
import sys
import time

sys.path.insert(0, r"C:\Users\zhen.luo\.research")
import _gs_yd_session as gs

LOG = "/data/tmp/ocean/logs/spinA300.log"
DEADLINE = time.time() + 26 * 3600


def poll():
    cli, sh = gs.connect_node("014")
    out = gs.run(
        sh,
        f"grep -c VERDICT {LOG} 2>/dev/null; "
        f"grep -E '^\\s+[0-9]+\\.[0-9]' {LOG} | tail -1; "
        f"pgrep -c -f 'tag spinA300' || echo 0",
        wait=12,
    )
    out += gs.to_ascii(gs.drain(sh, 4))
    cli.close()
    return out


while time.time() < DEADLINE:
    try:
        out = poll()
        lines = [l for l in out.splitlines() if l.strip() and "Quit" not in l]
        n_verdict = lines[0].strip() if lines else "?"
        last_row = lines[1].strip() if len(lines) > 1 else "?"
        n_proc = lines[-1].strip() if lines else "?"
        print(f"verdict={n_verdict} procs={n_proc} | {last_row}", flush=True)
        if n_verdict not in ("0", "?"):
            print("SPINA300_FINISHED", flush=True)
            sys.exit(0)
        if n_proc == "0":
            print("SPINA300_DEAD_WITHOUT_VERDICT", flush=True)
            sys.exit(2)
    except Exception as e:
        print(f"poll error: {e}", flush=True)
    time.sleep(600)

print("WATCH_TIMEOUT_26H", flush=True)
