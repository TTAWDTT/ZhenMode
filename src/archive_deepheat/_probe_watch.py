# -*- coding: utf-8 -*-
"""Visible background watcher for the spinA30probe run on node 014.

Polls the cluster gateway every 5 min; prints a heartbeat line each round
so the task output tail in the app's background-task panel shows live
progress. Exits (and wakes the session) when the run finishes (VERDICT in
log) or dies without finishing.
"""
import sys
import time

sys.path.insert(0, r"C:\Users\zhen.luo\.research")
import _gs_yd_session as gs

LOG = "/data/tmp/ocean/logs/spinA30probe.log"
DEADLINE = time.time() + 4 * 3600


def poll():
    cli, sh = gs.connect_node("014")
    out = gs.run(
        sh,
        f"grep -c VERDICT {LOG} 2>/dev/null; "
        f"grep -E '^\\s+[0-9]+\\.[0-9]' {LOG} | tail -1; "
        f"pgrep -c -f run_long_integration || echo 0",
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
        print(f"verdict_lines={n_verdict} procs={n_proc} | {last_row}",
              flush=True)
        if n_verdict not in ("0", "?"):
            print("PROBE_FINISHED", flush=True)
            sys.exit(0)
        if n_proc == "0":
            print("PROBE_DEAD_WITHOUT_VERDICT", flush=True)
            sys.exit(2)
    except Exception as e:  # gateway hiccup — keep polling
        print(f"poll error: {e}", flush=True)
    time.sleep(300)

print("WATCH_TIMEOUT", flush=True)
