# -*- coding: utf-8 -*-
"""Cluster run status board -- one command to see every long-running job.

!! MACHINE-SPECIFIC / NOT PORTABLE !!
This script queries a private GPU cluster through a private gateway helper
(``_gs_yd_session``). It is NOT part of the model and cannot run on a machine
without that helper: every cluster query fails there. ``--offline`` renders the
board from the registry alone and does work anywhere.

The registry is ``dashboard/runs.json`` -- the SAME file the web dashboard
(dashboard/server.js) reads, so the two views can never disagree. Add a run
there, not here.

Usage:
    python scripts/status_board.py             # query the cluster, rewrite status_zh.md
    python scripts/status_board.py --stdout    # print instead of writing
    python scripts/status_board.py --offline   # skip cluster queries (registry only)
"""
import io
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_MD = os.path.join(REPO, "status_zh.md")
REGISTRY = os.path.join(REPO, "dashboard", "runs.json")

HEALTH = {"max_u_warn": 5.0, "eta_warn": 10.0}   # red-flag thresholds

# Location of the private cluster gateway helper. Override with
# RESEARCH_HELPER_DIR if it lives elsewhere.
HELPER_DIR = os.environ.get(
    "RESEARCH_HELPER_DIR",
    os.path.join(os.path.expanduser("~"), ".research"))


def load_registry():
    """Read dashboard/runs.json and return its list of run dicts."""
    with io.open(REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _gateway():
    """Import the private cluster helper, with a readable failure."""
    sys.path.insert(0, HELPER_DIR)
    try:
        import _gs_yd_session as gs
    except ImportError as e:
        raise RuntimeError(
            "cluster helper _gs_yd_session not importable from %r (%s); "
            "re-run with --offline to render the registry without querying"
            % (HELPER_DIR, e))
    return gs


def parse_run(run, gs):
    """Return dict(tag, alive, verdict, last_row, warn, day) for one run."""
    tag, node, log = run["tag"], run["node"], run["log"]
    cli, sh = gs.connect_node(node)
    q = (f"grep VERDICT {log} 2>/dev/null | tail -1 | sed 's/\\x1b\\[[0-9;]*m//g'; "
         f"grep -E '^\\s*[0-9]+\\.[0-9]' {log} | tail -1; "
         f"pgrep -fc 'tag {tag}' || echo 0")
    out = gs.run(sh, q, wait=12)
    out += gs.to_ascii(gs.drain(sh, 4))
    cli.close()
    lines = [ln for ln in out.splitlines()
             if ln.strip() and "Quit" not in ln and not ln.endswith("~]")]
    verdict_line, last_row, n_proc = None, "", "0"
    if lines:
        verdict_line = lines[0].strip()
        for ln in lines[1:]:
            t = ln.strip()
            if t and t[0].isdigit() and "." in t.split()[0]:
                last_row = t
            elif t.isdigit():
                n_proc = t
                break
    d = dict(run)
    d.update({"verdict": None, "last_row": last_row,
              "alive": n_proc not in (0, "0"), "warn": []})
    if verdict_line and "VERDICT" in verdict_line:
        d["verdict"] = "PASS" if "PASS" in verdict_line else "FAIL"
    # health flags
    warn = []
    try:
        f = last_row.split()
        maxu, maxeta, nan = float(f[2]), float(f[4]), int(f[7])
        if maxu > HEALTH["max_u_warn"]:
            warn.append(f"max|u|={maxu:.2f}")
        if maxeta > HEALTH["eta_warn"]:
            warn.append(f"max|eta|={maxeta:.2f}m")
        if nan > 0:
            warn.append(f"NaN={nan}")
        d["day"] = float(f[0])
    except (IndexError, ValueError):
        pass
    d["warn"] = warn
    return d


def collect(offline):
    """Query every registered run; --offline skips the cluster entirely."""
    runs = load_registry()
    if offline:
        return [dict(r, alive=None, verdict=None, last_row="", warn=[],
                     error="offline") for r in runs]
    gs = _gateway()
    out = []
    for r in runs:
        try:
            out.append(parse_run(r, gs))
        except Exception as e:
            out.append(dict(r, alive=None, verdict=None, last_row="",
                            warn=[f"query failed: {e}"], error=str(e)))
    return out


def render(runs):
    now = time.strftime("%Y-%m-%d %H:%M") + " (本地)"
    lines = [
        "# ocean-solver 运行看板（status_zh.md）",
        "",
        "> 由 `scripts/status_board.py` 从 `dashboard/runs.json` 生成。",
        "> 人工只改 `dashboard/runs.json`；本文件每次刷新整体重写。",
        "",
        f"**生成时间**: {now} · 刷新命令: `python scripts/status_board.py`",
        "",
        "---",
        "",
        "## 当前活动运行",
        "",
    ]
    active = [r for r in runs if r.get("alive")]
    if not active:
        lines.append("（无活动运行 / 未查询集群）")
    else:
        lines.append("| 运行 | 节点/GPU | day | max\\|u\\| | max\\|eta\\| | 状态 |")
        lines.append("|---|---|---|---|---|---|")
        for r in active:
            f = r["last_row"].split()
            day = f[0] if f else "?"
            maxu = f[2] if len(f) > 2 else "?"
            eta = f[4] if len(f) > 4 else "?"
            st = "⚠ " + "; ".join(r["warn"]) if r["warn"] else "正常"
            lines.append(f"| {r['tag']} | {r['node']}/{r['gpu']} | {day} | "
                         f"{maxu} | {eta} | {st} |")
    lines += ["", "## 注册表中的全部运行", ""]
    for r in runs:
        total = r.get("total_days")
        prog = f"{r.get('day', '?')}/{total}" if total else str(r.get("day", "?"))
        v = r.get("verdict") or "无VERDICT"
        note = r.get("note", "")
        lines.append(f"- **{r['tag']}**（{r['node']}/{r['gpu']}）day {prog} — {v}"
                     + (f" — {note}" if note else ""))
    lines += ["", "---", ""]
    return "\n".join(lines) + "\n"


def main():
    offline = "--offline" in sys.argv
    runs = collect(offline)
    text = render(runs)
    if "--stdout" in sys.argv:
        print(text)
    else:
        with io.open(STATUS_MD, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"written {STATUS_MD}")
    for r in runs:
        flag = " WARN" if r["warn"] else ""
        print(f"  {r['tag']}: alive={r.get('alive')} verdict={r.get('verdict')}{flag}")


if __name__ == "__main__":
    main()
