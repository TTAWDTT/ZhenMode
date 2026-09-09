# -*- coding: utf-8 -*-
"""Cluster run status board — one command to see every long-running job.

Reads the RUNS registry below, queries each node via the research gateway
helper (_gs_yd_session), and regenerates status_zh.md in the repo root.

Usage:
    python scripts/status_board.py            # refresh status_zh.md
    python scripts/status_board.py --stdout   # print to terminal only

Adding a run: append an entry to RUNS (tag, node, gpu, log path, started).
The board handles the rest (progress parse, verdict check, liveness).
"""
import io
import os
import re
import sys
import time

sys.path.insert(0, r"C:\Users\zhen.luo\.research")
import _gs_yd_session as gs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_MD = os.path.join(REPO, "status_zh.md")

# ── Run registry: edit this table only ─────────────────────────────────
# (tag, node, gpu, logfile, started_local)
RUNS = [
    ("spinA100", "014", 0, "/data/tmp/ocean/logs/spinA100.log",
     "2026-09-09 16:15"),
    ("spinA30probe", "014", 0, "/data/tmp/ocean/logs/spinA30probe.log",
     "2026-09-09 13:02"),
]

HEALTH = {"max_u_warn": 5.0, "eta_warn": 10.0}   # red-flag thresholds


def sh_quote(s):
    return s


def query_node(node, cmds):
    """Run a list of shell commands on `node`; return list of outputs."""
    cli, sh = gs.connect_node(node)
    outs = []
    for c in cmds:
        out = gs.run(sh, c, wait=12)
        out += gs.to_ascii(gs.drain(sh, 4))
        outs.append(out)
    cli.close()
    return outs


def parse_run(tag, node, gpu, log, started):
    """Return dict(tag, alive, verdict, last_row, warn) for one run."""
    cli, sh = gs.connect_node(node)
    q = (f"grep VERDICT {log} 2>/dev/null | tail -1 | sed 's/\\x1b\\[[0-9;]*m//g'; "
         f"grep -E '^\\s*[0-9]+\\.[0-9]' {log} | tail -1; "
         f"pgrep -fc 'tag {tag}' || echo 0")
    out = gs.run(sh, q, wait=12)
    out += gs.to_ascii(gs.drain(sh, 4))
    cli.close()
    lines = [l for l in out.splitlines()
             if l.strip() and "Quit" not in l and not l.endswith("~]")]
    verdict_line, last_row, n_proc = None, "", "0"
    if lines:
        verdict_line = lines[0].strip()
        for l in lines[1:]:
            t = l.strip()
            if t and t[0].isdigit() and "." in t.split()[0]:
                last_row = t
            elif t.isdigit():
                n_proc = t
                break
    d = {"tag": tag, "node": node, "gpu": gpu, "started": started,
         "verdict": None, "last_row": last_row, "alive": n_proc not in (0, "0")}
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


def main():
    runs = []
    for r in RUNS:
        tag, node, gpu, log, started = r
        try:
            runs.append(parse_run(tag, node, gpu, log, started))
        except Exception as e:
            runs.append({"tag": tag, "node": node, "gpu": gpu, "error": str(e),
                         "alive": None, "verdict": None, "last_row": "",
                         "warn": [f"query failed: {e}"]})

    now = time.strftime("%Y-%m-%d %H:%M") + " (本地)"
    lines = [
        "# ocean-solver 运行看板（status_zh.md）",
        "",
        "> 本文件由 `scripts/status_board.py` 自动生成，是所有正在进行的长期运行的唯一观测入口。",
        "> 人工只改 `scripts/status_board.py` 顶部的 RUNS 注册表；本文件每次刷新会整体重写。",
        "",
        f"**生成时间**: {now} · 刷新命令: `python scripts/status_board.py`",
        "",
        "---",
        "",
        "## 当前活动运行",
        "",
    ]
    active = [r for r in runs if r["alive"]]
    if not active:
        lines.append("（无活动运行）")
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
    lines += ["", "## 非活动/最近运行", ""]
    done = [r for r in runs if not r["alive"]]
    if done:
        for r in done:
            v = r.get("verdict") or "无VERDICT"
            lines.append(f"- **{r['tag']}**（{r['node']}/{r['gpu']}）— "
                         f"{v}，最后行: `{r['last_row']}`")
    else:
        lines.append("（无）")
    lines += ["", "---", "", "*人工备注区（刷新保留手写内容需加入 status_board.py 模板）*"]
    text = "\n".join(lines) + "\n"
    if "--stdout" in sys.argv:
        print(text)
    else:
        with io.open(STATUS_MD, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"written {STATUS_MD}")
    for r in runs:
        flag = " ⚠" if r["warn"] else ""
        print(f"  {r['tag']}: alive={r['alive']} verdict={r['verdict']}{flag}")


if __name__ == "__main__":
    main()
