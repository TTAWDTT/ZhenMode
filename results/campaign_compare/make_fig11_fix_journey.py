# fig11: the fix journey — survival ladder 08-24 → 09-04.
# x = wall-clock completion date of each long run, y = days survived
# (symlog). Blue dashed verticals = decisive solver fixes. Story: every
# crash was diagnosed from terms_fn evidence and fixed; the survival
# ceiling climbed 6 d → 365 d in two weeks / 81 fix-feat commits.
# Image-blind: verified via font_audit + data assertions, never read back.
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))

def font_audit(fig, name):
    """Assert all text uses the serif/Times family (image-blind check)."""
    import matplotlib.font_manager as fm
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        f = t.get_fontfamily()
        if not f or f[0] not in ("serif", "Times New Roman"):
            bad.append((s[:30], f))
    if bad:
        raise AssertionError(f"{name}: non-serif text found: {bad[:5]}")
    print(f"[font-audit] {name}: OK (all serif/Times)")

plt.rcParams["font.family"] = "Times New Roman"

# (completion datetime, days survived, color, label, era)
RUNS = [
    ("2026-08-24 01:34",  90, "#2a7f62", "s1_90d\nrestore ON",  "regional"),
    ("2026-08-24 10:36",  15, "#c0504d", "smoke15",             "regional"),
    ("2026-08-24 13:06",   6, "#c0504d", "stream6",             "regional"),
    ("2026-08-24 19:24", 140, "#c0504d", "windblend200",        "regional"),
    ("2026-08-25 03:04", 340, "#c0504d", "sponge8",             "regional"),
    ("2026-08-25 07:31", 365, "#2a7f62", "sponge16\nPASS",      "regional"),
    ("2026-08-25 12:50", 120, "#c0504d", "norestore",           "regional"),
    ("2026-09-03 12:00", 365, "#c0504d", "fgate\nFAIL_DRIFT",   "global"),
    ("2026-09-03 14:00", 365, "#2a7f62", "glap\nrunner PASS",   "global"),
    ("2026-09-03 16:00", 365, "#2a7f62", "glap2\nA1/A2 PASS",   "global"),
    ("2026-09-04 10:37", 365, "#1f5fa8", "012\nPROD",           "global"),
]

# (fix commit datetime, label, label y)
FIXES = [
    ("2026-08-21 17:48", "biharmonic\nviscosity",        30),
    ("2026-08-22 01:47", "NCEP\nreal wind",              60),
    ("2026-08-29 12:05", "bulk flux\n(no-restore)",      90),
    ("2026-08-27 01:36", "polar cap\ntaper",            130),
    ("2026-08-31 17:39", "rho-PGF +\nmass sponge",      200),
    ("2026-09-03 01:18", "adv + Laplacian\nface gating", 260),
    ("2026-09-03 05:22", "ocean-only\nT_atm",            310),
    ("2026-09-04 07:07", "seasonal wind\n+ Med relax",   350),
]

fig, ax = plt.subplots(figsize=(15.5, 7.8), constrained_layout=True)

xs = [mdates.date2num(dt.datetime.strptime(r[0], "%Y-%m-%d %H:%M"))
      for r in RUNS]
ys = [r[1] for r in RUNS]

# era shading
ax.axvspan(xs[0] - 0.8, xs[6] + 0.8, color="#f2ecf8", zorder=0)
ax.axvspan(xs[6] + 0.8, xs[-1] + 0.8, color="#e8f0fa", zorder=0)
ax.annotate("regional spectral era (128^2 NW-Pacific)", (xs[0], 395),
            fontsize=10, color="#7a5fa0", fontweight="bold", va="top")
ax.annotate("global 1-deg FD era (360x120, real coastlines)",
            (xs[6] + 1.0, 395), fontsize=10, color="#1f5fa8",
            fontweight="bold", va="top")

# chronological connective tissue (two eras, not one line)
ax.plot(xs[:7], ys[:7], "-", color="0.55", lw=1.0, alpha=0.6, zorder=2)
ax.plot(xs[6:], ys[6:], "-", color="0.55", lw=1.0, alpha=0.6, zorder=2)

# fixes as verticals + labels
for d, lab, ytxt in FIXES:
    x = mdates.date2num(dt.datetime.strptime(d, "%Y-%m-%d %H:%M"))
    ax.axvline(x, color="#1f5fa8", lw=1.1, ls=(0, (3, 2)), alpha=0.55,
               zorder=1)
    ax.annotate(lab, (x, ytxt), xytext=(5, 0), textcoords="offset points",
                fontsize=8.4, color="#1f5fa8", fontweight="bold", va="top")

# runs as markers + labels
for (d, days, c, lab, era), x in zip(RUNS, xs):
    ax.plot(x, days, "o", ms=11 if era == "global" else 9, color=c,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(lab, (x, days), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=8, color="0.2")

ax.set_yscale("symlog", linthresh=50)
ax.set_yticks([6, 15, 50, 100, 200, 365])
ax.set_ylim(0, 430)
ax.set_yticklabels(["6", "15", "50", "100", "200", "365"])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylabel("days survived (symlog)")
ax.set_xlim(xs[0] - 1.0, xs[-1] + 1.0)
ax.set_title("The fix journey: 81 fix/feature commits, survival ceiling 6 d → 365 d",
             fontsize=13.5, fontweight="bold", pad=10)
ax.grid(alpha=0.25)

# data assertions (image-blind verification)
assert [r[1] for r in RUNS].index(6) == 2, "stream6 must be the floor"
assert max(r[1] for r in RUNS) == 365
assert len(FIXES) == 8

font_audit(fig, "fig11")
out = os.path.join(HERE, "fig11_fix_journey.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print("fig11 done: survival ladder", len(RUNS), "runs,", len(FIXES), "fixes")
