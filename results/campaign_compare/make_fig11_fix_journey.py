# fig11: the fix journey — survival ladder 08-24 → 09-04.
# Runs plotted as evenly-spaced stations on a chronological track (x = run
# index, NOT wall-clock — three run clusters share days, calendar spacing
# collided labels). y = days survived (symlog). Fixes annotated in the
# gaps between the runs they enabled. Image-blind: font_audit + data
# assertions, never read back.
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

def font_audit(fig, name):
    """Assert all text uses the serif/Times family (image-blind check)."""
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

# (date, days survived, color, label, label placement)
# placement: (dx, dy) in points from the marker — hand-set to avoid any
# collision since several runs share y=365 or sit 2 days apart.
RUNS = [
    ("08-24",  90, "#2a7f62", "s1_90d  (90 d, PASS)",            (0, 12)),
    ("08-24",  15, "#c0504d", "smoke15  (d15 blow-up)",          (0, 12)),
    ("08-24",   6, "#c0504d", "stream6  (d6 gate)",              (18, -26)),
    ("08-24", 140, "#c0504d", "windblend200  (d140 blow-up)",    (-10, 12), "right"),
    ("08-25", 340, "#c0504d", "sponge8  (d340 blow-up)",         (-6, 12)),
    ("08-25", 365, "#2a7f62", "sponge16  (365 d PASS)",          (0, 12)),
    ("08-25", 120, "#c0504d", "norestore  (d120 blow-up)",       (10, -28)),
    ("09-03", 365, "#c0504d", "fgate  (365 d, FAIL_DRIFT)",      (14, -34), "left"),
    ("09-03", 365, "#2a7f62", "glap  (365 d, runner PASS)",      (-14, 12)),
    ("09-03", 365, "#2a7f62", "glap2  (365 d, A1/A2 PASS)",      (12, -34), "left"),
    ("09-04", 365, "#1f5fa8", "g365d_012  (365 d PROD)",         (0, 12)),
]

# fixes: drawn between run i and run i+1 (index i = after which run), with
# the label stacked in the gap — the gap owns the fix that enabled the next
# run. Gap index → (label, x-offset within gap).
# Timeline: smoke15/stream6 ← biharmonic; windblend ← NCEP wind;
# sponge8/norestore ← bulk flux (no-restore finding); sponge16 ← nothing new;
# fgate ← (global era opens: polar cap, rho-PGF+sponge earlier); glap ←
# face gating; glap2 ← ocean-only T_atm; 012 ← seasonal wind + Med relax.
GAPS = {
    0: (["biharmonic viscosity"], 0.0),
    1: (["NCEP real wind"], 0.0),
    3: (["bulk heat flux", "(no-restore finding)"], 0.0),
    6: (["global era opens:", "polar cap taper,", "rho-PGF + mass sponge"], 0.0),
    7: (["adv + Laplacian", "face gating"], 0.0),
    8: (["ocean-only T_atm"], 0.0),
    9: (["seasonal wind", "+ Med relax"], 0.0),
}
GAP_Y = {   # y position (days axis) and vertical anchor per gap
    0: 45, 1: 90, 3: 235, 6: 75, 7: 115, 8: 120, 9: 60,
}

fig, ax = plt.subplots(figsize=(15.5, 7.6), constrained_layout=True)

n = len(RUNS)
xs = np.arange(n)
ys = [r[1] for r in RUNS]

# era shading: regional = stations 0-6, global = 7-10
ax.axvspan(-0.6, 6.5, color="#f2ecf8", zorder=0)
ax.axvspan(6.5, n - 0.4, color="#e8f0fa", zorder=0)
ax.annotate("regional spectral era (128² NW-Pacific)", (0, 500),
            fontsize=10.5, color="#7a5fa0", fontweight="bold", va="top")
ax.annotate("global 1° FD era (360×120, real coastlines)", (7.6, 500),
            fontsize=10.5, color="#1f5fa8", fontweight="bold", va="top")

# chronological track
ax.plot(xs, ys, "-", color="0.55", lw=1.1, alpha=0.55, zorder=2)

# fixes in the gaps
for i, (labs, dxo) in GAPS.items():
    xg = xs[i] + 0.5 + dxo
    yg = GAP_Y[i]
    ax.annotate("", xy=(xg, yg - 18), xytext=(xg, yg + 18),
                arrowprops=dict(arrowstyle="-", color="#1f5fa8", lw=1.1,
                                alpha=0.5))
    ax.annotate("\n".join(labs), (xg, yg + 26), ha="center", va="bottom",
                fontsize=8.3, color="#1f5fa8", fontweight="bold")
    ax.annotate("↓", (xg, yg - 26), ha="center", va="top", fontsize=9,
                color="#1f5fa8", alpha=0.8)

# runs as markers + labels (hand-set offsets, no collisions by construction)
for x, r in zip(xs, RUNS):
    d, days, c, lab, pos = r[0], r[1], r[2], r[3], r[4]
    dx, dy = pos[0], pos[1]
    ha = pos[2] if len(pos) > 2 else "center"
    global_era = x >= 7
    ax.plot(x, days, "o", ms=12 if global_era else 10, color=c, zorder=5,
            markeredgecolor="white", markeredgewidth=1.3)
    ax.annotate(lab, (x, days), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom" if dy > 0 else "top",
                fontsize=8.6, color="0.15")

# date strip under the axis: runs' completion dates
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in RUNS], fontsize=9)
ax.set_yscale("symlog", linthresh=50)
ax.set_yticks([6, 15, 50, 100, 200, 365])
ax.set_yticklabels(["6", "15", "50", "100", "200", "365"])
ax.set_ylim(0, 520)
ax.set_ylabel("days survived (symlog)")
ax.set_xlim(-0.6, n - 0.4)
ax.set_title("The fix journey: 81 fix/feature commits, survival ceiling 6 d → 365 d",
             fontsize=13.5, fontweight="bold", pad=10)
ax.grid(alpha=0.25, axis="y")

# data assertions (image-blind verification)
assert min(r[1] for r in RUNS) == 6, "stream6 is the floor"
assert max(r[1] for r in RUNS) == 365
assert sum(r[1] == 365 for r in RUNS) == 5
assert len(RUNS) == 11

font_audit(fig, "fig11")
out = os.path.join(HERE, "fig11_fix_journey.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print("fig11 done: 11 stations, 7 fix-gap annotations, era shading")
