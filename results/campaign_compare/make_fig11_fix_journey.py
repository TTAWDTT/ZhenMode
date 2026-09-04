# fig11: the fix journey — survival ladder 08-24 -> 09-04.
# Runs are stations on a chronological track (x = run index). y = days
# survived (symlog). Each fix is a CALLOUT box with a leader line pointing
# AT the run marker it enabled. Placement starts from a simple 3-row rule
# and is then cleaned by a deterministic pixel-space collision resolver
# (image-blind: bbox harness at the end must report 0 overlaps).
import os
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

# (date, days survived, color, label, label placement (dx, dy[, ha]))
RUNS = [
    ("08-24",  90, "#2a7f62", "s1_90d\n(90 d, PASS)",            (0, 13)),
    ("08-24",  15, "#c0504d", "smoke15\n(d15 blow-up)",          (0, 13)),
    ("08-24",   6, "#c0504d", "stream6\n(d6 gate)",              (0, 13)),
    ("08-24", 140, "#c0504d", "windblend200\n(d140 blow-up)",    (-10, 13), "right"),
    ("08-25", 340, "#c0504d", "sponge8\n(d340 blow-up)",         (-6, 13)),
    ("08-25", 365, "#2a7f62", "sponge16\n(365 d PASS)",          (0, 13)),
    ("08-25", 120, "#c0504d", "norestore\n(d120 blow-up)",       (8, -30)),
    ("09-03", 365, "#c0504d", "fgate\n(365 d, FAIL_DRIFT)",      (-2, -30)),
    ("09-03", 365, "#2a7f62", "glap\n(365 d, runner PASS)",      (0, 13)),
    ("09-03", 365, "#2a7f62", "glap2\n(365 d, A1/A2 PASS)",      (0, 13)),
    ("09-04", 365, "#1f5fa8", "g365d_012\n(365 d PROD)",         (0, 13)),
]

# fix callouts: (target run index, label, lane) — the fix enabled the run at
# `target`. lane 'top' = label above the plot body, 'bot' = in the lower band.
CALLS = [
    (1, "seasonal NCEP\nwind", "bot"),
    (2, "dynamic forcing\nas runtime arg", "bot"),
    (3, "wind-blend\ntransition", "bot"),
    (4, "mass-conserving\nsponge 8c", "top"),
    (5, "sponge widened\n8c -> 16c", "bot"),
    (6, "bulk heat flux\n(restore OFF)", "bot"),
    (7, "global era opens:\npolar cap | adjoint+rho PGF\n| GM closure + sponge", "bot"),
    (8, "adv + Laplacian\nface gating", "bot"),
    (9, "ocean-only\nT_atm", "bot"),
    (10, "seasonal wind\n+ Med relax", "bot"),
]

fig, ax = plt.subplots(figsize=(15.5, 8.6), constrained_layout=True)

n = len(RUNS)
xs = np.arange(n)
ys = [r[1] for r in RUNS]

# era shading: regional = stations 0-6, global = 7-10
ax.axvspan(-0.7, 6.5, color="#f2ecf8", zorder=0)
ax.axvspan(6.5, n - 0.3, color="#e8f0fa", zorder=0)
ax.text(0, 545, "regional spectral era (128^2 NW-Pacific)",
        fontsize=10.5, color="#7a5fa0", fontweight="bold", va="top")
ax.text(10.55, 545, "global 1deg FD era (360x120, real coastlines)",
        fontsize=10.5, color="#1f5fa8", fontweight="bold", va="top",
        ha="right")

# chronological track
ax.plot(xs, ys, "-", color="0.6", lw=1.3, alpha=0.65, zorder=2)

# ---- run markers + labels ------------------------------------------------
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

ax.annotate("pre-campaign foundations: spectral solver, JAX 12x, "
            "3D advection, biharmonic, semi-implicit PGF, NCEP loader",
            (6.45, 12), fontsize=8.2, color="0.35", style="italic",
            ha="right", va="center")

# ---- callouts: per-region tiered placement -------------------------------
# Three x-regions (left stations 0-3, mid 4-6, right 7-10); each callout
# goes into its region's tier stack (fixed y rows, x = marker x). Regions
# are chosen so the band is empty at those y values; a local nudge pass
# (next block) resolves any residual brush with run labels.
_TextCls = matplotlib.text.Text
callout_art = []
for k, (target, lab, lane) in enumerate(CALLS):
    xm, ym = xs[target], ys[target]
    t_ = ax.annotate(
        lab, xy=(xm, ym), xycoords="data",
        xytext=(xs[target], 30), textcoords="data",
        ha="center", va="center", fontsize=8.3, color="#1f5fa8",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#1f5fa8", lw=1.0,
                        alpha=0.7, connectionstyle="arc3,rad=0.22"),
        zorder=4)
    t_.set_alpha(0.0)   # invisible for measuring; painted by tier block
    callout_art.append(t_)
fig.canvas.draw()
ren = fig.canvas.get_renderer()
sizes = {k: _TextCls.get_window_extent(t_, ren)
         for k, t_ in enumerate(callout_art)}

TIERS = {
    "left":  [175, 215, 255],       # empty band above low markers (<=140)
    "mid":   [180, 220, 260],       # empty band below 340/365 labels
    "right": [55, 100, 145, 190],   # below the 340-365 cluster
}
REGION = {0: "left", 1: "left", 2: "left", 3: "left",
          4: "mid", 5: "mid", 6: "mid",
          7: "right", 8: "right", 9: "right", 10: "right"}
XWIN = {"left": (-0.55, 3.45), "mid": (3.55, 6.45), "right": (6.55, 10.55)}
used_rows = {r: 0 for r in TIERS}

# pack within each region: tallest callout claims the lowest (safest) row
by_region = {}
for k, (target, lab, lane) in enumerate(CALLS):
    by_region.setdefault(REGION[target], []).append(k)

for region, ks in by_region.items():
    ks.sort(key=lambda k: -sizes[k].height)
    xlo, xhi = XWIN[region]
    for row_i, k in enumerate(ks):
        t_, (target, lab, lane) = callout_art[k], CALLS[k]
        xm, ym = xs[target], ys[target]
        y_row = TIERS[region][min(row_i, len(TIERS[region]) - 1)]
        # x: marker x, clamped into the region window by half label width
        bb = sizes[k]
        w_data = (bb.width / (ax.transData.transform((1, 0))[0]
                              - ax.transData.transform((0, 0))[0]))
        cx = min(max(xm, xlo + w_data / 2 + 0.05), xhi - w_data / 2 - 0.05)
        t_.set_position((cx, y_row))
        t_.set_ha("center")
        t_.set_va("center")
        t_.set_alpha(1.0)
fig.canvas.draw()

# ---- local nudge: move a callout only within its region window ----------
fig.canvas.draw()
movable = {id(t) for t in callout_art}
inv2 = ax.transData.inverted()
region_of = {}
for k, (target, lab, lane) in enumerate(CALLS):
    region_of[id(callout_art[k])] = REGION[target]
win_of = {id(callout_art[k]): XWIN[REGION[CALLS[k][0]]]
          for k in range(len(CALLS))}
for _pass in range(80):
    ren2 = fig.canvas.get_renderer()
    boxes = [(t, _TextCls.get_window_extent(t, ren2))
             for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip()]
    clash = None
    for ti, bi in boxes:
        if id(ti) not in movable:
            continue
        for tj, bj in boxes:
            if tj is ti:
                continue
            ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ix > 2 and iy > 2:
                clash = (ti, bi, bj)
                break
        if clash:
            break
    if not clash:
        break
    ti, bi, bj = clash
    px, py = ax.transData.transform(ti.get_position())
    dy_px = (10 if bi.y0 >= bj.y0 else -10)
    dx_px = 0
    if abs((bi.y0 + bi.y1) / 2 - (bj.y0 + bj.y1) / 2) < 8:
        dx_px = 12 if bi.x0 >= bj.x0 else -12
        dy_px = 0
    nx, ny = inv2.transform((px + dx_px, py + dy_px))
    xlo, xhi = win_of[id(ti)]
    half_w = (bi.x1 - bi.x0) / 2 / (ax.transData.transform((1, 0))[0]
                                    - ax.transData.transform((0, 0))[0])
    nx = min(max(nx, xlo + half_w), xhi - half_w)
    ny = min(max(ny, 22), 445)
    ti.set_position((nx, ny))
    fig.canvas.draw()
print(f"local nudge: {_pass} passes")

# ---- final nudge pass: resolve any residual text-text overlaps ----------
fig.canvas.draw()
movable = {id(t) for t in callout_art}
inv2 = ax.transData.inverted()
for _pass in range(60):
    ren2 = fig.canvas.get_renderer()
    boxes = [(t, _TextCls.get_window_extent(t, ren2))
             for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip()]
    clash = None
    for ti, bi in boxes:
        if id(ti) not in movable:
            continue
        for tj, bj in boxes:
            if tj is ti:
                continue
            ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ix > 2 and iy > 2:
                clash = (ti, bi, bj)
                break
        if clash:
            break
    if not clash:
        break
    ti, bi, bj = clash
    px, py = ax.transData.transform(ti.get_position())
    if bi.y0 >= bj.y0:
        dy_px = 10; dx_px = 0
    else:
        dy_px = -10; dx_px = 0
    if abs((bi.y0 + bi.y1) / 2 - (bj.y0 + bj.y1) / 2) < 8:
        dx_px = 14 if bi.x0 >= bj.x0 else -14
        dy_px = 0
    nx, ny = inv2.transform((px + dx_px, py + dy_px))
    ny = min(max(ny, 22), 445)
    nx = min(max(nx, -0.55), n - 0.45)
    ti.set_position((nx, ny))
    fig.canvas.draw()
print(f"final nudge: {_pass} passes")

# ---- overlap harness (image-blind verification) --------------------------
fig.canvas.draw()
ren = fig.canvas.get_renderer()
texts = [(t.get_text().replace("\n", "/"), _TextCls.get_window_extent(t, ren))
         for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
ov = []
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        a, b = texts[i][1], texts[j][1]
        ix = min(a.x1, b.x1) - max(a.x0, b.x0)
        iy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if ix > 2 and iy > 2:
            ov.append((texts[i][0][:30], texts[j][0][:28]))
print(f"bbox harness: texts={len(texts)} overlaps={len(ov)}")
for p in ov:
    print("   OVERLAP:", p)
assert not ov, f"{len(ov)} text overlaps remain"

# data assertions (image-blind verification)
assert min(r[1] for r in RUNS) == 6, "stream6 is the floor"
assert max(r[1] for r in RUNS) == 365
assert sum(r[1] == 365 for r in RUNS) == 5
assert len(RUNS) == 11
assert all(c[0] in range(len(RUNS)) for c in CALLS)

font_audit(fig, "fig11")
out = os.path.join(HERE, "fig11_fix_journey.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print("fig11 done: 11 stations, 10 curve-anchored callouts, auto-resolved layout")
