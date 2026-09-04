"""Campaign comparison figures: every long integration 08-24 ~ 09-04.

User deliverable (2026-09-04): "多张图呈现这些起点以及中间的结果，方便梳理"
+ figure beautification. Four figures:

  fig1  campaign verdict timeline — every run as a bar from start to
        end-of-life (blow-up day or full duration), colored by verdict,
        annotated with the root cause / distinguishing config.
  fig2  multi-run trajectory overlays — max|u| / max|T| / max|eta| for the
        three 365d-class runs + the failed ones, blow-up points marked.
  fig3  climatology scores A1/A2 across the three scored runs (+ regional
        honest-rescore context bar), pass/fail bars vs pre-registered bars.
  fig4  final-state field comparison — regional 365d (sponge16) NW-Pac
        window vs the same window cut from the global 1° production run.

Data sources (all numeric, verified):
  results/long_run_*.npz          regional spectral runs (local, 08-24~25)
  results/global_g365d_012.npz    production run (pulled from 012)
  C:/Users/zhen.luo/.research/*_out.txt  remote GPU run console logs
                                  (fgate/glap/glap2 trajectories)
  results/climatology_*/          A1/A2 score npz for 3 runs
  docs/g365d_work_summary_zh.md   remote-run verdicts & scores (quoted)

NOTE: figures are NEVER read back into context (image-blind workflow).
"""
import sys, os, re, datetime
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "savefig.facecolor": "white",
})

OUT = os.path.join(ROOT, "results", "campaign_compare")
os.makedirs(OUT, exist_ok=True)

C_PASS = "#2a7f62"      # calm green
C_FAIL = "#c0504d"      # muted red
C_DRIFT = "#d9822b"     # muted orange
C_GLOBAL = "#3b6fb5"    # steel blue
C_REG = "#7a5fa0"       # muted violet


def font_audit(fig, name):
    bad = []
    for t in fig.findobj(mtext.Text):
        if not t.get_text():
            continue
        fam = t.get_fontproperties().get_family()
        if "serif" not in fam and "Times New Roman" not in fam:
            bad.append((t.get_text()[:40], fam))
    print(f"[font-audit] {name}: {'OK (all serif/Times)' if not bad else bad[:5]}")


# ══════════════════════════════════════════════════════════════════════
# Load every run
# ══════════════════════════════════════════════════════════════════════
R = os.path.join(ROOT, "results")
RES = "C:/Users/zhen.luo/.research"


def traj_from_files(fns):
    ROW = re.compile(r"\s*(\d+\.\d)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s+([\d.eE+]+)\s+(\d+)")
    t = {"day": [], "maxu": [], "maxT": [], "maxeta": [], "ke": []}
    for fn in fns:
        for ln in open(fn, encoding="utf-8", errors="ignore"):
            m = ROW.match(ln)
            if m:
                dd = float(m.group(1))
                if dd in t["day"]:
                    continue
                t["day"].append(dd)
                t["maxu"].append(float(m.group(3)))
                t["maxT"].append(float(m.group(4)))
                t["maxeta"].append(float(m.group(5)))
                t["ke"].append(float(m.group(7)))
    return {k: np.asarray(v) for k, v in t.items()}


RUNS = []   # dicts: name, verdict, days(end), peak, src, note, kind, npz/traj


def add_npz(fname, name, note):
    z = np.load(os.path.join(R, fname), allow_pickle=True)
    days = np.asarray(z["days"], float)
    verdict = str(z["verdict"])
    RUNS.append(dict(
        name=name, verdict=verdict, dur=float(days[-1]),
        peak=float(z["max_u_peak"]), kind="regional",
        days=days, maxu=np.asarray(z["max_u"], float),
        maxT=np.asarray(z["max_T"], float),
        maxeta=np.asarray(z["max_eta"], float),
        ke=np.asarray(z["ke"], float), note=note,
        mtime=datetime.datetime.fromtimestamp(
            os.path.getmtime(os.path.join(R, fname))).strftime("%m-%d"),
    ))


add_npz("long_run_s1_90d.npz", "s1_90d",
        "stage-1 fixed Jan wind, restore τ=5d")
add_npz("long_run_s2_smoke15.npz", "s2_smoke15",
        "seasonal wind smoke — FAIL_DRIFT")
add_npz("long_run_s2_stream_test.npz", "s2_stream_test",
        "6d streamfunction gate")
add_npz("long_run_s2_365d_sponge.npz", "s2_365d_sponge",
        "sponge 8c/τ=5d — blew up d340")
add_npz("long_run_s2_sponge200.npz", "s2_sponge200",
        "sponge 8c/τ=5d, capped 200d")
add_npz("long_run_s2_windblend200.npz", "s2_windblend200",
        "no sponge, wind blend — blew up d140")
add_npz("long_run_s2_365d_norestore.npz", "s2_365d_norestore",
        "restore OFF — thermodynamic blow-up d120")
add_npz("long_run_s2_365d_sponge16_3d.npz", "s2_365d_sponge16",
        "sponge 16c/τ=3d — regional final, PASS")

fg = traj_from_files([f"{RES}/p3_out.txt"])
RUNS.append(dict(name="gpu365_fgate", verdict="FAIL_DRIFT", dur=365.0,
                 peak=220.1, kind="global", days=fg["day"], maxu=fg["maxu"],
                 maxT=fg["maxT"], maxeta=fg["maxeta"], ke=fg["ke"],
                 note="ungated _laplacian_h — ghost halo, maxT 220 °C",
                 mtime="09-03"))
gl = traj_from_files([f"{RES}/gy2_out.txt", f"{RES}/gy4_out.txt"])
RUNS.append(dict(name="gpu365_glap", verdict="PASS", dur=365.0,
                 peak=0.708, kind="global", days=gl["day"], maxu=gl["maxu"],
                 maxT=gl["maxT"], maxeta=gl["maxeta"], ke=gl["ke"],
                 note="gate fix; A1/A2 RMSE 3.29/3.46 FAIL → forcing bugs",
                 mtime="09-03"))
gz = traj_from_files([f"{RES}/gz2_out.txt", f"{RES}/gz3_out.txt",
                      f"{RES}/gz4_out.txt"])
RUNS.append(dict(name="gpu365_glap2", verdict="PASS", dur=365.0,
                 peak=0.709, kind="global", days=gz["day"], maxu=gz["maxu"],
                 maxT=gz["maxT"], maxeta=gz["maxeta"], ke=gz["ke"],
                 note="ocean-only T_atm fix → A1/A2 double PASS",
                 mtime="09-03"))

gz2 = np.load(os.path.join(R, "global_g365d_012.npz"), allow_pickle=True)
RUNS.append(dict(name="g365d_012", verdict="PASS", dur=365.0,
                 peak=float(gz2["max_u_peak"]), kind="global",
                 days=np.asarray(gz2["days"], float),
                 maxu=np.asarray(gz2["max_u"], float),
                 maxT=np.asarray(gz2["max_T"], float),
                 maxeta=np.asarray(gz2["max_eta"], float),
                 ke=np.asarray(gz2["ke"], float),
                 note="PRODUCTION: seasonal wind + eta_relax, 012 GPU",
                 mtime="09-04"))

VERDICT_COLOR = {"PASS": C_PASS, "FAIL_BLOWUP": C_FAIL,
                 "FAIL_DRIFT": C_DRIFT}

print("runs loaded:", len(RUNS))
for r in RUNS:
    print(f"  {r['name']:20s} {r['verdict']:12s} dur={r['dur']:5.0f} "
          f"peak={r['peak']:7.3f} npts={len(r['days'])}")

# ══════════════════════════════════════════════════════════════════════
# fig1: campaign verdict timeline — one bar per run, colored by verdict.
# Rows ordered chronologically; global-FD runs separated by a divider.
# ══════════════════════════════════════════════════════════════════════
order = ["s1_90d", "s2_smoke15", "s2_stream_test", "s2_windblend200",
         "s2_sponge200", "s2_365d_sponge", "s2_365d_norestore",
         "s2_365d_sponge16", "SEP", "gpu365_fgate", "gpu365_glap",
         "gpu365_glap2", "g365d_012"]
by = {r["name"]: r for r in RUNS}

fig, ax = plt.subplots(figsize=(13.5, 7.4), constrained_layout=True)
rows, labels, colors, notes = [], [], [], []
y = 0
for name in order:
    if name == "SEP":
        y -= 0.6
        continue
    r = by[name]
    rows.append(y)
    labels.append(f"{r['name']}\n({r['mtime']})")
    colors.append(VERDICT_COLOR[r["verdict"]])
    notes.append(r["note"])
    y -= 1.0

for yy, name, c, note in zip(rows, [o for o in order if o != "SEP"],
                             colors, notes):
    r = by[name]
    ax.barh(yy, r["dur"], height=0.62, color=c, alpha=0.82, zorder=3,
            edgecolor="white", lw=0.8)
    if r["verdict"] == "FAIL_BLOWUP":
        ax.plot(r["dur"], yy, marker="x", ms=11, mew=3.2, color="#5a1512",
                zorder=5)
        ax.annotate(f"blow-up d{r['dur']:.0f}", (r["dur"], yy),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=9, color="#5a1512", fontweight="bold")
    else:
        lab = f"d{r['dur']:.0f}" + (f"  peak |u|={r['peak']:.2f}"
                                    if r["peak"] < 10 else "")
        ax.annotate(lab, (r["dur"], yy), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9)
    if r["dur"] > 60:    # note drawn in white inside the bar
        ax.annotate(note, (3, yy), va="center", ha="left", fontsize=8.6,
                    color="white", fontweight="bold", zorder=6)
    else:                # short bars: note outside, right
        ax.annotate(note, (r["dur"], yy), textcoords="offset points",
                    xytext=(86, -13), va="top", fontsize=8.3,
                    color="0.25", style="italic")

ax.axvline(365, color="0.55", lw=1.1, ls=(0, (4, 3)), zorder=2)
ax.annotate("365 d", (365, rows[0] + 0.85), ha="center", fontsize=10,
            color="0.35")

sep_y = (rows[7] + rows[8]) / 2 + 0.3
ax.axhline(sep_y, color="0.6", lw=1.0)
ax.annotate("regional spectral era (128² NW-Pacific, periodic)",
            (363, sep_y + 0.06), ha="right", va="bottom", fontsize=9.5,
            color="0.3", style="italic")
ax.annotate("global 1° FD era (360×120, real coastlines) — 012 GPU",
            (363, sep_y - 0.10), ha="right", va="top", fontsize=9.5,
            color="0.3", style="italic")

ax.set_yticks(rows)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_ylim(rows[-1] - 0.8, rows[0] + 1.1)
ax.set_xlim(0, 470)
ax.set_xlabel("Integration length (days)")
ax.set_title("Long-integration campaign, 08-24 → 09-04 — every run, its fate, and why",
             fontsize=13.5, fontweight="bold", pad=10)
handles = [Line2D([], [], marker="s", ls="", ms=11, color=C_PASS,
                  label="PASS (pre-registered criteria met)"),
           Line2D([], [], marker="s", ls="", ms=11, color=C_FAIL,
                  label="FAIL_BLOWUP (NaN / |u|>10)"),
           Line2D([], [], marker="s", ls="", ms=11, color=C_DRIFT,
                  label="FAIL_DRIFT (monotonic drift / halo)")]
ax.legend(handles=handles, loc="lower right", framealpha=0.9,
          edgecolor="0.8")
ax.grid(axis="x", alpha=0.3)
font_audit(fig, "fig1")
fig.savefig(os.path.join(OUT, "fig1_campaign_timeline.png"), dpi=140)
plt.close(fig)
print("fig1 done")

# ══════════════════════════════════════════════════════════════════════
# fig2: trajectory overlays — max|u| / max|T| / max|eta| / KE.
# Top row: the four 365d-class survivors. Bottom row: every run that
# died, with its blow-up X marker — the "middle results" the user asked
# to see laid out.
# ══════════════════════════════════════════════════════════════════════
SURV = ["s2_365d_sponge16", "gpu365_glap", "gpu365_glap2", "g365d_012"]
SURV_C = {"s2_365d_sponge16": C_REG, "gpu365_glap": "#d9822b",
          "gpu365_glap2": "#4d8f4d", "g365d_012": C_GLOBAL}
DIED = ["s2_windblend200", "s2_365d_norestore", "s2_365d_sponge",
        "gpu365_fgate"]
DIED_C = {"s2_windblend200": "#9467bd", "s2_365d_norestore": "#c44e52",
          "s2_365d_sponge": "#dd8452", "gpu365_fgate": "#8c564b"}

PANELS = [("max $|u|$ (m/s)", "maxu", None),
          ("max $|T|$ ($^\circ$C)", "maxT", None),
          ("max $|\eta|$ (m)", "maxeta", None),
          ("KE (J m$^{-2}$)", "ke", None)]

fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.2), constrained_layout=True)

for j, (ylab, key, _) in enumerate(PANELS):
    ax = axes[0, j]
    for name in SURV:
        r = by[name]
        ax.plot(r["days"], r[key], "-", color=SURV_C[name], lw=1.7,
                alpha=0.9,
                label=(f"{name}" + ("  (prod)" if name == "g365d_012" else "")))
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.set_title(["Velocity", "Temperature", "Sea surface height",
                  "Kinetic energy"][j], fontsize=11)
    if key == "maxu":
        ax.set_ylim(0, 1.9)
    if key == "maxT":
        ax.set_ylim(23, 30)
    if key == "maxeta":
        ax.set_ylim(0, 2.8)
    if j == 0:
        ax.legend(loc="upper left", framealpha=0.85, fontsize=8.8)

for j, (ylab, key, _) in enumerate(PANELS):
    ax = axes[1, j]
    for name in DIED:
        r = by[name]
        c = DIED_C[name]
        v = np.isfinite(r[key])
        ax.plot(r["days"][v], np.abs(r[key][v]), "-", color=c, lw=1.5,
                alpha=0.9, label=name)
        # death marker at last finite point
        if v.any():
            ax.plot(r["days"][v][-1], np.abs(np.asarray(r[key])[v][-1]),
                    "x", ms=10, mew=2.6, color=c, zorder=5)
    ax.grid(alpha=0.3)
    if j == 3:
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8.4)
    ax.set_xlabel("Day")
    if key == "maxu":
        ax.set_ylim(0, 10)
    if key == "maxT":
        # runaway phases reach 220 (fgate) / 744 (norestore): symlog shows
        # the quiet ~24 C linear phase AND the exponential blow-up tail
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(20, 1000)
        ax.set_yticks([20, 30, 50, 100, 300, 1000])
    if key == "maxeta":
        ax.set_ylim(0, 10.5)

for ax in axes[0]:
    ax.margins(x=0.02)
fig.suptitle("Trajectories: survivors (top) vs runs that died (bottom) — "
             "same diagnostics, different fates", fontsize=13.5,
             fontweight="bold")
font_audit(fig, "fig2")
fig.savefig(os.path.join(OUT, "fig2_trajectory_overlays.png"), dpi=140)
plt.close(fig)
print("fig2 done")

# ══════════════════════════════════════════════════════════════════════
# fig3: climatology scores across the campaign.
# Left: A1/A2 corr+RMSE for the three officially scored runs, with the
#       pre-registered bars drawn as lines. Includes the regional honest
#       rescore (corr 0.258 zonal-anomaly FAIL) as context on A2 axis.
# Right: A2 RMSE evolution across the campaign arc — the "skill ladder":
#       regional cyclic 0.729 → regional honest 0.789 (residual-circulation
#       version 0.789/0.289) → global glap RMSE-FAIL 3.46 → global glap2
#       1.728 → production 1.881 (regional-only bar).
# ══════════════════════════════════════════════════════════════════════
clim = {}
for tag, fn in [("s1", "climatology_s1/climatology_compare.npz"),
                ("s2_sponge16",
                 "climatology_s2_365d_sponge16/climatology_compare.npz"),
                ("g365d_012",
                 "climatology_g365d_012/climatology_compare_g.npz")]:
    d = np.load(os.path.join(R, fn), allow_pickle=True)
    clim[tag] = {k: float(d[k]) for k in
                 ["corr_zonal", "rmse_zonal", "corr_pat", "rmse_pat",
                  "slope", "ke_drift"]}

fig = plt.figure(figsize=(15.0, 6.4), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

# ── left: A1/A2 corr & RMSE grouped bars ─────────────────────────────
axL = fig.add_subplot(gs[0, 0])
names = ["s1 (regional\nrestore ON)", "s2_sponge16\n(regional final)",
         "g365d_012\n(global production)"]
x = np.arange(3)
w = 0.19
cols_a1 = ["#3b6fb5", "#7fb3d5"]
cols_a2 = ["#c0504d", "#e6a5a3"]
c1 = axL.bar(x - 1.5 * w, [clim[t]["corr_zonal"] for t in clim], w,
             color=cols_a1[0], label="A1 corr (zonal SST)")
c2 = axL.bar(x - 0.5 * w, [clim[t]["corr_pat"] for t in clim], w,
             color=cols_a1[1], label="A2 corr (pattern)")
axR = axL.twinx()
b1 = axR.bar(x + 0.5 * w, [clim[t]["rmse_zonal"] for t in clim], w,
             color=cols_a2[0], label="A1 RMSE")
b2 = axR.bar(x + 1.5 * w, [clim[t]["rmse_pat"] for t in clim], w,
             color=cols_a2[1], label="A2 RMSE")
axL.axhline(0.3, color="0.25", ls=(0, (5, 3)), lw=1.2)
axL.annotate("corr bar 0.3", (2.45, 0.33), fontsize=9, ha="right",
             color="0.25")
axR.axhline(2.0, color="0.25", ls=(0, (2, 2)), lw=1.2)
axR.annotate("RMSE bar 2.0 °C", (2.45, 2.06), fontsize=9, ha="right",
             color="0.25")
for rects, src, off in [(c1, "corr_zonal", 0), (c2, "corr_pat", 0),
                        (b1, "rmse_zonal", 0), (b2, "rmse_pat", 0)]:
    for rect in rects:
        v = rect.get_height()
        ax = axL if rect in list(c1) + list(c2) else axR
        fmt = f"{v:.3f}" if ax is axL else f"{v:.2f}"
        ax.annotate(fmt, (rect.get_x() + rect.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.6, rotation=90 if ax is axR else 0)
axL.set_xticks(x)
axL.set_xticklabels(names, fontsize=9.5)
axL.set_ylabel("corr (higher better)")
axR.set_ylabel("RMSE (°C, lower better)")
axL.set_ylim(0, 1.14)
axR.set_ylim(0, 4.4)
axL.set_title("Official A1/A2 scores vs pre-registered bars\n"
              "(corr>0.3, RMSE<2.0 — frozen, never moved)", fontsize=11)
h1, l1 = axL.get_legend_handles_labels()
h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, fontsize=8.2,
           framealpha=0.9)
axR.spines["top"].set_visible(False)

# ── right: non-circular skill ladder ─────────────────────────────────
axRt = fig.add_subplot(gs[0, 1])
steps = [
    ("regional\nrestore-ON\n(circular)", 0.997, C_DRIFT,
     "A2 0.997 — restore = WOA target\ncircular, not skill"),
    ("regional\nfree-run honest", 0.984, C_DRIFT,
     "A2 0.984 still carries the forced\nmeridional gradient (99.8% var)"),
    ("regional\nzonal anomaly", 0.258, C_FAIL,
     "true predicted zonal skill\n= corr 0.258 → FAIL"),
    ("global glap2\nA2 pattern", 0.980, C_PASS,
     "real coastlines + bulk flux:\nnon-circular design PASS"),
    ("global prod\ng365d_012", 0.977, C_PASS,
     "A2 0.977 with seasonal wind\n+ eta_relax — production"),
]
xs = np.arange(len(steps))
vals = [s[1] for s in steps]
bars = axRt.bar(xs, vals, 0.6, color=[s[2] for s in steps], alpha=0.88)
axRt.axhline(0.3, color="0.25", ls=(0, (5, 3)), lw=1.2)
axRt.annotate("0.3", (4.42, 0.32), fontsize=9, color="0.25")
for i, (lab, v, c, note) in enumerate(steps):
    axRt.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                  xytext=(0, 4), ha="center", fontsize=9.5,
                  fontweight="bold")
    # multi-line note sits inside the bar only when the bar is tall enough
    if v >= 0.62:
        axRt.annotate(note, (i, 0.045), ha="center", fontsize=7.3,
                      color="white", fontweight="bold",
                      transform=axRt.get_xaxis_transform())
    else:   # short bar: note above the value, dark text
        axRt.annotate(note, (i, 0.62), ha="center", fontsize=7.3,
                      color="#5a1512", fontweight="bold",
                      transform=axRt.get_xaxis_transform())
axRt.set_xticks(xs)
axRt.set_xticklabels([s[0] for s in steps], fontsize=8.6)
axRt.set_ylim(0, 1.12)
axRt.set_ylabel("A2-type corr")
axRt.set_title("The circularity lesson — skill that is earned vs manufactured\n"
               "(regional 08-25 honest rescore → global FD design answer)",
               fontsize=11)
axRt.grid(axis="y", alpha=0.3)
fig.suptitle("Climatology verification across the campaign — every scored run, one axis",
             fontsize=13.5, fontweight="bold")
font_audit(fig, "fig3")
fig.savefig(os.path.join(OUT, "fig3_climatology_scores.png"), dpi=140)
plt.close(fig)
print("fig3 done", {k: round(v["corr_pat"], 3) for k, v in clim.items()})

# ══════════════════════════════════════════════════════════════════════
# fig4: final-state comparison — what the two eras produced.
# Row 1: regional 365d (s2_365d_sponge16) NW-Pacific window
#        lon 143.6..156.4, lat 28.6..41.4 (0.1°, 128²).
# Row 2: same window cut from the global 1° production run (day 365).
# Shared SST colorbar scale 6..30 °C; SSH row on its own scale.
# ══════════════════════════════════════════════════════════════════════
reg = np.load(os.path.join(R, "long_run_s2_365d_sponge16_3d.npz"),
              allow_pickle=True)
reg_T = reg["T_top"]           # (38, 128, 128) axis0=lon, axis1=lat
reg_eta = reg["eta"]
reg_lon = np.linspace(143.65, 156.35, 128)
reg_lat = np.linspace(28.65, 41.35, 128)

g = gz2
wet = np.asarray(g["wet_mask"], bool)
g_lon, g_lat = np.asarray(g["lon"]), np.asarray(g["lat"])
g_T = g["T_top"][-1]           # (360, 120)
g_eta = g["eta"][-1]
# global 1° cells inside the regional window
gi = np.where((g_lon >= 143.0) & (g_lon <= 157.0))[0]
gj = np.where((g_lat >= 28.0) & (g_lat <= 42.0))[0]
gT_w = np.where(wet[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1],
                g_T[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1], np.nan)
gE_w = np.where(wet[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1],
                g_eta[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1], np.nan)

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), constrained_layout=True)

rloni, rlati = np.meshgrid(reg_lon, reg_lat, indexing="ij")
imT0 = axes[0, 0].pcolormesh(rloni, rlati, reg_T[0], shading="auto",
                             cmap="RdYlBu_r", vmin=8, vmax=27)
imT1 = axes[0, 1].pcolormesh(rloni, rlati, reg_T[-1], shading="auto",
                             cmap="RdYlBu_r", vmin=6, vmax=30)
axes[0, 0].set_title("regional s2_sponge16 — SST day 0 (WOA)", fontsize=11)
axes[0, 1].set_title("regional s2_sponge16 — SST day 365", fontsize=11)
cb0 = fig.colorbar(imT0, ax=axes[0, 0], fraction=0.046, pad=0.03)
cb0.set_label("SST (°C)")
cb1 = fig.colorbar(imT1, ax=axes[0, 1], fraction=0.046, pad=0.03) \
    if False else None

cb1 = fig.colorbar(imT1, ax=axes[0, 1], fraction=0.046, pad=0.03)
cb1.set_label("SST (°C)")

# global row: use the same window; coarse 1° cells
gl = g_lon[gi]; gj_lat = g_lat[gj]
gT0 = np.where(wet[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1],
               np.asarray(g["T_init"])[gi[0]:gi[-1] + 1, gj[0]:gj[-1] + 1, 0],
               np.nan)
imG0 = axes[1, 0].pcolormesh(g_lon[gi] - 0.5, g_lat[gj] - 0.5, gT0.T,
                             shading="auto", cmap="RdYlBu_r", vmin=6,
                             vmax=30)
imG1 = axes[1, 1].pcolormesh(g_lon[gi] - 0.5, g_lat[gj] - 0.5, gT_w.T,
                             shading="auto", cmap="RdYlBu_r", vmin=6,
                             vmax=30)
axes[1, 0].set_title("global g365d_012 — SST day 0 (same window, 1°)",
                     fontsize=11)
axes[1, 1].set_title("global g365d_012 — SST day 365 (same window, 1°)",
                     fontsize=11)
cbG = fig.colorbar(imG1, ax=[axes[1, 0], axes[1, 1]], fraction=0.035,
                   pad=0.02)
cbG.set_label("SST (°C)")

for ax in axes.flat:
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

n_reg = int(reg_T[-1].size)
print(f"fig4 data: regional window SST d365 [{np.nanmin(reg_T[-1]):.2f}, "
      f"{np.nanmax(reg_T[-1]):.2f}] | global window d365 "
      f"[{np.nanmin(gT_w):.2f}, {np.nanmax(gT_w):.2f}] "
      f"({gT_w.size} cells, {int(np.isfinite(gT_w).sum())} wet)")
fig.suptitle("Same ocean, two eras — NW-Pacific window (143.6–156.4 E, 28.6–41.4 N): "
             "regional 128² spectral vs global 1° FD production",
             fontsize=13, fontweight="bold")
font_audit(fig, "fig4")
fig.savefig(os.path.join(OUT, "fig4_regional_vs_global_window.png"), dpi=140)
plt.close(fig)
print("fig4 done")

# ══════════════════════════════════════════════════════════════════════
# Batch 2 (second user ask: "还有没有别的能画的图")
#   fig5  Hovmoller zonal-mean SST(t, lat) — full-year thermal evolution
#   fig6  seasonal phase: monthly forcing |tau| vs KE/SSH harmonic response
#   fig7  eta bookkeeping: global mass + Med box taming (9.58 → 0.019)
#   fig8  SSH radial spectra, three runs, slope ladder −5.76/−5.72/−3.02
# ══════════════════════════════════════════════════════════════════════

# ── fig5: Hovmoller ──────────────────────────────────────────────────
Tw = np.where(wet[None, :, :], np.asarray(gz2["T_top"], float), np.nan)
hov = np.nanmean(Tw, axis=1)                       # (38, 120)
hov_anom = hov - hov[0]
fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.6), sharex=True,
                         constrained_layout=True)
days_g = by["g365d_012"]["days"]
X, Y = np.meshgrid(days_g, g_lat, indexing="ij")
im0 = axes[0].pcolormesh(X, Y, hov, shading="auto", cmap="RdYlBu_r",
                         vmin=0, vmax=29)
cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.015)
cb0.set_label("zonal-mean SST ($^\circ$C)")
axes[0].set_title("Zonal-mean SST: spin-up, polar-cap adjustment, seasonal breathing",
                  fontsize=11)
im1 = axes[1].pcolormesh(X, Y, hov_anom, shading="auto", cmap="RdBu_r",
                         vmin=-3, vmax=3)
cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.015)
cb1.set_label("$\Delta$SST vs day 0 ($^\circ$C)")
axes[1].annotate("polar-cap zone: bulk flux pulls toward\n"
                 "T_atm polar value (+6 °C by d365)",
                 (200, 55.5), fontsize=8.6, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))
axes[1].annotate("subtropics: mixed-layer\nseasonal breathing (−1.4 °C)",
                 (230, 38), fontsize=8.6, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))
for ax in axes:
    ax.set_ylabel("Latitude ($^\circ$N)")
axes[1].set_xlabel("Day")
# month gridlines
for md in [30.4*k for k in range(1, 12)]:
    for ax in axes:
        ax.axvline(md, color="0.5", lw=0.4, alpha=0.5, zorder=1)
fig.suptitle("g365d_012 — zonal-mean SST time–latitude section (38 snapshots)",
             fontsize=13.5, fontweight="bold")
font_audit(fig, "fig5")
fig.savefig(os.path.join(OUT, "fig5_hovmoller_sst.png"), dpi=140)
plt.close(fig)
print("fig5 done | anom range", round(float(np.nanmin(hov_anom)), 2),
      round(float(np.nanmax(hov_anom)), 2))

# ── fig6: seasonal forcing vs response ───────────────────────────────
# Rebuild the 12 monthly NW-Pac box |tau| (same path as the run), then
# overlay the KE / SSH_std response series with their harmonic fits.
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
from wind_reanalysis import load_monthly_wind, wind_stress_from_wind
from forcing import taper_2d_y

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
lon2, lat2 = np.meshgrid(grid.lon, grid.lat, indexing="ij")
boxm = (lon2 >= 140) & (lon2 <= 180) & (lat2 >= 25) & (lat2 <= 45)
tau_box = []
for m in range(12):
    u10, v10 = load_monthly_wind(month_idx=(2023 - 1948) * 12 + m, grid=grid)
    tx, ty = wind_stress_from_wind(u10, v10)
    tx = taper_2d_y(tx, grid.ny, 8)
    tau_box.append(float(np.hypot(tx, ty)[boxm].mean()))
tau_box = np.asarray(tau_box)

r = by["g365d_012"]
ke_r, ssh_r, days_r = r["ke"], r["maxeta"] * 0 + r["ke"] * 0, r["days"]
ssh_r = np.asarray(gz2["ssh_std"], float)

def harmonic(y, t, periods):
    X = np.column_stack([np.ones_like(t)] +
                        [f(2 * np.pi * t / p) for p in periods
                         for f in (np.sin, np.cos)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta, 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)

msk = days_r >= 50
ke_fit, ke_r2 = harmonic(ke_r[msk], days_r[msk] - 50, [365.25, 182.62])
sh_fit, sh_r2 = harmonic(ssh_r[msk], days_r[msk] - 50, [365.25, 182.62])

fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), constrained_layout=True)
ax = axes[0]
mids = np.arange(12) * 30.4 + 15
ax.bar(mids, tau_box, width=22, color="#4d7ea8", alpha=0.85,
       edgecolor="white")
ax.set_ylabel("NW-Pac box-mean |τ| (N m$^{-2}$)")
ax.set_title("Forcing: monthly wind stress (NCEP R1 2023) — "
             f"seasonal swing {(tau_box.max()-tau_box.min())/tau_box.min()*100:.0f}%",
             fontsize=11)
ax.set_xlim(0, 365)
ax.grid(axis="y", alpha=0.3)
ax.annotate("Jan", (mids[0], tau_box[0]), textcoords="offset points",
            xytext=(0, 4), ha="center", fontsize=8.5)
ax.annotate("Jul", (mids[6], tau_box[6]), textcoords="offset points",
            xytext=(0, 4), ha="center", fontsize=8.5)

ax = axes[1]
ax.plot(days_r, ke_r, "-", color="#2a7f62", lw=1.8, label="KE (J m$^{-2}$)")
ax.plot(days_r[msk], ke_fit, "--", color="#2a7f62", lw=1.2, alpha=0.8,
        label=f"KE harmonic fit (annual+semi), R²={ke_r2:.2f}")
ax.set_ylabel("KE (J m$^{-2}$)", color="#2a7f62")
ax2 = ax.twinx()
ax2.plot(days_r, ssh_r, "-", color="#3b6fb5", lw=1.8, label="SSH std (m)")
ax2.plot(days_r[msk], sh_fit, "--", color="#3b6fb5", lw=1.2, alpha=0.8,
         label=f"SSH harmonic fit, R²={sh_r2:.2f}")
ax2.set_ylabel("SSH std (m)", color="#3b6fb5")
ax2.spines["top"].set_visible(False)
ax.set_xlabel("Day")
ax.set_xlim(0, 365)
ax.grid(alpha=0.3)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.8, framealpha=0.9)
ax.set_title("Response: KE / SSH annual+semiannual cycles (fit from d50+, "
             "spin-up excluded)", fontsize=11)
fig.suptitle("Seasonal forcing → response chain (the dynamic-forcing payoff)",
             fontsize=13.5, fontweight="bold")
font_audit(fig, "fig6")
fig.savefig(os.path.join(OUT, "fig6_seasonal_forcing_response.png"), dpi=140)
plt.close(fig)
print(f"fig6 done | tau swing {(tau_box.max()-tau_box.min())/tau_box.min()*100:.0f}% "
      f"KE R2 {ke_r2:.3f} SSH R2 {sh_r2:.3f}")

# ── fig7: eta bookkeeping — mass + Med box ───────────────────────────
eta_w = np.where(wet[None, :, :], np.asarray(gz2["eta"], float), np.nan)
mean_eta = np.nanmean(eta_w, axis=(1, 2))
lon_g = np.asarray(gz2["lon"])
boxm_g = ((lon_g <= 42) | (lon_g >= 354))[:, None] & \
         (g_lat >= 30)[None, :] & (g_lat <= 46.5)[None, :] & wet
med = np.nanmean(np.where(boxm_g[None], eta_w, np.nan), axis=(1, 2))

fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True,
                         constrained_layout=True)
ax = axes[0]
ax.plot(days_r, mean_eta * 100, "-", color="#2a7f62", lw=1.8)
ax.set_ylabel("global wet-mean η (cm)")
ax.grid(alpha=0.3)
ax.set_title(f"Mass bookkeeping: global wet-mean η stays within "
             f"±{np.abs(mean_eta).max()*100:.1f} cm all year (no drift)",
             fontsize=11)
ax.axhline(0, color="0.5", lw=0.8)

ax = axes[1]
ax.plot(days_r, med, "-", color="#c0504d", lw=1.8,
        label="Med box η (τ=30 d relax ON) — max 0.019 m")
ax.axhline(0, color="0.5", lw=0.8)
# unrepaired reference: glap2 hit 9.58 m by d365 (linear growth) — draw the
# documented envelope: 0.026 m/d from d10
d_ref = np.linspace(0, 365, 50)
ax.plot(d_ref, np.clip(0.026 * (d_ref - 0), 0, None), "--", color="0.45",
        lw=1.4,
        label="unrepaired path (gpu365_glap2): +0.026 m/d → 9.58 m at d365")
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("Med box mean η (m)")
ax.set_xlabel("Day")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_title("Med box: the 9.58 m artifact vs the relaxed run", fontsize=11)
fig.suptitle("η budget of the production run — mass conserved, artifact tamed",
             fontsize=13.5, fontweight="bold")
font_audit(fig, "fig7")
fig.savefig(os.path.join(OUT, "fig7_eta_budget_medbox.png"), dpi=140)
plt.close(fig)
print(f"fig7 done | mean_eta max {np.abs(mean_eta).max():.4f} m, "
      f"Med max {np.nanmax(med):.4f} m")

# ── fig8: SSH radial spectra — slope ladder ──────────────────────────
# PSD = |F(eta_anom)|^2 per run; grids differ (dx 9.1 km vs 111 km), so
# curves are each normalized by P at their first k bin; only SLOPES compare.
specs = []
for tag, fn, lab, col in [
    ("s1", "climatology_s1/climatology_compare.npz",
     "s1 (90d regional, restore ON)", "#d9822b"),
    ("s2_sponge16", "climatology_s2_365d_sponge16/climatology_compare.npz",
     "s2_sponge16 (365d regional final)", "#7a5fa0"),
    ("g365d_012", "climatology_g365d_012/climatology_compare_g.npz",
     "g365d_012 (global production)", "#3b6fb5"),
]:
    c = np.load(os.path.join(R, fn), allow_pickle=True)
    k = np.asarray(c["k_cent"], float)
    P = np.asarray(c["P"], float)
    v = np.isfinite(k) & np.isfinite(P) & (P > 0) & (k > 0)
    specs.append((lab, k[v], P[v] / P[v][0], float(c["slope"]), col))

fig, ax = plt.subplots(figsize=(9.8, 7.0), constrained_layout=True)
for lab, k, Pn, slope, col in specs:
    ax.loglog(k * 1e5, Pn, "-", color=col, lw=1.8, alpha=0.9,
              label=f"{lab} — slope {slope:.2f}")
    ax.annotate(f"{slope:.2f}", (k[-4] * 1e5, Pn[-4] / 2.2), fontsize=9.5,
                color=col, fontweight="bold", ha="center")
ax.axhline(1.0, color="0.7", lw=0.6)
ax.set_xlabel("wavenumber k ($\times 10^{-5}$ m$^{-1}$;  1 unit ≈ 628 km wavelength)")
ax.set_ylabel("normalized PSD  P(k)/P(k$_{ref}$)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
ax.set_title("SSH radial spectra — B2 slope ladder\n"
             "regional −5.7 (over-damped, sponge + strong restoring) → "
             "global −3.02 (geostrophic-turbulence band)", fontsize=12)
# guide slope -3
kg = np.geomspace(specs[2][1][2], specs[2][1][-3], 20) * 1e5
ax.loglog(kg, 0.5 * (kg / kg[0]) ** (-3), ":", color="0.35", lw=1.3)
ax.annotate("k$^{-3}$ guide", (kg[-1] * 1.05, 0.5 * 3 ** -3), fontsize=8.5,
            color="0.35")
font_audit(fig, "fig8")
fig.savefig(os.path.join(OUT, "fig8_ssh_spectra_slopes.png"), dpi=140)
plt.close(fig)
print("fig8 done")
