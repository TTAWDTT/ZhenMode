# -*- coding: utf-8 -*-
"""Phase A spinup probe analysis (cluster-side, self-contained).

Reads results/global_spinA30probe.npz (2D tables) and the yearly 3D snaps in
results/global_spinA30probe_3d/, computes the three probe criteria from
docs/spinup_plan_zh.md:
  (a) no blowup          -> VERDICT == PASS in the runner npz
  (b) deepT trend drop   -> mean T over wet-at-depth cells (z<=1000 m) per
                            yearly snap, linear trend vs the +0.045 C/yr
                            baseline (kappa_v=1e-5 10-yr run)
  (c) Med eta no-watchdog-> max|eta| over the run well below the 15 m limit

Deep-T uses the 3-D layer-resolved wet mask (wet3_push.npz, rebuilt from
ETOPO via the grid.py pipeline) so below-floor ghost fill is excluded.
Writes results/spinA30probe_analysis.json (+ .npz) and prints a verdict.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/data/tmp/ocean"
TAG = sys.argv[1] if len(sys.argv) > 1 else "spinA30probe"
NPZ = os.path.join(ROOT, "results", f"global_{TAG}.npz")
D3 = os.path.join(ROOT, "results", f"global_{TAG}_3d")
W3 = os.path.join(ROOT, "results", "wet3_push.npz")
OUT_JSON = os.path.join(ROOT, "results", f"{TAG}_analysis.json")
OUT_NPZ = os.path.join(ROOT, "results", f"{TAG}_analysis.npz")

R_EARTH = 6.371e6
BASELINE_DEEPT_C_PER_YR = 0.045   # kappa_v=1e-5 10-yr run (moc_tenyr_ms_gm)

snap_files = sorted(glob.glob(os.path.join(D3, "snap_*.npy")))

d = None
if os.path.exists(NPZ):
    d = np.load(NPZ, allow_pickle=True)
else:
    # mid-run: final npz not yet written — proceed with 3D snaps only
    print(f"NOTE: {NPZ} not found (run still in progress); "
          "verdict/max-eta from tables unavailable")

lat = d["lat"] if (d and npz_ok) else np.linspace(-59.5, 59.5, 120)
lon = d["lon"] if (d and npz_ok) else (np.arange(360) + 0.5)
z = d["z"] if (d and npz_ok) else np.array(
    [0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500, -1000, -2000,
     -4000], dtype=float)
# Day axis from the snap FILENAME index (snap_00029 -> day 29*365), which is
# correct for both fresh runs (index 0..N) and --restart-from resumes (index
# starts at n_prev_snaps). The runner npz `days` table is NOT usable mid-run:
# it only exists at completion, and a stale same-tag npz from an earlier era
# would silently mismatch the snap count.
days = np.array([int(os.path.basename(f)[5:10]) * 365.0
                 for f in snap_files])
# npz verdict/max-eta are only valid when the table was written by THIS run
# (final npz) — gate on last-day agreement to reject stale same-tag npz from
# an earlier era (table row count can legitimately differ from snap-file count
# on --restart-from resumes: the parent era's boundary snaps stay on disk).
npz_ok = (d is not None and len(d["days"]) > 0
          and abs(float(d["days"][-1]) - days[-1]) < 1.0)
verdict = str(d["verdict"]) if npz_ok else "RUNNING"
max_eta_all = float(np.max(np.abs(d["max_eta"]))) if npz_ok else float("nan")
# Parent-era max|eta| (same integration, pre-resume segment): the era-2 table
# starts at the resume day, so set PARENT_NPZ to merge it in.
PARENT_NPZ = os.environ.get("PARENT_NPZ", "")
if npz_ok and PARENT_NPZ and os.path.exists(PARENT_NPZ):
    dp = np.load(PARENT_NPZ, allow_pickle=True)
    if len(dp["max_eta"]):
        max_eta_all = max(max_eta_all,
                          float(np.max(np.abs(dp["max_eta"]))))
max_u_peak = float(d["max_u_peak"]) if d else float("nan")
ke = d["ke"] if d else None

# ── Atlantic sector mask ──
nlon, nlat = lon.size, lat.size
atl_i = np.zeros(nlon, dtype=bool)
atl_i[(lon >= 280.0)] = True
atl_i[(lon < 25.0)] = True
atl2 = np.broadcast_to(atl_i[:, None], (nlon, nlat)).copy()

# ── 3-D layer-resolved wet mask (exclude below-floor ghost fill) ──
w = np.load(W3)
wet3_3d = (w["wet3"] if "wet3" in w.files
           else w[w.files[0]]) > 0.5        # (360,120,nz)
wet3_deep = wet3_3d & (z <= -1000.0)[None, None, :]
wet3_deep_atl = wet3_deep & atl_i[:, None, None]
wet3_2k = wet3_3d & (z >= -2000.0)[None, None, :]

# ── MOC streamfunction from v ──
dz = np.empty(z.size)
dz[0] = z[0] - z[1]
dz[-1] = z[-2] - z[-1]
for k in range(1, z.size - 1):
    dz[k] = 0.5 * (z[k - 1] - z[k + 1])
dx = 2.0 * np.pi * R_EARTH * np.cos(np.deg2rad(lat)) / nlon
wet3_col = (wet3_3d[:, :, 0:1])   # column mask for MOC (v ghost is zeroed)


def moc_psi(v, lons=None):
    if lons is None:
        vI = np.where(wet3_col, v, 0.0)
    else:
        vI = np.where(lons[:, :, None] & wet3_col, v, 0.0)
    V = vI.sum(axis=0) * dx[:, None] * dz[None, :]
    return np.cumsum(V, axis=1) / 1.0e6


n = len(snap_files)
print(f"{n} 3D snaps, days {days[0]:.0f}..{days[-1]:.0f}, verdict={verdict}")

amoc = np.zeros(n)
deepT = np.zeros(n)
deepT_atl = np.zeros(n)
T2k = np.zeros(n)
for i, f in enumerate(snap_files):
    s = np.load(f)                       # (4, nx, ny, nz) = T,u,v,S
    psi_a = moc_psi(s[2], lons=atl2)
    jband = (lat >= 30.0) & (lat <= 60.0)
    kband = (z <= -300.0) & (z >= -3000.0)
    amoc[i] = psi_a[np.ix_(jband, kband)].max()
    T = s[0]
    deepT[i] = np.nanmean(np.where(wet3_deep, T, np.nan))
    deepT_atl[i] = np.nanmean(np.where(wet3_deep_atl, T, np.nan))
    T2k[i] = np.nanmean(np.where(wet3_2k, T, np.nan))

yrs = days / 365.0
# linear trend over yr2..end (skip yr0-1 spinup transient)
m = yrs >= 2.0
if m.sum() >= 3:
    slope_g = float(np.polyfit(yrs[m], deepT[m], 1)[0])
    slope_a = float(np.polyfit(yrs[m], deepT_atl[m], 1)[0])
else:
    slope_g = slope_a = float("nan")

m2 = yrs >= 2.0
amoc_mean = float(amoc[m2].mean()) if m2.sum() else float("nan")
amoc_last5 = (float(amoc[yrs >= yrs[-1] - 5.0].mean())
              if n >= 2 else float("nan"))

crit = {
    "a_no_blowup": verdict == "PASS" if npz_ok else None,
    "b_deept_trend": {
        "global_C_per_yr": slope_g,
        "atlantic_C_per_yr": slope_a,
        "baseline_C_per_yr": BASELINE_DEEPT_C_PER_YR,
        "reduction_factor": (BASELINE_DEEPT_C_PER_YR / slope_g
                             if slope_g and slope_g > 0 else None),
        "pass": bool(slope_g < 0.5 * BASELINE_DEEPT_C_PER_YR),
    },
    "c_med_eta": {
        "max_eta_m": max_eta_all,
        "watchdog_m": 15.0,
        "pass": bool(max_eta_all < 15.0) if npz_ok else None,
    },
}

out = {
    "tag": TAG,
    "verdict": verdict,
    "max_u_peak": max_u_peak,
    "n_snaps": n,
    "criteria": crit,
    "amoc": {
        "day0": float(amoc[0]), "final": float(amoc[-1]),
        "mean_yr2_end": amoc_mean, "last5yr_mean": amoc_last5,
    },
    "deepT": {
        "day0": float(deepT[0]), "final": float(deepT[-1]),
        "atl_final": float(deepT_atl[-1]), "T2k_final": float(T2k[-1]),
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
np.savez(OUT_NPZ, days=days, amoc=amoc, deepT=deepT, deepT_atl=deepT_atl,
         T2k=T2k, yrs=yrs)

print(json.dumps(out, indent=2))
print(f"saved {OUT_JSON}")
print(f"saved {OUT_NPZ}")
