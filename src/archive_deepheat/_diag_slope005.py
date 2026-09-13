"""Diagnose WHY slope005 wins: compare AMOC psi sections + Atlantic T(z)
profiles + deepT/OHC time series across era-2 tags, using the local npz files
in _ana_tmp/ (pulled earlier). No new runs — pure local analysis.
"""
import json
import numpy as np

TAGS = ["relax15", "relax60", "relax90", "slope005", "slope007", "from30"]

print("=" * 100)
print("LATE-ERA TRENDS (yr 170-194.7) + FINAL STATE  — from _ana_tmp npz")
print("=" * 100)
hdr = f"{'tag':<10} {'deepT C/yr':>11} {'OHC ZJ/yr':>11} {'AMOC30-60':>10} {'AMOC26N':>9} {'ratio':>7} {'aabw':>8} {'pmoc':>8} {'drake':>8}"
print(hdr)

for t in TAGS:
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{t}_analysis.npz")
    yrs = d["yrs"]
    sel = yrs >= 170.0
    if sel.sum() < 5:
        sel = yrs >= yrs[-1] - 20.0
    def tr(k):
        return float(np.polyfit(yrs[sel], d[k][sel], 1)[0])
    print(f"{t:<10} {tr('deepT'):>11.5f} {tr('ohc'):>11.2f} "
          f"{d['amoc'][-1]:>10.2f} {d['amoc_26n'][-1]:>9.2f} "
          f"{d['amoc_26n'][-1]/d['amoc'][-1]:>7.3f} "
          f"{d['aabw'][-1]:>8.2f} {d['pmoc'][-1]:>8.2f} {d['drake'][-1]:>8.2f}")

print()
print("=" * 100)
print("WHY does slope005 win? — stratification & deep-T state comparison")
print("=" * 100)
for t in TAGS:
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{t}_analysis.npz")
    z = d["prof_z"]
    Tg = d["prof_T_glb"]
    deep = z <= -1000
    print(f"\n{t}:  global T(z) at 0/300/1000/2000/4000 m:",
          " ".join(f"{Tg[np.argmin(np.abs(z-qq))]:6.2f}" for qq in [0, -300, -1000, -2000, -4000]))
    print(f"{'':10s} deepT(z<=1000m mean)={d['deepT'][-1]:6.3f} C   "
          f"OHC={d['ohc'][-1]:9.2f} ZJ   strat(0-100m)={d['strat'][-1]:5.3f} C/100m   "
          f"ke={d['ke'][-1]:8.1f} EJ   maxu={d['maxu'][-1]:5.2f}")

print()
print("=" * 100)
print("AMOC psi(y,z) structure at final snap — 26N vs 45N cell shape (Sv)")
print("=" * 100)
for t in ["relax15", "slope005", "slope007"]:
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{t}_analysis.npz")
    psi = d["psi_flat"].reshape(tuple(d["psi_shape"]))
    plat = d["prof_lat"]
    pz = d["prof_z"]
    i45 = int(np.argmin(np.abs(plat - 45.0)))
    i26 = int(np.argmin(np.abs(plat - 26.5)))
    k1000 = int(np.argmin(np.abs(pz + 1000.0)))
    k3000 = int(np.argmin(np.abs(pz + 3000.0)))
    print(f"\n{t}:  psi(lat=45N) z-profile 0..-4000m:",
          " ".join(f"{psi[i45, k]:6.1f}" for k in range(0, pz.size, 2)))
    print(f"{'':10s} psi(lat=26N) z-profile 0..-4000m:",
          " ".join(f"{psi[i26, k]:6.1f}" for k in range(0, pz.size, 2)))
    print(f"{'':10s} max psi@45N={psi[i45].max():6.1f} Sv   "
          f"max psi@26N={psi[i26].max():6.1f} Sv   "
          f"min psi@60S-30S={psi[(plat>=-60)&(plat<=-30)].min():6.1f} Sv")
