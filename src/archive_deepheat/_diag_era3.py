"""Era-3 verdict: rank slope003/004 vs era-1/2 chain. slope003 blew up at
day 48545 (yr 133) so compare era-matched windows: yr 100-133 for all tags
(era-2 tags ran to 194.7 yr; restrict the era-2 side to the same window).
"""
import numpy as np

Y0, Y1 = 100.0, 133.0   # common window: slope003 dies at 133
Y0b, Y1b = 161.0, 190.0 # late-era window for slope004/era-2 comparison

def series(tag):
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{tag}_analysis.npz")
    return d

print("=" * 108)
print(f"WINDOW yr {Y0}-{Y1} (slope003 whole life after 100):  deepT C/yr | OHC ZJ/yr | AMOC | 26N | ratio")
print("=" * 108)
for t in ["relax15", "slope007", "slope005", "slope004", "slope003"]:
    d = series(t)
    yrs = d["yrs"]
    sel = (yrs >= Y0) & (yrs <= min(Y1, yrs[-1]))
    if sel.sum() < 4:
        continue
    dt = np.polyfit(yrs[sel], d["deepT"][sel], 1)[0]
    do = np.polyfit(yrs[sel], d["ohc"][sel], 1)[0]
    print(f"{t:<10} {dt:>10.5f} {do:>11.2f} {d['amoc'][-1]:>8.2f} {d['amoc_26n'][-1]:>7.2f} "
          f"{d['amoc_26n'][-1]/d['amoc'][-1]:>7.3f}")

print()
print("=" * 108)
print(f"WINDOW yr {Y0b}-{Y1b} (late era; slope003 excluded - dead):")
print("=" * 108)
for t in ["relax15", "slope007", "slope005", "slope004"]:
    d = series(t)
    yrs = d["yrs"]
    sel = (yrs >= Y0b) & (yrs <= min(Y1b, yrs[-1]))
    if sel.sum() < 4:
        continue
    dt = np.polyfit(yrs[sel], d["deepT"][sel], 1)[0]
    do = np.polyfit(yrs[sel], d["ohc"][sel], 1)[0]
    print(f"{t:<10} {dt:>10.5f} {do:>11.2f} {d['amoc'][-1]:>8.2f} {d['amoc_26n'][-1]:>7.2f} "
          f"{d['amoc_26n'][-1]/d['amoc'][-1]:>7.3f}")

print()
# blowup forensics for slope003: eta & KE path in the last 10 yr
d = series("slope003")
yrs = d["yrs"]
print("slope003 final decade (maxu, ke, drake):")
for y in range(120, 134, 2):
    i = int(np.argmin(np.abs(yrs - y)))
    print(f"  yr{yrs[i]:6.1f}  maxu={d['maxu'][i]:5.2f}  ke={d['ke'][i]:9.1f} EJ  drake={d['drake'][i]:6.2f}  amoc={d['amoc'][i]:6.1f}")
print("\nslope004 final decade:")
for y in range(180, 190, 2):
    i = int(np.argmin(np.abs(yrs - y)))
    print(f"  yr{yrs[i]:6.1f}  maxu={d['maxu'][i]:5.2f}  ke={d['ke'][i]:9.1f} EJ  drake={d['drake'][i]:6.2f}  amoc={d['amoc'][i]:6.1f}  eta_sshstd~n/a")
