import numpy as np
d = np.load("results/global_tenyr_ms_gm.npz", allow_pickle=True)
print("keys:", list(d.keys()))
c = d["config"]
try:
    c = c.item()
except Exception:
    pass
if isinstance(c, dict):
    for k, v in c.items():
        print("  %-24s %r" % (k, v))
else:
    print(type(c))
    print(str(c)[:4000])
