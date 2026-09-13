import os, psutil
print("logical cores:", os.cpu_count())
print("physical cores:", psutil.cpu_count(logical=False))
print("total RAM GB: %.1f" % (psutil.virtual_memory().total / 1e9))
print("avail RAM GB: %.1f" % (psutil.virtual_memory().available / 1e9))
print("cpu load 1/5/15:", os.getloadavg() if hasattr(os, "getloadavg") else "n/a")
print("cpu percent:", psutil.cpu_percent(interval=1.0))
