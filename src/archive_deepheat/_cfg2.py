import numpy as np
c = str(np.load("results/global_tenyr_ms_gm.npz")["config"])
print(c.replace(", ", ",\n"))
