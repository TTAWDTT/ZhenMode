"""Where does the slope cap bite? Compare the relax15 (no cap) vs slope005
Atlantic T(z) profiles and the psi section shape in full, to characterize the
mechanism. Local npz only.
"""
import numpy as np

tags = ["relax15", "slope007", "slope005"]
z = None
prof = {}
psi = {}
lat = None
for t in tags:
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{t}_analysis.npz")
    z = d["prof_z"]
    prof[t] = (d["prof_T_glb"], d["prof_T_atl"])
    psi[t] = d["psi_flat"].reshape(tuple(d["psi_shape"]))
    lat = d["prof_lat"]

print("z      " + "".join(f"{t:>12s}" for t in tags) + "   (global T;  relax15 - slope005)")
for k in range(z.size):
    row = f"{z[k]:7.0f} "
    row += "".join(f"{prof[t][0][k]:12.3f}" for t in tags)
    row += f"{prof['relax15'][0][k] - prof['slope005'][0][k]:12.3f}"
    print(row)

print("\nAtlantic T(z):")
print("z      " + "".join(f"{t:>12s}" for t in tags))
for k in range(z.size):
    row = f"{z[k]:7.0f} "
    row += "".join(f"{prof[t][1][k]:12.3f}" for t in tags)
    print(row)

print("\npsi(lat) at z=-1500m (row across lat, every 5th point):")
k15 = int(np.argmin(np.abs(z + 1500.0)))
for i in range(0, lat.size, 5):
    print(f"{lat[i]:7.1f} " + "".join(f"{psi[t][i, k15]:12.1f}" for t in tags))
