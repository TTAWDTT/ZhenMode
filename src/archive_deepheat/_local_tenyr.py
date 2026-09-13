import os, glob, numpy as np
d = "results/global_tenyr_ms_gm_3d"
snaps = sorted(glob.glob(os.path.join(d, "*.npy")))
print("n snaps", len(snaps))
print("first/last:", snaps[0], snaps[-1])
a = np.load(snaps[0])
print("shape", a.shape, "dtype", a.dtype)
for nm, i in (("T",0),("u",1),("v",2),("S",3)):
    x = a[i]
    print(f"  {nm}: min {np.nanmin(x):.4f} max {np.nanmax(x):.4f} mean {np.nanmean(x):.4f}")
z = np.load("results/ckpt_tenyr_ms_gm.npz")
print("ckpt keys", list(z.keys()))
for k in z.keys():
    try:
        print("  ", k, np.asarray(z[k]).shape, np.asarray(z[k]).dtype)
    except Exception as e:
        print("  ", k, z[k])
