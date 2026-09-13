import os, glob
for pat in ("results/global_*_3d", "results/*spinG*", "results/*spinH*", "results/ckpt_*",
            "_ana_tmp", "results/*.npz"):
    m = sorted(glob.glob(pat))
    print(f"{pat}: {len(m)}")
    for x in m[:8]:
        print("    ", x, "(dir)" if os.path.isdir(x) else os.path.getsize(x))
