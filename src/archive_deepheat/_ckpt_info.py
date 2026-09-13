import numpy as np
d = np.load('results/ckpt_tenyr_ms_gm.npz')
print("cur_step:", d['cur_step'], "n_3d_snaps:", d['n_3d_snaps'])
print("days =", d['cur_step'] * 3600.0 / 86400.0)
