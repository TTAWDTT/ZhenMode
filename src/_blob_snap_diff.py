import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 28 snaps d0-d135 at 5d cadence: k = day/5
for k, day in [(22,110),(23,115),(24,120),(25,125),(26,130)]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')  # (4,360,120,14) T,u,v,S
    T, u, v, S = arr
    box = (slice(295,310), slice(65,80))
    Tb, ub, vb, Sb = T[box], u[box], v[box], S[box]
    print(f'day {day}: box T=[{Tb.min():.2f},{Tb.max():.2f}] u=[{ub.min():.3f},{ub.max():.3f}] '
          f'v=[{vb.min():.3f},{vb.max():.3f}] S=[{Sb.min():.3f},{Sb.max():.3f}]')
    iu = np.unravel_index(np.argmax(np.abs(ub)), ub.shape)
    print(f'   max|u| at abs(i={295+iu[0]}, j={65+iu[1]}, k={iu[2]}) val={ub[iu]:+.3f}')
    u0 = ub[:,:,0]
    iu0 = np.unravel_index(np.argmax(np.abs(u0)), u0.shape)
    print(f'   surf max|u| at abs(i={295+iu0[0]}, j={65+iu0[1]}) val={u0[iu0]:+.3f}')
