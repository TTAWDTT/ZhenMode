"""Is wet_mask_z column-uniform in k?  If yes, the per-level gate in
_divergence_h IS column-uniform and the k-sum should pull through -- yet
_algt Link 3 showed Fz[0] rms 8.4e-06 under masked m.  So either the mask
varies by level, or something else (dx/dy per-level) breaks the k-sum."""
import numpy as np, sys
sys.path.insert(0, 'src')
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import jax_solver_global as JS
import jax.numpy as jnp

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
p_ = PhysicsConfig()
_, _, _, p, _ = JS.make_solver_global(g, p_, 3600.0, return_params=True)

M = np.asarray(p.wet_mask_z)          # (nx,ny,nz)
print("wet_mask_z shape", M.shape)
# column-uniform?  compare each column against its own first level
m0 = M[:, :, :1]
diff = np.abs(M - m0)
print("max |m_k - m_0| over all cells/levels:", diff.max())
print("columns with any level differing from level 0:", int((diff.max(axis=2) > 0).sum()),
      "of", M.shape[0]*M.shape[1])

# how many levels does each column have?
nlev = M.sum(axis=2)
print("wet levels per column: min %d max %d   unique %s"
      % (nlev.min(), nlev.max(), np.unique(nlev)[:20]))
full = int((nlev == M.shape[2]).sum())
print("full-depth columns:", full, "of", nlev.size)

# A column is 'partial' if wet at surface but not at depth.  The gate matters
# only where two adjacent columns have DIFFERENT wet counts (their shared face
# is gated at the levels kept by only one).
# Count gated faces per level.
wm = M.astype(bool)
uf = wm * np.roll(wm, -1, axis=0)          # zonal face open
lx = ~(uf | (np.roll(wm, -1, axis=0) & ~wm) | (wm & ~np.roll(wm, -1, axis=0)))
# simpler: face exists (both cells in domain) but is closed
in_dom_x = np.roll(np.ones_like(wm, bool), 0, 0)  # always true (periodic)
closed_x = ~uf
print("per-level CLOSED zonal faces (k=0..%d):" % (M.shape[2]-1),
      closed_x.sum(axis=(0, 1)))
print("per-level wet cells (k=0..%d):", wm.sum(axis=(0, 1)))

# Does the closure of a face vary by level within the same column pair?
# For each (i,j) face, is closed_x[i,j,:] constant in k?
cx = closed_x
vary = (cx.max(axis=2) != cx.min(axis=2))
print("zonal faces whose open/closed status VARIES by level:", int(vary.sum()),
      "of", vary.size)
