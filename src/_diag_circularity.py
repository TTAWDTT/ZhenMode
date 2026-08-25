"""
Non-circularity diagnostic for the bulk-flux A1/A2 re-score.

god's watch-point (inbox 2026-08-25-god-to-pam-gate-passed-a1-circularity-watch):
  T_atm is the zonally-uniform meridional WOA SST profile. The bulk flux
  pulls model SST toward T_atm(y) on a ~30d timescale, so in equilibrium
  <SST_model>(y) ~= T_atm(y) = <WOA_SST>(y) BY CONSTRUCTION. A1 compares
  model zonal-mean SST(y) vs WOA zonal-mean SST(y) -> that correlation is
  dominated by the PRESCRIBED meridional gradient = residual-circular.

This script decomposes the skill into:
  - meridional-gradient part (prescribed via T_atm; residual-circular)
  - zonal-anomaly part (SST - <SST>(y); genuinely PREDICTED by the model:
    zonal fronts, basin-scale anomalies, seasonal phase/amplitude)

and reports the non-circular skill score honestly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import DEFAULT_CONFIG
from grid import make_grid
from woa_data import get_initial_fields

NPZ = r"C:\Users\zhen.luo\HarnessAgents\worktrees\pam-mt5l9102\results\long_run_s2_365d_bulk_norestore.npz"

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float('nan')
    return float(np.mean(a*b) / (sa*sb))

def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))

z = np.load(NPZ, allow_pickle=True)
days = z['days']; T_top = z['T_top']            # (nsnap, nx, ny)
t_end = days[-1]; mask = days >= (t_end - 90.0)
sst_clim = np.mean(T_top[mask], axis=0)         # (nx,ny) model SST climatology

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
T_init, _ = get_initial_fields(grid)
woa_sst = T_init[:, :, 0]                        # (nx,ny) WOA SST = validation target

# --- standard A1/A2 (as bench reports them; for reference) ---
sst_zonal_model = np.mean(sst_clim, axis=0)      # (ny,)
sst_zonal_woa = np.mean(woa_sst, axis=0)         # (ny,)
a1_corr = corr(sst_zonal_model, sst_zonal_woa)
a1_rmse = rmse(sst_zonal_model, sst_zonal_woa)

# A2: >2deg smoothed demeaned pattern (reuse bench's smooth)
from bench_climatology_compare import smooth_2d
sst_model_sm = smooth_2d(sst_clim, grid, deg=2.0)
sst_woa_sm = smooth_2d(woa_sst, grid, deg=2.0)
m = ocean
a_model = sst_model_sm[m] - sst_model_sm[m].mean()
a_woa = sst_woa_sm[m] - sst_woa_sm[m].mean()
a2_corr = corr(a_model, a_woa)
a2_rmse = rmse(sst_model_sm[m], sst_woa_sm[m])

print("=" * 72)
print("NON-CIRCULARITY DIAGNOSTIC — bulk-flux no-restore run (last 90d)")
print("=" * 72)
print("\n[Reference] standard A1/A2 (as bench reports):")
print(f"  A1 zonal-mean SST(y):  corr={a1_corr:.3f}  RMSE={a1_rmse:.3f} C")
print(f"  A2 2D pattern (>2deg): corr={a2_corr:.3f}  RMSE={a2_rmse:.3f} C")

# --- the decomposition ---
# Meridional-gradient component: zonally-uniform part <SST>(y).
# This is what T_atm prescribes -> residual-circular.
model_merid = np.broadcast_to(sst_zonal_model[None, :], sst_clim.shape)   # (nx,ny)
woa_merid = np.broadcast_to(sst_zonal_woa[None, :], woa_sst.shape)

# Zonal anomaly: SST - <SST>(y). Genuinely predicted (zonal structure).
model_zanon = sst_clim - model_merid   # (nx,ny)
woa_zanon = woa_sst - woa_merid

print("\n[Decomposition] meridional-gradient (prescribed via T_atm) vs zonal-anomaly (predicted):")
print(f"  model meridional-gradient std: {sst_zonal_model.std():.3f} C  (prescribed)")
print(f"  model zonal-anomaly std:       {model_zanon[ocean].std():.3f} C  (predicted)")
print(f"  WOA   meridional-gradient std: {sst_zonal_woa.std():.3f} C")
print(f"  WOA   zonal-anomaly std:       {woa_zanon[ocean].std():.3f} C")

# How much of A2's variance is the prescribed meridional gradient vs zonal?
tot_var = np.var(sst_model_sm[m])
merid_var = np.var(np.broadcast_to(np.mean(sst_model_sm, axis=0)[None,:], sst_clim.shape)[m])
zanon_var = tot_var - merid_var
print(f"\n  A2 smoothed-field variance split:")
print(f"    meridional-gradient fraction: {merid_var/tot_var*100:.1f}%  (prescribed -> residual-circular)")
print(f"    zonal-anomaly fraction:       {zanon_var/tot_var*100:.1f}%  (genuinely predicted)")

# --- non-circular skill: zonal-anomaly correlation ---
# Smoothed zonal anomaly (large-scale zonal structure only)
model_zanon_sm = smooth_2d(model_zanon, grid, deg=2.0)
woa_zanon_sm = smooth_2d(woa_zanon, grid, deg=2.0)
za_model = model_zanon_sm[m] - model_zanon_sm[m].mean()
za_woa = woa_zanon_sm[m] - woa_zanon_sm[m].mean()
za_corr = corr(za_model, za_woa)
za_rmse = rmse(model_zanon_sm[m], woa_zanon_sm[m])

print("\n[NON-CIRCULAR SKILL] zonal-anomaly SST(x,y) - <SST>(y), >2deg smoothed:")
print(f"  zonal-anomaly corr = {za_corr:.3f}")
print(f"  zonal-anomaly RMSE = {za_rmse:.3f} C")
print(f"  (this is the genuinely PREDICTED large-scale zonal structure)")

# --- seasonal-cycle phase/amplitude (also genuinely predicted) ---
# T_atm is TIME-INDEPENDENT (annual-mean WOA profile), so any seasonal
# cycle in model SST is genuinely produced by the model + seasonal wind.
sst_series = T_top[mask]                         # (nsnap, nx, ny)
domain_mean_series = np.array([np.mean(s[ocean]) for s in sst_series])
print("\n[SEASONAL CYCLE] (T_atm is annual-mean -> any cycle is model-produced):")
print(f"  domain-mean SST over last 90d: min={domain_mean_series.min():.3f} "
      f"max={domain_mean_series.max():.3f} range={float(np.ptp(domain_mean_series)):.3f} C")
print(f"  (nonzero range = model produces a seasonal cycle not present in T_atm)")

print("\n" + "=" * 72)
print("HONEST VERDICT:")
print(f"  A1/A2 standard corr ({a1_corr:.3f}/{a2_corr:.3f}) is DOMINATED by the")
print(f"  prescribed meridional gradient (T_atm) -> residual-circular.")
print(f"  Genuinely predicted zonal-anomaly corr = {za_corr:.3f}.")
print(f"  Report A1/A2 as 'forced-equilibrium skill with residual meridional")
print(f"  circularity; zonal structure corr = {za_corr:.3f}'.")
print("=" * 72)
