"""
Field visualization — intuitive maps + cross-sections + time series.

Runs a real WOA-seeded integration (wind + heat forcing) and produces
a multi-panel figure saved to results/fields.png, plus a numpy snapshot
snapshot.npz used by plot_animation.py.

This is the "直观的数据/图产出" (intuitive plots) deliverable:
  (a) Sea-surface temperature map
  (b) Temperature depth cross-section (zonal slice through domain mid-lat)
  (c) Salinity depth cross-section
  (d) Sea-surface height (free surface) map
  (e) Surface kinetic energy / |u| map
  (f) Density anomaly map, vertical velocity, bathymetry, time series
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, JaxState
from woa_data import get_initial_fields
from forcing import wind_stress_gyre, heat_flux_meridional

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics

    DT = 300.0
    N_STEPS = 24          # ~2 h — enough to show the flow evolving
    EVERY = 6             # snapshot every 6 steps (~30 min)

    # Real-world prescribed forcing
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    print(f"Grid {grid.nx}x{grid.ny}x{grid.nz}, dt={DT}s, "
          f"forcing=wind+heat, steps={N_STEPS} ({N_STEPS*DT/3600:.1f}h)")
    T_init, S_init = get_initial_fields(grid)
    print(f"WOA T range: [{T_init.min():.2f}, {T_init.max():.2f}] C")
    print(f"WOA S range: [{S_init.min():.2f}, {S_init.max():.2f}] PSU")

    step_fn, init_state, diagnostics = make_solver(
        grid, physics, DT, forcing=forcing)
    state = init_state(T_init=T_init, S_init=S_init)

    # Snapshot accumulator
    T_series, u_series, eta_series = [], [], []
    times = []

    for i in range(1, N_STEPS + 1):
        t0 = time.time()
        state = step_fn(state)
        if i % EVERY == 0 or i == 1:
            T_series.append(np.asarray(state.T))
            u_series.append(np.asarray(state.u))
            eta_series.append(np.asarray(state.eta))
            times.append(i * DT)
        dt_ms = (time.time() - t0) * 1000
        print(f"  step {i:3d}  T=[{float(state.T.min()):6.2f}, "
              f"{float(state.T.max()):6.2f}]  |u|max={float(jnp.max(jnp.abs(state.u))):8.3f} "
              f"({dt_ms:.0f} ms)")
    final = state

    # Diagnostics (rho, pressure, w) at final state
    rho, pressure, w = diagnostics(final)

    # ---- Figure ----
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle('Ocean Solver — Stabilized Fields (WOA2023 seed, wind+heat)',
                 fontsize=15, fontweight='bold', y=0.995)
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.4)

    lon, lat = grid.lon, grid.lat
    z = grid.z                                 # depths (m), negative down
    mask = grid.ocean_mask

    Tf = np.asarray(final.T)      # (nx,ny,nz)
    Sf = np.asarray(final.S)
    uf = np.asarray(final.u)
    vf = np.asarray(final.v)
    etaf = np.asarray(final.eta)

    def _map(ax, field, title, cmap, vmin=None, vmax=None):
        im = ax.pcolormesh(lon, lat, np.ma.masked_where(~mask, field),
                           cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (E)'); ax.set_ylabel('Latitude (N)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        return im

    # (a) SST map
    ax = fig.add_subplot(gs[0, 0])
    _map(ax, Tf[:, :, 0], '(a) SST (°C)', 'RdYlBu_r')

    # (b) T cross-section through mid-lat
    ax = fig.add_subplot(gs[0, 1])
    jmid = grid.ny // 2
    Tsec = Tf[:, jmid, :].T
    im = ax.pcolormesh(lon, z, Tsec, cmap='RdYlBu_r', shading='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Longitude (E)'); ax.set_ylabel('Depth (m)')
    ax.set_title('(b) T cross-section @ mid-lat (°C)', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # (c) S cross-section
    ax = fig.add_subplot(gs[0, 2])
    Ssec = Sf[:, jmid, :].T
    im = ax.pcolormesh(lon, z, Ssec, cmap='viridis', shading='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Longitude (E)'); ax.set_ylabel('Depth (m)')
    ax.set_title('(c) Salinity cross-section (PSU)', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # (d) SSH map
    ax = fig.add_subplot(gs[0, 3])
    _map(ax, etaf, '(d) Sea-surface height (m)', 'RdBu_r')

    # (e) Surface speed |u|
    speed = np.hypot(uf[:, :, 0], vf[:, :, 0])
    ax = fig.add_subplot(gs[1, 0])
    _map(ax, speed, r'(e) Surface speed |u| (m/s)', 'magma')

    # (f) Surface density anomaly
    ax = fig.add_subplot(gs[1, 1])
    rho_map = np.asarray(rho)[:, :, 0] - 1025.0
    _map(ax, rho_map, r'(f) Density anomaly $\rho-1025$ (kg/m³)', 'viridis')

    # (g) Vertical velocity at mid-depth
    ax = fig.add_subplot(gs[1, 2])
    wmap = np.asarray(w)[:, :, grid.nz // 2]
    _map(ax, wmap, r'(g) Vertical velocity w @ mid-depth (m/s)', 'RdBu_r')

    # (h) T/|u| envelope time series
    ax = fig.add_subplot(gs[1, 3])
    Tmin = [float(T.min()) for T in T_series]
    Tmax = [float(T.max()) for T in T_series]
    umax = [float(np.max(np.abs(u))) for u in u_series]
    tt = np.array(times) / 3600.0
    ax.fill_between(tt, Tmin, Tmax, alpha=0.3, color='#C44E52', label='T range')
    ax.plot(tt, Tmax, '-o', color='#C44E52', ms=4, label='T max')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('T range (°C)', color='#C44E52')
    ax.tick_params(axis='y', labelcolor='#C44E52')
    ax.set_title('(h) Envelope time series', fontsize=11, fontweight='bold')
    axb = ax.twinx()
    axb.plot(tt, umax, '-s', color='#4C72B0', ms=4, label='|u| max')
    axb.set_ylabel('max |u| (m/s)', color='#4C72B0')
    axb.tick_params(axis='y', labelcolor='#4C72B0')

    # (i) Bathymetry / mask map
    ax = fig.add_subplot(gs[2, 0])
    _map(ax, np.ma.masked_where(~mask, grid.depth), '(i) Bathymetry depth (m)', 'Blues_r')

    # (j) Mean KE + energy time series
    ax = fig.add_subplot(gs[2, 1:3])
    KE = [0.5 * float(np.mean(u ** 2)) for u in u_series]
    ax.plot(tt, KE, '-o', color='#55A868', ms=4)
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Mean KE (m²/s²)')
    ax.set_title('(j) Domain-mean kinetic energy', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (k) summary text panel
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    ax.text(0.02, 0.98,
            f'Grid {grid.nx}×{grid.ny}×{grid.nz}\n'
            f'dt={DT:.0f}s, steps={N_STEPS}\n'
            f'final T=[{Tf.min():.2f},{Tf.max():.2f}]°C\n'
            f'final |u|max={float(np.max(np.abs(uf))):.2f} m/s\n'
            'WOA2023 seed + wind/heat\n'
            'stabilized: Laplacian+biharmonic',
            va='top', ha='left', fontsize=11, family='monospace',
            transform=ax.transAxes)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'fields.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved {out}")

    snap = os.path.join(OUT_DIR, 'snapshot.npz')
    np.savez(snap, T=np.stack(T_series), u=np.stack(u_series),
             eta=np.stack(eta_series), times=np.array(times),
             lon=lon, lat=lat, z=z, mask=mask)
    print(f"Saved {snap} ({len(times)} snapshots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
