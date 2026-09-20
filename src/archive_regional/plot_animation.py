"""
Time-lapse animation of SST + surface speed from a saved run snapshot.

Reads results/snapshot.npz (written by plot_fields.py) and renders a
side-by-side animation of sea-surface temperature and surface speed
through time, saved as results/animation.gif.

Run:  python src/plot_animation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def main():
    snap_path = os.path.join(OUT_DIR, 'snapshot.npz')
    if not os.path.exists(snap_path):
        print(f"Missing {snap_path}. Run plot_fields.py first.")
        return 1
    d = np.load(snap_path)
    T = d['T']          # (n, nx, ny, nz)
    u = d['u']          # (n, nx, ny, nz)
    times = d['times']
    lon, lat = d['lon'], d['lat']
    mask = d['mask']
    n = T.shape[0]

    T_sfc = T[:, :, :, 0]                 # (n, nx, ny)
    speed = np.hypot(u[:, :, :, 0], u[:, :, :, 1]) if u.shape[-1] >= 2 \
        else np.abs(u[:, :, :, 0])

    Tmin, Tmax = float(T_sfc.min()), float(T_sfc.max())
    spd_max = float(speed.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ocean Solver — SST & Surface Speed Evolution', fontsize=14,
                 fontweight='bold')

    axT = axes[0]
    axS = axes[1]

    def _masked(f):
        return np.ma.masked_where(~mask, f)

    imT = axT.pcolormesh(lon, lat, _masked(T_sfc[0]), cmap='RdYlBu_r',
                         shading='auto', vmin=Tmin, vmax=Tmax)
    imS = axS.pcolormesh(lon, lat, _masked(speed[0]), cmap='magma',
                         shading='auto', vmin=0, vmax=spd_max)
    for ax in axes:
        ax.set_xlabel('Longitude (E)')
        ax.set_ylabel('Latitude (N)')
    axT.set_title('Sea-surface temperature (°C)')
    axS.set_title(r'Surface speed |u| (m/s)')
    fig.colorbar(imT, ax=axT, fraction=0.046, pad=0.03)
    fig.colorbar(imS, ax=axS, fraction=0.046, pad=0.03)

    time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=12)

    def update(k):
        imT.set_array(_masked(T_sfc[k]).ravel())
        imS.set_array(_masked(speed[k]).ravel())
        time_text.set_text(f't = {times[k] / 3600.0:.2f} h'
                           f'   ({k + 1}/{n})')
        return imT, imS, time_text

    anim = FuncAnimation(fig, update, frames=n, interval=450, blit=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'animation.gif')
    anim.save(out, writer='pillow', dpi=110)
    print(f"Saved {out} ({n} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
