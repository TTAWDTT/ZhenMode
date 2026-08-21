"""
Plot benchmark results: speedup, wave speed accuracy, dt convergence.

Generates a 2x2 figure saved to results/bench_plot.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Measured data (from bench_compare.py run 2026-08-21) ──

# 1. Timing
ms_numpy = 280.7
ms_jax = 74.6
speedup = ms_numpy / ms_jax  # 3.76x
sim_wall_np = 1069
sim_wall_jax = 4019

# 2. Wave speed accuracy
c_theory = 198.09
c_measured = 186.46
wave_err_pct = abs(c_measured - c_theory) / c_theory * 100  # 5.87%
mass_drift = 7.11e-15

# 3. dt convergence (reference dt=10s, t=1800s)
dt_vals = np.array([30.0, 60.0, 120.0, 300.0])
dT_vals = np.array([2.373213e-11, 5.972645e-11, 1.343476e-10, 3.793925e-10])
du_vals = np.array([1.464241e-07, 3.688526e-07, 8.241563e-07, 2.281947e-06])
deta_vals = np.array([7.734370e-07, 1.858775e-06, 3.794026e-06, 8.248205e-06])

# Empirical convergence rate
rate = np.log(dT_vals[-1] / dT_vals[0]) / np.log(dt_vals[-1] / dt_vals[0])  # ~1.20

# ── Plot ──

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Ocean Solver Benchmark Results\n'
             f'Grid: 128×128×14  |  CPU (x86-64, x64)  |  JAX 0.11.1',
             fontsize=14, fontweight='bold', y=0.98)
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# ── (a) Speed: numpy vs JAX bar chart ──
ax1 = fig.add_subplot(gs[0, 0])
labels = ['numpy', 'JAX (JIT)']
times = [ms_numpy, ms_jax]
colors = ['#4C72B0', '#55A868']
bars = ax1.bar(labels, times, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
ax1.set_ylabel('ms / step', fontsize=12)
ax1.set_title('(a) Per-step Wall Time', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(times) * 1.25)
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.03,
             f'{t:.1f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Speedup annotation
ax1.annotate(f'{speedup:.2f}× speedup',
             xy=(0.5, 0.85), xycoords='axes fraction', ha='center',
             fontsize=14, fontweight='bold', color='#C44E52',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#C44E52'))
ax1.grid(axis='y', alpha=0.3)

# ── (b) Wave speed: measured vs theory ──
ax2 = fig.add_subplot(gs[0, 1])
labels2 = ['Theory\n$\\sqrt{gH}$', 'Measured']
speeds = [c_theory, c_measured]
colors2 = ['#4C72B0', '#DD8452']
bars2 = ax2.bar(labels2, speeds, color=colors2, width=0.5, edgecolor='white', linewidth=1.2)
ax2.set_ylabel('Wave speed (m/s)', fontsize=12)
ax2.set_title('(b) Surface Gravity Wave Speed', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(speeds) * 1.2)
for bar, s in zip(bars2, speeds):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(speeds) * 0.03,
             f'{s:.2f} m/s', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.annotate(f'Error: {wave_err_pct:.2f}%\nMass drift: {mass_drift:.2e}',
             xy=(0.5, 0.78), xycoords='axes fraction', ha='center',
             fontsize=11, fontweight='bold', color='#C44E52',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#C44E52'))
ax2.grid(axis='y', alpha=0.3)

# ── (c) dt convergence: error vs dt (log-log) ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.loglog(dt_vals, dT_vals, 'o-', color='#4C72B0', linewidth=2, markersize=8,
           label='$\\max|\\Delta T|$')
ax3.loglog(dt_vals, du_vals, 's--', color='#55A868', linewidth=2, markersize=8,
           label='$\\max|\\Delta u|$')
ax3.loglog(dt_vals, deta_vals, '^:', color='#DD8452', linewidth=2, markersize=8,
           label='$\\max|\\Delta \\eta|$')

# Reference line: O(dt^2)
dt_fine = np.linspace(dt_vals[0], dt_vals[-1], 50)
ref_line = dT_vals[0] * (dt_fine / dt_vals[0]) ** 2
ax3.loglog(dt_fine, ref_line, 'k--', alpha=0.3, linewidth=1.5, label='$O(\\Delta t^2)$ reference')

ax3.set_xlabel('$\\Delta t$ (s)', fontsize=12)
ax3.set_ylabel('Max error vs $\\Delta t=10$s reference', fontsize=12)
ax3.set_title('(c) Time-step Convergence (t=1800s)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, which='both', alpha=0.3)
ax3.annotate(f'Empirical rate: $O(\\Delta t^{{{rate:.2f}}})$',
             xy=(0.55, 0.15), xycoords='axes fraction',
             fontsize=11, fontweight='bold', color='#C44E52',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#C44E52'))

# ── (d) Summary text panel ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
summary_text = (
    "Benchmark Summary\n"
    "===============================\n\n"
    f"  Grid: 128 x 128 x 14 (~230k pts)\n\n"
    f"  -- Speed --\n"
    f"  numpy:     {ms_numpy:.1f} ms/step\n"
    f"  JAX:       {ms_jax:.1f} ms/step\n"
    f"  Speedup:   {speedup:.2f}x\n"
    f"  JIT compile: ~0.92 s (one-time)\n\n"
    f"  -- Accuracy --\n"
    f"  Wave speed error: {wave_err_pct:.2f}%\n"
    f"  Mass drift:       {mass_drift:.2e}\n"
    f"  Conv. rate: O(dt^{rate:.2f})\n\n"
    f"  -- Error Cost (dt=300 vs dt=10) --\n"
    f"  max|dT|:   {dT_vals[-1]:.2e}\n"
    f"  max|du|:   {du_vals[-1]:.2e}\n"
    f"  max|deta|: {deta_vals[-1]:.2e}"
)
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', edgecolor='#CCCCCC'))

# Save
os.makedirs('results', exist_ok=True)
out_path = 'results/bench_plot.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close(fig)
