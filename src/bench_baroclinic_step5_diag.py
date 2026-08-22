"""Step 5 diagnostic: WHY does the modeled mesoscale SLA pattern not match?

Step 5 (bench_baroclinic_step5.py) returned a negative band-passed correlation
(-0.118 at 10d spin-up). A single number does not tell us WHERE the mismatch
lives. This diagnostic decomposes the mismatch by horizontal scale so we can
answer, with evidence:

  Q1  Are model and observed SLA even in the SAME spectral band?
      -> 1D radial-wavenumber power spectra, model SSH vs observed SLA.
  Q2  Is there any scale band where model correlates positively with obs?
      -> band-passed correlation swept over L_LO (retention width) and over
         a coarse band set; report the best-correlating band.
  Q3  What is the spectral nature of the mismatch?
      -> cross-spectral squared coherence and phase between model & obs,
         revealing whether the anti-correlation is a phase problem (eddies at
         wrong location) or an amplitude/scale problem.

All inputs are post-processing of the ALREADY-computed Step 5 model field
(results/step5_eta_10d.npz) and the cached observed SLA — no new spin-up.

Usage:
    python src/bench_baroclinic_step5_diag.py \
        --eta results/step5_eta_10d.npz --month 2023-01
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_CONFIG
from grid import make_grid
from bench_baroclinic_step5 import band_pass, load_obs_sla, L_HI, L_LO

DEG_PER_KM = 1.0 / 111.0


def radial_spectrum(field, dx):
    """1D radial-averaged power spectrum of a 2D field (periodic FFT).

    Returns (k_cyc [/km], P(k)) where P is summed power at each |k|, and the
    field is pre-tapered + de-meaned to reduce leakage. k in cycles per km so
    wavelength (km) = 1/k.
    """
    f = np.nan_to_num(field, nan=0.0).astype(np.float64)
    f = f - f[np.isfinite(field)].mean() if np.isfinite(field).any() \
        else f - f.mean()
    nx, ny = f.shape
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    ft = f * wx[:, None] * wy[None, :]
    F = np.fft.fftshift(np.fft.fft2(ft))
    PSD = np.abs(F) ** 2
    # wavenumbers cycles per unit: use minimum grid spacing for k normalization
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))   # [1/m]
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)                    # [1/m]
    k_cyc = K * 1e3                                    # [1/km] (1/m * 1000 m/km)
    PSD_binned = PSD * wx[:, None] * wy[None, :]       # approximate taper weight
    # radial bins (log-spaced to cover 1..~50 km^-1 => 0.0005..0.05)
    k_edges = np.geomspace(k_cyc[k_cyc > 0].min(), k_cyc.max(), 40)
    k_c = 0.5 * (k_edges[:-1] + k_edges[1:])
    P = np.zeros_like(k_c)
    for i in range(len(k_c)):
        m = (k_cyc >= k_edges[i]) & (k_cyc < k_edges[i + 1])
        P[i] = PSD_binned[m].sum()
    return k_c, P


def corr_band(a, b, mask, l_lo, l_hi, dx_km, land):
    """Band-pass corr between fields in [l_lo, l_hi] km."""
    bp_a = band_pass(a, l_hi, l_lo, dx_km, land)
    bp_b = band_pass(np.nan_to_num(b, nan=0.0), l_hi, l_lo, dx_km, land)
    v = mask & np.isfinite(bp_a) & np.isfinite(bp_b) & np.isfinite(b)
    a_ = bp_a[v]; b_ = bp_b[v]
    a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
    den = np.sqrt((a_ * a_).mean() * (b_ * b_).mean())
    r = float((a_ * b_).mean() / den) if den > 0 else float('nan')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", default="results/step5_eta_10d.npz")
    ap.add_argument("--month", default="2023-01")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    land = ~ocean
    dx_km = grid.dx / 1000.0
    dx_m = grid.dx

    if not os.path.exists(args.eta):
        print(f"ERROR: model field cache {args.eta} not found. Run "
              f"bench_baroclinic_step5.py --save-eta {args.eta} first.")
        sys.exit(1)
    model_eta = np.load(args.eta)['eta'].astype(np.float64)

    cache = f"data/t3_sla_{args.month}.npz"
    sla_obs = load_obs_sla(cache, grid)

    print("=" * 68)
    print("STEP 5 DIAGNOSTIC — WHY THE MESOSCALE PATTERN DOESN'T MATCH")
    print("=" * 68)
    print(f"model SSH ocean std = {np.std(model_eta[ocean]):.4f} m")
    print(f"obs   SLA ocean std = {np.nanstd(sla_obs[ocean]):.4f} m")

    # ── Q1: radial spectra (model vs obs) ──
    kmod, Pmod = radial_spectrum(model_eta, dx_m)
    kobs, Pobs = radial_spectrum(sla_obs, dx_m)
    # normalize spectra to unit integral for slope comparison
    Pmod_n = Pmod / np.sum(Pmod)
    Pobs_n = Pobs / np.sum(Pobs)
    print("\n  Q1  Radial power spectra (model SSH vs obs SLA, normalized):")
    print("      wl(km)   k(/km)     model        obs      ratio")
    for i in range(0, len(kmod)):
        wl = 1.0 / kmod[i] if kmod[i] > 0 else np.inf
        if 20 < wl < 900:
            print(f"      {wl:7.1f}  {kmod[i]:.4e}  {Pmod_n[i]:.3e}  "
                  f"{Pobs_n[i]:.3e}  {Pmod_n[i]/max(Pobs_n[i],1e-30):6.2f}")

    # ── Q2: correlation sweep by band ──
    print("\n  Q2  Band-passed model-obs correlation vs retention width:")
    valid = ocean & np.isfinite(sla_obs)
    bands = [
        (50.0, 600.0,   "meso  50-600 "),
        (80.0, 600.0,   "meso  80-600 "),
        (100.0, 600.0,  "meso 100-600 "),
        (50.0, 300.0,   "meso  50-300 "),
        (120.0, 600.0,  "meso 120-600 "),
        (150.0, 600.0,  "wide 150-600 "),
    ]
    for lo, hi, label in bands:
        r = corr_band(model_eta, sla_obs, valid, lo, hi, dx_km, land)
        print(f"      {label} km -> corr {r:+.3f}")

    # raw reference
    e = model_eta[valid]; o = sla_obs[valid]
    e = e - e.mean(); o = o - o.mean()
    den = np.sqrt((e * e).mean() * (o * o).mean())
    print(f"      full raw        -> corr {float((e*o).mean()/den):+.3f}")

    # ── Q3: cross-spectral coherence + phase ──
    print("\n  Q3  Cross-spectral coherence & phase (model vs obs):")
    # coherence & phase at the band-passed field via band-limited cross-spectrum
    cm = band_pass(model_eta, L_HI, L_LO, dx_km, land)
    co = band_pass(np.nan_to_num(sla_obs, nan=0.0), L_HI, L_LO, dx_km, land)
    v = valid & np.isfinite(cm) & np.isfinite(co)
    fm = np.nan_to_num(cm, nan=0.0).astype(np.float64); fm = fm - fm[v].mean()
    fo_ = np.nan_to_num(co, nan=0.0).astype(np.float64); fo_ = fo_ - fo_[v].mean()
    FM = np.fft.fftshift(np.fft.fft2(fm * np.hanning(grid.nx)[:, None] * np.hanning(grid.ny)[None, :]))
    FO = np.fft.fftshift(np.fft.fft2(fo_ * np.hanning(grid.nx)[:, None] * np.hanning(grid.ny)[None, :]))
    Pmm = np.abs(FM) ** 2
    Poo = np.abs(FO) ** 2
    Pmo = FM * np.conj(FO)
    kx = np.fft.fftshift(np.fft.fftfreq(grid.nx, d=dx_m)) * 1e3
    ky = np.fft.fftshift(np.fft.fftfreq(grid.ny, d=dx_m)) * 1e3
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_edges = np.geomspace(K[K > 0].min(), K.max(), 14)
    k_c = 0.5 * (k_edges[:-1] + k_edges[1:])
    print("      wl(km)   coh^2    phase(deg)")
    for i in range(len(k_c)):
        m = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if m.sum() == 0:
            continue
        s_pmm = Pmm[m].sum(); s_poo = Poo[m].sum(); s_pmo = np.abs(Pmo[m]).sum()
        coh = float(s_pmo ** 2 / (s_pmm * s_poo)) if s_pmm * s_poo > 0 else float('nan')
        ph = float(np.angle(np.sum(Pmo[m])))
        wl = 1.0 / k_c[i] if k_c[i] > 0 else np.inf
        if 20 < wl < 900:
            print(f"      {wl:7.1f}  {coh:5.3f}   {np.degrees(ph):+7.1f}")

    print("\n  Diagnostic complete. Interpret with care: band-pass uses DoG "
          "with reflect padding; coherence/phase are at tilted band only.")


if __name__ == "__main__":
    main()
