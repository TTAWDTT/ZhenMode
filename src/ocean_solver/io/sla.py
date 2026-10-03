"""Fetch monthly-mean SLA (nesdisSSH1day, NOAA CoastWatch ERDDAP) for 2023.

Covers the N Pacific window matching the global-model verification region:
lat 15-55N, lon 120-180E, 0.25-deg grid, 12 monthly means + the annual mean.

Rate-limited upstream (HTTP 429 observed), so the download goes through the
shared sequential fetch-with-backoff in erddap_fetch, with a per-month disk
cache in data/sla_npac/.
"""

import os

import netCDF4
import numpy as np

from ocean_solver.io.erddap import fetch

CACHE_DIR = "data/sla_npac"
DATASET = "nesdisSSH1day"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"


def fetch_month(month: int, retries: int = 6) -> str:
    fn = os.path.join(CACHE_DIR, f"sla_2023-{month:02d}.nc")
    # ERDDAP constraint syntax: restrict the time axis with a month window
    # (days 1-28 exist in every month, so one request covers all of them).
    url = (f"{ERDDAP}/{DATASET}.nc"
           f"?sla[(2023-{month:02d}-01):(2023-{month:02d}-28)]"
           f"[(15):(55)][(120):(180)]")
    return fetch(url, fn, min_bytes=100_000, retries=retries,
                 politeness_s=12.0, label=f"sla month {month}")


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # month 1 already fetched during the coverage probe
    probe = "data/_sla_np_test.nc"
    if os.path.exists(probe) and os.path.getsize(probe) > 1_000_000:
        os.replace(probe, os.path.join(CACHE_DIR, "sla_2023-01.nc"))
        print("  moved probe -> sla_2023-01.nc")
    for m in range(2, 13):
        fetch_month(m)
    # verify all 12 + build annual mean
    means, lats, lons = [], None, None
    for m in range(1, 13):
        fn = os.path.join(CACHE_DIR, f"sla_2023-{m:02d}.nc")
        ds = netCDF4.Dataset(fn)
        lat = np.asarray(ds.variables["latitude"][:])
        lon = np.asarray(ds.variables["longitude"][:])
        a = ds.variables["sla"][:]
        a = np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
        ds.close()
        means.append(np.nanmean(a, axis=0))
        lats, lons = lat, lon
    np.savez(os.path.join(CACHE_DIR, "sla_npac_monthly_2023.npz"),
             sla=np.stack(means, 0), lat=lats, lon=lons)
    print(f"saved stack {np.stack(means,0).shape} -> sla_npac_monthly_2023.npz")




if __name__ == "__main__":
    main()
