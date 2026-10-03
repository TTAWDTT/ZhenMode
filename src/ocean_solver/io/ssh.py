"""Fetch 2012 monthly-mean ABSOLUTE SSH (erdTAssh1day, TOPEX/Jason-era).

erdTAssh1day covers 1992-10 -> 2012-12, so 2012 is the freshest year.
Constrained to lat 15-55N, lon 120-180E (N Pacific verification window).
Altitude axis must be pinned to 0.0. Downloads go through the shared
429-safe sequential fetch in erddap_fetch.
"""

import os

import netCDF4
import numpy as np

from ocean_solver.io.erddap import fetch

CACHE_DIR = "data/sla_npac"
DATASET = "erdTAssh1day"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"


def fetch_month_day(month: int, day: int, retries: int = 6) -> str:
    fn = os.path.join(CACHE_DIR, f"ssh_abs_2012-{month:02d}.nc")
    url = (f"{ERDDAP}/{DATASET}.nc"
           f"?ssh[(2012-{month:02d}-{day:02d})][(0.0):(0.0)]"
           f"[(15):(55)][(120):(180)]")
    return fetch(url, fn, min_bytes=50_000, retries=retries,
                 politeness_s=10.0, label=f"ssh_abs {month:02d}-{day:02d}")


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # mid-month days, all present in the daily record
    days = [15, 15, 15, 15, 16, 16, 16, 16, 15, 16, 15, 16]
    means, lats, lons = [], None, None
    for m in range(1, 13):
        fn = fetch_month_day(m, days[m - 1])
        ds = netCDF4.Dataset(fn)
        lat = np.asarray(ds.variables["latitude"][:])
        lon = np.asarray(ds.variables["longitude"][:])
        a = np.ma.filled(ds.variables["ssh"][:], np.nan)
        a = a.reshape(-1, a.shape[-2], a.shape[-1])[0]
        ds.close()
        means.append(a)
        lats, lons = lat, lon
    np.savez(os.path.join(CACHE_DIR, "ssh_abs_npac_monthly_2012.npz"),
             ssh=np.stack(means, 0), lat=lats, lon=lons)
    print(f"saved {np.stack(means,0).shape} -> ssh_abs_npac_monthly_2012.npz")




if __name__ == "__main__":
    main()
