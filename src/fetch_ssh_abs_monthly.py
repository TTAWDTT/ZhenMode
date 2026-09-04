"""Fetch 2012 monthly-mean ABSOLUTE SSH (erdTAssh1day, TOPEX/Jason-era).

erdTAssh1day covers 1992-10 -> 2012-12, so 2012 is the freshest year.
Constrained to lat 15-55N, lon 120-180E (N Pacific verification window).
Altitude axis must be pinned to 0.0. Sequential fetch, 429-safe backoff.
"""
import os
import time
import urllib.request

import netCDF4
import numpy as np

CACHE_DIR = "data/sla_npac"


def fetch_month_day(month: int, day: int, retries: int = 6) -> str:
    fn = os.path.join(CACHE_DIR, f"ssh_abs_2012-{month:02d}.nc")
    if os.path.exists(fn) and os.path.getsize(fn) > 50_000:
        return fn
    url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdTAssh1day.nc"
           f"?ssh[(2012-{month:02d}-{day:02d})][(0.0):(0.0)][(15):(55)][(120):(180)]")
    for k in range(retries):
        try:
            r = urllib.request.urlopen(url, timeout=240)
            data = r.read()
            with open(fn, "wb") as f:
                f.write(data)
            print(f"  {fn}: {len(data)/1e3:.0f} KB", flush=True)
            time.sleep(10)
            return fn
        except Exception as e:  # noqa: BLE001
            wait = 30 * (k + 1)
            print(f"  attempt {k+1} ({e}); backoff {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed for {month}")


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
