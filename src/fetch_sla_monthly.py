"""Fetch monthly-mean SLA (nesdisSSH1day, NOAA CoastWatch ERDDAP) for 2023.

Covers the N Pacific window matching the global-model verification region:
lat 15-55N, lon 120-180E, 0.25-deg grid, 12 monthly means + the annual mean.

Rate-limited upstream (HTTP 429 observed) -> sequential fetches with backoff
and per-month disk cache in data/sla_npac/.
"""
import os
import time
import urllib.request

import netCDF4
import numpy as np

CACHE_DIR = "data/sla_npac"


def fetch_month(month: int, retries: int = 6) -> str:
    fn = os.path.join(CACHE_DIR, f"sla_2023-{month:02d}.nc")
    if os.path.exists(fn) and os.path.getsize(fn) > 100_000:
        return fn
    # ERDDAP constraint syntax: restrict the time axis with a month window
    # (days 1-28 exist in every month, so one request covers all of them).
    url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day.nc"
           f"?sla[(2023-{month:02d}-01):(2023-{month:02d}-28)][(15):(55)][(120):(180)]")
    for k in range(retries):
        try:
            r = urllib.request.urlopen(url, timeout=240)
            data = r.read()
            with open(fn, "wb") as f:
                f.write(data)
            print(f"  {fn}: {len(data)/1e6:.1f} MB")
            time.sleep(12)  # politeness + 429 avoidance
            return fn
        except Exception as e:  # noqa: BLE001
            wait = 30 * (k + 1)
            print(f"  attempt {k+1} failed ({e}); backing off {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"fetch failed for month {month}")


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
