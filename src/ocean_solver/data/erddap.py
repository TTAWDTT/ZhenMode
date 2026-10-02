"""Shared ERDDAP download helper for the altimetry fetchers.

Both fetch_sla_monthly.py and fetch_ssh_abs_monthly.py pull gridded products
from NOAA CoastWatch ERDDAP and hit the same upstream failure mode (HTTP 429
under load), so the retry / backoff / disk-cache policy lives here once
instead of being copied into each fetcher.
"""

import os
import time
import urllib.request

from ocean_solver._compat import preserve_legacy_names

POLITENESS_S = 10.0        # sleep after a good fetch (429 avoidance)
BACKOFF_BASE_S = 30.0      # wait = BACKOFF_BASE_S * (attempt + 1)
TIMEOUT_S = 240


def fetch(url, path, min_bytes, retries=6, politeness_s=POLITENESS_S, label=""):
    """Download ``url`` to ``path``, returning ``path``.

    An existing file larger than ``min_bytes`` is reused, so a partial
    download is retried rather than trusted. A failed attempt waits
    ``BACKOFF_BASE_S * (attempt + 1)`` seconds and the last one raises.
    """
    if os.path.exists(path) and os.path.getsize(path) > min_bytes:
        return path
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
                data = response.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"  {path}: {len(data) / 1e6:.2f} MB", flush=True)
            time.sleep(politeness_s)
            return path
        except Exception as e:  # noqa: BLE001 - any network failure is retryable
            wait = BACKOFF_BASE_S * (k + 1)
            print(f"  attempt {k + 1} failed ({e}); backing off {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed for {label or url}")

preserve_legacy_names(globals(), 'erddap_fetch')
