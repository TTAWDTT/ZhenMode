# -*- coding: utf-8 -*-
"""Dashboard curves for runs launched WITHOUT --save-3d (cluster-side).

The full `_spinup_probe_analysis.py` needs yearly 3-D snaps, which only exist
when a run was started with --save-3d.  The mixing-matrix runs were launched
without it, so their `<tag>_analysis.npz` was never produced and every chart
on the dashboard rendered blank.

This builds the same 1-D npz payload from two sources that DO exist:
  - results/_drift_traj.csv   (tag,day,yr,ohc,deepT,sst) — written by the
    drift logger off the yearly checkpoints
  - results/global_<TAG>.npz  (max_u / max_eta / ke / ssh_std tables) — only
    after the run completes

Series that genuinely need the 3-D velocity field (AMOC, Drake, profiles, the
ψ section) are omitted; the frontend hides a card when none of its keys are
present, so those just stay hidden until a --save-3d re-run exists.

Usage: python results/_curves_from_drift.py <TAG> [<TAG> ...]
"""
import csv
import os
import sys

import numpy as np

ROOT = "/data/tmp/ocean"
DRIFT = os.path.join(ROOT, "results", "_drift_traj.csv")


def load_drift(tag):
    rows = []
    if not os.path.exists(DRIFT):
        return rows
    with open(DRIFT) as f:
        for r in csv.DictReader(f):
            if r.get("tag") != tag:
                continue
            try:
                rows.append((float(r["yr"]), float(r["ohc"]),
                             float(r["deepT"]), float(r["sst"])))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows


def interp_to(x_src, y_src, x_dst):
    xs, ys = np.asarray(x_src, float), np.asarray(y_src, float)
    if xs.size < 2:
        return None
    return np.interp(x_dst, xs, ys)


def build(tag):
    rows = load_drift(tag)
    npz_path = os.path.join(ROOT, "results", f"global_{tag}.npz")
    d = None
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)

    # prefer the run-npz day table (authoritative, spans the whole run); fall
    # back to the checkpoint cadence when the run is still in flight
    if d is not None and "days" in d.files and d["days"].size:
        days = np.asarray(d["days"], float)
        yrs = days / 365.0
    elif rows:
        yrs = np.asarray([r[0] for r in rows], float)
        days = yrs * 365.0
    else:
        return None

    out = {"days": days, "yrs": yrs}

    if rows:
        dyr = [r[0] for r in rows]
        out["ohc"] = interp_to(dyr, [r[1] for r in rows], yrs)
        out["deepT"] = interp_to(dyr, [r[2] for r in rows], yrs)
        out["sst"] = interp_to(dyr, [r[3] for r in rows], yrs)
    else:
        out["ohc"] = out["deepT"] = out["sst"] = None

    if d is not None:
        for src, dst in (("max_u", "maxu"), ("max_eta", "maxeta"),
                         ("ke", "ke"), ("ssh_std", "sshstd"),
                         ("max_T", "maxT")):
            if src in d.files and d[src].size:
                v = np.asarray(d[src], float)
                if v.size == yrs.size:
                    out[dst] = v
                elif v.size > 1:
                    out[dst] = interp_to(np.asarray(d["days"], float) / 365.0,
                                         v, yrs)

    # criteria the UI badges, computed from whatever series we have
    m = yrs >= 2.0
    if out.get("deepT") is not None and m.sum() >= 3:
        sg = float(np.polyfit(yrs[m], out["deepT"][m], 1)[0])
    else:
        sg = float("nan")
    if out.get("ohc") is not None and m.sum() >= 3:
        so = float(np.polyfit(yrs[m], out["ohc"][m], 1)[0])
    else:
        so = float("nan")
    max_eta_all = (float(np.max(np.abs(d["max_eta"])))
                   if d is not None and "max_eta" in d.files
                   and d["max_eta"].size else float("nan"))
    verdict = str(d["verdict"]) if d is not None and "verdict" in d.files \
        else "RUNNING"

    out["crit_slope_g"] = np.array([sg])
    out["crit_slope_ohc"] = np.array([so])
    out["crit_max_eta"] = np.array([max_eta_all])
    out["crit_b_pass"] = np.array(
        [1.0 if np.isfinite(sg) and sg < 0.0225 else 0.0])
    out["crit_d_pass"] = np.array(
        [1.0 if np.isfinite(so) and abs(so) < 2.0 else 0.0])
    out["verdict_flag"] = np.array([1.0 if verdict == "PASS" else 0.0])

    payload = {}
    for k, v in out.items():
        if v is None:
            continue
        arr = np.asarray(v, float)
        if arr.ndim == 1 and arr.size:
            payload[k] = arr
    return payload


def main():
    tags = sys.argv[1:]
    if not tags:
        print("usage: _curves_from_drift.py <TAG> [...]")
        return
    for tag in tags:
        try:
            payload = build(tag)
        except Exception as e:  # noqa: BLE001 — one bad tag must not stop the rest
            print(f"{tag}: FAILED {e!r}")
            continue
        if not payload:
            print(f"{tag}: no data (no drift rows, no npz)")
            continue
        outp = os.path.join(ROOT, "results", f"{tag}_analysis.npz")
        np.savez(outp, **payload)
        n = payload["yrs"].size
        print(f"{tag}: {n} pts -> {outp}  "
              f"(yr {payload['yrs'][0]:.0f}..{payload['yrs'][-1]:.0f}, "
              f"keys={sorted(payload)})")


if __name__ == "__main__":
    main()
