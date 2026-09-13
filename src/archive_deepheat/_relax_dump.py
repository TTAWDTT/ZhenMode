#!/usr/bin/env python3
"""Compact criteria dump for the unreviewed spinC_* analyses."""
import glob
import json
import os

os.chdir("/data/tmp/ocean/results")
tags = ["spinC_relax15", "spinC_relax60", "spinC_relax90",
        "spinC_from30", "spinC_slope005", "spinC_slope007"]
for t in tags:
    p = t + "_analysis.json"
    print("=" * 78)
    print("===== " + t + " =====")
    if not os.path.exists(p):
        print("  MISSING")
        continue
    with open(p) as fh:
        d = json.load(fh)
    c = d.get("criteria", {})
    print("  verdict={}  max_u_peak={}  n_snaps={}".format(
        d.get("verdict"), d.get("max_u_peak"), d.get("n_snaps")))
    for lab, key in [("deepT", "b_deept_trend"), ("eta", "c_med_eta"),
                     ("ohc", "d_ohc_trend"), ("sss", "e_sss_drift")]:
        print("  {:6s} {}".format(lab, c.get(key)))
    f = c.get("f_amoc_sanity", {})
    if f:
        print("  amoc   final={} ratio={}".format(
            f.get("amoc_final_sv"), f.get("amoc26_ratio")))
    print("  amoc   {}".format(d.get("amoc")))
    print("  deepT  {}".format(d.get("deepT")))
    print("  warn   {}".format(d.get("warnings")))
