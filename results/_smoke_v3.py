# -*- coding: utf-8 -*-
"""Smoke-test the v3 analysis additions against synthetic snaps."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "results", "smoke_3d"), exist_ok=True)
nx = 12
lat = np.linspace(-59.5, 59.5, 36)      # 1-deg-ish: bands j26/jband nonempty
ny = lat.size
z = np.array([0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500,
              -1000, -2000, -3000, -4000], float)
nz = z.size
nz = z.size
lat = np.linspace(-59.5, 59.5, ny)
np.savez(os.path.join(tmp, "results", "wet3_push.npz"),
         wet3=np.ones((nx, ny, nz)))
for i in range(12):
    T = 15 + 0.05 * i - z * 0.004
    S = 34.8 - 0.001 * i + z * 0.0001
    T3 = np.broadcast_to(T[None, None, :], (nx, ny, nz)).copy()
    S3 = np.broadcast_to(S[None, None, :], (nx, ny, nz)).copy()
    v = 0.05 * np.cos(np.deg2rad(lat))[None, :, None] * np.ones((nx, ny, nz))
    u = 0.01 * np.ones((nx, ny, nz))
    np.save(os.path.join(tmp, "results", "smoke_3d", f"snap_{i:05d}.npy"),
            np.stack([T3, u, v, S3]).astype(np.float64))
np.savez(os.path.join(tmp, "results", "global_smoke.npz"),
         days=np.arange(12) * 365.0, verdict=np.array("PASS"),
         max_eta=np.linspace(1.0, 1.5, 12), max_u_peak=np.array(0.8),
         lat=lat, lon=np.arange(nx) + 0.5, z=z, ssh_std=np.ones(12))

sys.argv = ["x", "smoke"]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "an", os.path.abspath("results/_spinup_probe_analysis.py"))
m = importlib.util.module_from_spec(spec)
# redirect every path constant to the sandbox BEFORE the module body runs
# (exec runs the source top-level directly) — D3 is read at top level so it
# must be patched in the source, not after
src = open(os.path.abspath("results/_spinup_probe_analysis.py"),
           encoding="utf-8").read()
src = src.replace('ROOT = "/data/tmp/ocean"', f'ROOT = r"{tmp}"')
D3p = os.path.join(tmp, "results", "smoke_3d")
W3p = os.path.join(tmp, "results", "wet3_push.npz")
NPZp = os.path.join(tmp, "results", "global_smoke.npz")
OJp = os.path.join(tmp, "results", "smoke_analysis.json")
ONp = os.path.join(tmp, "results", "smoke_analysis.npz")
src = src.replace('os.path.join(ROOT, "results", f"global_{TAG}_3d")',
                  f'r"{D3p}"')
src = src.replace('os.path.join(ROOT, "results", "wet3_push.npz")', f'r"{W3p}"')
src = src.replace('os.path.join(ROOT, "results", f"global_{TAG}.npz")',
                  f'r"{NPZp}"')
src = src.replace('os.path.join(ROOT, "results", f"{TAG}_analysis.json")',
                  f'r"{OJp}"')
src = src.replace('os.path.join(ROOT, "results", f"{TAG}_analysis.npz")',
                  f'r"{ONp}"')
code = compile(src, "spinup_probe_analysis_v3", "exec")
exec(code, m.__dict__)

d = np.load(os.path.join(tmp, "results", "smoke_analysis.npz"))
key_list = list(d.files)                 # copy before closing the handle
vals = {k: (d[k].tolist() if d[k].ndim else d[k].item()) for k in key_list}
d.close()
for k in ["crit_slope_ohc", "crit_slope_sss", "crit_d_pass", "crit_e_pass",
          "amoc26_ratio", "prof_z", "prof_T_atl", "prof_S_glb", "prof_lat"]:
    assert k in key_list, k
psi_n = 1
for sdim in vals["psi_shape"]:
    psi_n *= sdim
assert psi_n == ny * nz
assert vals["psi_shape"] == [ny, nz]
j = json.load(open(os.path.join(tmp, "results", "smoke_analysis.json")))
for k in ["d_ohc_trend", "e_sss_drift", "f_amoc_sanity"]:
    assert k in j["criteria"], k
assert j["profiles"]["T_atl"] and len(j["profiles"]["z"]) == nz
print("SMOKE OK — ohc %.3f ZJ/yr, sss %.5f psu/yr, d_pass %s e_pass %s, "
      "psi %s, T_atl surface %.2f" % (
          vals["crit_slope_ohc"][0], vals["crit_slope_sss"][0],
          vals["crit_d_pass"][0], vals["crit_e_pass"][0],
          str(vals["psi_shape"]), j["profiles"]["T_atl"][0]))
shutil.rmtree(tmp, ignore_errors=True)   # np.load in the analysis holds a
                                         # handle; win32 unlink would fail
