"""Convert the final/mean MOM6 surface T on the shared grid and score it."""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import netCDF4

from benchmark_metrics import score_snapshot


def main():
    p = ArgumentParser()
    p.add_argument("--prog", required=True)
    p.add_argument("--geometry", default="ocean_geometry.nc")
    p.add_argument("--reference-npz", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--steady-days", type=float, default=0.0)
    p.add_argument("--model", default="MOM6")
    p.add_argument("--run-id", default="")
    args = p.parse_args()

    geo = netCDF4.Dataset(args.geometry)
    lat = np.asarray(geo["geolat"][:], dtype=float)
    lon = np.asarray(geo["geolon"][:], dtype=float)
    wet = np.asarray(geo["wet"][:], dtype=bool)
    prog = netCDF4.Dataset(args.prog)
    temp = prog_variable = prog["Temp"]
    # MOM6 writes (time, z, y, x).  If multiple time levels exist, average the
    # requested final steady window; otherwise use the final record.
    days = np.asarray(prog["time"][:], dtype=float)
    if days.size > 1 and args.steady_days > 0:
        keep = days >= days[-1] - args.steady_days
    else:
        keep = np.ones(days.shape, dtype=bool)
    sst = np.mean(np.asarray(temp[keep, 0, :, :], dtype=float), axis=0)
    ref = np.load(args.reference_npz, allow_pickle=True)
    reference = np.asarray(ref["T_init"], dtype=float)[:, :, 0]
    ref_ocean = np.asarray(ref["wet_mask"], dtype=bool)
    ref_lat = np.asarray(ref["lat"], dtype=float)
    ref_lon = np.asarray(ref["lon"], dtype=float)
    # The shared grid is exact; assert against the reference grid.
    if ref_lat.shape != lat.shape or ref_lon.shape != lon.shape:
        raise RuntimeError(f"grid mismatch: MOM6 {lat.shape}, reference {ref_lat.shape}")
    if not np.allclose(ref_lat, lat, atol=1e-6) or not np.allclose(ref_lon, lon, atol=1e-6):
        raise RuntimeError("MOM6 grid centers differ from reference")
    ocean = ref_ocean & wet
    result = score_snapshot(sst, reference, ocean, ref_lat, ref_lon)
    result.update({
        "model": args.model,
        "run_id": args.run_id,
        "source_prog": str(Path(args.prog)),
        "time_records": int(days.size),
        "time_first": float(days[0]) if days.size else None,
        "time_last": float(days[-1]) if days.size else None,
        "steady_window_days": ([float(days[keep][0]), float(days[-1])]
                               if days.size > 1 else [None, None]),
        "n_mom6_wet": int(wet.sum()),
        "n_reference_wet": int(ref_ocean.sum()),
        "n_scored": int(ocean.sum()),
    })
    out = Path(args.out or Path(args.prog).with_name("mom6_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
