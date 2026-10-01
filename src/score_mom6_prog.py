"""Convert the final/mean MOM6 surface T on the shared grid and score it."""
import json
from argparse import ArgumentParser
from pathlib import Path

import netCDF4
import numpy as np

from benchmark_metrics import _relative_drift
from score_external_model import score_external_field


def main():
    p = ArgumentParser()
    p.add_argument("--prog", required=True)
    p.add_argument("--geometry", default="ocean_geometry.nc")
    p.add_argument("--reference-npz", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--steady-days", type=float, default=0.0)
    p.add_argument("--model", default="MOM6")
    p.add_argument("--run-id", default="")
    p.add_argument("--stats", default=None)
    args = p.parse_args()

    result = score_external_field(
        args.prog, variable="temp", reference_path=args.reference_npz,
        geometry=args.geometry, lat_var="lath", lon_var="lonh", wet_var="wet",
        steady_days=args.steady_days)
    if args.stats:
        with netCDF4.Dataset(args.stats) as stats:
            result["heat_drift_percent"] = _relative_drift(np.asarray(stats["Heat"][:], dtype=float))
            result["salt_drift_percent"] = _relative_drift(np.asarray(stats["Salt"][:], dtype=float))
    result.update({
        "model": args.model,
        "run_id": args.run_id,
        "source_prog": str(Path(args.prog)),
        "n_mom6_wet": result["n_model_wet"],
    })
    out = Path(args.out or Path(args.prog).with_name("mom6_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
