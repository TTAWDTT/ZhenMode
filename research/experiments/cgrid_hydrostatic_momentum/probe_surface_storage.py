"""Preregistered scalar compiled storage experiment, NOT an ocean trajectory."""
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]


def run(volume, increment, count, compensated=False, barrier=False):
    def body(unused_index, state):
        stock, correction = state
        if compensated:
            adjusted = increment - correction
            following = stock + adjusted
            if barrier:
                following = jax.lax.optimization_barrier(following)
            correction = (following - stock) - adjusted
            return following, correction
        return stock + increment, correction

    return jax.lax.fori_loop(0, count, body, (volume, jnp.asarray(0., jnp.float64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain precision evidence")
    hashes = {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    volume = 3.5e11
    for count in (100000, 1000000):
        for direction in (-1., 1.):
            increment = direction * np.spacing(volume) / 32.
            with localcontext() as context:
                context.prec = 60
                expected = Decimal.from_float(volume) + count * Decimal.from_float(increment)
                for name, compensated, barrier in (("plain", False, False), ("compensated", True, False), ("compensated_barrier", True, True)):
                    kernel = jax.jit(lambda stock, amount: run(stock, amount, count, compensated, barrier))
                    stock, correction = kernel(jnp.asarray(volume), jnp.asarray(increment))
                    stock.block_until_ready()
                    represented = Decimal.from_float(float(stock)) - Decimal.from_float(float(correction))
                    error = represented - expected
                    row = {"steps": count, "direction": direction, "method": name, "initial_volume_m3": volume,
                           "increment_m3": increment, "source_total_m3": float(count * Decimal.from_float(increment)),
                           "stored_stock_m3": float(stock), "stored_correction_m3": float(correction),
                           "exact_represented_error_m3": float(error)}
                    rows.append(row)
                    print(row, flush=True)
    naive = [row for row in rows if row["method"] == "plain"]
    controlled = [row for row in rows if row["method"] == "compensated_barrier"]
    if any(abs(row["exact_represented_error_m3"]) < .99 * abs(row["source_total_m3"]) for row in naive):
        raise ValueError("registered lost-source witness did not reproduce")
    if any(row["exact_represented_error_m3"] != 0. for row in controlled):
        raise ValueError("controlled compensated representation not exact on binary fixture")
    unchanged = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    report = {"status": "ARITHMETIC_WITNESS_COMPLETE_NOT_PHYSICAL_QUALIFICATION", "backend": jax.default_backend(),
              "source_sha256": hashes, "source_hashes_unchanged": unchanged, "runs": rows,
              "scope": "no_geometry_flux_momentum_FCT_restart_or_climate_qualification"}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not unchanged:
        raise ValueError("probe source changed")


if __name__ == "__main__":
    main()
