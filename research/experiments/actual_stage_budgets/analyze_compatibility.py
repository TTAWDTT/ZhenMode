"""Compare audited states with the frozen, pre-instrumentation kernel."""
import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from jax_solver_global import JaxStateG
from stage_budgets import make_budget_step
from tests.support.fd.horizontal_diffusion import _parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/stage_compatibility.json")
    parser.add_argument("--ref", default="d9ffce9")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{args.ref}:src/jax_solver_global.py"],
                                     cwd=ROOT, text=True, encoding="utf-8")
    old = types.ModuleType("pre_budget_solver")
    old.__file__ = str(ROOT / "src/jax_solver_global.py")
    sys.modules[old.__name__] = old
    exec(compile(source, old.__file__, "exec"), old.__dict__)
    results = []
    for dtype in (jnp.float64, jnp.float32):
        for ice in (False, True):
            _, params, volume = _parameters(65., land=True)
            params = params._replace(dynamic_ice=ice)
            params = params._replace(**{name: value.astype(dtype) for name, value in params._asdict().items()
                                      if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
            random = np.random.default_rng(2718)
            shape = volume.shape
            wet = np.asarray(params.wet_mask_z)
            state = JaxStateG(
                jnp.asarray(0.01 * random.normal(size=shape) * wet, dtype=dtype),
                jnp.asarray(0.01 * random.normal(size=shape) * wet, dtype=dtype),
                jnp.asarray(15. + random.normal(size=shape), dtype=dtype),
                jnp.asarray(35. + 0.1 * random.normal(size=shape), dtype=dtype),
                jnp.zeros(shape[:2], dtype=dtype), jnp.full(shape[:2], 1e-4 if ice else 0., dtype=dtype))
            previous = jax.jit(lambda current: old._step_impl(current, params))(state)
            audited, _ = make_budget_step(params)(state)
            maximum_errors = {}
            for name in state._fields:
                np.testing.assert_allclose(getattr(audited, name), getattr(previous, name),
                                           rtol=32. * np.finfo(np.dtype(dtype)).eps, atol=1e-12)
                maximum_errors[name] = float(np.max(np.abs(np.asarray(getattr(audited, name), dtype=float)
                                                           - np.asarray(getattr(previous, name), dtype=float))))
            results.append({"dtype": np.dtype(dtype).name, "dynamic_ice": ice,
                            "maximum_absolute_errors": maximum_errors})
    report = {"scope": "read_only_instrumentation_compatibility_not_climate_qualification",
              "frozen_kernel": args.ref, "results": results,
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in current_source_files(ROOT, [ROOT / "src/jax_solver_global.py", ROOT / "src/stage_budgets.py",
                                             ROOT / "tests/test_horizontal_tracer_diffusion.py", Path(__file__).resolve()]).items()}}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
