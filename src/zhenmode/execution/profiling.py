"""Observe the production executable without changing its numerical operators."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import replace


class StepTimings:
    """Separate lowering/compilation from synchronized executable calls."""

    def __init__(self):
        self.compilations = []
        self.steps = []
        self.io_s = 0.0

    def factory(self, native_factory):
        def observed(*args, **kwargs):
            import jax

            result = list(native_factory(*args, **kwargs))
            for index in (0, 5):
                if index >= len(result):
                    continue
                native = result[index]
                cache = {}

                def step(*pos, _native=native, _index=index, _cache=cache, **kw):
                    leaves, tree = jax.tree_util.tree_flatten((pos, kw))
                    arrays = [value if hasattr(value, "shape") and hasattr(value, "dtype") else jax.numpy.asarray(value) for value in leaves]
                    signature = (tree, tuple((tuple(value.shape), str(value.dtype), bool(getattr(value, "weak_type", False))) for value in arrays))
                    jax.block_until_ready((pos, kw))
                    if signature not in _cache:
                        start = time.perf_counter()
                        lowered = _native.lower(*pos, **kw)
                        lower_s = time.perf_counter() - start
                        start = time.perf_counter()
                        _cache[signature] = lowered.compile()
                        self.compilations.append({"entry_index": _index, "lower_s": lower_s,
                                                  "compile_s": time.perf_counter() - start})
                        print("STEP_COMPILED", self.compilations[-1], flush=True)
                    start = time.perf_counter()
                    updated = _cache[signature](*pos, **kw)
                    jax.block_until_ready(updated)
                    self.steps.append(time.perf_counter() - start)
                    return updated
                result[index] = step
            return tuple(result)
        return observed

    @contextmanager
    def observe(self, services):
        # Patch only the isolated worker's IO dispatch, and always restore it.
        import zhenmode.model.io.records as records
        import zhenmode.model.runtime.run_loop as loop

        originals = []
        for module, name in ((loop, "write_final_records"), (loop, "save_restart"), (records, "_save_snapshot_file")):
            native = getattr(module, name)
            def timed(*args, _native=native, **kwargs):
                start = time.perf_counter()
                try:
                    return _native(*args, **kwargs)
                finally:
                    self.io_s += time.perf_counter() - start
            originals.append((module, name, native))
            setattr(module, name, timed)
        try:
            yield replace(services, make_solver_global=self.factory(services.make_solver_global))
        finally:
            for module, name, native in originals:
                setattr(module, name, native)

    def report(self):
        steady = self.steps[3:]
        return {"compile_s": sum(item["compile_s"] for item in self.compilations),
                "lower_s": sum(item["lower_s"] for item in self.compilations),
                "integration_s": sum(self.steps), "io_s": self.io_s,
                "step_count": len(self.steps), "warmup_discarded_steps": min(3, len(self.steps)),
                "steady_step_count": len(steady),
                "steady_step_median_s": statistics.median(steady) if steady else None,
                "step_times_s": self.steps, "compilations": self.compilations,
                "scope": "Main FD executable only: synchronized calls exclude forcing interpolation, monitors and IO. Compile excludes diagnostic JITs. IO covers final result, checkpoints and snapshots, not input reads or logs. Components are not exhaustive end-to-end partitions."}
