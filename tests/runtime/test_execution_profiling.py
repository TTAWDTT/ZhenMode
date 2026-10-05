"""Independent arithmetic witness for the executable observer, not ocean skill."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode.execution.profiling import StepTimings


def test_profiled_calls_match_numpy_and_unobserved_jit():
    advance = jax.jit(lambda state, forcing: state * jnp.float32(.5) + forcing)
    observer = StepTimings()
    wrapped = observer.factory(lambda: (advance, None, None, None, None, advance))()[5]
    value = jnp.asarray([4, 8], dtype=jnp.float32)
    reference = np.array([4, 8], dtype=np.float32)
    native = value
    for _ in range(5):
        forcing = jnp.array([1, 2], dtype=jnp.float32)
        value = wrapped(value, forcing)
        native = advance(native, forcing)
        reference = reference * np.float32(.5) + np.array([1, 2], dtype=np.float32)
    np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(value, native)
    report = observer.report()
    assert report["step_count"] == 5 and report["steady_step_count"] == 2
    assert len(report["compilations"]) == 1
    assert report["integration_s"] == sum(report["step_times_s"])


def test_io_observers_restore_on_failure():
    import zhenmode.model.io.records as records
    import zhenmode.model.runtime.run_loop as loop

    before = (loop.write_final_records, loop.save_restart, records._save_snapshot_file)
    services = SimpleNamespace(make_solver_global=lambda: ())  # Not a dataclass: restoration still required.
    with pytest.raises(TypeError):
        with StepTimings().observe(services):
            pass
    assert before == (loop.write_final_records, loop.save_restart, records._save_snapshot_file)
