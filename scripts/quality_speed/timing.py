"""Explicit synchronized lifecycle timing, independent of any solver/driver."""
from time import perf_counter


def measure_lifecycle(*, setup, compile_cold, warmup, integrate, diagnostics_io,
                      synchronize, pair_id, clock=perf_counter):
    """Callbacks pass their result forward; warmup must return a reset initial state.

    Run each trial in a fresh process with an isolated/disabled persistent cache.
    compile_cold includes trace/lower/compile; warmup excludes measured steps.
    Synchronize must block on every state AND diagnostic leaf. Integration must
    include production per-step monitoring; diagnostics_io writes actual outputs.
    No exceptions are converted into zero seconds or a successful measurement.
    """
    total_start = clock()
    phases = {}
    current = None
    for name, operation in [('setup', setup), ('cold_compile', compile_cold),
                            ('warmup', warmup), ('integration', integrate),
                            ('diagnostics_io', diagnostics_io)]:
        start = clock()
        current = operation(current)
        synchronize(current)
        phases[name] = clock() - start
    phases['total'] = clock() - total_start
    phases['phase_status'] = {name: {'status': 'measured'} for name in phases}
    phases['pair_id'] = pair_id
    return phases, current
