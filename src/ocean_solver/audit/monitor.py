"""Shared device-side state checks and first-rejection budget monitoring."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.audit.schema import accumulate_budget


class StateMetrics(NamedTuple):
    max_u: jax.Array
    max_velocity: jax.Array
    max_eta: jax.Array
    finite: jax.Array
    failure: jax.Array


@jax.jit
def classify_state(state, velocity_limit=10., eta_limit=15., inclusive=False):
    """Codes: 0 accepted, 1 nonfinite state, 3 velocity, 4 surface height."""
    max_u = jnp.max(jnp.abs(state.u))
    velocity = jnp.maximum(max_u, jnp.max(jnp.abs(state.v)))
    eta = jnp.max(jnp.abs(state.eta))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in state]))
    bad_velocity = jnp.where(inclusive, velocity >= velocity_limit, velocity > velocity_limit)
    bad_eta = jnp.where(inclusive, eta >= eta_limit, eta > eta_limit)
    failure = jnp.where(~finite, 1, jnp.where(bad_velocity, 3, jnp.where(bad_eta, 4, 0)))
    return StateMetrics(max_u, velocity, eta, finite, failure)


def make_monitored_advance(step, zero_budget, audited=False, transport_tolerances=None):
    """Retain the debug runner's inclusive limits and first-rejection tuple contract."""
    if transport_tolerances is not None:
        if not audited or len(transport_tolerances) != 2 or not all(
                np.isfinite(value) and value > 0. for value in transport_tolerances):
            raise ValueError("transport_tolerances require auditing and two finite positive limits")

    def classify(state, ledger):
        metrics = classify_state(state, inclusive=True)
        finite_ledger = jnp.all(jnp.stack([jnp.all(jnp.isfinite(values)) for values in ledger.values()]))
        failure = jnp.where(~metrics.finite, 1, jnp.where(~finite_ledger, 2, metrics.failure))
        if transport_tolerances is not None:
            continuity, matching = transport_tolerances
            transport = ledger["transport_consistency_max"]
            transport_failure = jnp.where(jnp.abs(transport[0]) > continuity, 5,
                                          jnp.where(jnp.abs(transport[2]) > matching, 6, 0))
            failure = jnp.where(failure == 0, transport_failure, failure)
        return metrics.max_velocity, metrics.max_eta, metrics.finite & finite_ledger, failure

    @jax.jit
    def advance(current, count):
        initial_velocity, initial_eta, initial_finite, initial_failure = classify(current, zero_budget)
        initial = (current, initial_velocity, initial_eta, initial_finite, zero_budget,
                   jnp.asarray(0), jnp.asarray(0), initial_failure, current, zero_budget)

        def attempt(carry):
            previous, peak_velocity, peak_eta, finite, totals, accepted, attempted, _, rejected, rejected_ledger = carry
            if audited:
                updated, ledger = step(previous)
            else:
                updated, ledger = step(previous), zero_budget
            new_totals = accumulate_budget(totals, ledger) if audited else totals
            velocity, eta, step_finite, failure = classify(updated, ledger)
            total_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(values)) for values in new_totals.values()]))
            failure = jnp.where((failure == 0) & ~total_finite, 2, failure)
            valid = failure == 0
            accepted_state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), updated, previous)
            accepted_totals = jax.tree.map(lambda new, old: jnp.where(valid, new, old), new_totals, totals)
            rejected = jax.tree.map(lambda new, old: jnp.where(valid, old, new), updated, rejected)
            rejected_ledger = jax.tree.map(lambda new, old: jnp.where(valid, old, new), ledger, rejected_ledger)
            return (accepted_state, jnp.maximum(peak_velocity, velocity), jnp.maximum(peak_eta, eta),
                    finite & step_finite & total_finite, accepted_totals, accepted + valid.astype(accepted.dtype),
                    attempted + 1, failure, rejected, rejected_ledger)

        def monitored_step(index, carry):
            return jax.lax.cond(carry[7] == 0, attempt, lambda unchanged: unchanged, carry)

        return jax.lax.fori_loop(0, count, monitored_step, initial)

    return advance
