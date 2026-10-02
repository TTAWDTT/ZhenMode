"""Legacy state and snapshot metrics outside numerical integration."""

from __future__ import annotations

from integration_monitor import classify_state
from ocean_solver.fd.backend import np


def state_is_finite(state):
    return bool(classify_state(state).finite)


def total_kinetic_energy(state, ocean_mask):
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    ke = 0.5 * np.sum((u**2 + v**2) * ocean_mask[:, :, None])
    return float(ke)
