"""Installation and full-state checkpoint contracts."""
import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import all_wet_grid

from jax_solver_global import JaxStateG


def test_checkpoint_saves_every_state_field(tmp_path):
    from run_long_integration_global import _save_checkpoint

    grid = all_wet_grid(nx=8, ny=8, nz=4)
    shape = (8, 8, 4)
    state = JaxStateG(jnp.zeros(shape), jnp.zeros(shape), jnp.full(shape, -1.8),
                      jnp.full(shape, 35.), jnp.zeros((8, 8)), jnp.full((8, 8), 0.5))
    path = tmp_path / "checkpoint.npz"
    _save_checkpoint(path, state, grid, 12, 2)
    with np.load(path) as saved:
        for field in state._fields:
            np.testing.assert_array_equal(saved[field], np.asarray(getattr(state, field)))
        assert saved["cur_step"] == 12
        assert saved["n_3d_snaps"] == 2


def test_legacy_checkpoint_cannot_silently_reset_dynamic_ice():
    from run_long_integration_global import _load_checkpoint_state

    grid = all_wet_grid(nx=8, ny=8, nz=4)
    checkpoint = {name: np.zeros((8, 8, 4)) for name in ("u", "v", "T", "S")}
    checkpoint.update(eta=np.zeros((8, 8)), grid_nx=8, grid_ny=8, grid_nz=4)
    legacy = _load_checkpoint_state(checkpoint, grid, jnp.float32)
    np.testing.assert_array_equal(legacy.ice, 0.)
    with pytest.raises(ValueError, match="ice thickness"):
        _load_checkpoint_state(checkpoint, grid, jnp.float32, dynamic_ice=True)


def test_checkpoint_rejects_malformed_field_shape():
    from run_long_integration_global import _load_checkpoint_state

    grid = all_wet_grid(nx=8, ny=8, nz=4)
    checkpoint = {name: np.zeros((8, 8, 4)) for name in ("u", "v", "T", "S")}
    checkpoint.update(eta=np.zeros((8, 7)), grid_nx=8, grid_ny=8, grid_nz=4)
    with pytest.raises(ValueError, match="eta shape"):
        _load_checkpoint_state(checkpoint, grid, jnp.float64)
