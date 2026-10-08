"""Native payload axis controls and independent extensive surface inventories."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.coupling.sis2 import FLUX_FIELDS, SURFACE_FIELDS, decode_reply
from zhenmode.model.config import CP0_TEOS10, G_EARTH, RHO_0, PhysicsConfig
from zhenmode.model.solver.dynamics.pressure import _compute_pressure_gradient
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.physics.sis2 import apply_sis2_exchange, extract_frazil
from zhenmode.model.solver.state import JaxStateG


def test_native_axes_exact_size_and_category_coverage(tmp_path):
    shape = (2, 3, 6)
    path = tmp_path / "reply"
    payload = []
    for n in FLUX_FIELDS:
        payload.append(np.arange(6.0).reshape((2, 3), order="F") + FLUX_FIELDS.index(n))
    for n in SURFACE_FIELDS:
        value = np.arange(36.0).reshape(shape, order="F")
        payload.append(np.full(shape, 1 / 6) if n == "part_size" else value)
    payload += [
        np.full((2, 3), 35.0),
        np.ones((2, 3)),
        np.array([10.0, -20.0, 3.0, 334000.0, 3991.86795711963, 905.0]),
    ]
    raw = b"".join(a.astype("<f8").tobytes(order="F") for a in payload)
    path.write_bytes(raw)
    decoded = decode_reply(path, shape, fluxes=True, interval_seconds=2)
    assert decoded["interval_seconds"] == 2
    assert decoded["flux_u"][1, 2] == 5
    assert decoded["t_surf"][1, 2, 5] == 35
    np.testing.assert_array_equal(decoded["stocks"], [10.0, -20.0, 3.0])
    path.write_bytes(raw + b"unexpected")
    with pytest.raises(ValueError, match="size"):
        decode_reply(path, shape, fluxes=True)
    path.write_bytes(b"\x00" * len(raw))
    # Wet area with no category coverage must be rejected.
    offset = (len(FLUX_FIELDS) * 6 + len(SURFACE_FIELDS) * 36 + 6) * 8
    broken = bytearray(raw)
    broken[len(FLUX_FIELDS) * 6 * 8 : (len(FLUX_FIELDS) * 6 + 36) * 8] = b"\x00" * (36 * 8)
    assert offset < len(broken)
    path.write_bytes(broken)
    with pytest.raises(ValueError, match="cover"):
        decode_reply(path, shape, fluxes=True)


@pytest.mark.parametrize("direction", [-1, 1])
def test_sis2_mass_salt_enthalpy_with_surface_displacement(direction):
    """Freeze/melt signs, SR units, mass enthalpy and unused frazil are separate."""
    params = SimpleNamespace(
        wet_mask=jnp.array([[1.0, 0.0], [1.0, 1.0]]), dz_node=jnp.full((2, 2, 3), 4.0)
    )
    state = JaxStateG(
        jnp.zeros((2, 2, 3)),
        jnp.zeros((2, 2, 3)),
        jnp.full((2, 2, 3), -1.5),
        jnp.full((2, 2, 3), 34.0),
        jnp.full((2, 2), 0.2),
        jnp.zeros((2, 2)),
    )
    fields = {n: jnp.zeros((2, 2)) for n in FLUX_FIELDS}
    fields["interval_seconds"] = 2
    # Manufacture fluxes in SI, not by calling model functions.
    for n, value in {
        "lprec": direction * 0.003,
        "fprec": 0.001,
        "flux_q": 0.0001,
        "flux_salt": -direction * 0.00002,
        "flux_t": 20.0,
        "flux_lw": 70.0,
        "flux_lh": 5.0,
        "sw_vis_dir": 30.0,
        "enth_mass_in_ocn": -40.0,
        "enth_mass_out_ocn": 10.0,
        "frazil_left": 15.0,
    }.items():
        fields[n] = jnp.full((2, 2), value)
    fields["thermo_constants"] = jnp.array([334000.0, 3991.86795711963, 905.0])
    updated, inputs, height = jax.jit(lambda s: apply_sis2_exchange(s, fields, params, 2.0))(state)
    before_h = 4.0 + np.asarray(state.eta)
    after_h = 4.0 + np.asarray(updated.eta)
    mass = RHO_0 * (np.asarray(updated.eta) - np.asarray(state.eta))
    salt = RHO_0 / 1000 * (after_h * np.asarray(updated.S[..., 0]) - before_h * 34.0)
    heat = RHO_0 * CP0_TEOS10 * (after_h * np.asarray(updated.T[..., 0]) - before_h * -1.5)
    expected = [
        2 * (direction * 0.003 + 0.001 - 0.0001),
        2 * direction * 0.00002,
        2 * (30 + 70 - 20 - 5 - 334000 * 0.001) + 40 - 10 - 15,
    ]
    wet = np.asarray(params.wet_mask) > 0.5
    for actual, target, ledger in zip((mass, salt, heat), expected, inputs, strict=True):
        np.testing.assert_allclose(actual[wet], target, atol=2e-8, rtol=0)
        np.testing.assert_allclose(np.asarray(ledger)[wet], target, atol=1e-12, rtol=0)
    for n in JaxStateG._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(updated, n))[0, 1], np.asarray(getattr(state, n))[0, 1]
        )
    assert np.min(np.asarray(height)) > 0


def test_exchange_refuses_wrong_interval_and_exhausted_surface():
    params = SimpleNamespace(wet_mask=jnp.ones((2, 2)), dz_node=jnp.ones((2, 2, 3)))
    state = JaxStateG(*(jnp.zeros((2, 2, 3)) for _ in range(4)),
                      jnp.zeros((2, 2)), jnp.zeros((2, 2)))
    fields = {name: jnp.zeros((2, 2)) for name in FLUX_FIELDS}
    fields.update(interval_seconds=2, thermo_constants=jnp.array([334000., CP0_TEOS10, 905.]))
    with pytest.raises(ValueError, match="differs from native"):
        apply_sis2_exchange(state, fields, params, 3.)
    # Also reject a mismatch when the interval is a runtime JAX scalar.
    advance = jax.jit(lambda s, f: apply_sis2_exchange(s, f, params, 2.))
    wrong = fields | {"interval_seconds": jnp.asarray(3.)}
    assert not np.isfinite(np.asarray(advance(state, wrong)[0].T)).all()
    for evaporation in [RHO_0 / 2, RHO_0]:
        exhausted = fields | {"flux_q": jnp.full((2, 2), evaporation)}
        updated, _, height = advance(state, exhausted)
        assert np.all(np.asarray(height) <= 0)
        assert not np.isfinite(np.asarray(updated.T)).all()


def test_native_link_compiler_identity_is_bound(tmp_path, monkeypatch):
    import shlex

    from zhenmode.baselines.mom6 import omip2
    from zhenmode.coupling import sis2
    from zhenmode.provenance.sources import sha256_file

    files = {name: tmp_path / name for name in ["mpif90", "gfortran"]}
    for name, path in files.items():
        path.write_text(name)
    monkeypatch.setattr(sis2.shutil, "which", lambda name: str(files[name]) if name in files else name)
    monkeypatch.setattr(omip2, "_command", lambda command: "compiler-v1" if command[-1] == "--version" else shlex.quote(str(files["gfortran"])))
    recipe = {"tools": {name: {"path": str(path), "sha256": sha256_file(path), "version": "compiler-v1"} for name, path in files.items()}}
    assert sis2._link_tools(recipe) == recipe["tools"]
    files["gfortran"].write_text("changed compiler")
    with pytest.raises(ValueError, match="differs from recorded"):
        sis2._link_tools(recipe)


def test_pressure_diagnostics_use_the_same_dynamic_load_as_the_step():
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    _, initialize, diagnostics, _, _, _ = make_solver_global(
        grid, PhysicsConfig(), dt=1., return_params=True, dynamic_forcing=True,
        polar_cap_rows=0, polar_cap_taper=0)
    state = initialize()
    load = jnp.broadcast_to(jnp.arange(8.)[:, None] * 100., (8, 8))
    base_pressure = diagnostics(state)[1]
    loaded_pressure = diagnostics(state, surface_pressure_pa=load)[1]
    np.testing.assert_allclose(loaded_pressure - base_pressure,
                               np.broadcast_to(np.asarray(load)[..., None], (8, 8, 4)), atol=1.e-8)


def test_frazil_is_water_enthalpy_increase_not_ice_thickness():
    params = SimpleNamespace(
        wet_mask=jnp.array([[1.0, 0.0], [1.0, 1.0]]), dz_node=jnp.full((2, 2, 3), 2.0)
    )
    state = JaxStateG(
        jnp.zeros((2, 2, 3)),
        jnp.zeros((2, 2, 3)),
        jnp.full((2, 2, 3), -2.5),
        jnp.full((2, 2, 3), 35.0),
        jnp.full((2, 2), 0.1),
        jnp.zeros((2, 2)),
    )
    raised, deficit = jax.jit(lambda s: extract_frazil(s, params))(state)
    wet = np.asarray(params.wet_mask) > 0.5
    assert np.all(np.asarray(deficit)[wet] > 0)
    actual_energy = RHO_0 * CP0_TEOS10 * 2.1 * (np.asarray(raised.T[..., 0]) + 2.5)
    np.testing.assert_allclose(actual_energy[wet], np.asarray(deficit)[wet], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(raised.T[..., 1:], state.T[..., 1:])
    np.testing.assert_array_equal(raised.ice, state.ice)
    assert np.asarray(deficit)[0, 1] == 0


@pytest.mark.parametrize("geometry", ["legacy", "fixed_partial_v1"])
def test_ice_surface_pressure_drives_flow_and_balances_sea_level(geometry):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(
        grid,
        z=np.array([0.0, -5.0, -20.0, -1000.0]),
        dz=np.array([5.0, 15.0, 980.0]),
        depth=np.full((8, 8), 1000.0),
    )
    load = np.broadcast_to(1000 * np.sin(2 * np.pi * np.arange(8)[:, None] / 8), (8, 8)).copy()
    _, initialize, _, p, _ = make_solver_global(
        grid,
        PhysicsConfig(),
        dt=1.0,
        return_params=True,
        surface_pressure_pa=load,
        column_geometry=geometry,
        conservative_kv=True,
        localize_conv=True,
        polar_cap_rows=0,
        polar_cap_taper=0,
    )
    state = initialize(T_init=jnp.full((8, 8, 4), 15.0), S_init=jnp.full((8, 8, 4), 35.0))
    acceleration = jax.jit(lambda s: _compute_pressure_gradient(s, p))
    ax, _ = acceleration(state)
    assert np.max(np.abs(np.asarray(ax))) > 1e-9
    # Hydrostatic inverse-barometer balance must cancel the imposed load,
    # at every depth, rather than creating a momentum or volume source.
    balanced = state._replace(eta=-jnp.asarray(load) / (RHO_0 * G_EARTH))
    for value in acceleration(balanced):
        np.testing.assert_allclose(value, 0.0, atol=1e-15, rtol=0)
