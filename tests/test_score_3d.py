import numpy as np

from score_external_3d import _vertical_mask
from score_solver_3d import score_solver_3d


def test_vertical_mask_removes_below_seafloor_layers(tmp_path):
    z = np.array([0.0, -100.0])
    depth = np.array([[500.0, 50.0], [50.0, 500.0]])
    ocean2d = np.ones((2, 2), dtype=bool)
    mask = _vertical_mask(depth, ocean2d, z)
    assert mask[0, 0, 1]
    assert not mask[0, 1, 1]
    assert mask.sum() == 6


def test_solver_3d_excludes_ghost_layers(tmp_path, monkeypatch):
    # One shallow column has a ghost layer at 100m; it must not be scored.
    z = np.array([0.0, -100.0])
    depth = np.array([[500.0, 50.0], [50.0, 500.0]])
    reference = np.full((2, 2, 2), 10.0)
    T = np.full_like(reference, 10.0)
    T[0, 1, 1] = 90.0
    snap = np.stack([T, np.zeros_like(T), np.zeros_like(T), np.full_like(T, 35.0)])
    npz_path = tmp_path / "run.npz"
    np.savez(
        npz_path,
        T_init=reference,
        S_init=np.full_like(reference, 35.0),
        wet_mask=np.ones((2, 2)),
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        z=z,
    )
    snap_path = tmp_path / "snap.npy"
    np.save(snap_path, snap)
    from netCDF4 import Dataset
    geom_path = tmp_path / "geom.nc"
    with Dataset(geom_path, "w") as ds:
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        v = ds.createVariable("D", "f8", ("lat", "lon"))
        v[:] = depth.T
    result = score_solver_3d(
        str(npz_path), str(snap_path), depth_file=str(geom_path), depth_var="D"
    )
    assert result["global_3d"]["raw_rmse"] == 0.0
    assert result["global_3d"]["n"] == 6
    assert result["depth_mask_applied"]

def test_solver_3d_requires_depth_mask(tmp_path):
    import numpy as np

    from score_solver_3d import score_solver_3d
    ref = np.full((2, 2, 1), 10.0)
    npz_path = tmp_path / "run.npz"
    np.savez(
        npz_path,
        T_init=ref,
        S_init=np.full_like(ref, 35.0),
        wet_mask=np.ones((2, 2)),
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        z=np.array([0.0]),
    )
    snap_path = tmp_path / "snap.npy"
    np.save(snap_path, np.stack([ref, np.zeros_like(ref), np.zeros_like(ref), np.full_like(ref, 35.0)]))
    result = score_solver_3d(str(npz_path), str(snap_path))
    assert result["depth_mask_applied"] is False
    assert result["verdict"] == "FAIL"

def test_solver_3d_uses_snapshot_window(tmp_path):
    import numpy as np

    from score_solver_3d import score_solver_3d
    z = np.array([0.0, -100.0])
    ref = np.full((2, 2, 2), 10.0)
    T0 = ref.copy()
    T0[0, 0, 0] = 8.0
    T1 = ref.copy()
    T1[0, 0, 0] = 12.0
    snap0 = np.stack([T0, np.zeros_like(T0), np.zeros_like(T0), np.full_like(T0, 35.0)])
    snap1 = np.stack([T1, np.zeros_like(T1), np.zeros_like(T1), np.full_like(T1, 35.0)])
    npz_path = tmp_path / "run.npz"
    np.savez(
        npz_path,
        T_init=ref,
        S_init=np.full_like(ref, 35.0),
        wet_mask=np.ones((2, 2)),
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        z=z,
    )
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    np.save(snap_dir / "snap_00000.npy", snap0)
    np.save(snap_dir / "snap_00001.npy", snap1)
    result = score_solver_3d(
        str(npz_path), snap_dir=str(snap_dir), snap_days=10.0, steady_days=20.0
    )
    assert result["n_snapshots"] == 2
    assert result["global_3d"]["raw_rmse"] == 0.0


