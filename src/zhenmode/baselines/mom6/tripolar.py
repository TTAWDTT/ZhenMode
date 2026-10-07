"""Candidate MOM supergrid preparation; no ocean mask, mapping or run claim.

Lengths use shortest great-circle arcs on a sphere. Areas use the FMS default
line integral along edges linear in longitude/latitude, with paired pole
vertices. This is an explicit metric convention, not the unknown convention
that generated the archived OM_1deg areas. Source angular positions are held.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.provenance.sources import (
    load_json,
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)

RADIUS_M = 6371000.0
SOURCE_GRID_SHA256 = "247c01a410e88760ca724edba4447aec2adb17c06fc769beaa0742236c342666"


def _xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def _distance(a, b):
    # atan2 is stable at both coincident and antipodal positions.
    return RADIUS_M * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def _integral(lon, lat):
    dlon = np.roll(lon, -1, axis=-1) - lon
    # Leave ordinary small differences untouched: modulo around pi loses
    # significant bits in small cells near the fold.
    dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)
    dlon = np.where(dlon < -np.pi, dlon + 2 * np.pi, dlon)
    other = np.roll(lat, -1, axis=-1)
    return (
        np.abs(
            np.sum(
                -dlon * np.sin((lat + other) / 2) * np.sinc((other - lat) / (2 * np.pi)), axis=-1
            )
        )
        * RADIUS_M**2
    )


def _pole_area(lon, lat):
    # FMS fix_lon: a pole has incoming and outgoing meridians, not an
    # arbitrary single longitude. Duplicate isolated poles before pairing.
    points = list(zip(lon, lat))
    if all(abs(b) == 90 for _, b in points):
        return 0.0
    fixed = []
    for i, (a, b) in enumerate(points):
        if abs(b) == 90:
            previous, following = points[i - 1], points[(i + 1) % len(points)]
            if previous[1] == b and following[1] == b:
                continue
            if previous[1] != b:
                fixed.append((previous[0], b))
            if following[1] != b:
                fixed.append((following[0], b))
        else:
            fixed.append((a, b))
    if len(fixed) < 3:
        return 0.0
    a, b = np.deg2rad(np.asarray(fixed).T)
    return float(_integral(a, b))


def supergrid_metrics(lon, lat):
    """Regenerate metrics under the declared spherical FMS area convention."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    if (
        lon.ndim != 2
        or lon.shape != lat.shape
        or min(lon.shape) < 3
        or not np.isfinite(lon).all()
        or not np.isfinite(lat).all()
        or np.any(abs(lat) > 90)
    ):
        raise ValueError("finite matching spherical vertex coordinates are required")
    v = _xyz(lon, lat)
    dx, dy = _distance(v[:, :-1], v[:, 1:]), _distance(v[:-1], v[1:])
    a = np.stack((lon[:-1, :-1], lon[:-1, 1:], lon[1:, 1:], lon[1:, :-1]), axis=-1)
    b = np.stack((lat[:-1, :-1], lat[:-1, 1:], lat[1:, 1:], lat[1:, :-1]), axis=-1)
    area = _integral(np.deg2rad(a), np.deg2rad(b))
    for j, i in np.argwhere(np.any(abs(b) == 90, axis=-1)):
        area[j, i] = _pole_area(a[j, i], b[j, i])
    return dx, dy, area


def extend_southern_mercator(lon, lat, south_boundary):
    """Prepend even rows by continuing the measured southern Mercator step."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 2 or min(lon.shape) < 5:
        raise ValueError("matching supergrid coordinate arrays are required")
    if not np.isfinite(south_boundary) or not -89 < south_boundary < 0:
        raise ValueError("southern boundary must be finite and between -89 and 0 degrees")
    n = min(31, lat.shape[0])
    if (
        not np.isfinite(lon).all()
        or not np.isfinite(lat).all()
        or np.any(abs(lat) > 90)
        or not np.allclose(lat[:n], lat[:n, :1], rtol=0, atol=1e-12)
        or not np.allclose(lon[:n], lon[:1], rtol=0, atol=1e-12)
        or np.any(abs(lat[:n]) >= 90)
    ):
        raise ValueError("southern source sector must be regular finite longitude/latitude")
    u = np.arcsinh(np.tan(np.deg2rad(lat[:n, 0])))
    step = float(np.median(np.diff(u)))
    if step <= 0 or not np.allclose(np.diff(u), step, rtol=1e-11, atol=0):
        raise ValueError("southern source sector is not uniform in Mercator latitude")
    target = np.arcsinh(np.tan(np.deg2rad(south_boundary)))
    rows = max(0, int(np.ceil((u[0] - target) / step)))
    rows += rows % 2
    if rows > 256:
        raise ValueError("southern continuation exceeds 256 supergrid rows")
    extra = np.rad2deg(np.arctan(np.sinh(u[0] - step * np.arange(rows, 0, -1))))
    return (
        np.concatenate((np.broadcast_to(lon[0], (rows, lon.shape[1])), lon)),
        np.concatenate((np.broadcast_to(extra[:, None], (rows, lat.shape[1])), lat)),
        rows,
    )


def prepare_tripolar_grid(acquisition_file, south_boundary, output):
    """Write an identity-bound candidate; whole-cell coastline is a later gate."""
    acquisition_file, output = Path(acquisition_file), Path(output)
    acquisition = load_json(acquisition_file)
    if (
        acquisition.get("host") != "ftp.gfdl.noaa.gov"
        or acquisition.get("remote_path") != "/perm/Alistair.Adcroft/MOM6-testing/OM_1deg.tgz"
        or acquisition.get("status") != "official_archive_retrieved_and_regular_members_extracted"
    ):
        raise ValueError("official OM_1deg acquisition record is required")
    row = next(
        (
            r
            for r in acquisition["files"]
            if Path(r["path"]).as_posix().replace("\\", "/") == "files/OM_1deg/INPUT/ocean_hgrid.nc"
        ),
        None,
    )
    if row is None or row["sha256"] != SOURCE_GRID_SHA256:
        raise ValueError("unsupported source grid identity")
    source = acquisition_file.parent / row["path"].replace("\\", "/")
    if sha256_file(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
        raise ValueError("source grid identity mismatch")
    archive = acquisition_file.parent / acquisition["archive_path"]
    if (
        sha256_file(archive) != acquisition["archive_sha256"]
        or archive.stat().st_size != acquisition["bytes"]
    ):
        raise ValueError("source archive identity mismatch")
    with netCDF4.Dataset(source) as data:
        x, y = (np.asarray(data[k][:], dtype=float) for k in ("x", "y"))
    if x.shape != (641, 721) or y.shape != x.shape:
        raise ValueError("OM_1deg angular supergrid shape mismatch")
    x, y, rows = extend_southern_mercator(x, y, south_boundary)
    v = _xyz(x, y)
    seam = float(np.max(np.linalg.norm(v[:, 0] - v[:, -1], axis=-1)))
    fold = float(np.max(np.linalg.norm(v[-1] - v[-1, ::-1], axis=-1)))
    if seam > 1e-10 or fold > 1e-10:
        raise ValueError("periodic seam or tripolar fold mismatch")
    dx, dy, area = supergrid_metrics(x, y)
    ny, nx = area.shape
    coarse_area = area.reshape(ny // 2, 2, nx // 2, 2).sum(axis=(1, 3))
    analytic_domain_area = 2 * np.pi * RADIUS_M**2 * (1 - np.sin(np.deg2rad(y[0, 0])))
    residual = float(abs(area.sum() / analytic_domain_area - 1))
    if not np.all(coarse_area > 0) or residual > 1e-10:
        raise ValueError("candidate cell areas fail positivity or analytic domain closure")
    identities = {
        n: sha256_file(p)
        for n, p in source_paths(source_root(__file__), production_source_modules()).items()
    }
    report = {
        "schema_version": 1,
        "status": "mom_tripolar_metric_candidate_prepared",
        "source_grid_sha256": row["sha256"],
        "source_acquisition_sha256": sha256_file(acquisition_file),
        "source_archive_sha256": acquisition["archive_sha256"],
        "publisher_archive_checksum_verified": False,
        "metric_convention": "sphere_6371000_gc_lengths_fms_lonlat_integral_area_v1",
        "earth_radius_m": RADIUS_M,
        "requested_south_boundary_degN": float(south_boundary),
        "actual_south_boundary_degN": float(y[0, 0]),
        "added_supergrid_rows": rows,
        "native_shape_xy": [nx // 2, ny // 2],
        "area_m2": float(area.sum()),
        "analytic_domain_area_relative_residual": residual,
        "seam_unit_sphere_residual": seam,
        "north_fold_unit_sphere_residual": fold,
        "source_angular_positions_preserved": True,
        "source_metrics_preserved": False,
        "ocean_mask_prepared": False,
        "initial_fields_prepared": False,
        "MOM6_read_verified": False,
        "execution_ready": False,
        "completed_timesteps": 0,
        "geography_certified": False,
        "climate_qualification": False,
        "sources": identities,
    }
    # Validate receipt serialization and input stability before publishing.
    json.dumps(report, allow_nan=False)
    if (
        sha256_file(source) != row["sha256"]
        or sha256_file(archive) != acquisition["archive_sha256"]
        or sha256_file(acquisition_file) != report["source_acquisition_sha256"]
    ):
        raise ValueError("tripolar source changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "ocean_hgrid.nc"
    with netCDF4.Dataset(target, "w", format="NETCDF3_64BIT_OFFSET") as data:
        for name, size in (("nx", nx), ("ny", ny), ("nxp", nx + 1), ("nyp", ny + 1)):
            data.createDimension(name, size)
        for name, value, dims, units in (
            ("x", x, ("nyp", "nxp"), "degrees_east"),
            ("y", y, ("nyp", "nxp"), "degrees_north"),
            ("dx", dx, ("nyp", "nx"), "m"),
            ("dy", dy, ("ny", "nxp"), "m"),
            ("area", area, ("ny", "nx"), "m2"),
        ):
            variable = data.createVariable(name, "f8", dims)
            variable.units = units
            variable[:] = value
        data.metric_convention = report["metric_convention"]
        data.source_grid_sha256 = row["sha256"]
        data.qualification = (
            "candidate metrics only; no mask, forcing, initial fields or actual MOM6 read"
        )
    report["output"] = {
        "path": target.name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }
    (output / "tripolar_grid.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return report
