"""Bounded CF JRA55-do reads on an explicitly prepared native rectangular grid.

Instantaneous fields interpolate in physical seconds. Interval means integrate
their actual bounds without smoothing. No download, implicit regridding,
calendar conversion, or missing-weather fallback occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.model.solver.physics.air_sea import AirState
from zhenmode.provenance.sources import load_json, sha256_file

TIME_UNITS = "seconds since 1970-01-01 00:00:00"
# Field -> (CF variable, exact SI spellings, cadence, interpretation, height).
FIELDS = {
    "wind_u": ("uas", ("m s-1", "m/s"), 10800, "instant", 10),
    "wind_v": ("vas", ("m s-1", "m/s"), 10800, "instant", 10),
    "temperature_k": ("tas", ("K",), 10800, "instant", 10),
    "specific_humidity": ("huss", ("1", "kg kg-1"), 10800, "instant", 10),
    "pressure_pa": ("psl", ("Pa",), 10800, "instant", None),
    "shortwave_down": ("rsds", ("W m-2", "W/m2"), 10800, "mean", None),
    "longwave_down": ("rlds", ("W m-2", "W/m2"), 10800, "mean", None),
    "rain": ("prra", ("kg m-2 s-1",), 10800, "mean", None),
    "snow": ("prsn", ("kg m-2 s-1",), 10800, "mean", None),
    "runoff": ("friver", ("kg m-2 s-1",), 86400, "mean", None),
    "calving": ("licalvf", ("kg m-2 s-1",), 86400, "mean", None),
}


def _exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} fields must be exactly {sorted(keys)}")


def _seconds(values, variable):
    calendar = getattr(variable, "calendar", None)
    if calendar not in {"gregorian", "proleptic_gregorian", "standard"}:
        raise ValueError("JRA requires Gregorian time, not a repeating or 360_day calendar")
    dates = netCDF4.num2date(values, variable.units, calendar=calendar)
    return np.asarray(netCDF4.date2num(dates, TIME_UNITS, calendar="proleptic_gregorian"), dtype=float)


def conservative_rectilinear_weights(source_lon_bounds, source_lat_bounds,
                                     target_lon_bounds, target_lat_bounds):
    """Explicit first-order spherical remap of global rectangular cell means.

    Bounds are degrees, ordered, contiguous and cover the whole sphere.
    Gaussian latitude bands are supported; curvilinear/tripolar grids are not.
    Returns (latitude, longitude) matrices mapping data[lat,lon] as Y@data@X.T.
    The operation covers all cells. Applying an ocean mask/routing land fluxes
    is a separate physical operation and needs its own budget/receipt.
    """
    def bounds(values, latitude):
        values = np.array(values, dtype=float, copy=True)
        if (values.ndim != 2 or values.shape[1] != 2 or len(values) == 0
                or not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0])
                or not np.allclose(values[:-1, 1], values[1:, 0], atol=1e-9, rtol=0)):
            raise ValueError('remapping requires finite ordered contiguous cell bounds')
        if latitude:
            if not np.allclose([values[0, 0], values[-1, 1]], [-90, 90], atol=1e-9, rtol=0):
                raise ValueError('latitude bounds must cover both poles in degrees')
            return np.sin(np.deg2rad(values))
        if not np.isclose(values[-1, 1] - values[0, 0], 360, atol=1e-9, rtol=0):
            raise ValueError('longitude bounds must cover 360 degrees')
        values -= np.floor(values[0, 0] / 360) * 360
        return values

    source_x, target_x = bounds(source_lon_bounds, False), bounds(target_lon_bounds, False)
    source_y, target_y = bounds(source_lat_bounds, True), bounds(target_lat_bounds, True)

    def overlap(source, target):
        return np.maximum(0, np.minimum(target[:, 1, None], source[None, :, 1])
                          - np.maximum(target[:, 0, None], source[None, :, 0]))

    x = sum(overlap(source_x + shift, target_x) for shift in (-360, 0, 360))
    y = overlap(source_y, target_y)
    x /= np.diff(target_x, axis=1)
    y /= np.diff(target_y, axis=1)
    if not np.allclose(x.sum(axis=1), 1, atol=1e-12, rtol=0) or not np.allclose(y.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError('source bounds leave uncovered target cells')
    return y, x


def remap_rectilinear_means(values, weights):
    """Use declared weights, rejecting missing cells instead of renormalizing."""
    y, x = weights
    data = np.ma.asarray(values)
    if (data.ndim != 2 or data.shape != (y.shape[1], x.shape[1])
            or np.ma.is_masked(data) or not np.isfinite(data).all()
            or not np.isfinite(y).all() or not np.isfinite(x).all()
            or np.any(y < 0) or np.any(x < 0)
            or not np.allclose(y.sum(axis=1), 1, atol=1e-12, rtol=0)
            or not np.allclose(x.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError('invalid remapping weights or missing/mismatched source means')
    return y @ np.asarray(data) @ x.T


def bilinear_rectilinear_weights(source_lon, source_lat, target_lon, target_lat):
    """Periodic geographic components/state; refuse latitude extrapolation."""
    def linear(source, target, periodic):
        source, target = np.asarray(source, float), np.asarray(target, float)
        if (source.ndim != 1 or target.ndim != 1 or len(source) < 2 or len(target) == 0
                or not np.isfinite(source).all() or not np.isfinite(target).all()
                or np.any(np.diff(source) <= 0)):
            raise ValueError('bilinear coordinates must be finite ordered vectors')
        count = len(source)
        if periodic:
            if source[-1]-source[0] >= 360:
                raise ValueError('longitude contains a duplicate seam or more than one globe')
            target = (target-source[0]) % 360 + source[0]
            source = np.r_[source, source[0]+360]
        elif target.min() < source[0] or target.max() > source[-1]:
            raise ValueError('bilinear latitude extrapolation is forbidden')
        left = np.clip(np.searchsorted(source, target, side='right')-1, 0, len(source)-2)
        alpha = (target-source[left])/(source[left+1]-source[left])
        weights = np.zeros((len(target), count))
        weights[np.arange(len(target)), left] = 1-alpha
        weights[np.arange(len(target)), (left+1) % count] += alpha
        return weights
    return linear(source_lat, target_lat, False), linear(source_lon, target_lon, True)


@dataclass(frozen=True)
class _Record:
    path: Path
    index: int
    time: float
    bounds: tuple[float, float] | None
    transpose: bool


class JRA55Forcing:
    """Stream a checked window; only sampled records enter memory.

    Manifest paths are relative to the manifest. Multiple chronological shards
    are supported. A transformed product needs separate original/weight receipts
    before creating this native manifest. Test fixtures are labelled manufactured.
    """

    def __init__(self, manifest_path, *, lon, lat, wet_mask, start_seconds, end_seconds):
        self.manifest_path = Path(manifest_path).resolve()
        document = load_json(self.manifest_path)
        _exact_keys(document, {"schema_version", "product", "version", "data_kind", "files"}, "forcing")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("forcing schema_version must be 1")
        if document["product"] != "JRA55-do" or document["version"] != "1.4.0":
            raise ValueError("forcing product/version must be JRA55-do/1.4.0")
        if document["data_kind"] not in {"observed", "manufactured"}:
            raise ValueError("data_kind must explicitly distinguish observed/manufactured")
        self.data_kind = document["data_kind"]
        self.start, self.end = float(start_seconds), float(end_seconds)
        if not np.isfinite([self.start, self.end]).all() or self.start >= self.end:
            raise ValueError("forcing window must be finite with start < end")
        self.lon, self.lat = np.array(lon), np.array(lat)
        mask = np.asarray(wet_mask)
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("wet mask must contain only finite zero/one values")
        self.wet = mask.astype(bool)
        if (self.lon.ndim != 1 or self.lat.ndim != 1 or
                not np.isfinite(self.lon).all() or not np.isfinite(self.lat).all() or
                len(self.lon) == 0 or len(self.lat) == 0 or
                np.any(np.diff(self.lon) <= 0) or np.any(np.diff(self.lat) <= 0) or
                self.wet.shape != (len(self.lon), len(self.lat)) or not self.wet.any()):
            raise ValueError("native lon/lat and wet grid are invalid")
        self.records = {name: [] for name in FIELDS}
        self.receipts, self.file_stats = [], {}
        if not isinstance(document["files"], list) or not document["files"]:
            raise ValueError("forcing files must be a nonempty list")
        seen = set()
        for entry in document["files"]:
            _exact_keys(entry, {"field", "path", "sha256", "bytes", "source_url", "license"}, "forcing file")
            name = entry["field"]
            if name not in FIELDS or any(not isinstance(entry[key], str) or not entry[key].strip()
                                         for key in ("path", "source_url", "license")):
                raise ValueError("unknown field or missing file source/license")
            path = (self.manifest_path.parent / entry["path"]).resolve()
            if (name, path) in seen:
                raise ValueError("duplicate forcing field/file")
            seen.add((name, path))
            if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
                raise ValueError("forcing bytes must be a positive integer")
            if not path.is_file():
                raise FileNotFoundError(f"missing forcing file: {path}")
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"forcing file identity mismatch: {path}")
            self._read_axis(name, path)
            self.file_stats[path] = path.stat()
            self.receipts.append(entry | {"resolved_path": str(path)})
        self.times, self.bounds = {}, {}
        for name, records in self.records.items():
            _, _, cadence, interpretation, _ = FIELDS[name]
            if not records:
                raise ValueError(f"missing forcing field: {name}")
            records.sort(key=lambda record: record.time)
            times = np.array([record.time for record in records])
            if interpretation == "instant":
                if len(times) < 2 or not np.allclose(np.diff(times), cadence, rtol=0, atol=1e-6):
                    raise ValueError(f"instant cadence/gap/duplicate: {name}")
                first, last = times[0], times[-1]
            else:
                bounds = np.array([record.bounds for record in records])
                if (not np.allclose(bounds[:, 1] - bounds[:, 0], cadence, rtol=0, atol=1e-6) or
                        not np.allclose(bounds[1:, 0], bounds[:-1, 1], rtol=0, atol=1e-6) or
                        not np.all((times >= bounds[:, 0]) & (times <= bounds[:, 1]))):
                    raise ValueError(f"mean bounds/gap/overlap: {name}")
                first, last = bounds[0, 0], bounds[-1, 1]
                self.bounds[name] = bounds
            if first > self.start or last < self.end:
                raise ValueError(f"forcing window not covered without extrapolation: {name}")
            self.times[name] = times

    def _read_axis(self, name, path):
        variable_name, units, _, interpretation, height = FIELDS[name]
        with netCDF4.Dataset(path) as ds:
            if getattr(ds, "source_id", None) != "MRI-JRA55-do-1-4-0":
                raise ValueError("NetCDF source_id does not match the declared version")
            if getattr(ds, "data_kind", "observed") != self.data_kind:
                raise ValueError("NetCDF data_kind does not match manifest")
            variable = ds[variable_name]
            if variable.units not in units:
                raise ValueError(f"wrong SI units for {name}: {variable.units}")
            if height is not None:
                h = ds["height"]
                if h.units != "m" or h.size != 1 or float(h[:].item()) != height:
                    raise ValueError(f"{name} must be measured at {height}m")
            for axis, expected, allowed in (("lon", self.lon, {"degrees_east"}),
                                            ("lat", self.lat, {"degrees_north"})):
                coordinate = ds[axis]
                if (coordinate.dimensions != (axis,) or coordinate.units not in allowed or
                        np.ma.is_masked(coordinate[:]) or not np.array_equal(coordinate[:], expected)):
                    raise ValueError(f"native {axis} grid mismatch; no implicit remapping")
            if variable.dimensions not in {("time", "lat", "lon"), ("time", "lon", "lat")}:
                raise ValueError("forcing dimensions must identify time/lon/lat explicitly")
            transpose = variable.dimensions == ("time", "lat", "lon")
            times = _seconds(ds["time"][:], ds["time"])
            if times.ndim != 1 or not np.isfinite(times).all() or len(times) == 0:
                raise ValueError("invalid forcing time axis")
            if interpretation == "mean":
                bounds_name = getattr(ds["time"], "bounds", None)
                if bounds_name not in ds.variables or getattr(variable, "cell_methods", None) != "time: mean":
                    raise ValueError("mean flux requires explicit time bounds and time: mean")
                bounds = _seconds(ds[bounds_name][:], ds["time"])
                if bounds.shape != (len(times), 2) or not np.isfinite(bounds).all():
                    raise ValueError("invalid mean flux bounds")
            else:
                if getattr(variable, "cell_methods", "time: point") != "time: point":
                    raise ValueError("instant field must not be a temporal mean")
                bounds = None
            for index, time in enumerate(times):
                self.records[name].append(_Record(path, index, time,
                    None if bounds is None else tuple(bounds[index]), transpose))

    def _values(self, name, record):
        stat = record.path.stat()
        before = self.file_stats[record.path]
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
            raise ValueError("forcing file changed after verification")
        with netCDF4.Dataset(record.path) as ds:
            data = ds[FIELDS[name][0]][record.index]
            data = data.T if record.transpose else data
            array = np.ma.asarray(data).filled(np.nan)
            if not np.isfinite(array[self.wet]).all():
                raise ValueError(f"missing/nonfinite wet forcing: {name}")
            # Dry cells are outside the physical problem, not missing-weather fallback.
            result = np.where(self.wet, array, 0)
            wet_values = result[self.wet]
            if name == "specific_humidity" and not np.all((wet_values >= 0) & (wet_values < 0.1)):
                raise ValueError("specific humidity outside kg/kg physical range")
            if name == "temperature_k" and not np.all((wet_values > 180) & (wet_values < 340)):
                raise ValueError("air temperature outside Kelvin physical range")
            if name == "pressure_pa" and not np.all((wet_values > 50000) & (wet_values < 120000)):
                raise ValueError("sea level pressure outside Pa physical range")
            if FIELDS[name][3] == "mean" and np.any(wet_values < 0):
                raise ValueError(f"negative downward input: {name}")
            return result

    def sample(self, time_seconds, *, interval_end_seconds):
        """Weather at the stated stage; flux means over the stated integration interval."""
        start, end = float(time_seconds), float(interval_end_seconds)
        if not np.isfinite([start, end]).all() or not self.start <= start < end <= self.end:
            raise ValueError("sample interval is outside the checked forcing window")
        values = {}
        for name, records in self.records.items():
            if FIELDS[name][3] == "instant":
                right = min(np.searchsorted(self.times[name], start, side="right"), len(records) - 1)
                left = right - 1
                fraction = (start - records[left].time) / (records[right].time - records[left].time)
                values[name] = ((1 - fraction) * self._values(name, records[left])
                                + fraction * self._values(name, records[right]))
            else:
                bounds = self.bounds[name]
                begin = np.searchsorted(bounds[:, 1], start, side="right")
                finish = np.searchsorted(bounds[:, 0], end, side="left")
                integral = np.zeros(self.wet.shape)
                for index in range(begin, finish):
                    lo, hi = records[index].bounds
                    integral += (min(end, hi) - max(start, lo)) * self._values(name, records[index])
                values[name] = integral / (end - start)
        return AirState(**values)
