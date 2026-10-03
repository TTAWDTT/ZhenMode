"""Explicit-format preflight readers bound to immutable byte snapshots.

This module is separate from the scientific/legacy WOA reader. Unknown twins
are rejected rather than assigned a role from the requested CLI argument.
"""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from netCDF4 import Dataset

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.io.data_quality import normalize_source


@dataclass(frozen=True)
class InputSnapshot:
    path: Path
    content: bytes
    sha256: str

    @classmethod
    def read(cls, path):
        resolved = Path(path).resolve(strict=True)
        content = resolved.read_bytes()
        snapshot = cls(resolved, content, hashlib.sha256(content).hexdigest())
        snapshot.verify_unchanged()
        return snapshot

    def identity(self):
        return {'filename': self.path.name, 'bytes': len(self.content),
                'sha256': self.sha256, 'binding': 'immutable_read_snapshot'}

    def verify_unchanged(self):
        try:
            with self.path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as error:
            raise ValueError('input changed or disappeared after snapshot read') from error
        if digest != self.sha256:
            raise ValueError('input changed after snapshot read')


def load_snapshot_npz(snapshot):
    try:
        with np.load(BytesIO(snapshot.content), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except (ValueError, OSError, EOFError, BadZipFile) as error:
        raise ValueError('unsupported or corrupted NPZ snapshot') from error


def load_climatology_snapshot(snapshot, variable):
    """Recognize a typed twin or a WOA-compatible, metadata-described NetCDF."""
    if snapshot.content.startswith(b'PK'):
        raw = load_snapshot_npz(snapshot)
        format_name = np.asarray(raw.get('source_format'))
        if format_name.shape != () or format_name.item() != 'ocean.woa_twin.v1':
            raise ValueError('unsupported twin format identity')
        _, identity_ok, encoding_ok = normalize_source(raw, variable)
        if not identity_ok or not encoding_ok:
            raise ValueError('unsupported twin variable, units, format or missing encoding')
        return raw
    role, name = {'T': ('temperature', 't_an'), 'S': ('salinity', 's_an')}[variable]
    try:
        with Dataset('input_snapshot', memory=snapshot.content) as dataset:
            source = dataset.variables[name]
            if source.dimensions != ('time', 'depth', 'lat', 'lon'):
                raise ValueError('unsupported NetCDF dimension contract')
            if source.shape[0] != 1:
                raise ValueError('unsupported NetCDF time dimension')
            if '_FillValue' not in source.ncattrs():
                raise ValueError('unsupported NetCDF missing encoding: no _FillValue')
            # netCDF4 applies its explicit mask/packing metadata before conversion.
            data = np.ma.asarray(source[0], dtype=float).filled(np.nan)
            raw = {key: np.ma.asarray(dataset.variables[key][:], dtype=float)
                   for key in ('lon', 'lat', 'depth')}
            raw.update(data=data, source_format='woa-compatible-netcdf.v1',
                       variable=role, units=getattr(source, 'units', None),
                       missing_encoding='netcdf_masked', fill_value=source._FillValue,
                       longitude_units=getattr(dataset.variables['lon'], 'units', None),
                       latitude_units=getattr(dataset.variables['lat'], 'units', None),
                       depth_units=getattr(dataset.variables['depth'], 'units', None),
                       depth_positive=getattr(dataset.variables['depth'], 'positive', None))
    except (OSError, KeyError, RuntimeError) as error:
        raise ValueError('unsupported climatology format or variable identity') from error
    _, identity_ok, encoding_ok = normalize_source(raw, variable)
    if not identity_ok or not encoding_ok:
        raise ValueError('unsupported NetCDF variable, units or missing encoding')
    return raw

preserve_legacy_names(globals(), 'input_sources')
