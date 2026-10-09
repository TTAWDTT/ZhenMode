"""Verified JRA weather point -> prescribed six-hour mean stress for controlled water."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from zhenmode.execution.runs import write_json
from zhenmode.model.inputs.forcing.jra55 import FIELDS
from zhenmode.model.solver.physics.air_sea import AirState, open_water_fluxes
from zhenmode.preparation.forcing import _original_field
from zhenmode.provenance.sources import load_json, package_source_hashes, sha256_file


def prepare(acquisition, output, *, latitude=45.0, longitude=180.0, sst_c=15.0):
    """Use the existing checked source reader; no downloader, regrid or missing fallback."""
    acquisition = Path(acquisition)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    received = load_json(acquisition)
    data_kind = received.get("data_kind", "observed")
    if (
        received["product"] != "JRA55-do"
        or received["version"] != "1.4.0"
        or received["year"] != 1958
        or received["execution_status"] != "completed"
        or data_kind not in ("observed", "manufactured")
        or set(received["files"]) != {v[0] for v in FIELDS.values()}
        or received["verified"] != {k: v["sha256"] for k, v in received["files"].items()}
    ):
        raise ValueError("requires complete identity-verified JRA55-do v1.4.0 1958 receipt")
    if not (-90 < latitude < 90 and 0 <= longitude < 360 and 0 < sst_c < 40):
        raise ValueError("invalid warm-water point request")
    source = package_source_hashes(__file__)
    weather = {}
    identities = {}
    position = None
    axes = None
    times = None
    with ExitStack() as stack:
        for field in ("wind_u", "wind_v", "temperature_k", "specific_humidity", "pressure_pa"):
            ds, time, _, records, birth = _original_field(
                stack, [acquisition], [received], field, data_kind
            )
            lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
            if axes is not None and (
                not np.array_equal(lat, axes[0]) or not np.array_equal(lon, axes[1])
            ):
                raise ValueError("weather grids disagree")
            axes = (lat, lon)
            iy = int(np.argmin(abs(lat - latitude)))
            ix = int(np.argmin(abs((lon - longitude + 180) % 360 - 180)))
            selected = (float(lat[iy]), float(lon[ix]))
            if position is not None and selected != position:
                raise ValueError("weather point locations disagree")
            position = selected
            variable = FIELDS[field][0]
            dates = [str(ds["time"].units), str(ds["time"].calendar)]
            sample = ds[variable][:3, iy, ix]
            if np.ma.is_masked(sample) or not np.isfinite(sample).all():
                raise ValueError("missing point weather")
            if times is not None and not np.array_equal(time[:3], times):
                raise ValueError("weather clocks disagree")
            times = time[:3]
            weather[field] = np.asarray(sample, dtype=float)
            print(f"validated_weather={field} selected_records=3", flush=True)
            identities[field] = dict(
                original=birth, raw_clock=dates, record_indices=[0, 1, 2], source_kind=data_kind
            )
        import netCDF4

        decoded = netCDF4.num2date(ds["time"][:3], ds["time"].units, calendar=ds["time"].calendar)
        if [str(date) for date in decoded] != [
            "1958-01-01 00:00:00",
            "1958-01-01 03:00:00",
            "1958-01-01 06:00:00",
        ]:
            raise ValueError("unexpected source date window")
        if not (
            np.all((weather["temperature_k"] > 200) & (weather["temperature_k"] < 340))
            and np.all((weather["specific_humidity"] >= 0) & (weather["specific_humidity"] < 0.1))
            and np.all((weather["pressure_pa"] > 50000) & (weather["pressure_pa"] < 120000))
        ):
            raise ValueError("invalid SI weather values")
        zeros = np.zeros(3)
        air = AirState(
            **weather,
            shortwave_down=zeros,
            longwave_down=zeros,
            rain=zeros,
            snow=zeros,
            runoff=zeros,
            calving=zeros,
        )
        flux = open_water_fluxes(air, sst_c, 0.0, 0.0)
        tx, ty = np.asarray(flux.tau_x), np.asarray(flux.tau_y)
        if not np.isfinite(tx).all() or not np.isfinite(ty).all():
            raise ValueError("invalid derived stress")
        weights = np.array([0.25, 0.5, 0.25])
        result = dict(
            schema="jra-point-mean-stress-v1",
            data_kind=data_kind + "_weather_derived_stress",
            requested_point={"latitude": latitude, "longitude": longitude},
            actual_point={"latitude": position[0], "longitude": position[1]},
            source_dates=[str(date) for date in decoded],
            elapsed_seconds=[0, 10800, 21600],
            atmospheric_samples={k: v.tolist() for k, v in weather.items()},
            stress_samples_N_m2={"tau_x": tx.tolist(), "tau_y": ty.tolist()},
            tau_x_N_m2=float(weights @ tx),
            tau_y_N_m2=float(weights @ ty),
            weights=weights.tolist(),
            derivation="LY2009/Gill existing bulk implementation; SST fixed, ocean current zero; trapezoidal mean of three stress samples",
            assumptions={
                "sst_C": sst_c,
                "ocean_current_m_s": [0, 0],
                "spatial_representation": "one nearest original grid point, uniform prescribed stress over controlled water",
                "other_fluxes": "not applied",
            },
            sources=identities,
            acquisition_sha256=sha256_file(acquisition),
            package_source_sha256=source,
            industrial_qualified=False,
        )
    # Recheck only the five weather originals actually used, not all unused forcing roles.
    for field in weather:
        identity = received["files"][FIELDS[field][0]]
        if sha256_file(acquisition.parent / identity["filename"]) != identity["sha256"]:
            raise ValueError("weather source changed")
    if package_source_hashes(__file__) != source:
        raise ValueError("wind derivation code changed")
    write_json(output, result, create=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latitude", type=float, default=45.0)
    parser.add_argument("--longitude", type=float, default=180.0)
    parser.add_argument("--sst-c", type=float, default=15.0)
    args = parser.parse_args(argv)
    prepare(
        args.acquisition,
        args.output,
        latitude=args.latitude,
        longitude=args.longitude,
        sst_c=args.sst_c,
    )
    return 0
