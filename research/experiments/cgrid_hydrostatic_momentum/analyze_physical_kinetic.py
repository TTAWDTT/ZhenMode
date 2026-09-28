"""Independent NumPy probe for the registered physical horizontal KE metric."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def angular_gram(phi_s, delta_phi, delta_lambda, order):
    phi_n = phi_s + delta_phi
    sin_width = np.sin(phi_n) - np.sin(phi_s)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mu = (nodes + 1.) * .5
    weights = weights * .5
    phi = np.arcsin(np.sin(phi_s) + mu * sin_width)
    w = (phi - phi_s) / delta_phi
    beta = (delta_phi / delta_lambda) * (mu - w) / np.cos(phi)
    gs = (1. - mu) * np.cos(phi_s) / np.cos(phi)
    gn = mu * np.cos(phi_n) / np.cos(phi)
    weighted_basis = np.sqrt(weights[:, None]) * np.stack((-beta, beta, gs, gn), axis=-1)
    angular = weighted_basis.T @ weighted_basis
    xi_nodes, xi_weights = np.polynomial.legendre.leggauss(8)
    xi_nodes, xi_weights = (xi_nodes + 1.) * .5, xi_weights * .5
    xbasis = np.stack((1. - xi_nodes, xi_nodes), axis=-1)
    xgram = np.einsum("q,qi,qj->ij", xi_weights, xbasis, xbasis)
    angular[:2, :2] += xgram
    return angular


def cell_gram(phi_s, delta_phi, delta_lambda, intervals, area, order):
    """Integrate a cell's four normal-velocity coefficients over wet overlaps."""
    angular = angular_gram(phi_s, delta_phi, delta_lambda, order)
    gram = np.zeros((4, 4), dtype=np.float64)
    for left in range(4):
        for right in range(4):
            overlap = max(0., min(intervals[left][1], intervals[right][1]) - max(intervals[left][0], intervals[right][0]))
            if overlap == 0.:
                continue
            gram[left, right] = area * overlap * angular[left, right]
    return gram


def direct_field_integral(phi_s, delta_phi, delta_lambda, intervals, area, coefficients):
    """Integrate physical components from mapped Q on separate depth partitions."""
    phi_n = phi_s + delta_phi
    sine_width = np.sin(phi_n) - np.sin(phi_s)
    radius = np.sqrt(area / (delta_lambda * sine_width))
    nodes, weights = np.polynomial.legendre.leggauss(96)
    fraction, weights = .5 * (nodes + 1.), .5 * weights
    longitude_nodes, longitude_weights = np.polynomial.legendre.leggauss(7)
    longitude = .5 * (longitude_nodes + 1.)[:, None]
    longitude_weights *= .5
    phi = np.arcsin(np.sin(phi_s) + fraction * sine_width)
    weight = sine_width / (delta_phi * np.cos(phi))
    primitive = (phi - phi_s) / delta_phi
    density = coefficients * radius * np.array([delta_phi, delta_phi, delta_lambda * np.cos(phi_s), delta_lambda * np.cos(phi_n)])
    breaks = np.unique(np.asarray(intervals).reshape(-1))
    energy = 0.
    for start, end in zip(breaks[:-1], breaks[1:]):
        middle = .5 * (start + end)
        trace = density * np.array([left < middle < right for left, right in intervals])
        mapped_east = weight * ((1. - longitude) * trace[0] + longitude * trace[1])
        mapped_north = (1. - fraction) * trace[2] + fraction * trace[3] + (trace[1] - trace[0]) * (fraction - primitive)
        east = mapped_east * np.cos(phi) / (radius * sine_width)
        north = mapped_north / (radius * delta_lambda * np.cos(phi))
        energy += .5 * area * (end - start) * np.sum(longitude_weights[:, None] * weights[None, :] * (east ** 2 + north ** 2))
    return energy


def canonical_probe():
    phi_s, delta_phi, delta_lambda, area = .31, .12, .2, 7.3
    interval_sets = [[(0., 2.), (0., 2.), (0., 2.), (0., 2.)],
                     [(0., 1.4), (.3, 2.), (0., 2.), (.6, 2.)]]
    results = []
    for intervals in interval_sets:
        grams = [cell_gram(phi_s, delta_phi, delta_lambda, intervals, area, order) for order in (24, 48, 96)]
        errors = [float(np.max(np.abs(grams[i] - grams[i + 1])) / max(np.max(np.abs(grams[i + 1])), 1.)) for i in range(2)]
        gram = grams[-1]
        mixed = float(np.max(np.abs(gram - np.diag(np.diag(gram)))))
        eigen_min = float(np.linalg.eigvalsh(gram)[0])
        coeff = np.array([1., -.7, .4, 1.2])
        energy = .5 * coeff @ gram @ coeff
        direct = direct_field_integral(phi_s, delta_phi, delta_lambda, intervals, area, coeff)
        diagonal_only = .5 * np.sum(np.diag(gram) * coeff ** 2)
        heights = np.array([end - start for start, end in intervals])
        south_fraction = (np.sin(phi_s + .5 * delta_phi) - np.sin(phi_s)) / (np.sin(phi_s + delta_phi) - np.sin(phi_s))
        shares = np.array([.5, .5, south_fraction, 1. - south_fraction])
        proxy_common = .5 * area * np.sum(shares * heights * coeff ** 2)
        proxy_half = .5 * area * 2. * np.sum(shares * coeff ** 2)
        uniform = np.array([1., 1., 0., 0.])
        checker = np.array([1., -1., 0., 0.])
        uniform_ke = .5 * uniform @ grams[-1] @ uniform
        checker_ke = .5 * checker @ grams[-1] @ checker
        full_column = intervals == interval_sets[0]
        uniform_expected = .5 * area * (intervals[0][1] - intervals[0][0]) if full_column else None
        uniform_error = (abs(uniform_ke - uniform_expected) if full_column else 0.)
        results.append({"quadrature_orders": [24, 48, 96], "relative_differences": errors,
                        "converged_1e-12": max(errors) <= 1e-12, "max_mixed_entry": mixed,
                        "minimum_eigenvalue": eigen_min, "symmetric": bool(np.array_equal(gram, gram.T)),
                        "positive_semidefinite": eigen_min >= -1e-12, "same_coefficients_gram_ke": float(energy),
                        "direct_field_integral": float(direct), "direct_gram_relative_error": float(abs(direct - energy) / abs(direct)),
                        "diagonal_only_same_field_rejected": bool(abs(diagonal_only - direct) > 1e-12 * abs(direct)),
                        "common_wet_lumped_proxy": float(proxy_common), "half_prism_kinematic_proxy": float(proxy_half), "uniform_zonal_ke": float(uniform_ke),
                        "uniform_zonal_expected_full_wet": uniform_expected, "checkerboard_ke": float(checker_ke),
                        "uniform_zonal_absolute_error": float(uniform_error),
                        "checkerboard_same_face_lumped_proxy": float(.5 * area * np.sum(shares * heights * checker ** 2)), "gram": gram.tolist()})
    canonical_ok = all(r["converged_1e-12"] and r["symmetric"] and r["positive_semidefinite"]
                       and r["max_mixed_entry"] > 0 and r["uniform_zonal_absolute_error"] <= 1e-12
                       and r["checkerboard_ke"] > 0 and r["direct_gram_relative_error"] <= 1e-12
                       and r["diagonal_only_same_field_rejected"] for r in results)
    return {"status": "PENDING", "canonical_checks": "PASS" if canonical_ok else "FAIL",
            "scope": "canonical_cell_integral_only_no_actual_reference_snapshots",
            "results": results,
            "limitations": ["Actual-Q eight-snapshot integration and host verification are pending.",
                            "Canonical probe does not claim full energy, dynamics, or century qualification."]}


def validate_report(report_path):
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    try:
        from verify_physical_velocity_reference import verify_physical_report
        verify_physical_report(report)
    except Exception as error:
        problems.append(f"host verifier rejected report: {error}")
    for run in report.get("runs", []):
        snapshot = (ROOT / run["snapshot_path"]).resolve()
        if not snapshot.is_file():
            problems.append(f"missing snapshot: {snapshot}")
            continue
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != run.get("snapshot_sha256"):
            problems.append(f"snapshot hash mismatch: {snapshot}")
            continue
        with np.load(snapshot, allow_pickle=False) as data:
            required = {"initial_volume", "final_volume", "last_previous_volume", "last_east_flux", "last_north_flux",
                        "last_vertical_flux", "last_trace_side_top", "last_trace_side_bottom", "last_trace_side_density",
                        "last_trace_cell_top", "last_trace_cell_height", "area", "thickness", "longitude_edges", "latitude_edges", "interfaces"}
            missing = required.difference(data.files)
            if missing:
                problems.append(f"snapshot missing arrays {sorted(missing)}: {snapshot}")
                continue
            for key in required:
                array = data[key]
                if array.dtype != np.float64 or not np.all(np.isfinite(array)):
                    problems.append(f"snapshot invalid shape/dtype/finiteness for {key}: {snapshot}")
            volume_shape = data["last_previous_volume"].shape
            expected_shapes = {"initial_volume": volume_shape, "final_volume": volume_shape,
                               "last_east_flux": volume_shape, "last_north_flux": volume_shape,
                               "last_vertical_flux": volume_shape[:-1] + (volume_shape[-1] + 1,),
                               "last_trace_side_top": volume_shape + (4,),
                               "last_trace_side_bottom": volume_shape + (4,),
                               "last_trace_side_density": volume_shape + (4,),
                               "area": volume_shape[:2], "thickness": volume_shape,
                               "longitude_edges": (volume_shape[0] + 1,), "latitude_edges": (volume_shape[1] + 1,),
                               "interfaces": (volume_shape[-1] + 1,)}
            for key, expected_shape in expected_shapes.items():
                if data[key].shape != expected_shape:
                    problems.append(f"snapshot shape mismatch for {key}: {snapshot}")
    return {"report": str(path), "report_status": report.get("status"), "runs": len(report.get("runs", [])), "validation_errors": problems}


def snapshot_ke(data, order=96):
    area = data["area"]
    lon = np.radians(data["longitude_edges"])
    lat = np.radians(data["latitude_edges"])
    dlon, dphi = np.diff(lon), np.diff(lat)
    dsin = np.diff(np.sin(lat))
    radius = np.sqrt(area / (dlon[:, None] * dsin[None, :]))
    tops, bottoms = data["last_trace_side_top"], data["last_trace_side_bottom"]
    density = data["last_trace_side_density"]
    coeff = np.zeros_like(density)
    coeff[..., 0] = density[..., 0] / (radius[..., None] * dphi[None, :, None])
    coeff[..., 1] = density[..., 1] / (radius[..., None] * dphi[None, :, None])
    coeff[..., 2] = density[..., 2] / (radius[..., None] * dlon[:, None, None] * np.cos(lat[None, :-1, None]))
    coeff[..., 3] = density[..., 3] / (radius[..., None] * dlon[:, None, None] * np.cos(lat[None, 1:, None]))
    height_shape = data["thickness"].shape
    latitude_count, longitude_count = area.shape[1], area.shape[0]
    height = np.maximum(bottoms - tops, 0.)
    z_overlap = np.maximum(0.,
                           np.minimum(bottoms[..., :, None], bottoms[..., None, :])
                           - np.maximum(tops[..., :, None], tops[..., None, :]))
    volume = data["last_previous_volume"]
    east_q, north_q = data["last_east_flux"], data["last_north_flux"]
    side_h = height
    east_area = radius[..., None] * dphi[None, :, None] * side_h[..., 1]
    north_area = radius[..., None] * dlon[:, None, None] * np.cos(lat[None, 1:, None]) * side_h[..., 3]
    u_east = np.divide(east_q, east_area, out=np.zeros_like(east_q), where=east_area > 0.)
    north_q_faces = np.zeros_like(north_area)
    north_q_faces[:, 1:] = north_q
    v_north = np.divide(north_q_faces, north_area, out=np.zeros_like(north_area), where=north_area > 0.)
    next_area = np.roll(area, -1, axis=0)
    common_east_mass = .5 * (area + next_area)[..., None] * side_h[..., 1]
    mid_lat = .5 * (lat[:-1] + lat[1:])
    north_fraction = ((np.sin(lat[1:]) - np.sin(mid_lat)) / np.diff(np.sin(lat)))[None, :]
    south_fraction = 1. - north_fraction
    common_north_mass = np.zeros(height_shape, dtype=np.float64)
    half_north_mass = np.zeros(height_shape, dtype=np.float64)
    if latitude_count > 1:
        common_north_mass[:, :-1] = (area[:, :-1] * north_fraction[:, :-1]
                                      + area[:, 1:] * south_fraction[:, 1:])[..., None] * side_h[:, :-1, :, 3]
        half_north_mass[:, :-1] = (north_fraction[:, :-1, None] * volume[:, :-1]
                                    + south_fraction[:, 1:, None] * volume[:, 1:])
    east_dual_velocity = u_east
    common_proxy = .5 * float(np.sum(common_east_mass * east_dual_velocity ** 2)
                               + np.sum(common_north_mass * v_north ** 2))
    half_east_mass = .5 * (volume + np.roll(volume, -1, axis=0))
    half_prism_proxy = .5 * float(np.sum(half_east_mass * u_east ** 2)
                                  + np.sum(half_north_mass * v_north ** 2))
    angular = np.empty(area.shape + (4, 4), dtype=np.float64)
    for longitude_index in range(longitude_count):
        for latitude_index in range(latitude_count):
            angular[longitude_index, latitude_index] = angular_gram(float(lat[latitude_index]), float(dphi[latitude_index]), float(dlon[longitude_index]), order)
    gram = area[..., None, None, None] * z_overlap * angular[..., None, :, :]
    energy = .5 * float(np.einsum("ijks,ijkst,ijkt->", coeff, gram, coeff))
    eigenvalues = np.linalg.eigvalsh(gram)
    gram_scale = np.max(np.abs(gram), axis=(-1, -2))
    positive = np.all(eigenvalues[..., 0] >= -(1e-12 + 64. * np.finfo(np.float64).eps) * gram_scale)
    if not positive or not np.array_equal(gram, np.swapaxes(gram, -1, -2)) or energy < 0.:
        raise ValueError("actual physical Gram positivity/symmetry failed")
    gram_off_diagonal = gram.copy()
    diagonal_indices = np.arange(4)
    gram_off_diagonal[..., diagonal_indices, diagonal_indices] = 0.
    return {"physical_gram_ke_m5_s2": energy, "common_wet_physical_dual_proxy_m5_s2": common_proxy,
            "half_prism_kinematic_proxy_m5_s2": half_prism_proxy, "quadrature_order": order,
            "positive_semidefinite": bool(positive), "mixed_entries_nonzero": bool(np.any(gram_off_diagonal != 0.))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="optional updated reference report")
    parser.add_argument("--out", help="write JSON to a new path; existing files are never overwritten")
    args = parser.parse_args()
    result = canonical_probe()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.report:
        result["reference_report_sha256"] = hashlib.sha256(Path(args.report).read_bytes()).hexdigest()
        result["reference_validation"] = validate_report(args.report)
        if result["reference_validation"]["validation_errors"]:
            result["canonical_checks"] = "FAIL"
        else:
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
            calculated = []
            for run in report["runs"]:
                snapshot = ROOT / run["snapshot_path"]
                with np.load(snapshot, allow_pickle=False) as data:
                    calculated.append({"geometry": run["geometry"], "velocity_dtype": run["velocity_dtype"],
                                       "disturbed": run["disturbed"], "snapshot_sha256": run["snapshot_sha256"],
                                       "order48": snapshot_ke(data, 48), "order96": snapshot_ke(data, 96)})
            result["actual_reference_groups"] = calculated
            quadrature_ok = all(abs(item["order48"]["physical_gram_ke_m5_s2"] - item["order96"]["physical_gram_ke_m5_s2"])
                                <= 1e-12 * abs(item["order96"]["physical_gram_ke_m5_s2"])
                                for item in calculated)
            actual_ok = (len(calculated) == 8 and quadrature_ok
                         and all(np.isfinite(value) for item in calculated for order_key in ("order48", "order96")
                                 for value in item[order_key].values() if isinstance(value, (float, int)))
                         and result["reference_validation"]["report_status"] == "PASS" and result["canonical_checks"] == "PASS"
                         and all(item["order96"]["positive_semidefinite"] and item["order96"]["mixed_entries_nonzero"] for item in calculated))
            result["actual_reference_status"] = "PASS" if actual_ok else "FAIL"
            if actual_ok:
                result["status"] = "PASS"
                result["scope"] = "canonical_and_last_actual_q_horizontal_field_metric_selection_not_force_or_time_closure"
                result["limitations"] = ["Energies are per reference density (m5/s2), not Joules without multiplication by rho0.",
                                         "Different quadratures are not retrospective bugs in the old discrete energy norm.",
                                         "No matched nonlinear force/pressure/rotation/time/production or century qualification."]
            else:
                result["status"] = "PENDING"
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        output = Path(args.out)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result.get("canonical_checks") == "FAIL" or result.get("actual_reference_status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
