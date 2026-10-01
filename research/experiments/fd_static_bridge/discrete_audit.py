"""Audit original discrete masses; do not reconstruct unsampled bottom regions."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def audit(
    depth,
    reference_weights,
    wet_mask,
    values,
    eta,
    Tref,
    Sref,
    alpha,
    beta,
    rho0,
    gravity,
    terrain_depth,
    discrete_bottom,
    control_interfaces,
    source_sha,
    terrain_sha,
):
    for v in [eta, Tref, Sref, alpha, beta, rho0, gravity, terrain_depth, discrete_bottom]:
        if np.asarray(v).shape != () or not np.isfinite(v):
            raise ValueError("finite scalar required")
    for name, sha in [("source", source_sha), ("terrain", terrain_sha)]:
        sha = np.asarray(sha)
        size = 40 if name == "source" else 64
        if (
            sha.shape != ()
            or sha.dtype.kind != "U"
            or len(sha.item()) != size
            or any(c not in "0123456789abcdef" for c in sha.item())
        ):
            raise ValueError("invalid identity")
    d, w, mask, v, z = map(
        np.asarray, [depth, reference_weights, wet_mask, values, control_interfaces]
    )
    if (
        d.ndim != 1
        or len(d) < 2
        or w.shape != d.shape
        or mask.shape != d.shape
        or v.shape != (len(d), 4)
    ):
        raise ValueError("shape")
    if (
        not all(np.isfinite(a).all() for a in [d, w, mask, v, z])
        or d[0] != 0
        or np.any(np.diff(d) <= 0)
    ):
        raise ValueError("finite ordered nodes required")
    if (
        not np.all((mask == 0) | (mask == 1))
        or mask[0] != 1
        or np.any(np.diff(mask) > 0)
        or np.any(w <= 0)
    ):
        raise ValueError("contiguous wet mass required")
    active = mask == 1
    h = w * mask
    h[0] += eta
    if (
        np.any(h[active] <= 0)
        or min(alpha, beta, rho0, gravity, terrain_depth, discrete_bottom) <= 0
    ):
        raise ValueError("nonpositive geometry/EOS")
    if np.any(v[active, :2] < [-5, 0]) or np.any(v[active, :2] > [45, 50]):
        raise ValueError("tracer bounds")
    count = int(active.sum())
    if z.size and (z.shape != (count + 1,) or np.any(np.diff(z) >= 0) or z[0] != eta):
        raise ValueError("invalid declared interfaces")
    stock = h[:, None] * v
    stock[:, 2:] = (w * mask)[:, None] * v[:, 2:]
    rho = rho0 * (-alpha * (v[:, 0] - Tref) + beta * (v[:, 1] - Sref)) * mask
    pressure = (
        np.r_[0, np.cumsum(gravity * 0.5 * (rho[:-1] + rho[1:]) * np.diff(d))]
        + rho0 * gravity * eta
    )
    return dict(
        qualification_passed=False,
        mode="original_discrete_mass_only",
        eos_identity=dict(
            alpha=float(alpha),
            beta=float(beta),
            rho0=float(rho0),
            gravity=float(gravity),
            Tref=float(Tref),
            Sref=float(Sref),
        ),
        water_m=float(h.sum()),
        reference_column_m=float((w * mask).sum()),
        T_S_reference_u_v_inventory=stock.sum(axis=0).tolist(),
        wet_node_depth_m=float(d[count - 1]),
        terrain_depth_m=float(terrain_depth),
        discrete_bottom_m=float(discrete_bottom),
        unsampled_depth_to_discrete_bottom_m=float(max(0, discrete_bottom - d[count - 1])),
        declared_interfaces_present=bool(z.size),
        declared_interface_mass_difference_m=None
        if not z.size
        else (-np.diff(z) - h[active]).tolist(),
        original_node_pressure_Pa=pressure[:count].tolist(),
        pressure_defined_node_domain_m=[float(d[0]), float(d[count - 1])],
        continuous_bottom_profile_recovered=False,
        historical_source_independently_verified=False,
        declared_source_sha=str(np.asarray(source_sha).item()),
        declared_terrain_sha=str(np.asarray(terrain_sha).item()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    snapshot = Path(args.input).read_bytes()
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as packet:
        report = audit(**{k: packet[k] for k in packet.files})
    report["input_sha256"] = hashlib.sha256(snapshot).hexdigest()
    with open(args.output, "x") as f:
        json.dump(report, f, indent=2)
