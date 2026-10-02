"""Bounded synthetic/static evidence for the actual 14-slot force interface."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import inventory_pressure as p
from .force_contour_oracle import contour, stock_pe_direction
from .force_geometry import CONTRACT, BarotropicOperator, UnsupportedGeometry, static_contours
from .force_geometry_cases import columns


def evidence():
    inputs = columns(shear=True, external_pressure=(100., 130., 90.))
    op = BarotropicOperator(*inputs)
    rate = op.rates()
    certificates = op.contour_certificate()
    direct_PE = stock_pe_direction(op.profile, op.footprint.area_m2, rate.hdot, rate.stock_transport, rate.etadot)
    base = BarotropicOperator(*columns())
    old = p.certify_force_consumption(base.profile, base.requests, base.walls, footprint=base.footprint)
    stratified = columns(eta=(0., .2), bottom=-10., periodic=False,
                         temperature=lambda lo, hi: 15. + .2 * .5 * (lo + hi))
    diagnostics = static_contours(*stratified)
    p1_ratios = [abs(r['contour_force_N'] - contour(stratified[0], r['segment'])['force_N']) /
                 (r['bound_N'] + contour(stratified[0], r['segment'])['bound_N']) for r in diagnostics]
    refusals = {}
    for name, case in [('stable_P1_consumption', stratified),
                       ('step_bottom', columns(eta=(0., .2), bottom=(-10., -12.), periodic=False))]:
        try:
            BarotropicOperator(*case)
        except UnsupportedGeometry as exc:
            refusals[name] = str(exc)
        else:
            raise RuntimeError(f'{name} was not refused')
    source_files = ['force_geometry.py', 'force_contour_oracle.py', 'force_geometry_cases.py', 'force_geometry_evidence.py']
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files}
    return dict(contract=CONTRACT, domain='manufactured declared rectangular footprint; full 14-slot stocks',
                source_sha256=hashes, time_step_executed=False, accepted_state_returned=False,
                default_production_entry_changed=False, original_physical_profile_recovered=False,
                periodic_H123=dict(new_force_N=base.force_total().tolist(),
                                   old_Cmin_accepted=old['accepted'], old_Cmin_failure_N=old['incompatibility_N']),
                nonzero_case=dict(slots=14, columns=3, segments=len(op.segments),
                                  column_flux=op.column_flux_diagnostics(),
                                  eta_rate_m_s=rate.etadot.tolist(),
                                  band_R_m_s=rate.relative_flux[:, 3].tolist(),
                                  max_deep_h_rate_m_s=float(np.max(abs(rate.hdot[:, 3:]))),
                                  max_deep_stock_rate=float(np.max(abs(rate.stock_transport[:, 3:]))),
                                  max_pressure_momentum_rate=float(np.max(abs(rate.pressure_momentum))),
                                  full_stock_transport_budget=rate.transport_budget,
                                  stock_PE_direction_W=direct_PE, energy=rate.energy,
                                  kinetic_transport=rate.kinetic_transport,
                                  max_contour_roundoff_ratio=max(r['roundoff_ratio'] for r in certificates),
                                  midpoint=op.midpoint_identity()),
                stable_P1_flat_static=dict(segments=len(diagnostics), max_oracle_roundoff_ratio=max(p1_ratios),
                                           force_consumption_qualified=False),
                explicit_refusals=refusals,
                scope_missing=['general P1 geometry-bound Ctranspose/stock-PE/ALE work coupling', 'step-bottom cut dual and solid faces',
                               'finite-step predict/12fast/replay integration on this geometry',
                               'FCT/filter/mixing/biharmonic/wind/heat/rotation/drag adapters',
                               'real grid momentum-dual geometry', 'fixed-endpoint order and equal-error speed comparison'])


if __name__ == '__main__':
    print(json.dumps(evidence(), indent=2, allow_nan=False))
