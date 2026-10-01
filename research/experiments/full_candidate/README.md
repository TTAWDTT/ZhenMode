# Production-grid inventory candidate (isolated, not production-qualified)

`step.py:migrate(fd, params, source_sha=..., forcing_sha=...)` consumes actual
`GlobalOceanGrid`/`nodal_dual_v1` geometry once. Deep reference masses remain
unchanged; shallow columns use their real reference band bottom (a two-wet-node
column ends at -10 m, not -22.5 m). Top tracer inventories and reference
momentum are conserved during migration before actual-mass velocity recovery.

`advance(state, params, grid, forcing_sha=..., max_subcycles=256)` is executable
on the explicitly covered configuration: spherical wet faces, pressure response,
deep continuity inverse and top/deep exchange, bounded P1 top remap, heat/bulk
heat/wind, exact rotation, conservative Laplacian/vertical mixing/convection,
linear bottom drag and closed walls. It returns accept/report or a deep rollback.
`save`/`load` bind state, source, parameter/forcing snapshot, candidate code,
original geometry and integer step. No private arrays or driver changes.

This is **inventory KDK with linear half stages**, explicitly requested through
`params.process_time_scheme='inventory_kdk_with_linear_halves_v1'`. It rejects an
original stage schedule, active unsupported closures/filters and changed forcing
or parameters. No original physics may be disabled to obtain a real-data pass.
Biharmonic, GM/Redi, ice, mixed-layer deposition/restoring, sponge/eta/polar and
original nonlinear filters, and original process subcycle clocks still block
production use. This list is computed before the first update.

The source/tracer budget is independent of measured diffusion changes. Pressure,
rotation, drag and wall momentum impulses are accounted separately from closed
transport. This does **not** prove total energy closure or the original time
order >=1.9; no real failure replay, industrial comparison or long integration
has been run. `qualification_passed` remains false. Do not call this a real
1-degree repair. Next work is adapting the actual enabled original processes
and temporal schedule, not creating another flat two-column model.

Tests: `tests/test_full_candidate_geometry.py` uses production spherical grid and
parameter builders, variable wet depth/land, static affine density, wave/forcing,
rotation/mixing, deep response/cross-band GCL, changed controls, corrupt restart
and pre-call byte rollback. Small configuration: 4x3x6, dt30 s, 3 fast steps;
synthetic inputs only. CPU costs and accurate remote head/CI are reported in the
handoff; these tests do not replace frozen real-data verification.
