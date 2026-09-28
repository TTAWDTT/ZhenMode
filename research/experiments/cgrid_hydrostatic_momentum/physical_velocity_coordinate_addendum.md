# Exact runtime point coordinates, not reassociated host endpoints

2026-09-29, registered after the first physical-frame reference and BEFORE
the coordinate recording fix/repeat. Retain the original c08754d report at
`results/industrial_alignment/cgrid_physical_velocity_reference.json`.
All eight original linear histories pass, but host point qualification FAILS:
independently recomputed top can differ at the last bits and select the other
side of a discontinuous wet-contact support. Using the stored top/height alone
still fails at bottom endpoints. Neither failure proves wrong bulk dynamics.

JAX documents that compiler algebraic simplification can change floating-point
results: https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs.
The host must evaluate the SAME queried coordinates, not its reassociated
version. Add actual depth arrays returned by the SAME compiled step for each
top/bottom/middle evaluation. Preserve actual component results and masks.

Before evaluating any field, independently reconstruct geometry from V and
spherical area; require declared point depths to match registered0/1/.38123
fractions within64eps*(abs(top)+height). This stored-geometry rounding floor
is ONLY for coordinate provenance, not for velocities, divergence, sources
or any existing gate. Record the coordinate contract in the new reference.
No coordinate clipping, support widening, gate loosening, state or Q edits.

Repeat SAME eight100x60s groups at a clean commit under a NEW result basename.
Host metric/frame/continuity and deliberate coordinate corruption must reject.
Retain both first failures and all prior stage/inventory/wet/dual gates.
The old incomplete endpoint record is not upgraded by inference. Whole
nonlinear/ALE/production, century/climate and industrial qualification remain
unachieved. This is an evidence-format correction, not a production fix.
