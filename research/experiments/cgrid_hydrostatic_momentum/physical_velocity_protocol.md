# Physical velocity frame and kinetic representation before nonlinear migration

Registered2026-09-29 BEFORE new witnesses/core edits. Previous goal turn is
PROGRESS:4113bcc/4bc1b42 actual metric/dual-Q integration,498 tests and eight
short references verified. Full industrial objective remains active/unmet.

Read [MITgcm momentum/metric/energy sections2.14](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html),
[MOM6 coordinate-specific kinematics](https://mom6.readthedocs.io/en/main/api/generated/pages/Specifics.html),
[Piola normal mapping](https://defelement.org/finite-elements.html),
[compatible geophysical finite elements, sections5--6](https://arxiv.org/pdf/1605.00551),
[slip-boundary compatible dynamics](https://arxiv.org/pdf/1801.00691) and
[MITgcm moving free surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html).
Finite-volume discrete KE and continuous reconstructed-field KE need not be
identical. FE methods can require coupled mass solves; wall conditions are
not guaranteed by interior skew algebra. These sources DO NOT validate the
project's moving 3D cut-cell field or authorize changing mass alone.

## Frozen mean-Q physical frame, not a completed time discretization

Current qx/qy/qz are mapped transport fluxes. Div(q)=D/H in a primary prism,
top/bottom material qz=0. Top layer D is column horizontal outflow; lower
D=0 up to participating-flux arithmetic. A direct qz/area interpretation as
Eulerian downward w has WRONG surface kinematics and nonzero top-layer fluid
divergence when D is nonzero. This is a rejected interpretation of a properly
scoped existing transport representation, not a claim the old driver used it.

Define phi(mu)=asin(sin(phi_s)+mu*delta_s), R inferred from actual geometry
meridional width/delta_phi (custom radius supported), delta_lambda from zonal
width/(R*cos(phi_mid)). The horizontal physical components are
u=qx*cos(phi)/(R*delta_s), v=qy/(R*delta_lambda*cos(phi)).
Cell area=A; alpha=(z-t)/H. Only the top cell moves; source Rv is the SAME
explicit volume source entering actual V, interpreted as signed top inflow.
Let D0 equal actual net Q in top cell and zero in lower cells. Then

grid_down=(D0-Rv)/A*(1-alpha),
relative_down=qz/A+Rv/A*(1-alpha),
absolute_down=qz/A+D0/A*(1-alpha).

At surface w_abs-grid_down=Rv/A; without source it is a material surface.
At fixed internal interfaces grid velocity iszero and w_abs=sharedQz/A.
At physical bottom Qz=0 =>w_abs=0; step-wall normal horizontal flux stayszero.
In bulk div(u,v,w_abs)=(D-D0)/(A*H)=0, including rain/evaporation as BOUNDARY
inflow, not a fictitious distributed compressibility. Inflow source replaces
the transport API's explicit cell source representation only in this physical
VIEW; no inventory, flux, source or state update is changed. No source added.

These fields describe actual macrostep-mean Q on FROZEN starting geometry,
with source/mean surface motion implied by actual continuity. They are not
the exact endpoint state velocity, full ALE/GCL time update, nonlinear flux,
pressure/wall reaction or kinetic/buoyancy conservation proof. Pole endpoints
remain unsupported; no accepted cosine floor. Only top volume sources valid.

Wire this physical reconstruction into actual linear-step output/acceptance
using SAME Q, geometry and source; use it for ensuing nonlinear transport.
Reject unsupported sources, inconsistent lower continuity and malformed
geometry rather than return a finite but false physical velocity.

## Registered gates

1. Two naive qz/A surface-velocity witnesses must FAIL before new API; use
   irregular5x4 partial/land/moving geometry, signed nonuniform Q, custom R.
   Keep old-interpretation failure and the old transport contract intact.
2. Physical horizontal normals at ALL four contacts match actual Q/contact
   area; blocked wet wedges and geographic walls have exactlyzero normal
   speed. Both sides match across open interfaces. Scalar and trailing-sample
   queries, actual X64 geometry/Q, no fictitious point velocity from units.
3. At top, absolute-minus-grid matches signed Rv/A, bottom is impermeable;
   each fixed internal vertical interface has same actual Qz/A. Positive and
   negative source and no-source controls, plus source-only uniform rest.
4. Independent physical spherical divergence via JVP and centered FD away
   from support breaks <=1e-12 participating depth-flux/A plus64eps; FD1e-6
   relative. Naive unlifted vertical field must fail for nonzero top D. Grid
   surface speed must match actual V increment with1e-12 transported amount
   plus64eps stored-V rounding; do not normalize by total ocean volume.
5. Shapes/dtypes/finiteness/positive metrics, land/lower volume source, failed
   lower continuity, bad coordinates and unsupported poles reject. Actual
   source/no-source momentum64/32 cases retain prior state arithmetic. Four
   frozen fixtures only; no bitwise claim on all actual trajectories.
6. Kinetic selection: independently integrate physical horizontal field on
   depth contact partitions/longitude polynomial/latitude area quadrature.
   Test symmetric positive Gram action and explicitly detect nonzero mixed
   terms before promoting diagonal mass. Compare common-wet/half-prism
   diagonal proxies on SAME reconstructed velocity/Q, not endpoint velocities.
   Difference is NOT a retrospective failure of the old discrete KE norm.
   Do not assert exact field KE equals every industrial FV scheme's quadrature.
7. Adjacent/full/MMS/installed active path. SAME eight ETOPO100x60s references;
   keep old reports/gates, source and snapshot hashes, physical field stage
   flags and last point/frame/source data. Host independent geometry/physical
   frame audit plus deliberate metric/source/absolute/grid/bulk/stage corruption
   must reject. Last snapshots/recorded flags are NOT all-step independent replay.

Next use a justified physical velocity/kinetic representation with wall forces
and matched pressure/rotation/fast/source/time equations to implement ACTUAL
nonlinear momentum and kinetic/buoyancy exchange, not an indefinite alternative.
Complete actual EOS/real sources/mixing/ice/initial/restart/driver migration,
global topology, century/sensitivity, independent climate/forecast, fair GPU/
distributed, whole adjoint/learning and engineering. No industrial completion.
