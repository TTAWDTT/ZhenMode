# Physical momentum rectangles and common-overlap Coriolis repair

Registered2026-09-29 againstf6bb886, BEFORE implementation/results. Previous
goal turn made progress: implemented active linear3D pressure/shared-Q, corrected
local adjoint, verified eight short references/440 tests, and exposed a physical
partial-face Coriolis FAIL. Full industrial objective remains active and intact.

## Primary research and boundaries

Read [MITgcm horizontal grid](https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html),
[partial cells](https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html),
sections2.14.2/2.15 of [momentum algorithms](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html),
and actual [mom_vi_coriolis.F](https://github.com/MITgcm/MITgcm/blob/master/pkg/mom_vecinv/mom_vi_coriolis.F)
branches0--3. Horizontal u/v dual areas and wet thicknesses are distinct from
tracer areas. Velocity averaging, thickness weighting and energy-preserving
forms are different options, not automatic properties of a C grid. The raw
source endpoint failed; the official GitHub code view supplied the actual
Fortran. [MOM6](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_Coriolis.html)
also uses thickness/PV-dependent coupling, with energy/enstrophy and vanishing
layers treated separately. These are primary constraints, NOT a literal copy
or a claim that the chosen new operator is an existing MOM6/MITgcm scheme.

Reject (a) keeping wrong sqrt-volume interpolation because it preserves a
quadrature, (b) uncoupled velocity averaging that loses energy symmetry,
(c) force clipping/damping, and (d) multiplying coefficients by a minimum height
without defining its physical area, wet interval and momentum mass.

## Chosen geometric weak rotation, with explicit energy change of definition

Face velocity is piecewise constant on its adjacent horizontal half-cell
rectangles, restricted vertically to the face's ACTUAL common wet interval.
Undefined/dry face velocity is zero. Longitude midpoints split cell areas in
half. Latitude midpoints split sin(latitude) areas, not necessarily in half.
Let A_S/A_N be exact south/north half areas of each scalar cell. For east face
e between cell i and i+1, momentum mass is
M_e=0.5*(A_i+A_(i+1))*H_e. For north face n, it is
M_n=(A_N,j+A_S,j+1)*H_n. This is the volume of the defined wet rectangles.
Do not substitute face_area*center_distance or sqrt of neighbor mass ratios.

Each u/v pair meets on ONE quadrant of their shared scalar cell. Its area is
0.5*A_N or0.5*A_S; its vertical thickness is
max(min(face bottoms)-max(face tops),0), including unequal moving tops.
K_en is the integral of f over that common volume. Scalar/horizontal prescribed
f is piecewise constant per scalar cell. Default f=2*Omega*sin(latitude) has
exact half-band means Omega*(sin(phi_mid)+sin(phi_edge)). Same K enters BOTH
equations: du=K*v/M_e and dv=-K*u/M_n, accumulated over all intersecting pairs.
Thus sum(M_e*u*du+M_n*v*dv)=0 by paired cancellation. In sqrt(M) coordinates
C_en=K_en/sqrt(M_e*M_n) and the reverse is its exact negative transpose.
Implicit midpoint/adjoint-safe CG is retained, WITHOUT changing solver gates.

This explicitly replaces the D39 energy quadrature with physical rectangle
volumes; the old report/definition stays archived and is NOT relabeled. A
direct independent host quadrant-intersection oracle must compute masses/K
and a dense physical-velocity matrix without calling the new helper.
Pressure and rotation share one physical face-intersection helper. Coupled
rotations receive the actual initial V, and freeze the SAME geometry during
this reference macrostep; no second inconsistent eta/face interpretation.

## Fixed gates before running the candidate

1. The original1m/30m thin-east witness and its thin-north counterpart must
   match +/-f*2 within1e-8 relative, unchanged. Reproduce old amplification
   from frozenf6bb886 into a separate output, preserving original JSON/log.
2. Physical dual masses, wet faces and rotation tendencies agree with an
   independent quadrant/interval oracle within1e-12 relative/absolute for
   irregular spherical cells, land, one-cell columns and unequal moving tops.
   Constant-f tendency cannot exceed abs(f)*max_neighbor_speed by more than
   1e-12 relative arithmetic error; this follows from nonnegative overlaps
   partitioning a dual volume, not an applied cap.
3. Weighted work and pure64 midpoint energy change <=1e-12; stored32 <=2e-6.
   Dense implicit system matches within1e-12; invalid/capped solves reject.
   Local velocity and moving-volume FD/JVP <=1e-6 and VJP dot <=1e-12 away
   from wet-overlap transitions. Adjoint-safe zero start/atol remains required.
4. All prior common-depth pressure, counted-once mean/shear force, physical
   source closure, constant/bounded contents and surface identity gates remain.
   Re-run SAME eight real-ETOPO references,100x60s/BT4, unchanged coefficients,
   disturbance0.1 and dtype64/32 policy. New output/provenance identifies the
   physical-rectangle rotation; keep old reports. Full suite/lint/MMS/install.

## Required next production work, not completion claims

This repairs linear rotation consistency, NOT full momentum dynamics. The
existing center-distance pressure/fast-wave quadrature is not automatically
the same as exact momentum rectangles; full pressure/kinetic/potential work
compatibility is still REQUIRED, along with a dual mass continuity equation
and translated scalar Q for nonlinear momentum. Variable-face geometry stays
frozen only within this reference macrostep. Complete EOS/forcing/mixing/ice,
nonlinear split coupling, initialization/restart and actual driver cutover must
follow. No PV/enstrophy, arbitrary cut-cell accuracy, whole-model energy or long
adjoint claim. Century/climate/forecast, full polar topology, GPU/distributed
and every industrial-roadmap gate remain unresolved.
