# Wet-contact trace reconstruction on the actual shared flux path

Registered2026-09-29 BEFORE new regressions/core edits. Previous goal turn is
PROGRESS:0056386 retained12-group half-prism kinematics and rejected plain RT0
wall spreading; current core0eb56ed has470 regressions and eight short linear
references, but no implemented nonlinear momentum or production cutover.

Re-read [cut-cell capacities sections3--4](https://arxiv.org/html/2211.10698),
[Herbin et al. dual mass/kinetic conditions](https://www.numdam.org/item/10.1051/m2an/2017055.pdf)
and [MITgcm nonlinear free surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html).
Stationary Cartesian and shock/internal-energy results do not establish moving
spherical ocean dynamics. Current half-prism averages conserve integrated
mass but plain RT0 violates step-wall normal traces. Do not change mass alone,
inject geometry sources, broaden wet faces or claim skew algebra proves force.

## Actual implementation step, not another unused diagnostic

Implement a trace-compatible enriched primary flux reconstruction, using the
SAME MomentumGeometry wet intervals and VolumeFluxes already used by V/N.
Wire it into actual linear_momentum_surface_step acceptance and return it for
the next dual-momentum implementation. Keep current D40/D41 mass, pressure,
rotation, fast solver, state arithmetic and sources unchanged. This is a
spatial transport construction on the existing FROZEN macrostep geometry,
not a new time-dependent surface/velocity/EOS interpretation.

Coordinates xi=longitude fraction and mu=normalized sin(latitude) area
coordinate; physical positive-down depth z in a cell[t,b], H=b-t. Directions
are W,E,S,N, with north/south closed boundary fluxes zero. For each horizontal
side its integrated flux F has actual shared interval[a,c]. Define
T(z)=F/(c-a) within[a,c], zero elsewhere; zero-width intervals require F=0.
It is a face-integrated flux per unit depth, NOT point physical velocity.
For open latitude faces, the reconstruction is constant in longitude; for
longitude faces it is constant in mu, not latitude arc length. This explicit
lowest-order transverse distribution must be tested for physical accuracy
in the later nonlinear discretization, not silently treated as exact u.

qx=(1-xi)*T_W+xi*T_E, qy=(1-mu)*T_S+mu*T_N.
D=F_E-F_W+F_N-F_S+Q_bottom-Q_top is actual cell net outward Q.
Let I_side(z)=integral_t^z T_side(s)ds, exactly a clipped interval length
times F/(c-a). Then
qz(z)=Q_top+(z-t)*D/H-I_E+I_W-I_N+I_S.
Hence d_xi qx+d_mu qy+d_z qz=D/H, qz(t)=Q_top,
qz(b)=Q_bottom, and all horizontal face integrals equal actual F with zero
normal trace on blocked wall wedges. Internal qz is required by divergence,
not a compensating inventory or momentum source. No independent geometry
evolution/remap; no actual velocity or whole-step energy claim.

## Direct gates before core edit

1. Regressions must first fail absent API. Analytic scalar-host step-wall,
   moving-top, land and both geographic walls in irregular5x4 geometry with
   interfaces[0,7,21,50], depths50/[1,1]=22/[2,2]=3/[3,1]=0. Signed prescribed
   Q obey actual wet masks/material interfaces. Use64 geometry/Q explicitly.
2. Check all side traces: integrate over ACTUAL shared interval and compare
   to recorded F at1e-12 relative plus64-eps participating flux rounding;
   zero normal flux above/below contact, including on the deeper neighbor.
   Verify both sides use identical contact traces. Plain full-height control
   must fail on bottom and moving-top wedges without changing tolerance.
3. Independently integrate piecewise horizontal traces and vertical primitive
   at cell top/bottom and interior depths. Endpoint error <=1e-12 participating
   flux plus64-eps rounding; exact mapped divergence <=1e-12 participating
   depth-flux, no ocean-stock normalization. JVP away from breakpoints must
   match analytic divergence and centered FD at relative1e-6.
4. All-wet uniform-height case reduces to conventional mapped RT0 at1e-12.
   Material top/bottom Q stays zero; internal interface Q is shared from both
   layers. No query treats derived q as physical velocity.
5. Inputs require explicit X64,64 geometry/Q, correct shapes; invalid/nonfinite
   flux, closed-face Q, depleted/invalid geometry and invalid query coordinates
   reject via ValueError/static checks or JIT-safe validity flags. No clipping
   of bad input into accepted state. Queries are scalar or cell-shaped with
   optional trailing sample dimension, coordinates confined to wet cells.
6. Actual coupled result exposes reconstruction, checks its validity, retains
   exact same Q and V/N/local source accounting and old force/pressure gates.
   Compare frozen0eb56ed states numerically on the same fixture, not a new
   mass/force or hidden dissipation. Run adjacent/full/MMS/isolated installation.

## Same real cases and independent rejection

Replay SAME eight real ETOPO100x60s, BT4, V/N64, velocity64/32 cases at clean
new source hashes. New report retains prior inventory/pressure/rotation/surface
gates plus actual trace validity/endpoint gates at every accepted step. Keep
old reports. Hash last-step returned trace intervals/densities/net-Q/endpoints
alongside actual Q and old V, so a host scalar-depth oracle can independently
check contact support, flux totals and vertical primitive. Add deliberate
trace, net-Q and stage corruptions; they must reject without gate relaxation.
Recorded step gates are not an independent replay of every intermediate step.

After spatial traces pass, implement actual moving dual momentum with stated
velocity support and wall reactions, matched physical Coriolis/pressure and
C=sum(S^2/M) fast coupling. Spherical metrics, vertical/source momentum,
kinetic/buoyancy work and nonlinear temporal/positivity accuracy must be
qualified together, then ACTUAL production initial/restart/driver migration,
EOS/real forcing/mixing/ice. No indefinite alternative kernel. Complete global
topology, century/sensitivity, independent climate/forecast, fair GPU/distributed,
whole-model adjoint/learning and engineering remain required and unachieved.
