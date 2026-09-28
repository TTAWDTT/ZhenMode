# Actual shared Q and moving momentum-dual continuity

Registered2026-09-29, BEFORE diagnosis or core/API edits. Previous goal turn is
PROGRESS:9a2f219 repaired actual pressure operators and aab483c retained failure,
466-test/8-reference evidence. Full industrial objective remains unchanged.

Read [Herbin et al.2018 equations3.5--3.10](https://www.numdam.org/item/10.1051/m2an/2017055.pdf):
kinetic balance requires a LOCAL dual-mass equation and matched momentum flux.
Their regular MAC averages accompany LINEAR half-primary-cell masses; they do
not prove our min-height moving common-wet rectangle masses obey that equation.
Read [MITgcm nonlinear surface/momentum](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)
and [actual r-star calculation](https://github.com/MITgcm/MITgcm/blob/master/model/src/calc_r_star.F).
Its staggered surface thickness and vertical-coordinate treatment differ from
our fixed-z moving top. Do not copy its coordinate formulas while keeping our
old physical volumes or infer a complete nonlinear scheme from a skew force.

## Actual coupling observability is a prerequisite, not the final repair

The existing linear step computes exact matched layer fluxes from actual fast
substep mean Q, then discards those fluxes in its result. Nonlinear momentum
MUST use that same Q, not rebuild from endpoint velocity or post-step geometry.
Before API edit add tests that require the returned east/north/vertical
VolumeFluxes and independently check local V change, explicit source accounting,
column mean-Q equality and closed faces. Add this already-used immutable tuple
to MomentumResult without changing arithmetic, correction or state storage.
Do not construct a second Q for the audit. Float64 inventory/shared-Q with32/64
layer velocity is explicit. Local V update acceptance includes only64 rounding
floor64*eps*max(V_old,V_new), plus1e-12 of the absolute transported/source amount.
This floor reflects stored V subtraction, not relaxed force/work acceptance.

## Registered falsification of naive ordinary-MAC transplantation

Same grid12x6, longitude0:360 and latitude+/-60, plus irregular5x4 longitude
[0,37,131,206,298,360]/latitude[-60,-27,-3,15,56]. Physical depth50 except
irregular partial/land[1,1]=22,[2,2]=3,[3,1]=0; interfaces[0,7,21,50]. State
V/N64, two constant tracers12/35, velocities0, density anomaly0. One macrostep
dt60, BT4, gravity0 and Coriolis0 isolates mass/geometry from all forces.
Uniform rain increases every wet surface by0.2m; isolated rain increases only
cell[1,2] by0.2m. Volume source=A*delta_eta/dt, content source=source*[12,35],
only top layer. Both are explicit inputs, not repairs after a failing step.

Use actual returned shared Q to form independent primary change
deltaV=dt*(source-divQ), including vertical Q. Verify observed scalar V and
whole source-adjusted volume budget. Actual dual mass change is M(V_new)-M(V_old)
using independent host interval/half-sphere integrals, not the new audit formula.

Naive MAC predicts east change0.5*(deltaV_K+deltaV_E) and north change
fraction_N,K*deltaV_K+fraction_S,N*deltaV_N, masked only at actual closed duals.
Compare top-layer local residuals; normalized gate1e-12 of absolute change plus
the stated64-eps inventory floor. Uniform-source controls must agree, isolated
sources must not be relabeled as pass if geometry terms are missing. Record
signed inventory changes, fractions, worst residual and floor separately; do
not normalize solely by the entire unchanged ocean inventory.

Independent kinematic explanation, NOT a production momentum fix: each wet
half contributes beta*V_K with beta=(half-horizontal-area/A_K)*H_common/H_K.
The exact finite-increment product identity is
delta(beta*V)=beta_mid*deltaV+V_mid*delta(beta).
Check observed dual change against both terms with unchanged change/floor gate.
The second term records moving support and exchanges with excluded wet
rectangles. It is NOT a legitimate arbitrary source to add to momentum merely
to make mass/energy close. A conservative physical mapping requires actual
geometric boundary fluxes/remapping and complementary wet regions, or a
re-registered consistent coordinate/velocity support; primary flux contraction
alone is insufficient when beta changes. Record omitted wet volume fractions
per component and variation from source-forced free-surface displacement.

If naive mapping fails and product controls pass, retain FAIL and do not add
nonlinear momentum using ordinary averages, global mass/velocity correction,
stronger damping, removed shallow cells or changed tolerances. Research/select
proper physical moving-dual representation/fluxes and matched pressure/rotation
split TOGETHER before implementation. Fixed-geometry pressure and static thin
Coriolis passes remain valid at their stated scope, not full nonlinear physics.
Save hashes, source revision, all controls and actual shared-flux arrays; old
reports remain historical. The API change must replay direct/adjacent suites,
MMS, installation and the SAME eight real reference groups at clean new hashes.
Full production migration/EOS/sources/mixing/ice/global topology, century and
independent climate/forecast, fair GPU/distributed and whole adjoint/learning
still define success. This diagnostic must lead to actual production work,
not endless expansion of an unused prototype.
