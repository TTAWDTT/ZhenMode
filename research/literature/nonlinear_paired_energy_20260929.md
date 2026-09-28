# Nonlinear moving kinetic/buoyancy coupling: research before the next action

2026-09-29. Current paired kernel is an actual frozen-M linear reference;
this note does NOT promote it or redefine the full industrial goal.

## Primary sources and application limits

1. [Wimmer, Cotter and Bauer](https://arxiv.org/pdf/1901.06349), section3:
   depth-weighted kinetic energy makes shallow-water Hamiltonian cubic. Their
   energy-conserving time scheme integrates Hamiltonian variations along the
   state segment; ordinary midpoint on both variables is NOT generally energy
   conserving. Upwind depth/velocity terms are paired in the bracket. This is
   not evidence that our standalone scalar FCT automatically fits that bracket.
2. [Thetis hydrostatic DG](https://gmd.copernicus.org/articles/11/4359/2018/):
   hydrostatic momentum, tracer transport, free-surface/ALE and mode splitting
   are a combined space/time method. Its conservation/second-order results
   belong to its own spaces, limiter and moving-grid stages, not our cut-wet
   contacts. Cannot borrow its convergence claims by naming our update ALE.
3. [MOM6 ALE source](https://mom6.readthedocs.io/en/dev-gfdl/api/generated/source/MOM__ALE_8F90.html):
   velocity remapping diagnoses kinetic changes. Optional baroclinic KE
   correction explicitly assumes unchanged total depth and column velocity
   integral. Our free-surface volume changes and rain are not that remap;
   blindly importing its rescaling would be an unjustified energy repair.
4. [MOM6 momentum budget](https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Closing_momentum_budget.html):
   separate rotation/nonlinear vorticity, kinetic gradient, pressure, mixing
   and remapping tendencies. The cookbook notes absent online vertical-remap
   diagnostics; a residual label is not independent proof of its physical cause.
5. [Lundgren, Helanow and Ahlkrona](https://arxiv.org/pdf/2409.00972), sections2/4:
   variable-density buoyancy/tracer terms require matching work to conserve
   kinetic plus potential energy. Its nonhydrostatic continuous FE formulation
   is NOT a direct formula for our hydrostatic discrete partial cells.

## Project-derived consequence, not a literature theorem for our code

Current physical kinetic measure is K(V,u)=u^T*M(V)*u/2. Therefore

    dK/dt = u^T*M(V)*du/dt + u^T*(dM/dt)*u/2.

Frozen-M midpoint closes ONLY the first term against rotation/pressure/held
force work. Current actual code separately records endpoint moving-M energy.
It cannot disappear merely because the frozen residual is tiny. Contact
overlaps contain min/max, so M is piecewise affine in top volumes. The energy
is cubic on a fixed contact branch, with branch changes along a moving state
segment. Fixed two-point quadrature is not automatically exact across those
changes. Wet collapse introduces an additional DOF/connectivity problem.

For nonlinear H(V,N,u), require a discrete gradient identity

    H1-H0 = gradient_bar(H) dot (state1-state0)

and paired conservative/skew or explicitly dissipative flux/force terms.
An arbitrary -Mdot*u/2 force can make an energy expression cancel while STILL
giving the wrong momentum transport. It is not an acceptable substitute for
actual advection, spherical metric forces, material boundary work or sources.
Likewise common-depth hydrostatic well balance does not itself prove pairing
with FCT's ACTUAL scalar flux and gravitational potential-energy quadrature.

## Next preimplementation requirements

Derive actual conservative horizontal AND vertical momentum transport in the
qualified physical basis, including true side-support jumps, spherical metric
terms and material source velocities; match the SAME actual Q/V/N time levels.
Derive buoyancy pressure work and potential-energy transfer from actual scalar
fluxes/EOS, including identifiable mixing/limiter work rather than requiring
unphysical zero dissipation. Independently verify momentum, angular/metric and
kinetic/buoyancy identities and nonlinear space/time convergence; do not only
test an energy cancellation or full-wet shallow-water surrogate.
Then migrate actual initialization, checkpoints, source/physics and production
driver. Partial coasts, global/polar topology, real-forcing century/sensitivity,
independent climate/forecast and fair GPU/full-adjoint remain required.
Register a concrete nonlinear protocol after this derivation and before edits.
