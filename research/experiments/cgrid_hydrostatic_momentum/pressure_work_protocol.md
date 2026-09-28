# Pressure/continuity work with the actual physical momentum mass

Registered2026-09-29 after physical-overlap repairf26d686, BEFORE this diagnostic.
The overlap protocol explicitly leaves pressure/kinetic/potential work and dual
continuity unresolved. Do not infer a whole energy budget from rotation alone.

Read [MITgcm sections2.12--2.14](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html):
continuity uses shared volume fluxes and hydrostatic/gravity work must pair
kinetic and potential conversion. Together with its distinct horizontal u/v
areas, this motivates an independently measurable requirement for OUR physical
rectangle masses, not transplantation of its default point-distance formulas.

Frozen-geometry gravity/continuity stage, no density anomaly, Coriolis, sources,
drag, mixing or nonlinear advection. For physical column masses M_e/M_n and
face areas S_e/S_n, use actual Q=S*u in the scalar divergence. Surface potential
energy is0.5*rho0*g*sum(A*eta^2); its rate is rho0*g*sum(eta*(-div Q)). Kinetic
pressure power is rho0*sum(M*u*G). Their sum MUST vanish within1e-12 of the sum
of absolute cell/face powers. No cancellation against a huge stored inventory.

Measure actual fast pressure acceleration by calling current
subcycle_barotropic with eta, ZERO velocity, dt_sub1s/nsub1 and g9.81. Eta then
does not change before the returned velocity, so it isolates the real pressure
gradient without temporal differencing of giant energies. Evaluate its work
against independent nonzero velocity/current Q on the SAME frozen wet faces.

Two fixed cases: regular12x6 with longitude edges0:360, latitude edges+/-60,
and irregular5x4 edges longitude[0,37,131,206,298,360], latitude[-60,-27,-3,15,56].
All columns50m, interfaces[0,7,21,50]. eta=0.2*cos(2*pi*i/nx)*cos(phi_center),
east velocity0.01*sin(2*pi*i/nx), north0.02*sin(phi_north_edge), closed faces0.
Compute current V and common physical faces; preserve those actual masses.

Algebraic positive control, NOT a production fix: compatible acceleration
G_e=-g*S_e*delta_e(eta)/M_e, G_n=-g*S_n*delta_n(eta)/M_n with the SAME Q. Check
that the paired summation closes under the same1e-12 gate. Record both controls,
power in W with rho0=1025, signed residual, normalized residual, CFL/valid flag,
and source/runtime hashes. Preserve FAIL without removing coarse/irregular cells,
relaxing tolerance or switching kinetic mass back to face_area*distance.

If current gradient fails but the algebraic control passes, update hydrostatic
contact-force normalization and fast surface gradient TOGETHER to the compatible
mass/face operator, re-register analytic force and MMS/order checks, and replay
the same old/new reference matrices before production cutover. This diagnostic
does NOT prove full buoyancy work, dual-mass time evolution, nonlinear advection,
FB's temporal energy behavior or full-mode energy conservation. They remain
required, along with EOS/real sources/mixing/ice/restart and the entire industrial
roadmap. Never use the short test as century/climate/forecast qualification.

## Repair selection and accuracy gates, registered BEFORE implementation

Previous goal turn supplied the century/climate explanation and inspected new
evidence but did not change implementation; revalidate it as no implementation
progress. Full455 regressions finished successfully. The independently measured
regular/irregular gravity work residuals are0.001343973494/0.003075742265 of
absolute power, FAIL against1e-12; the unchanged algebraic controls are below
4.1e-17. Preserve those original artifacts and the repaired-overlap eight cases.

Research actual [MITgcm surface-gradient source](https://github.com/MITgcm/MITgcm/blob/master/model/src/calc_grad_phi_surf.F)
and [horizontal areas](https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html),
plus [MOM6 finite-volume pressure](https://mom6.readthedocs.io/en/main/api/generated/modules/mom_pressureforce_fv.html).
MITgcm uses center-distance gradients; this is not proof of compatibility with
OUR exact spherical rectangle masses. MOM6 integrates finite-volume pressure,
but its vertical coordinate/EOS differs; do not claim verbatim transplantation.

Choose the mass-adjoint of OUR shared-Q divergence: layer acceleration is
minus contact area times the same-depth pressure difference divided by rho0
times the physical momentum volume. The fast acceleration uses the same
horizontal ratio L_face/A_dual. Factor exact spherical south/north half areas,
east/north dual areas and face lengths once in finite_volume; rotation geometry,
hydrostatic contact force and barotropic pressure/CFL must share it. Keep actual
point center distances and scalar reconstruction widths unchanged: changing
those to make pressure pass would corrupt the tracer operator. Do not introduce
a legacy-gradient default or fix only the linear3D caller while leaving the
public fast-wave component mismatched. No state correction, artificial damping,
mass redefinition, precision promotion or relaxed existing gate is permitted.

Before core edits add regressions for the same two work cases; random work on
irregular partial/land/moving-top geometries; common-depth pressure integrals
divided by independently integrated half-cell rectangle mass; constant-density
surface-load equality with the actual fast gradient; unchanged thin-face
Coriolis and constant-tracer/source/surface identities. Gates are1e-12 for64
power/force identities, existing2e-6 for32 stored momentum,1e-6 FD/JVP and1e-12
JVP/VJP. Independent expectations must not call the new geometry helper.

Spatial accuracy is a separate preregistered gate. Use regular24x12,48x24,96x48
longitude-global grids over latitude+/-60 and constant50m depth with interfaces
[0,7,21,50]. Represent cos(lambda)*cos(phi) as exact scalar cell means under
spherical area, not point samples. Compare fast accelerations to analytic
g*sin(lambda_face)/R and g*cos(lambda_center)*sin(phi_face)/R. For hydrostatic
density equal to these cell means and eta0, multiply each analytic acceleration
by the common layer midpoint/rho0. Use physical-dual-area/volume weighted RMS
errors separately for east/north; EACH of the two2:1 ratios must be>=3.5.
Closed-wall faces are omitted because their prescribed velocity is zero, not
the unconstrained analytic velocity. No claim of arbitrary irregular-grid
second-order accuracy follows from this regular smooth MMS. Save measured
errors/ratios regardless of PASS/FAIL and retain previous point-distance force
expectations explicitly as historical tests of a different quadrature.

Then repeat the unchanged thin-face probe, pressure-work diagnostic, direct
adjacent/full suites, isolated installation and the same eight real inputs at
the repaired source hashes. The old physical-mass/pressure mismatch remains
reproducible in the committed pre-repair code. Full buoyancy-energy exchange,
moving dual-mass continuity, nonlinear momentum, actual thermodynamics/sources,
polar topology, production cutover and the whole industrial roadmap remain
requirements, not exclusions from the final goal.
