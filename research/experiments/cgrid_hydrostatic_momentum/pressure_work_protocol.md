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
