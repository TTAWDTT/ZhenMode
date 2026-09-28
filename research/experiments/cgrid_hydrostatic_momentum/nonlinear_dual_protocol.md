# Actual nonlinear dual-mass/transport/surface candidate

2026-09-29, registered BEFORE probes/tests/core edits. Previous goal turn is
PROGRESS: shared scoring plus completed real-reference audits, not industrial
completion. Current worktree2c5d071 is clean; no validation process is live.
Full nonlinear/buoyancy/production/global/century/climate/forecast/GPU/adjoint
objective remains unchanged; this stage is not a smaller substitute goal.

## Research and method decision

[Herbin, Latche and Nguyen](https://www.numdam.org/item/10.1051/m2an/2017055.pdf),
section3.1 equations3.5/3.6, requires dual mass continuity driven by the SAME
primary mass flux to derive momentum/kinetic budgets. Their Cartesian explicit
compressible theorem is NOT a theorem for our moving spherical hydrostatic
cells. [MITgcm flux momentum](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)
sections2.14.1/2.14.6/2.14.7 pairs volume-flux advection, curvature, rotation,
pressure and kinetic diagnostics; [nonlinear free surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)
also changes the geometry and tracer/momentum tendencies together. Wimmer's
[compatible FE scheme](https://arxiv.org/pdf/1901.06349) changes both bracket
and time derivatives; scalar FCT alone is not a compatible FE theorem.

Current physical-field L2 M(V) is a valid frozen norm, but changing support
introduces mixed terms: M1-M0 is not necessarily PSD for positive local rain.
Preregister an independent48-quadrature small irregular partial-cell witness,
raising only cell(1,0) by0.5m. Compare eigenvalues of physical M1-M0 with actual
full-half-prism mass increments. A negative physical eigenvalue, well above
64eps cancellation scale, rejects treating M1-M0 as added-fluid kinetic mass;
it does NOT retroactively invalidate the frozen paired method.

Select conservative FV dual momentum inventories m*u and kinetic norm
K_FV=sum(m*u^2)/2, with m the ALREADY actual spherical full-half-prism maps of
V. This is an explicit METHOD/NORM change, not a silently substituted physical
L2 mass. The reconstructed physical field remains a separate diagnostic and
requires accuracy/convergence qualification. Re-derive pressure, rotation,
curvature, source, wall and time coupling together; no old acceleration passed
to a new inverse mass and no physical/dual energy names interchanged.

## Project-derived complete discrete step

Use actual primary face Q=S(V_mid)*u_star, bottom-up fixed-z material Qz,
and actual metric half-prism dual Q. Same Q advances V/N through checked FCT.
Only top cells move at this stage; no wet topology changes or pole endpoints.
All inventories/geometry/Q/solves64; momentum storage32/64, cast work separate.

For each component's dual mass0/1, set

    u_star=(sqrt(m1)*u1+sqrt(m0)*u0)/(sqrt(m1)+sqrt(m0)).

This gives the exact identity

    K1-K0 = sum(u_star*delta(m*u)-u_star^2*delta(m)/2).

Advance actual momentum, not velocity plus arbitrary Mdot damping:

    delta(m*u)/dt = -div_dual(Q_dual*u_face_star)
                       + C_mid*u_star + g*B_mid^T*eta_bar + F_held
                       + P_positive_source + R_negative_dual*u_star,
    A*delta(eta)/dt = -B_mid*u_star + R_column,
    eta_bar=(eta0+eta1)/2.

Central velocity face value=(u_star_i+u_star_j)/2 conserves K via dual mass
continuity. Optional upwind value adds known dissipation
dt*sum(abs(Q_dual)*(u_star_i-u_star_j)^2)/2. All horizontal AND vertical
faces participate; do not invent momentum from new V after a scalar step.

Positive incoming sources require an EXPLICIT physical east/north source
velocity per primary cell. Map positive R, R*u_in and R*u_in^2/2 with the SAME
dual map. Negative sources remove bulk momentum at u_star, separately from
positive injection even if they cancel within a dual cell. Source work is
incoming KE minus exact inelastic mixing and outgoing KE, not zero by fiat.

Use cell-centered east/north mean interpolation T and its EXACT transpose:
C_mid=T^T*V_mid*(f+U_mid*tan(phi)/R)*J*T, J(U,V)=(V,-U).
This rotation/curvature is skew in the SELECTED FV norm and consistent with
spherical component equations on refinement. Exact discrete physical axial
momentum/global Noether invariance is NOT inferred from skewness; test/report
it and remaining metric truncation separately. A zero-curvature switch is only
an explicit isolated transport/source test mode, not the global production case.
The held baroclinic pressure is an actual FORCE from common wet contacts,
not the old acceleration; full EOS/FCT buoyancy PE conversion remains required.

Solve layer u_star and eta_bar TOGETHER, lagging only nonlinear coefficient
updates. D=m1+sqrt(m0*m1), E0=m0+sqrt(m0*m1) yield a linear inner block:
D*u_star-dt*(C+gB^T-Adv+R_negative)*u_star=E0*u0+dt*(F+P_in).
Surface block is2*A*eta_bar+dt*B*u_star=2*A*eta0+dt*R.
Use increment GMRES with true residuals and outer fixed-point convergence;
independently recompute full final momentum/continuity/energy gates. No
unconverged iterate, negative mass, invalid FCT stage or hidden clip accepts.

Closed normal DOFs remain zero, but compute the required momentum reaction
impulse explicitly from the constrained row, not a zero-normal-flow assertion.
It has zero work at its constrained u_star; retain its signed impulse. Cast
work, incoming/outgoing/mixing, advection dissipation, held-force work and
surface source work remain separate from unresolved thermodynamic conversion.

## Preimplementation gates

1. Retain actual old rain witness FAIL and independent norm increment probe.
2. Independent NumPy loop/dense oracle verifies dual momentum flux, all3D
   face signs, mass/source maps, work identity and wall reaction. Invalid
   incoming source declaration/geometry/dtype/shape/negative mass fails closed.
3. Isolated uniform rain from old witness: actual top speed dilutes correctly,
   relative axial source impulse below1e-12 plus64eps floor; evaporation removes
   bulk momentum, positive nonzero incoming velocity changes momentum and KE
   by independently computed injection/mixing. No one-cell-only source fix.
4. Actual central/upwind steps on irregular partial/land/moving cells with
   shear and signed sources: SAME Q V/N and full momentum residual at1e-11
   participating impulse plus64eps stored-momentum floor; energy at1e-11
   participating change/work plus64eps initial/final energies. Deliberate
   endpoint-Q, source-sign, omitted vertical advection and transpose defects
   reject independent oracle. Casts are not manufactured dissipation.
5. Nonlinear time refinement and smooth spatial consistency, metric/rotation
   convergence, and physical axial diagnostic are required before promotion.
   True real-geometry states get new candidate reports/hashes, not recycled
   prior physical-L2 trajectories. Adjacent/full/installed checks follow.

Implement ONE actual candidate nonlinear entry that evolves layer momentum,
surface and V/N; former paired-L2 method remains explicitly a historical
frozen reference, not another production mainline. The CLI remains legacy
until initialization/restart/physics/driver migration is complete. Never
upgrade this candidate to full climate/century or energy/EOS qualification.
