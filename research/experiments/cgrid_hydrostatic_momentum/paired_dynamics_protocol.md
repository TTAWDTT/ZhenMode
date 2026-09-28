# Paired physical kinetic mass, rotation and surface dynamics

2026-09-29. Register before implementation; the full industrial goal remains
unchanged. Previous goal turn answered the century qualification question but
made no authoritative implementation change (NO PROGRESS toward dynamics).

## Research and selection

- [Natale, Shipton and Cotter](https://arxiv.org/pdf/1605.00551), sections4--5:
  compatible pressure/divergence, velocity mass and rotation must be paired;
  a globally coupled mass has a dense inverse, so the diagonal Helmholtz
  mobility cannot simply be reused. Spatial energy identities alone do not
  prove time-discrete conservation or nonlinear stability.
- [MITgcm algorithm](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html):
  finite-volume kinetic quadratures are legitimate alternatives. A different
  kinetic norm is not retrospective proof that the old norm was a bug.
- [JAX GMRES](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html):
  convergence info is a placeholder; independently evaluate the actual block
  residual. Use matrix-free application and diagonal energy scaling, not a
  dense global inverse or unchecked success flag.

Choose the horizontal L2 norm of the already qualified physical wet-contact
field for this migration path. This exposes explicit partial-contact support,
cross terms and latitude metrics, rather than silently promoting the kinematic
half-prism masses. Old linear FV references remain clearly named baselines;
the NEW paired reference must actually advance inventory and layer momentum.
This is an explicit numerical-method change, not a bitwise-compatible view.

For local normal coefficients (uW,uE,vS,vN), longitude xi and latitude area
coordinate mu, U=((1-xi)uW,xi*uE) and
V=(-beta*uW,beta*uE,gS*vS,gN*vN), with
beta=dphi/dlambda*(mu-(phi-phiS)/dphi)/cos(phi),
gS=(1-mu)*cos(phiS)/cos(phi), gN=mu*cos(phiN)/cos(phi).
Each basis is supported only on its ACTUAL common wet interval. Integrate
Mij=integral(Ui*Uj+Vi*Vj)dV and
Cij=integral(f*(Ui*Vj-Vi*Uj))dV using the SAME interval overlaps/metrics.
Default f=2*Omega*sin(phi) is integrated at latitude quadrature nodes.
Closed normal coefficients are excluded, not assigned invented fluid mass.

Let B*u be column outflow from SAME physical face areas; B^T is its Euclidean
transpose. Frozen-step equations are

    M du/dt = C*u + g*B^T*eta + F
    A deta/dt = -B*u + R
    E = (u^T M u + g*eta^T A eta)/2
    dE/dt = u^T F + g*eta^T R.

F is the existing common-depth hydrostatic FORCE (old acceleration multiplied
by its declared common-wet mass), not old acceleration applied with new M.
It is held external work in this stage; closed kinetic/buoyancy exchange is
NOT claimed. Use a coupled implicit-midpoint block solve at EACH fast substep,
including rotation, pressure, source and surface continuity. The actual
substep-midpoint Q average drives the SAME closed Q, extensive FCT inventory,
wet/dual flux and physical velocity view. No end-state-Q substitution, layer
mean/deviation repair, old diagonal mobility or fabricated explicit CFL bound.

## Gates and counterexamples

1. Independent NumPy quadrature/dense assembly: M symmetric positive on open
   coefficients, mixed entries retained, C skew, B transpose/work cancellation;
   normalized discrepancies <=1e-12. Compare24/48 quadrature at <=1e-12.
   Cover partial/land/moving-top, irregular latitude/longitude and custom R.
2. Independent dense midpoint oracle: state/Q/work/residual <=1e-11 relative
   with explicit64 roundoff floors. Test simultaneous rotation+gravity+force
   and signed top source. Reject diagonal-only mass and endpoint-Q witnesses.
3. True scaled solve residual <=1e-12*rhs_norm +64eps*(rhs_norm+solution_norm).
   Solver nonconvergence fails closed; the solve does not repair the state.
   Frozen energy/work residual <=1e-11 relative with measured arithmetic floor.
   Float32 momentum cast work is separately reported, NOT counted as physical
   dissipation or silently hidden by a larger double-precision conservation gate.
4. Actual coupled inventory update: volume/free-surface difference <=1e-12 m
   +1e-12*eta scale; inventories/Q/source64, momentum32/64. Constant tracer,
   signed sources, bounds and global inventory conservation; masks/dtype,
   malformed metrics, unsupported poles and nonconvergence reject.
5. Time refinement on fixed geometry: midpoint second-order ratios >=3.5
   for a nontrivial coupled solution; local tangent/FD consistency <=1e-5.
   These are NOT full-model adjoint or nonlinear accuracy qualifications.
6. Actual real ETOPO reference using NEW kernel, not read-only reconstruction:
   initially a bounded paired-step trajectory with recorded source hashes,
   exact solve/energy/source/Q records and host independent endpoint replay.
   Preserve any failure rather than relaxing gates or changing geometry.

## Scope and required next migration

M/contact geometry are frozen within each macrostep. Report endpoint geometry
kinetic-mass change separately; do NOT claim the frozen energy identity closes
moving-M, wall reaction, nonlinear advection, buoyancy, mixing or sea-ice work.
Zero wall-normal flow is a kinematic condition, not a wall-force qualification.
No production/century/climate/forecast/GPU/distributed/full-adjoint promotion.
After this paired dynamic step, implement ACTUAL nonlinear transport/moving
kinetic-buoyancy exchange and ACTUAL production initialization/restart/driver
cutover; do not proliferate unused alternative prototypes. Complete all
remaining industrial roadmap requirements with real independent evidence.
