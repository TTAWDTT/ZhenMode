# Physical control volumes and shared time-integrated transport

Registered 2026-09-29 before implementation, against mainline `4aeb5de`.
This is a required physical foundation for the complete industrial roadmap,
not a second qualified ocean solver or a replacement of the final objective.

## Research and choice

Primary sources, read 2026-09-29:

- [MITgcm finite volumes](https://mitgcm.readthedocs.io/en/latest/algorithm/finitevol-meth.html): cell content and shared boundary fluxes, not node weights masquerading as physical volume.
- [MITgcm vertical grid](https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html): distinguish centers, interfaces and partial cell/face geometry.
- [MITgcm C grid](https://mitgcm.readthedocs.io/en/latest/algorithm/c-grid.html): horizontal velocities live on faces; existing collocated states cannot simply be relabeled.
- [MITgcm moving surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html): use the same thickness and time-level transport in h and h*C, and account explicitly for fresh-water tracer content.
- [MOM6 coupling](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html): layer transport sums must match the barotropic time mean actually driving eta. Its flow-dependent PPM/ALE fits are not directly transplanted into fixed-z cells.
- [MITgcm advection](https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html): donor-cell flux is a useful conservative low-order foundation but too diffusive as a final climate scheme. Higher-order bounded transport remains required.

Implement explicit spherical cell edges, positive-down vertical interfaces,
partial bottom cells and intersecting wet faces. Store V [m3] and N=V*C,
not a fixed-node concentration inventory. Shared oriented fluxes Q [m3/s]
update both V and N. East faces are periodic; meridional boundaries and
material top/bottom are closed. Surface water and content enter as separate,
explicitly supplied sources, not an implicit rain temperature or salt flux.

For the first fixed-z moving-top implementation, lower-layer volume is fixed.
Interior vertical transports are obtained bottom-up from horizontal net
outflow. The top interface is material, and top-layer volume changes by the
column horizontal divergence plus explicit sources. Invalid loss of a wet
cell or an excessive donor outflow must reject the step, not clip or refill.

A C-grid linear gravity-wave reference advances eta from OLD substep face
transports, then velocity from NEW eta. It returns the sum of transports
actually used, not the final velocity surrogate. Layer transports are matched
locally to this column mean by a common face-velocity correction. The same
layer fluxes then drive volume and tracer contents. The reference uses static
face geometry, no Coriolis, stratified momentum, wind, mixing or sea ice. It
demonstrates the required coupling identity, not full physical qualification.

## Registered gates and negative controls

1. Exact spherical area, and sum(h_k)=bathymetric depth for 50m columns,
   15m steps, land and depths beyond the old 4000m last node. Float64 relative
   geometry error <=1e-13. Depth below the last interface, malformed edges,
   nonfinite/negative bathymetry fail explicitly; no depth floor or truncation.
2. Open face heights equal the intersection of adjacent layers. Closed face
   NaN velocity sentinels cannot contaminate valid fluxes; open NaNs must fail.
3. Conservative donor transport uses the same Q in V and N. Closed-domain
   global content residual normalized by sum(abs(N_old))+sum(abs(dt*source))
   <=1e-12 for float64 and <=2e-6 for float32 over the registered trajectory.
   No global correction, clipping or fixed-node proxy in the pass metric.
4. A spatially constant concentration remains constant while eta changes,
   <=1e-12 relative float64 / <=2e-6 float32. Nonuniform source-free
   concentration remains within incoming global extrema at valid donor CFL.
   Sources may change extrema and are not falsely judged source-free.
5. Explicit water/content sources satisfy independent hand-computed totals,
   including freshwater dilution with zero salt-content source. Material
   top/bottom flux, dry content, dry sources, invalid shapes and NaNs reject.
   Donor dt*outflow/V>1, wet-cell depletion and gravity-wave CFL outside a
   conservative sufficient bound reject; no automatic timestep repair.
6. The actual substep-mean continuity identity and coupled top-volume eta
   agree <=1e-12 float64 / <=2e-6 float32 (normalized by incoming eta and
   accumulated change, not a vanishing tendency). Retain a nontrivial
   endpoint-instead-of-mean negative control with error >1e-5.
7. Local JVP/finite-difference relative error <=1e-6 and VJP dot identity
   <=1e-12 in float64 away from flow-sign transitions. These are not
   whole-model adjoint, branch-crossing or long-horizon gradient claims.
8. Real ETOPO-derived 2-degree geometry, both the prior smoothed/floored
   fixture and an unsmoothed 10m-floor diagnostic fixture: 100 coupled
   reference steps with constant and patterned two-tracer fields, float64
   and float32. Report exact edges, native depth and all invalid gates. If
   tiny partial cells reject a timestep, retain the failure before choosing
   a preregistered smaller dt (60s, then 10s, then 1s); do not change depth.
9. Existing full tests, configured lint, MMS and installed module imports
   must remain valid. Old production outputs stay scientifically unqualified.

The initial missing-module test failure will be retained before implementing
the new component. Direct expected geometry, loop-based flux budgets, source
totals and substep reference are independent of the implementation.

Execution contract clarified before real-grid runs: V is the primary state;
eta for the next step is diagnosed from its top-cell volume, rather than
maintaining a second, independently rounded volume state. Both the returned
barotropic eta and the incoming diagnosed eta still face the unchanged local
identity gate, with no offset/refill added. Step reductions apply only to
registered outflow/wave CFL failures, never to improve budget/precision scores.
The extra flux JVP initially used a perturbation too small relative to Earth-
scale volumes (derivative norm 3.94e-9, finite-difference cancellation error
8.94e-13). Its direction is rescaled to the native flux magnitude, away from
sign changes; the <=1e-6 derivative gate is unchanged.

## Mainline migration, still required

The new modules are core components for migration, not a production flag that
silently reuses collocated momentum. Cutover requires C-grid 3D momentum,
well-balanced hydrostatic pressure across unequal partial-cell centers,
Coriolis/metric terms, higher-order bounded tracer fluxes, compatible mixing,
heat/freshwater/ice sources, production accumulated physical budgets, and
explicit node-to-cell initialization/checkpoint schema migration. Tiny cells
need a researched stability strategy, not hidden bathymetry modification.
Only after that cutover can real-forcing spin-up, century and independent
climate/forecast gates qualify the current model. All full-roadmap dimensions
remain active; no century, climate, GPU or industrial PASS follows this phase.
