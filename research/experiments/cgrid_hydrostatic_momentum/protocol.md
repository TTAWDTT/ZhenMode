# Common-depth pressure and actual three-dimensional linear momentum coupling

Registered 2026-09-29 against20d5b82, before implementation or new results.
Previous goal turn was progress: explicit64 inventories/shared FCT, eight real
component references, retained failures,409 tests and isolated installed step.
Full industrial roadmap remains active; production cutover is NOT complete.

## Primary research and applicability

- [MOM6 pressure](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_PG.html): evaluate hydrostatic contact forces on common control-volume edges, not a difference of pressure at unequal cell depths.
- [Engwirda/Kelley/Marshall](https://arxiv.org/pdf/1608.05271), sections3--6: vertical mean reconstruction and common rectilinear integration; exact balance depends on reconstructing the actual stratification. This paper uses pressure coordinates/nonlinear thermodynamics and is not a literal fixed-z implementation.
- [MOM6 Coriolis](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_Coriolis.html): staggered cross-component interpolation needs compatible thickness weights; energy and enstrophy properties are distinct, not automatic from calling a scheme C-grid.
- [MOM6 barotropic coupling](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Momentum_Equations.html): depth-integrated baroclinic forces must enter the fast mode, and free-surface/internal-pressure coupling cannot simply be dropped.
- [MITgcm pressure algorithm](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html): prognostic momentum, diagnostic hydrostatic pressure and continuity/tracers require consistent coupling, geometry and time levels.

Read actual pressure-paper reconstruction/common-rectilinear equations and
equilibrium experiments, not only its abstract. Legacy code uses node pressure
integration/collocated gradients and cannot be relabeled as these physical cells.

## Chosen migration stage, not final replacement

Add actual layer east/north C-grid velocities, pressure forces, conservative
weighted Coriolis rotation and a coupled step using existing shared mean-Q/FCT.
This stage is a frozen-pressure/frozen-face LINEAR-momentum reference. Nonlinear
momentum transport, complete thermodynamics/mixing/ice and nonlinear free-surface
substep geometry are still required for the final production cutover. Do not
qualify this reference as a complete ocean or silently drop those requirements.

Physical geometry keeps explicit vertical interfaces and latitude edges in64.
Inventories/pressure/2D fast-wave arithmetic are64; 3D velocity storage may be32
only explicitly. The existing eight-reference mixed policy/defaults remain
unchanged. No pure32, GPU, whole-model memory or performance claim.

Pressure uses fixed-z layer tops/bottoms with moving top -eta and partial bottom.
For each layer/face, integrate each adjacent column's pressure over the SAME
physical wet interval: max(tops) through min(bottoms). Face areas use that
intersection. Do not compare at unequal centers or extrapolate into dry water.
Only the top volume may change; deeper volume/geometry inconsistencies reject.

An explicit global reference density anomaly R(z)=a+b*z+c*z^2 is represented by
its exact cell averages, including partial cells. Reconstruct the residual
cell-average anomaly using metric-weighted PLM, one-sided at wet-column ends.
Integrate hydrostatic residual pressure analytically. The common R(z) pressure
cancels analytically, but RESTORE its column-dependent surface load
-g*primitive_R(-eta). This is not removing a horizontal mean or suppressing
physical eta/density forces. Default reference is zero. Exact preservation is
claimed only for represented reference profiles/linear residuals, not arbitrary
nonlinear EOS or unresolved vertical structures.

Coriolis uses face weights W=wet_face_area*center_distance as the discrete
kinetic-energy quadrature, not an assertion that W is exact physical dual volume.
Use local four-face interpolation in sqrt(W)-scaled velocities and its EXACT
negative transpose. Closed pairs do not couple. Implicit midpoint solves an
SPD eliminated system; independently recompute its residual and reject failures.
No damping is added to make rotation look stable. Check consistency as well as
energy: an arbitrary skew operator is not automatically the physical Coriolis.

Coupling: half rotation; common-depth pressure; face-area-weighted mean force
fed ONCE into the fast mode; zero-mean layer-force predictor and midpoint layer
transport; match to actual fast mean Q; FCT V/N update; endpoint layers match
fast endpoint velocity; second half rotation. No independent duplicate eta or
barotropic state is carried forward. Freeze current physical wet faces within
this reference macrostep; do not call it fully nonlinear split-explicit physics.

## Gates, registered before results

1. A naive pressure-at-center negative control MUST generate nonzero spurious
   force for horizontally identical linear stratification/unequal partial cells.
   Common-depth result <=1e-12 m/s2 on represented equilibria (including quadratic
   reference, land/one-cell columns). Nonzero manufactured density/eta signals
   agree with independent analytic face integrals within1e-12 relative/absolute.
2. Physical moving-top face intersections agree with hand geometry. Dry density
   sentinels cannot enter wet forces. Reject NaN wet density, bad shapes/dtypes,
   depleted volume, non-top volume changes, nonfinite/invalid constants.
3. Coriolis tendency has weighted work residual <=1e-12; positive/negative signs,
   constant-f uniform interior consistency and independent dense matrix/midpoint
   solve match. Rotation energy change <=1e-12 pure64, <=2e-6 stored32; residual
   <=1e-12 of the original RHS (with an explicit arithmetic floor), checked
   independently of CG's return. Failed/capped solves are not accepted.
4. Coupled rest/background and manufactured force response test BOTH mean and
   layer velocities, preventing a no-op pressure path or double-counted force.
   Pure64 content budgets/constants <=1e-12, stored32 velocities <=2e-6 for
   velocity gates; physical V/N stay64. Check pressure, both rotations, fast CFL,
   donor/FCT gates, source closure and closed material boundaries independently.
   New near-zero-eta test uses physical surface error in m: <=1e-12 +1e-12*eta
   scale, not an impossible1e-20m normalized threshold. Existing nonzero-wave
   component gates are not loosened or reclassified by this distinct protocol.
5. Local JVP/FD <=1e-6 and VJP dot <=1e-12 away from face/limiter transitions;
   includes pressure, rotation and the active coupled predictor. Not long adjoint.
6. Real ETOPO geometries unchanged, both64 and stored32 3D velocities, represented
   quadratic stratification at rest and nonzero manufactured density disturbance:
   eight groups,100 dt60 macrosteps, BT4. Store inventories/layer velocities,
   geometric/pressure/Coriolis/surface metrics and runtime hashes; retain FAILs.
   Synthetic density-only reference is not observed climate or full EOS validation.
7. Full suite/lint/MMS/install must pass. Preserve old surface/precision failures.
   Next milestone must bring nonlinear momentum/real physics into the production
   path, not indefinitely expand an unused separate reference. Century/climate,
   forecast, full polar topology, GPU/distributed and whole adjoint remain required.
