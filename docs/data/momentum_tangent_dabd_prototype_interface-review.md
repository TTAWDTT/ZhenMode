# External Ft-only producer interface review

2026-10-13. Static review only by the prior nonauthor algebra/interface verifier. No prototype/model execution, no new data opened, no src/installed files edited. Reviewed declaration.json, prototype.py, run.py and finite algebra FINAL_VERDICT.json first; supporting original installed function source is being read for contracts. No numerical outcome is known or used for design.

## Intended scope
Standalone DEVELOPMENT Ft-only tangent with unchanged original normal Bn, fixed uniform meridional weights, same original periodic longitude projector; read-only external copies of advection and instantaneous momentum owners; no native stepping or production implementation grade.

## Operator/shape/unit check
Pending final transcription.

## Frozen entry-point/source-copy contract
Pending final transcription.

## Gate coverage and deviations
Pending final transcription. Initial visible concern: declaration requests direct Bn/control and current synthetic routine appears to contain only tangent F matrix fixtures. No numeric run is authorized by this incomplete review yet.

## Ruling
Pending static review and any declared small repairs.

## Static findings, before any numerical run

### Operator, dimensions and units
For a field of shape (Nx,Ny,Nz), axis0 is periodic longitude and axis1 is the nonperiodic meridional line. The source binary symmetric Fourier mask has shape (Nx,1,1), broadcasts across latitude/depth, and defines the already permitted orthogonal longitude projector P in exact arithmetic. ft_only preserves the original source FFT/mask/real-IFFT sequence. The helper requires Ny>=4, and n4 controls plus Ny12/Ny24 satisfy that domain. R has shape (Nx,Ny-2,Nz) and is computed from immutable values as (x[j+2]-x[j+1])-(x[j+1]-x[j]). Padding r by (0,2), (1,1), (2,0) with respective coefficients 1,-2,1 yields adj_i=r_i-2r_(i-1)+r_(i-2) with missing r=0. Consequently the return is exactly (F_t tensor P)field over real arithmetic. Each of the finite differences, pads and additions is O(Nx*Ny*Nz); the meridional operation is O(Ny) per fiber. Full helper cost also includes the original FFT, so O(Ny) is not a claim that the entire helper costs linear work in every dimension. No dx/dy factor belongs in this dimensionless filter; coefficients act on tendencies with units m/s^2.

normal=True returns old_filter(field,p,normal=True) before new tangent operations, so original Bn and its separately imposed odd wall role are preserved by construction. This permits numerical equivalence checks without selecting Fn. No strict-even tangent assumption is imported.

The declared fields use physical y in meters and d=-z in meters: Y=pi*y/Ly and D=d/H are dimensionless. Psi has units m^2/s; v=-Psi_z=(.14/H)sinY(1-2D), w_up=Psi_y=(.14*pi/Ly)cosY*D(1-D). Thus v_y+w_z=0 and both v and w are physical velocity components. u_y=pi/Ly(-.003sinY+.002cosY), u_z=-(.004+.002D)/H, v_y=.14*pi/(H*Ly)cosY(1-2D), v_z=.28/H^2*sinY are correct. Ru=-v*u_y-w_up*u_z and Rv=-v*v_y-w_up*v_z have acceleration units. X-constant manufacture removes zonal derivative but does not suppress the new tangent sin/cos/quadratic terms. No field coefficient is selected from measured errors.

p.dz_node is a precomputed fixed positive vertical quadrature, not evolving volume h. Even if its endpoint weights differ across depth, contraction and tangent-sum preservation remain valid by applying the horizontal theorem independently at every vertical level k and multiplying its inequality by fixed h_k>0, then summing over k. This needs h_k constant across each horizontal fiber; an explicit positivity/horizontal constancy assertion before comparison was requested. The resulting norm is a fixed separable metric, not a dynamic eta/h theorem. RMS uses the same fixed weights in numerator and denominator. Analytic w errors and original dual-face native-w errors must remain separately named; the latter cannot establish correctness of physical-node w.

### Frozen owner contract and coverage
Installed _advection_flux_form body is acquired verbatim with inspect.getsource, then compiled/executed in a shallow copy of transport's globals with exactly the _dealias_h_fd binding changed to ft_only. Original module globals are never assigned. Installed _compute_momentum_tendency body is likewise copied verbatim and receives only the copied advection binding in a copied global namespace. Its pressure, native vertical-velocity/Fz, derivatives, diffusion/forcing owners are the original function objects. This is external source-copy construction, not a new installed native integrator. Source text hashes are retained; 'AST-copy' here refers to compiling the exact acquired function source, not a separate independently tested AST transformer. Package 133-source SHA256 before/after and original helper identity are explicit gates. Loaded module paths must lie under the frozen installed directory. No test invokes the solver step returned by make_solver_global; it constructs params/init and calls instantaneous owners only. Scalar, GM, pressure and Fz functions are not substituted.

With f=0, eta=0, flat/all-wet Cartesian geometry, horizontally constant T=15/S=35 and zero diffusion/drag/wind/heat, the existing momentum owner should equal its advective tendency. The <=2e-12 m/s^2 checks test that assertion against both original and copied entrypoints before accepting their filtering interpretation. The exact constant-u zero owner test precedes the nonconstant comparison and exercises source owners. The native raw_u uses the original derivatives and original native w, matching the new copied advective owner before filtering; contraction is claimed for this instantaneous tendency, never for a time step. Native-w positioning error is retained descriptively. The polynomial-in-depth u has a different endpoint approximation error under old bare vertical derivative; no new fourth-order momentum derivative is claimed.

The two-entry timing pilot evaluates the same actual compute callset at Ny12 with no oracle residual displayed or compared. It cannot produce physics acceptance; results and timing-only labels keep that separate. One finite Ny12/Ny24 run follows a measured plan, with no retuning or repeated completed indices. Row journal must not preexist; result/log/resource files use exclusive creation. If either strictly-improved analytic-w u RMS check fails, data/result are retained and process returns failure; no production proposal follows. Fixed comparison is a DEVELOPMENT manufactured check, not a holdout or global physical grade.

run.py launches only this external script via the original Linux process-group supervisor, CPU affinity 1 / wall180s / aggregate sampled RSS4096MiB, GPU disabled. The supervisor is a sampled aggregate-RSS limit rather than a hard cgroup cap; preserve that resource receipt wording. It propagates child exit code. I did not execute either file.

## Pre-numeric repairs requested to root
1. Declaration promises a direct n4 Bn control but current synthetic() contains only tangent F matrix controls. Add the original normal=True helper output against an independently assembled n4 B=(I+Lodd/4)^2 before model/oracle work, with the original longitude low-mode or constant input. This verifies the retained old normal contract without introducing Fn.
2. Add explicit spec-pinned T1 fixture [-1/8,9/8,33/8,71/8] rather than relying only on constructed F@q for the quadratic test. Existing F@q matrix equality is useful but does not alone exercise an external expected-answer fixture.
3. Record measured min(e0 output), and if reporting notTVD=True, record the measured finite-line TV increase on e0; current labels are literal -1/16/True. These are overclaim exclusions, not design scores. If no TV measurement is intended, label notTVD as already-proved algebra exclusion rather than a synthetic result.
4. Assert p.dz_node positive, fixed and constant along horizontal fibers before weighted norm/sum gates. Hardcoded flat grid makes this true in the intended profile, but the consumer should enforce the hypothesis it consumes.

These are small frozen-control/contract repairs, not a theorem refutation and not permission to change fields, scores or designs. Until repaired, static scope is confirmed but declaration-to-control coverage remains GAP. Any numeric execution now would precede the full declared control gate. Root has been notified before any output was generated; no new data was read.

## Addendum01 re-review and static pre-numeric seal — 2026-10-13
Only revised prototype.py and pre-execution-addendum01.json were reopened; no generated scientific output or prototype execution was accessed. The prior initial-gap record is retained above rather than erased.

The four requested synthetic/contract gaps are now closed in code before any declared comparison outcome:
- The normal branch is tested on every n4 coordinate basis vector against independently assembled Lodd with diagonal [-3,-2,-2,-3], offdiagonal 1 and B=(I+Lodd/4)^2. X-constant test inputs pass the original longitude zero mode, so the comparison isolates original normal Bn. These tests call the exact old-helper delegate through ft_only(normal=True), do not introduce Fn, and retain actual residuals.
- Quadratic input [0,1,4,9] is now checked against the external dyadic T1 fixture [-1/8,9/8,33/8,71/8], with its computed vector retained.
- e0 actual output now supplies minimum=-1/16 and finite-line TV=17/16>1; notTVD is calculated from the measured inequality, rather than a literal unobserved result. This catches overclaims without scoring/selecting the design.
- Before weighted norm/sum gates, h is asserted positive and constant along latitude, with Cartesian cos_lat=1 and the manufactured eta constructed as an all-zero array. The frozen nodal grid uses a common one-dimensional z/d vector, so p.dz_node's fixed vertical weights also remain independent of longitude. The current assertions do not constitute a general arbitrary-h profile validator: h latitude equality alone would not qualify longitude-varying weights, and max(eta)==0 alone would not certify every arbitrary eta entry zero. These do not affect this fixed profile because its h is fixed by the common nodal column geometry and its eta is explicitly jnp.zeros. Broader profiles would require new hypotheses/gates. The weighted checks occur only in the finite comparison branch; the timing pilot makes no weighted-contract success claim.

The reviewed change adds only the predeclared n4 normal fixture, T1/negative/TV output receipts, and domain assertions. It does not alter Ft coefficient, retained normal Bn, manufactured fields, Nx/Ny/Nz/H/Lx/Ly/dt, analytic formulas, strict-improvement threshold, normal equivalence tolerance, source-copy boundaries, or design-selection rationale. Original installed and production modules are still unchanged by the script's operations. Resource launcher and all finite data retention/fail-closed mechanisms remain as previously reviewed.

STATIC PRE-NUMERIC GATE: CONFIRMED for this frozen external standalone Ft-only tangent/unchanged-Bn DEVELOPMENT producer and its planned timing-only Ny12 pilot followed by the separately measured-plan Ny12/Ny24 finite instantaneous comparison. This is code/scope review, not an executed control PASS. Any numerical control, resource, equality, or improvement failure must remain a retained failed run and cannot be overridden by this seal. Root may now run the declared timing pilot. No new native time integration, source edit, dynamic-volume energy, positivity/TVD, production consumer/physics qualification, changed score, full CI or SOTA claim is permitted by this seal. Final numerical gate remains unexecuted at review time.
