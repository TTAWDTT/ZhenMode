# Physical momentum rectangles and compatible pressure: repair review

2026-09-29. Protocols were committed before each diagnostic/implementation.
Rotation selectionb55a130, implementationf26d686; pressure-work protocol62d9b8d,
measured failure/accuracy selectionfcd9761, coupled repair9a2f219. No push.
This changes actual numerical operators, not only tests or acceptance wording.
The full industrial objective remains active and incomplete.

## Physical rotation, including the original counterexample

The old sqrt(W) four-face interpolation preserved its chosen algebraic energy
but amplified thin-face Coriolis by3.2386. The original1m/30m case, original
FAIL report and frozenf6bb886 replay remain retained. Hard clipping, damping,
larger viscosity or excluding shallow cells would not fix physical consistency.

Integrate Coriolis over the actual shared wet quadrant rectangles. Exact
spherical north/south half areas are distinct except in special symmetric
cells; contact height is the actual intersection of both wet intervals. The
pair coupling K=f_mean*A_quadrant*H_overlap acts as +K*v/M_e and -K*u/M_n.
Every pair cancels physical work. The prescribed f is piecewise constant per
scalar cell; default Earth f uses exact half-band integrals of2*Omega*sin(phi).
The implicit midpoint matrix keeps its exact negative-transpose counterpart.
The previously corrected zero CG start and zero absolute tolerance are retained.

Energy quadrature explicitly changes from D39's face_area*point_distance to
physical wet half-cell rectangle mass. This is a physical definition change,
not a claim that historical energy proofs used this mass. An independent host
scalar-quadrant oracle verifies the geometry and physical-velocity matrix on
irregular partial/land/moving-top fixtures. The repaired thin-face force is
0.0020000000000000005m/s2 against0.002, relative2.22e-16 at the unchanged1e-8
gate; rotation energy change1.69e-16 and true solve residual8.01e-16 pass.
No force cap or additional dissipation was introduced.

46 direct cases and455 full regressions passed at this rotation stage. Eight
same-input real references also passed atf26d686 and their independent inventory
verifier plus five negative controls completed BEFORE the next core changes.
Raw overlap and frozen-failure artifacts remain separate from today's results.

## Newly diagnosed pressure/continuity work: retained FAIL then repair

Physical rotation does not make the pressure force correct. On spherical and
irregular cells, physical dual mass is generally NOT face_area*point_distance.
The preregistered frozen gravity test compares actual fast acceleration against
the SAME Q=S*u in scalar continuity, with surface potential0.5*rho0*g*sum(A*eta^2)
and kinetic power rho0*sum(M*u*G). No giant stored-energy normalization is used.

Actual regular/irregular relative work residuals were0.001343973494 and
0.003075742265 against1e-12. Signed residuals were-169667861 and-285286010W;
offline compatible controls closed below4.1e-17. The old report is retained as
`results/industrial_alignment/cgrid_pressure_work_current.json/.log`; FAIL
remains historical evidence, not overwritten by a later PASS.

Re-read actual MITgcm surface-gradient source and distinct horizontal areas,
its pressure/continuity/energy equations, and MOM6 analytic FV pressure. MITgcm's
point-distance formula is not proof for OUR exact rectangle masses; MOM6's
coordinate/EOS differs. The compatible operator here is independently derived:
G_layer=-S*delta(P)/(rho0*M), G_fast=-g*(L_face/A_dual)*delta(eta).
`horizontal_momentum_geometry` factors exact half/dual areas and face lengths;
rotation masses, contact pressure, fast gravity and wave CFL share these values.
The PUBLIC fast-wave component is repaired, not only a special3D caller.
Actual center distances and scalar reconstruction widths remain unchanged.
Current wet heights still cancel in the horizontal force/mass ratio; no
state repair, extra damping, promoted state or legacy-gradient default is added.

Before core edits, independent new regressions produced7 FAIL/3 PASS. Afterward
106 adjacent direct tests pass; the final11 force tests include an additional
moving-volume fast-gradient FD/JVP and JVP/VJP check. The new expectations use
independently integrated half-cell rectangles and contact-pressure integrals,
not the new geometry helper. Two old point-distance expectations are explicitly
replaced because the intended physical quadrature changed; spatial MMS and the
old failing work diagnostic independently constrain that change.

At9a2f219 the SAME two actual work cases close to3.016e-17/4.087e-17. Spatial
MMS independently represents cos(lambda)*cos(phi) as exact spherical scalar
cell means on24x12/48x24/96x48 grids. Fast and hydrostatic east error ratios are
3.98672/3.99638; north3.87809/3.93518, each above the precommitted3.5. This
supports regular smooth second-order pressure, NOT arbitrary irregular-cut-cell
or full time/space dynamics order. Original absolute work gate remains1e-12.

## Same real inputs, new source-qualified references

Eight SAME real-ETOPO groups complete at CLEAN9a2f219: smoothed80/floor500 and
unsmoothed0/floor10; layer momentum64/32; rest and0.1 density disturbance.
Grid180x66x14, latitude edges+/-66,100x60s with BT4 and V/N/pressure/fast64.
This is6000 simulated seconds per group, NOT a climate/century run. Synthetic
active density is diagnosed from actual transported inventory each macrostep;
it is not a full thermodynamic EOS or observed temperature benchmark.

All8 pass. Maximum content-budget relative error4.95323e-18, actual volume
budget differences[-0.00537109375,0.0130615234375]m3, surface identity5.41234e-16m,
bound excursion2.34479e-13, constant-tracer error2.31217e-13, lower-height
relative error2.22030e-16. Largest stored32 rotation energy error2.18338e-9
passes the unchanged2e-6 gate; solve residual1.19047e-16 passes1e-12. Rest
speeds below8.9e-16m/s; disturbed peaks0.008758/0.009255m/s. No inference of
climate improvement follows from changes to these synthetic short-run speeds.

Independent verifier recomputes physical geometry and V/N inventory snapshots,
checks dtype/duration/groups, every recorded stage gate and source/snapshot
hashes. Five duration/dtype/stage/snapshot/duplicate negative controls reject.
Recorded stage flags are NOT independent replay of every operator. Current
verifier additionally requires the physical-pressure formulation/protocol and
new direct-test source hashes; older eight-case reports qualify their historical
code only. It must not silently accept old reports as proof of today's method.

MMS of the unchanged legacy production FD solver still passes with4.30 ratio.
An isolated installed package, loaded outside the source path, executes an
active physical-mass momentum/shared-V/N step with irregular/land/partial cells:
rotation energy2.51e-16, surface identity5.86e-16m, inventory budget passes.
Generated untracked build was checked for tracked/reparse paths and removed
only at the verified workspace build path. Final full suite466 PASS with23
existing warnings in440.06s is recorded in
`results/industrial_alignment/cgrid_pressure_mass_full_tests.log`; lint, diff
check, research-state YAML and60 local documentation links also pass.

Raw new outputs are `results/industrial_alignment/cgrid_momentum_mass_reference.json`
and hashed snapshots, `cgrid_pressure_work_mass_mms.json`,
`cgrid_partial_rotation_mass_repair.json`, matching logs and verifier output.
Runtime source hashes independently match current files. All implementation,
protocols, tests and reviews are committed; bulky local results remain ignored.

## Required next integration, not permission to stop at components

Production driver remains legacy FD. Physical mass is frozen during each
linear reference macrostep; full buoyancy-energy conversion and FB temporal
energy behavior are not certified. New nonlinear momentum must use actual
scalar Q translated to the actual moving momentum dual volumes, including
partial faces, physical remapping/geometry terms and kinetic energy exchange.
Naive averages of primary-cell fluxes cannot be assumed correct for min-height
common wet supports. First derive/verify this mass balance and dynamics, then
complete thermodynamic EOS, real sources/mixing/ice, split coupling, initialization
and restart, and switch the ACTUAL production path. Do not indefinitely extend
an unused prototype or qualify current production from component PASS.

The next research entry is Herbin et al.'s staggered mass/kinetic balance:
pressure is a divergence transpose and dual mass flux must obey its own local
balance. Its regular MAC half-diamonds differ from OUR spherical partial wet
supports; its Euler internal-energy correction is NOT ocean thermodynamics.
Full buoyancy budget, global/polar geometry, production engineering, real
century/sensitivities, independent climate/forecast, fair GPU/distributed and
full adjoint/constrained learning remain necessary in the unchanged roadmap.

Sources read before major changes:
- [MITgcm pressure/continuity and hydrostatic energy](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)
- [MITgcm horizontal grid](https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html)
- [MITgcm actual surface gradient](https://github.com/MITgcm/MITgcm/blob/master/model/src/calc_grad_phi_surf.F)
- [MOM6 finite-volume pressure](https://mom6.readthedocs.io/en/main/api/generated/modules/mom_pressureforce_fv.html)
- [Herbin et al. staggered dual-mass and energy balance](https://www.numdam.org/item/10.1051/m2an/2017055.pdf)
