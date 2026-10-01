# Latitude-metric wet traces and ACTUAL paired half-prism fluxes

Registered2026-09-29 BEFORE regressions/core edits. Previous implementation
goal progressed tod7e24b9; subsequent century-proof discussion is planning,
not new physical qualification. This turn revalidated results and archived
483 tests/eight short references ine377d9d. Industrial objective stays active.

Re-read [contravariant Piola mapping](https://defelement.org/finite-elements.html),
[RT normal integral moments](https://defelement.org/elements/raviart-thomas.html),
[MITgcm spherical velocity definitions](https://mitgcm.readthedocs.io/en/latest/examples/global_oce_latlon/global_oce_latlon.html),
[dual mass/kinetic conditions](https://www.numdam.org/item/10.1051/m2an/2017055.pdf),
[stationary cut-cell capacities](https://arxiv.org/html/2211.10698) and
[moving surface coupling](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html).
Published mappings/Cartesian schemes do NOT validate this moving spherical
ocean construction. Formulas below are a project-derived candidate to test,
not a sourced theorem claiming the entire nonlinear scheme works.

## Specific physical defect and paired construction

Existing d7 trace is constant in mu=normalized sin(latitude) area. On a
longitude face physical normal speed is qx*cos(phi)/(R*delta_s). Given
F/H=U*R*delta_phi, that trace does NOT reproduce a constant physical U
across the latitude span. Do not label the prior correctly scoped mapped
construction wrong for its own normal integral/divergence contract.

For phi_s<phi_n strictly inside(-pi/2,pi/2), delta_s=sin(phi_n)-sin(phi_s),
delta_phi=phi_n-phi_s, phi(mu)=asin(sin(phi_s)+mu*delta_s), define
w(mu)=delta_s/(delta_phi*cos(phi(mu))), W(mu)=(phi(mu)-phi_s)/delta_phi.
Integral_0^1 w=1, and W(a)=1/2 where a is south geographic-half AREA fraction.
Use old wet traces T_W/E/S/N(z), primitive qz and netD from wet_trace_protocol:

qx=w(mu)*[(1-xi)*T_W+xi*T_E].
qy=(1-mu)*T_S+mu*T_N+(T_E-T_W)*(mu-W(mu)).
qz remains EXACT old clipped vertical primitive.

The extra qy vanishes at mu0/1, preserves actual north/south contact integrals
and cancels the added horizontal divergence. Longitude contact Q integrates
exactly and blocked traces remain zero. Physical normal U on that face is
constant when actual face data represent constant U. This is frozen SPATIAL
reconstruction, NOT true moving fluid point velocities, ALE time closure,
pressure/wall reactions, full vector momentum accuracy or full-physics PASS.
Pole endpoints are explicitly unsupported here, not rescued by a cosine floor.
Invalid queries keep false validity; no accepted coordinate/force clipping.

Integrated dual Q MUST change together with qx, not reuse old area splits.
Primary F_E/W/N/S are actual stored Q. All half-prisms including wall and
blocked wedges belong to kinematic support; no velocity force promotion yet.
East dual mass=.5*(V+V_east); its fluxes remain old half averages.
North dual mass at boundaryj=b_(j-1)*V_(j-1)+a_j*V_j, including outer wall halves.
North dual transverse east Q takes HALF each adjacent primary east Q (arc
half, not area a/b). Its northward flux at each PRIMARY geographic center is
q_center=b*F_S+a*F_N+(a-.5)*(F_E-F_W), with both external wall entries zero.
Vertical dual Q retains area-weighted a/b halves. These together commute
with mass mapping. The same source map is required later in momentum time
coupling. Do NOT change current D40/D41 masses/forces/fast solver alone.

Implement returned dual flux/mass from ACTUAL shared Q and check commutation
in ACTUAL linear-step acceptance; do not leave an offline-only alternative.
These returned masses are kinematic full half-prisms, distinct from current
force masses. Next nonlinear momentum must select velocity/wall support and
paired force, rotation and C=sum(S^2/M) fast modes/time coupling together.

## Gates registered before seeing numerical outcome

1. New constant physical longitude-normal U regression FAILS old d7 mapped
   trace; include broad asymmetric high-latitude cells, custom R, partial
   wet wall and moving tops. New candidate reproduces U at relative1e-12
   plus64-eps local arithmetic; retain old failure without blaming old scope.
2. Normal contacts integrate to original Q by independent latitude/depth
   integration, blocked wedges exactlyzero. Longitude-normal derivatives,
   latitude correction and old vertical primitive sum to D/H at1e-12 of
   participating depth-flux plus64-eps, with JVP/FD away from breakpoints.
   An omitted qy correction must FAIL; changing qx alone is rejected.
3. Independently integrate physical half-prisms and center/dual surfaces.
   Returned dual stock equals primary stock INCLUDING both walls, shared
   divergence commutes at1e-12 participating flux plus64-eps. Test regular,
   irregular partial/land/moving cases and unrestricted signed vertical Q
   with zero material interfaces. Old area split/new qcenter MUST FAIL.
4. Actual source/no-source32/64 steps return dual Q from SAME primal Q,
   local dual increment=dt*(mapped source-dual divergence) within1e-12
   transported amount plus64eps*(old/new stored mass). Do not inject sources.
   Bad shapes/dtypes/nonfinite geometry/flux, invalid contact and pole
   reconstruction reject. Match old states on four frozen fixtures only.
5. Adjacent/full/MMS/installed active step. Replay SAME eight ETOPO100x60s
   references, preserve old reports/gates. Hash returned metric trace/dual
   snapshots and independently verify LAST contact/metric/half-mass/Q and
   all recorded stage gates; deliberately corrupt metric, paired dual flux,
   mass and stage and require rejection. Not all-step independent replay.

After this spatial pairing, implement ACTUAL nonlinear momentum with geometric
wall support, sources, spherical vector metric work, compatible pressure/
rotation/fast/time coupling and kinetic/buoyancy exchange, then ACTUAL driver
initialization/checkpoint/EOS/real sources/mixing/ice migration. Global topology,
century/sensitivity, independent climate/forecast, fair GPU/distributed, full
adjoint/learning and engineering remain required. No substitute small PASS.
