# Two independent dt300 mechanical observations

The existing six-hour records cannot close mechanical work: their saved budget is moving temperature heat/salt, not kinetic/potential energy. The source/record audit is [mechanical_record_audit.md](mechanical_record_audit.md). This follow-up executed exactly two independent instrumented original dt300 attempts (t0 and6h), each paired with one uninstrumented equivalence witness. They are **not a continuation trajectory**. No other step, new six-hour/30h run or solver modification was performed.

## Protocol, source and equivalence

Frozen protocol SHA256: `ee4731e0418efc5b3456cc6b44000813b45642d5a858314275f2236bcb1de980`. Both attempts were accepted by the original gates. All six attempted and returned fields are finite and differ by at most2.13162821e−14 from the paired original execution; validity/boolean checks match. Arithmetic observations are not byte-identical; retaining auxiliary outputs may change compilation scheduling, but the cause of these differences was not independently established; the <=1e−11 equivalence bound was fixed before execution. Original mathematical AST is preserved after stripping diagnostic additions; the generated source diff and35-module manifest are published.

Reference nodal weights and H_sw/dz_norm contracts match exactly. Using rho0=1025kg/m3, A=dx*dy and h0=dz_node*wet, Kref=.5Σrho0*A*h0*(u²+v²), Kbt uses the same reference-depth mean, Kshear=Kref−Kbt. Eeta=.5*rho0*g*ΣA*wet*eta². All energies below are J. Kmat−Kref=.5*rho0*ΣA*wet*eta*(u_surface²+v_surface²) is a separate geometry diagnostic; it is not another observed work source.

| Starting checkpoint | Incoming Kref | Attempted Kref | Incoming Eeta | Attempted Eeta | Incoming Kmat−Kref | Attempted Kmat−Kref |
|---|---:|---:|---:|---:|---:|---:|
|t0|0.000000000e+00|9.879390548e+14|0.000000000e+00|6.509813030e+12|0.000000000e+00|-8.484445374e+08|
|6h|4.232861529e+17|4.268153892e+17|2.454497603e+17|2.461861555e+17|-3.567964520e+15|-3.553690469e+15|

## Retained stage changes

| Stage | t0 ΔKref J | 6h ΔKref J | Interpretation |
|---|---:|---:|---|
|first_drag|0.000000000e+00|-6.909716986e+14|isolated first150s exponential bottom drag|
|first_material_L|0.000000000e+00|-1.308860744e+15|combined joint viscosity/masks/sponge-off/shear rotation|
|N_predictor|1.899793873e+14|6.743897896e+15|combined two-stage actual nonlinear residual; pressure/wind/advection not individually resolved|
|predictor_L|-2.691548572e+12|-1.294328408e+15|combined second retained momentum L, not the discarded tracer branch|
|fast|9.675163611e+14|1.015931688e+15|12 actual25s fast updates plus entrance wall constraint|
|tracer_finalize_velocity_unchanged|0.000000000e+00|0.000000000e+00|T/S are retained; computed tracer-branch velocities are discarded|
|final_drag|-1.668651450e+14|-9.364323892e+14|isolated final150s exponential bottom drag|
|final_wall|0.000000000e+00|0.000000000e+00|final normal mask|

At t0, old-velocity pressure power is0 because velocity is0, but the finite update generates Kref. The retained N predictor adds1.89979e14J and fast adds9.67516e14J before final drag. This is why old-state velocity times forcing is an invalid finite-step work estimate. At6h the combined N stage adds6.74390e15J and both retained L stages lose energy; **the combined N increase does not identify its pressure/advective/wind share**, and a negative combined L change alone does not prove isolated-viscosity dissipativity.

## Twelve actual fast substeps

Every fast substep records incoming/masked/kick/final mean u/v, all eta drift endpoints, density forcing, eta gradient and both actual face sets. Full3D fast velocity is reproducible from the observed predictor shear and mean increments; final lift error is0. Impulses and work use actual increments with the kick endpoint average, not fitted sources.

| Fast total over one dt300 | t0 J | 6h J |
|---|---:|---:|
|density_work_J|9.745985821e+14|1.799596708e+15|
|wind_work_J|-3.006366650e+11|-1.074127903e+13|
|eta_pressure_work_J|-6.385203412e+12|-7.716584237e+14|
|coriolis_work_J|3.288418162e-06|-7.391208520e-04|
|delta_Eeta_drift_J|6.509813030e+12|7.363951922e+14|
|eta_filter_delta_E_J|0.000000000e+00|0.000000000e+00|
|velocity_postmap_delta_K_J|0.000000000e+00|0.000000000e+00|
|kick_mask_delta_K_J|0.000000000e+00|0.000000000e+00|

Fast entrance wall masking removes3.96381e11J at t0 and1.26532e12J at6h; it is separately included in the full3D fast stage, never labeled pressure or Coriolis work. Actual kick input masks and post-kick damping/cap maps are zero-energy changes in these probes. Configured sponge, eta relaxation and polar cap are off. The final outer wall map changes no energy after the fast constraint.

| Eta-pressure/drift decomposition | t0 J | 6h J |
|---|---:|---:|
|eta_pressure_plus_drift_defect_J|1.246096182e+11|-3.526323143e+13|
|frozen_shear_eta_drift_J|8.194644348e+10|-3.526173914e+13|
|eta_pressure_plus_barotropic_drift_defect_J|4.266317467e+10|-1.492283314e+09|

Frozen-shear faces are actual faces minus reference-depth mean faces, retaining spherical cos(latitude) and closed walls. Their contribution to eta drift energy is separated explicitly. The remaining mean-pressure/drift defect is a measured finite-step splitting quantity; original drift-kick-drift does not guarantee exact preservation of the unmodified quadratic Hamiltonian. **Neither this defect nor density pressure work is assigned a fitted physical source or labeled spurious energy.**

## Checks and limits

Both probes pass every frozen arithmetic check: actual kick increments and component works reconstruct the update; Coriolis work is within the zero-work bound; both eta drifts reconstruct into mean/shear face parts; reference-energy state telescoping and3D lift match; closed exterior fast faces have zero flux; isolated bottom-drag and wall maps have non-positive reference-energy changes. Kick increment residuals are at most2.060e−18m/s at t0 and1.581e−17m/s at6h. Maximum per-kick work reconstruction residual is0.09375J and19.15625J respectively, within the prospectively defined4096eps bounds. Detailed per-step terms and bounds are in the scalar JSON; these are algebraic contracts, not physical qualification.

Projection trace calls were0 for both original/instrumented paths. This confirms that this static traced branch contains no column-projection call; it is not a dynamic execution counter in general. Tracer face matching remains distinct from a velocity Poisson projection.

Fast wind work is negative in both probes; **total full-step wind work remains unknown**, because slow residual wind work was not isolated. Density pressure work is positive and large in the fast part, but a compatible buoyancy/vertical-coordinate potential-energy functional and thermal/salt/mixing source budget have not been established. It therefore remains an unclosed conversion. Existing sensible-heat J stocks cannot fill this gap. Slow RK/Heun RHS terms, raw versus dealiased advection and isolated viscosity/shear-rotation contributions are explicitly missing.

Decision: no counterexample to the observed fast algebra, Coriolis zero-work, wall flux or isolated drag/mask dissipation was found. There is **no complete physical mechanical-energy closure or demonstrated pseudoenergy repair**. Next hypothesis-driven review should isolate the actual slow N pressure/shear/advection/wind work and establish compatible density PE before choosing an initialization or boundary repair. These data do not demonstrate the physical suitability of the original initialization or closed southern boundary, or resolve the30h failure.

Total bounded execution including two uninstrumented witnesses:255.152721s, peak working set2547302400bytes and sampled private commit2500493312bytes, singleCPU;600s/4GiB guards did not trigger. Both phases exited0. Stage arrays and original data remain private; the PR publishes only source-generation/worker/analysis scripts, a frozen protocol, source diff/hashes and scalar evidence. No original Goal/system permissions/reset card/merge change.


## Validation before publication

Three new AST projection contracts and three existing geometry contracts passed (6tests,0.45s); full-repository ruff and diff checks passed. Independent read-only code, numerical-result and final-publication reviews found no blocking issue. New-head full CI is reported separately and is not presumed successful.
