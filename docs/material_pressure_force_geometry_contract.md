# Flat-bottom sigma momentum-dual force contract

## 2026-10-01: scope and source lineage

This is a new research interface `flat_bottom_sigma_dual_shared_stock_force_v1`,
branched from inventory-pressure correction
`e68025c9286c6e3cb65840dabd661399067d2638`. It consumes the actual `ColumnStocks`
and inventory-density P1 `Profile`, together with a complete declared rectangular
footprint. It does not consume a caller's force certificate. Geometry, profiles,
face requests and areas are copied, validated and bound internally.

The initial qualification is manufactured flat-bottom, constant-density static
geometry and instantaneous rates. All 14 slots carry h/IT/IS/Mu/Mv. No accepted
state or actual step is returned, and production entry points are unchanged.
This interface is deliberately a new sigma-dual discretization. It is not old
common-physical-depth Cmin, old symmetric_fast_v3, original FD sample recovery,
or a demonstrated equivalent of the historical solver's momentum geometry.
Old Cmin failures and archive-pressure rejection receipts remain unchanged.

The pre-existing 8x4x6 dynamic module remains separately tested. Its documented
30 s nonzero synthetic step changes eta by 6.8070786483e-6 m and deep IT/IS/Mu/Mv
by 5.9730858766e-6, 6.9030647865e-8, 18.0425450568 and 6.5039259398,
respectively. That receipt still sets `qualification_passed=false`, including
the unresolved total PE/ALE compatibility. This static interface has not been
substituted into that predict/12fast/replay integrator.

## Geometry and shared operator

Each wet column has height H=eta-b with the same fixed b. Convert every actual
vertical interface to sigma=(z-b)/H. Union both incident columns' sigma cuts.
Each interval [sigma_lo,sigma_hi] defines a trapezoid between column centers,
with horizontal transverse length L and linearly interpolated height H(x).
Its side levels are b+H_left*sigma and b+H_right*sigma. All four boundaries are
part of its physical contour, including internal sloping sigma cuts.

For each segment, S=L*(H_left+H_right)/2*(sigma_hi-sigma_lo),
Q=C(u)=S*(u_left+u_right).normal/2. Here the incident velocities are the actual
P0 layer stock velocities Mu/(rho0*h), Mv/(rho0*h) at that sigma interval.
Column Q is the unchanged sum of these segment Q values. A separate depth-mean
formula is checked as a diagnostic; disagreement rejects and never repairs Q.

The pressure potential for the qualified constant total density rho is
lambda=Delta(p_ext)+rho*g*Delta(eta). Its transpose gives the segment force
-S*lambda*normal, allocated equally to the two incident stock velocity DOFs
before dividing by each bound column area. This is the declared dual's weak
momentum allocation. Exact original column-local traction is not asserted.
The same coefficients are used for C, its transpose, water and stock fluxes.
There is no limiter or filter modification of Q in this instantaneous subset.

The independent physical contour is

    left:   +L integral(p_left dz)
    right:  -L integral(p_right dz)
    top:    +L Delta(H) sigma_hi integral_0^1 p(x,sigma_hi) dx
    bottom: -L Delta(H) sigma_lo integral_0^1 p(x,sigma_lo) dx

For constant rho, their sum is -S*lambda. Internal sigma-cut contributions
telescope. The constructor recomputes this contour with an independent scalar
seven-point Gauss oracle and refuses local traction incompatibility before a
pressure operator is available. Two-point Gauss integrates the candidate P1
contour. No residual is used to fit forces or redefine solid reactions.
This complete-contour choice follows the finite-volume pressure construction
described in [MOM6 Discrete Pressure Gradient](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_PG.html).

## Full-column transport, work and PE direction

Horizontal divergence D uses only these shared Q. eta_dot=-sum D. The moving
top m=min(3,n) layer fractions f=h/(eta-band_bottom) are fixed at the bound
state; band_bottom must be the actual interface m. Top hdot=f*eta_dot, deep
hdot=0. Relative downward ALE flux R has exactly zero surface/bottom values.
Deep bottom-up R_k=R_(k+1)+D_k is matched to top recurrence
R_(k+1)=R_k-D_k-f_k*eta_dot. A mismatch rejects, without changing Q.

Horizontal Q and vertical area*R use one shared upwind donor for IT, IS, Mu and
Mv. All active stocks, including deep stocks, receive conservative flux rates.
Constant-specific T/S/u/v rates equal their specific value times hdot.
Instantaneous upwind kinetic transport has the separately named nonnegative
loss rho0*sum(abs(volume_flux)*abs(Delta(u))^2)/2. This is not an unexplained
physical energy residual or a finite-step conservation claim.

PE is the existing rho0 free-surface PE plus density-anomaly gravitational PE.
Its independent stock directional oracle uses the actual h/stock rates,
moving interface centers and density P1 moments. It recomputes density means
from stock/EOS, secants and density-bound limiter branches independently.
Each layer's anomaly moment is rho_mean*h*zcenter + density_slope*h^3/12.
The right directional chain includes ds*h^3/12 + s*h^2*hdot/4, minmod zero and
equal-magnitude kinks, and the zero-slope/zero-room clipping corner. This
directional derivative is not a general-P1 linear gradient for Ctranspose.

Numerically near-uniform inputs still must pass full inventory directional
pairing. The constructor's local density tolerance alone never qualifies
their rate consumption. rates() rejects a mismatch against the collapsed
barotropic direction or the pressure-work pairing, using an independent local
operation scale. The demonstrated +/-T1e-10 near-uniform input is refused.
Extreme finite velocities whose squares overflow are refused, and every
returned array and scalar ledger must be finite.

For variable external pressure, pressure_power + PE_dot = external_power,
where external_power=-sum(area*p_ext*eta_dot) is listed separately. Uniform
external-pressure shifts preserve pressure force and volume transport.
The fixed-mass one-second impulse is algebra only: it computes actual M_before,
M_after, their true velocity midpoint, and work=-lambda*C(midpoint_u). It
returns only scalar evidence, never an advanced state.

## General P1 and stepped bottom: concrete continuation requirements

General stable density-P1 profiles already have a static contour diagnostic on
the full 14-slot geometry. Its explicitly declared horizontal density lift is
linear in x between the two inventory P1 values at equal sigma. Pressure sides
are determined by inventory means/slopes; top/bottom pressures follow that lift
and full hydrostatic integration. This model fills defined CVs, not unobserved
original sample support. CV P1 data alone does not uniquely specify a horizontal
lift, momentum dual or physical-grid area. [MOM6 Vertical Reconstruction](https://mom6.readthedocs.io/en/main/api/generated/pages/Vertical_Reconstruction.html)
describes the distinction between layer inventories and reconstruction.

General P1 consumption remains explicitly refused. A continuation must bind a
dual geometry to the same water/stock/ALE operator, construct its stock-PE
pressure potential from the directional chain, and show its transpose equals
the independent side/top/bottom contour. Density minmod and endpoint bounds
are nonsmooth, so a single reused linear gradient is not justified at kinks.
The new directional oracle and contour diagnostic supply two separate sides
of that required comparison; they do not prove their equality.

Stepped bottoms remain refused. For a 14-slot coastal dual, use a cut polygon
bounded by the shared wet aperture, physical top, both original vertical sides,
and each vertical solid step face plus its horizontal bottom segments. Record
wet/dry and step reactions by directly integrating those actual solid pieces.
The contour orientation and area/stock velocity metric must come from this
same polygon. A linear ramp between unequal bottoms would replace the declared
stair geometry, so it is not accepted here. Pressure outside one column's wet
CV requires declared horizontal/support reconstruction; it cannot be inferred
from a force imbalance. Known dry neighbors are bound impermeable adjacencies
with Q=0, while their physical wall pressure is retained in the existing
independent boundary ledger.

Missing adapters for this new instantaneous subset are FCT/filter, stock
diffusion/biharmonic, wind/heat, rotation/drag and finite-step predict/12fast/
replay. None is silently skipped from an original production configuration.
No coastal/global integration, GPU, real dt300/600, order>=1.9, industrial
readiness or equal-error speed improvement is claimed.

## Reproduction and bounded resources

Use the existing project-local environment (Python 3.13.5, NumPy 2.1.3,
pytest 8.3.4, ruff 0.15.7); this change adds no dependencies.

```powershell
.\.venv\Scripts\python.exe scripts\run_bounded_research_tests.py tests\test_pressure_force_geometry.py -q
.\.venv\Scripts\python.exe scripts\run_bounded_research_tests.py --module research.experiments.material_top_band.force_geometry_evidence
.\.venv\Scripts\python.exe scripts\run_bounded_research_tests.py tests\test_material_top_band_dynamic.py tests\test_material_real_geometry.py tests\test_inventory_pressure.py tests\test_pressure_force_geometry.py -q
.\.venv\Scripts\python.exe -m ruff check research\experiments\material_top_band\force_geometry.py research\experiments\material_top_band\force_contour_oracle.py research\experiments\material_top_band\force_geometry_cases.py research\experiments\material_top_band\force_geometry_evidence.py tests\test_pressure_force_geometry.py
```

The runner enforces one CPU, 180 s and 4 GiB on the actual interpreter job.
The public scalar receipt in `material_pressure_force_geometry_evidence.json`
binds source hashes, numerical witnesses, resources and review status. It
contains manufactured scalars only. No private paths, input arrays or archives
were accessed for this batch. The local environment lacks JAX, so the unrelated
JAX integration suite was not executed or counted as a pass.

## 2026-10-01: frozen validation and review corrections

The final focused regression passed 171 tests (139 existing plus 32 new) in
72.75 s pytest / 74.537 s Job wall, with peak interpreter RSS 103219200 B and
aggregate Job private memory 92909568 B. The scalar CLI took 0.914 s Job wall,
RSS 39358464 B and private memory 39313408 B. Both enforced one CPU, 180 s,
4 GiB and returned exit zero without a bound stop. Targeted lint passed.
All numerical cases/probes in this batch used under 300 s aggregate Job wall.

The periodic heights (1,2,3) witness has new total force
(4.092726157978177e-12,0) N while the preserved old Cmin failure is
(-20110.5,0) N. The 14-slot shear/variable-external-pressure case has column
Q=(0.1819791666666667,0.31232638888888886,0.2544444444444445) m3/s,
band R=(-0.02875,0.028977272727272727,-0.01142045454545455) m/s,
maximum deep stock rate 0.21606534090909113 and exactly zero deep hdot.
Global water-rate residual is 5.204170427930421e-18 m3/s; the largest absolute
IT/IS/Mu/Mv rate residual is 1.0547118733938987e-15 in its respective units.

Pressure power is 151.1282986111111 W, independently differentiated full
stock PE direction -146.6390624999999 W, and external-pressure power
4.489236111111108 W. Pairing uses the locally derived 1.5785220439410191e-10 W
roundoff bound; residual/bound is 0.00047263823486774304. No physical
truncation error is inferred from that arithmetic identity. Fixed-mass
unit-impulse KE change is 368920.1448890583 J and true midpoint work
368920.14488905825 J, bound 4.424649811560555e-8 J. The 27-segment stable-P1
static contour has maximum independent quadrature residual/bound
0.028854000599705564 and still refuses force consumption.

The independent inspector found two initial high-priority qualification
counterexamples: nonfinite scalar ledgers from finite extreme velocity, and a
near-uniform density input outside the simplified barotropic PE tangent.
Both are now explicit rejection regressions. The initial mean-only oracle's
number is not reported as a measured general-P1 PE derivative. The final
oracle includes the full P1 directional chain and adds controls for minmod
kinks, density-bound double corners, strictly inactive negative room, and
fixed physical affine density on moving top cells. Absolute operation scales
are formed before cancellation of individual moment contributions. The
inspector's two bounded synthetic probes totaled 0.623 s Job wall; they did
not run an archive, model or accepted step.

## 2026-10-01: final independent code gate

The independent inspector confirmed the five frozen Python file hashes and
reported code gate clean within the static, constant-density, common-flat-bottom
sigma-dual interface and instantaneous full-stock rates. Final reinspection
was read-only, with no additional probes. Its remaining receipt wording note
was corrected: `force_geometry_time_steps_executed=0` describes the new
interface, while the 171-test regression does exercise pre-existing dynamic
module synthetic small steps. No general stratified/stepped force consumption
or finite-step/industrial/speed qualification follows from this gate.
