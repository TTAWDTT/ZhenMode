# Standing-wave v0: frozen engineering screening, not industrial qualification

Owner: independent acceptance branch `codex/standing-wave-v0-acceptance`.
This directory contains a contract generator/scorer, not a replacement integrator.
Production code and benchmark_gate.py are untouched. Initial implementation base:
`8ad3552ff651cf45dfec7ae9a119ac75eef28890`.

## Immediate coarse handoff

```sh
python research/experiments/standing_wave_v0/score.py freeze --case coarse --out coarse-contract.json
python research/experiments/standing_wave_v0/score.py score --contract coarse-contract.json --output ocean-coarse.npz --report ocean-coarse-score.json
python research/experiments/standing_wave_v0/score.py score --contract coarse-contract.json --output mom-coarse.npz --report mom-coarse-score.json
```

`score` raises on invalid/incomplete data; scientific engineering-screen failure
writes the failing metrics and exits 1. No automatic run, expansion or tuning.
Nagi's input adapter `d3c830b09852af6654f00ccbc82a8097a514aec7`,
`research/experiments/industrial_flat_f0/prepare_inputs.py --f0 0
--output-directory INPUT`, matches the coarse dimensions, H, g, rho, T/S,
point-sampled ocean initial eta and interface heights. Before the first run, its
MOM file writer must apply sinc(1/64) to the top eta interface for true cell-mean
initialization; leave ocean initial eta untouched. Current published adapter
does not yet make this native-sampling correction. It must export runtime outputs
in the schema below; preparation alone is not a model run.

Only two coarse runs are currently released: one full ocean process and one
pinned MOM6 process, each one CPU/rank, 600 s total wall, aggregate descendant
peak <=4 GiB, output <=128 MiB. External runner owns timeout and aggregate
memory enforcement, including JIT/MPI descendants; declarations are not guards.
Parent owns total budget and conditional release of medium/fine/half/restart.
No repeated timing runs or extra reference grids are authorized by this file.

## Physical and actual time contracts

f=0; flat H=100 m; all wet; x periodic, y closed/free-slip; Ly=100000 m,
Lx=32000 sqrt(9.81*100)=1002269.4248554128 m. Constant physical metric, not
an equatorial approximation to the global sphere. x/y are metres, never labels
masquerading as longitude/latitude. nx=64/128/256, ny=8, four control volumes.
Ocean nodes [0,-100/3,-200/3,-100] and initial dual widths
[100/6,100/3,100/3,100/6]. MOM interfaces [eta,-100/6,-50,-500/6,-100].

T=15 degC, S=35 psu, reference density1025 kg/m3, g9.81 m/s2. Linear EOS:
drhodT=-.205 kg/m3/degC, drhodS=.779 kg/m3/psu, drhodp=0. All forcing,
restoring, drag, explicit mixing, enhanced convection, GM/Redi, ice, sponge and
polar averaging are zero. Full momentum, continuity and tracer evolution stay
active. Unavoidable numerical filters/time damping are recorded, not hidden.

Ocean calls the existing full `make_solver_global`: column_geometry
nodal_dual_v1, process_time_scheme legacy, mode_split False, conservative_kv
True, localize_conv True; project_adv_vel/monotone_adv/fct_adv False;
polar_cap_rows/taper 0. Zero PhysicsConfig coefficients including r_bot/cd.
Do not enable the separate research dynamics kernel or change native operators.
Original legacy nodal weights total133 1/3 m with these nodes while H_sw100 m;
that different geometric inventory cannot be reported as a matching100 m run.
This opt-in existing configuration is a new declared full-model baseline, not
historical-production repair qualification.

Pinned MOM6 f49a00096df607b48354603e2398e14e189fd62e, FMS
527a42f3ff36d75aac68b65757b35f73a74146bb. Cartesian flat domain, linear EOS,
BOUSSINESQ true, no regridding/bulk/buffer layers, full DO_DYNAMICS true and
OFFLINE false. SPLIT true, SPLIT_RK2B false; DT=DT_THERM=DT_FORCING=DTBT
100/50/25 s. DTBT is positive, not an adaptive negative CFL fraction. Record
actual barotropic/tracer/forcing steps and complete MOM_parameter_doc.all.
Do not set USE_RK2 as if it selects split RK2: pinned MOM.F90 reads that flag
only in unsplit mode. Reject resolved options that differ; no silent substitution.
Spherical/Mercator tc1, density layers, wind and SST restore are not this case.
Runtime viability and resolved option support must be checked by nagi before
running; a failed configuration is reported, not changed after results arrive.

Source checks: full ocean wall mask is only normal v: `_free_surface_step_fd`
and `_step_impl` preserve tangential u at first/last y rows when filters are off.
A one-step full native regression in tests verifies nonzero uniform u survives.
MOM's v is at wall faces; ocean's boundary v nodes are constrained. The wave
has no y-dependence/v, so this difference does not alter the target solution;
it does not qualify other wall-flow cases.

## Exact sampling and output

k=2pi/Lx; omega=2pi/32000. A=.01 m (medium half-control .005 m):
eta=A cos(kx) cos(omega*t), u=A sqrt(g/H) sin(kx) sin(omega*t), v=0.
Initially u=v=0. Ocean eta is point sampled at centres; MOM eta is the
exact cell mean at centres, cosine times sinc(1/nx). Its top interface must
be initialized with that mean, not the point value. The two different native
arrays represent the same continuous physical initial condition. This change
must precede either run, never be selected from observed results. Scorer rejects
MOM eta labelled node or ocean eta labelled cell_mean. Native u is point-sampled
at its own nodes/faces. Generic `exact()` also supports cell-mean u for oracle
checks, but those labels are not valid for these pinned native runtime layouts.
No common interpolated grid is used for scoring; analytic values are computed
at each native location. The eta centres are (i+.5)dx,(j+.5)dy.

One period32000 s; steps320/640/1280, dt100/50/25. Save33 instantaneous states:
t=0,1000,...32000; not daily diagnostics or time averages. Full T/S, h and
layer u/v must be exported, not SST or vertically averaged velocities alone.

NPZ `allow_pickle=False`, float64 numerical arrays, scalar Unicode JSON
`metadata`; snapshot_kind must be instantaneous. Arrays (x-major then y-major order; layers last):

* time[33]; x_eta,y_eta,area[nx*8]; eta[33,nx*8].
* h,T,S[33,nx*8,4]; h metres, sums to H+eta in every column. For ocean, h is
  the declared physical dual geometry derived from nodal geometry and eta;
  this export does not assert a prognostic h equation absent from the solver.
* x_u,y_u,width_u[nU], u,volume_u[33,nU,4]; x_v,y_v[nV],v,volume_v similarly.
  volume_u/v are dual-volume kinetic quadrature in m3, not mass in kg.
  `collocated`: coordinates and volume equal eta cells. `cgrid`: periodic
  u faces x=i*dx,y centres, nU=nx*8; v at x centres,y=j*dy,nV=nx*9.
  u dual h averages adjacent cells; interior v likewise, wall v half-cell.
  All coordinates/quadrature are independently validated from cell geometry.
  u width0 for node, dx for cell_mean. No duplicate periodic u endpoint.

Metadata exact fields are defined/tested in `validate()`; use those literal
names. Mandatory units, physical domain/EOS/boundary, contract canonical hash,
actual dt/steps, CPU/ranks/aggregate peak, model/source/executable/config/input
hashes, full-dynamics and zero-process declarations. `recorded_numerics` contains
nonempty time_scheme,transport,filters,vertical_coordinate,substeps strings and
`resolved_options` equal frozen ocean_options or mom_time_options. Do not label
meters as degrees. Record executable hash on actual bytes (for Python ocean,
use the full solver module hash and keep entrypoint/config hash in manifest).
`integration_s` includes all complete native steps and synchronization;
initialization/JIT separately; total_wall_s includes initialization/integration
and output. A scorer cannot independently prove truthful process receipts;
retain native resolved params, launch logs and input/output hashes for review.
NPZ and decoded NPZ each <=128 MiB; no missing times/fields/nonfinite values.

`trajectory={kind:continuous,processes:1}` for main/half.
Restart receipt: kind restart,processes2,split_s16000,checkpoint_sha256 hex64,
full_native_state true. Checkpoint must be native complete state, including
auxiliary barotropic/history state. Export combined full33-state trajectory
from first+resumed processes. Receipt validation is not proof of checkpoint
completeness; reviewer verifies native registration and logs. Scorer compares
all exported layer u/v,T/S,h,eta and kinetic quadrature. config hash excludes
input amplitude and restart path, which are identified separately.

## Predeclared engineering thresholds and their limits

Fixed scale normalized eta/u time-space RMS; trapezoidal time weighting and
native area/dual-volume weighting. No instantaneous exact field divisor at
zero crossings. Modal cosine eta coefficient and sine u coefficient form
z(t)=a_eta+i*a_u; phase angle(z exp(-i omega t)), amplitude |z|. Modal phase
is not a fitted single end snapshot. KE=.5rho(sum(Vu*u²)+sum(Vv*v²));
PE=.5rho*g*sum(area*eta²); energies in J. No tracer units mixed into energy.
Initial point-sampled E0=rho*g*A²*Lx*Ly/4; cell-mean version has sinc² factor.

Engineering RMS10/5/2%, phase.05 rad, amplitude2%, energy2% are **事前工程
容许值**, not derived rigorous global error bounds or an industrial standard.
They are deliberately looser than the following consistency scales, while
still able to reject gross timing/metric/pressure errors:

* x centered difference: k_h=sin(k dx)/dx, leading frequency error
  -(k dx)²/6; over a period phase magnitude approximately .01009/.00252/
  .000631 rad for64/128/256. C-grid ideal gradient/divergence instead has
  k_h=2sin(k dx/2)/dx and leading relative error -(k dx)²/24.
* omega*dt=.019635/.009817/.004909. A second-order oscillator dispersion
  can scale as2pi*(omega dt)²/24 (~.000101/.0000252/.00000631 rad), but
  the complete ocean legacy splitting is not proven globally second order.
  Forward/backward variable staggering can cause O(omega dt) component
  offsets; these estimates justify screening scale, not theorem-level PASS.
* A/H=1e-4, half5e-5; finite-amplitude effects can accumulate at order
  2pi A/H≈.000628. Half-control normalized RMS difference <=.002 is an
  engineering ceiling with margin, not a rigorous nonlinear error bound.
* fp64 epsilon2.22e-16. Fine1280 steps give stepwise gamma≈2.84e-13;
  one summation over8192 control volumes gives gamma≈1.82e-12. A nominal
  10 local operations per step plus reduction gives ~4.7e-12;1e-10 allows
  ~20x margin. This is an operation-scale estimate, not a counted/proven
  bound on all solver operations. No original conservation gate is relaxed.
  Mass relative<=1e-10; constant T/S absolute<=1e-10 degC/psu respectively,
  with scales15/35 and complete wet mask. If actual operation/conditioning
  evidence invalidates this allowance, report it; do not tune after failure.

Also require max |v|/U<=1e-8; no nonpositive h or NaN. Fine phase/amplitude/
energy thresholds currently applied conservatively to each individual run.
Adjacent-resolution error improvement>=1.5 only when errors exceed1e-3;
this series assessment remains reviewer-owned, not an implemented automatic
industrial convergence gate. Half-control and restart use `compare` CLI:

```sh
python research/experiments/standing_wave_v0/score.py compare --contract medium-contract.json --reference medium.npz --candidate half-medium.npz --half --report half-comparison.json
python research/experiments/standing_wave_v0/score.py compare --contract medium-contract.json --reference medium.npz --candidate restarted-medium.npz --report restart-comparison.json
```

Restart max normalized eta/u/v/h difference<=1e-10, tracer absolute<=1e-10.
Half amplitude uses normalized eta and u RMS <=.002. Both require same model,
source, executable, config and native sampling grids. These controls require
parent release, not just generation of their contracts.

All reports force industrial_qualified=false. Passing this screen alone is
never production/industrial quality or equal-error speedup. Final quality must
be no worse than pinned MOM6's **measured convergence reference on this same
problem**. The three resolution curves, actual timing and numerical floors
must be reviewed before identifying overlapping error levels for a speed
comparison. No interpolation/extrapolation outside qualified overlap, no
Python oracle/integrator timing standing in for a mature model.

## Reproducible small verification and primary sources

```sh
python -m pytest -q tests/test_standing_wave_v0.py
ruff check research/experiments/standing_wave_v0/score.py tests/test_standing_wave_v0.py
```

Synthetic exact arrays are scorer tests only. The single native full step is a
wall-mask regression, not a wave run, 1-degree simulation or speed benchmark.

* [Pinned MOM grid/init example](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/.testing/tc1/MOM_input)
* [Pinned MOM state initialization/file interfaces](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/initialization/MOM_state_initialization.F90)
* [Pinned linear EOS parameters](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/equation_of_state/MOM_EOS.F90)
* [Pinned full dynamics/time-scheme selection](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/core/MOM.F90)
