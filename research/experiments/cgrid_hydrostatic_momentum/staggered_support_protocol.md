# Physical staggered-support selection before nonlinear implementation

Registered2026-09-29 AFTER the retained ordinary-MAC failure and BEFORE this
candidate's diagnosis or any new core geometry/force implementation. Previous
phase0eb56ed/b1dff15 completed actual-Q coupling observability,470 regressions
and eight independently audited linear references, not nonlinear physics.

## Primary research and applicability

- [Herbin et al.2018,3.5--3.10](https://www.numdam.org/item/10.1051/m2an/2017055.pdf):
  conservative staggered momentum requires matched dual mass and flux. Its
  ordinary half-cell formula is not a proof for changing common-wet supports.
- [MITgcm2.14](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html):
  translated mass fluxes and pressure/buoyancy pairing; spherical metric forces
  remain necessary. Published algebra is not interchangeable with our masses.
- [Actual MITgcm r-star code](https://github.com/MITgcm/MITgcm/blob/master/model/src/calc_r_star.F):
  staggered height averages accompany whole-column coordinate changes. Reject
  copying one average into fixed-z inventory without coordinate/pressure work.
- [Quiros Rodriguez et al. cut-cell capacities](https://arxiv.org/html/2211.10698),
  sections2.2--2.3,3--4: naive interpolated geometry can destroy even first-order
  boundary accuracy. Additional staggered volume/surface capacities and boundary
  values are needed. The study is stationary Cartesian incompressible geometry;
  it does NOT qualify our spherical moving surface or nonlinear time coupling.
- [Arrufat et al. mass-momentum-consistent VOF](https://arxiv.org/pdf/1811.12327),
  section3.4/Algorithm1: staggered fractions come from the primary reconstruction;
  independently evolved shifted geometries can drift apart. Its two-phase
  Cartesian/interface machinery is NOT a ready ocean implementation.

These motivate a specified HALF-PRISM trial, not a mass repair after a failed
run. Alternative true space-time cut supports would need complementary wet
wedges and geometric boundary flux; r-star/ALE would need a full coordinate
migration. Do not add delta(beta)*V as an arbitrary momentum source.

## Trial support, including walls and partial-bottom wedges

Define staggered control volumes as unions of the two neighboring physical
wet HALF primary prisms, not their min-height intersection. Bottoms remain the
actual primary partial depths; tops are actual primary piecewise-constant eta.
East M=.5V_K+.5V_E. North M=b_K*V_K+a_N*V_N, where a/b are exact south/north
spherical half-area fractions, a+b=1. Include wall dual HALF prisms and closed
interior face prisms explicitly; do not remove their wet volume. Their normal
velocity constraint and wall reaction must be specified in the later actual
momentum scheme. This support/velocity reconstruction differs from D40; its
mass, energy and force quadrature must NOT silently replace qualified D40 data.

For this kinematic trial, reconstruct each primary flux by mapped RT0 on
reference coordinates(longitude,sin(latitude),local vertical coordinate).
Its face-integrated flux is the actual recorded Q. This is a specified
lowest-order flux reconstruction, not endpoint velocity averaging and not a
claim of accurate nonlinear velocities near stepped bottoms. Integrating
that field on dual boundaries gives east-component Q as longitude half sums;
north-component east/vertical Q as exact spherical half sums. Its flux at a
primary latitude center is b_j*Q_south+a_j*Q_north. Both physical wall fluxes
are zero; north duals number ny+1, including both walls. This construction
must satisfy div_dual Q_dual=L(div_primary Q) LOCALLY. Sources use the same L.

Important pre-diagnosis physical limit: plain cell-wise RT0 only matches the
INTEGRATED primary face Q. On a stepped bottom it spreads a nonzero Q across
the full primary face height, including a wall wedge. Register its rejected
wall-flux magnitude abs(Q)*(1-H_contact/H_primary) from BOTH sides. It must
not be promoted to a valid wall condition merely because integrated dual
mass closes. A trace-compatible enriched reconstruction would restrict each
boundary trace to actual shared intervals and provide an internal vertical
correction for its divergence; that enriched reconstruction is NOT supplied
by this offline diagnostic. Record unqualified wall/velocity reconstruction
explicitly rather than claiming a production representation has been chosen.

## Inputs, independent gates and mandatory force coupling

Reuse ALL four clean moving-dual snapshots and ALL eight current real-grid
last-step actual-Q snapshots, not selected easier cases. Verify their hashes
and original current-Q report before analysis. Do not recompute Q from u.
Float64 host analysis; no core edits or new state evolution in this trial.

1. Independently integrate every wet half prism by scalar-index spherical
   interval loops. Compare proposed masses at old/new volumes to1e-12 relative
   plus1e-6m3 absolute integration rounding. Check sum of ALL dual masses equals
   actual primary stock in each direction, including wall/closed prisms.
2. Check dual-divergence commutation to1e-12 of absolute participating flux
   plus64-eps actual participating-flux roundoff; no ocean-stock normalization.
3. Actual last-step M change must equal dt*(Lsource-div_dual Q), with unchanged
   per-cell1e-12 transported/source amount plus64-eps stored-inventory floor.
   Rain cases must retain old common-wet FAIL, not relabel it under new support.
4. Corrupt dual north internal flux by1e8m3/s and require rejection; replace
   spherical latitude-center weights by1/2 and require a nonuniform-grid
   counterexample. Uniform Cartesian-like half fractions are only controls.
5. With contact area S and proposed M, derive exact column mobility
   C=sum_k S_k^2/M_k and minimum-kinetic-energy transport mode e_k=S_k/M_k.
   For Q=sum S*u, u_fast=e*Q/C, u_slow=u-u_fast. Check mass-weighted cross work
   and KE decomposition to1e-12 of participating energy. SAME pressure must
   drive dQ/dt=-g*C*delta_eta and layer du/dt=-g*S/M*delta_eta. Independent
   contact/work transpose check1e-12; no claim of actual migrated fast solver.
   Quantify old uniform-velocity/area-mean decomposition and old fast capacity
   on the new support: they must not be reused merely because mass closed.

Positive candidate geometry/kinematics alone does NOT select production
cutover. Before core migration, derive wet/closed support velocity extension,
wall reactions, full partial-face Coriolis/pressure accuracy, vertical and
spherical metric momentum, positivity/CFL, sources and nonlinear time split;
pre-register analytic accuracy, pressure/buoyancy/kinetic work and whole-step
gates. No candidate passes just by being skew or by changing its energy norm.
Then implement ACTUAL coupled nonlinear momentum and driver/initial/restart,
EOS/real sources/mixing/ice; retain all historic data. Complete century,
climate/forecast, global topology, GPU/distributed and whole adjoint/learning
roadmap remains required. This selection must not become an endless unused
prototype or a substitute for that final state.
