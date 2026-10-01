# Static inventory-mean hydrostatic pressure contract

## 2026-10-01: authority and scope

`inventory_mean_p1_hydrostatic_v1` consumes the original fourteen-slot
`nodal_dual_stock_mean_bridge_v1` inventories from accepted step **353**.
The pressure branch starts at PR20 commit
`b05c21b24aed569f9df4c9f8e56d7f4a33b196c2`; PR20's geometry files and scalar
evidence remain unchanged. This is a static pressure representation and
physical-force qualification gate. No archived model state or accepted state
is advanced, no real timestep is run, and no production entry point changes.

The inventory authority is `T = IT/h`, `S = IS/h`,
`u = Mu/(rho0*h)`, `v = Mv/(rho0*h)`. All stocks, source arrays, ice and deep
stocks retain their geometry-contract meaning. Wet-prefix counts, per-column
eta/bottom, dry columns and all fourteen slots remain explicit. Original
node-zero-above-surface and unsampled-bottom flags are retained in the receipt.
The newly defined control-volume profile does **not** recover missing
observations or reproduce the original finite-difference profile.

Historical EOS constants are bound exactly:
`Tref=15`, `Sref=35`, `alpha=2e-4`, `beta=7.6e-4`,
`rho0=1025 kg/m^3`, `g=9.81 m/s^2`.
The different EOS of PR19's six-layer manufactured dynamic slice is not used.

## 2026-10-01: reconstruction and pressure

For each active control volume, retain its tracer mean and reconstruct
`T(z)=Tbar+sT*(z-zcenter)` and `S(z)=Sbar+sS*(z-zcenter)`.
Interior slopes use adjacent center secants with minmod. Both end cells use
one-sided center secants; these recover an affine profile exactly with at
least two active cells. A single cell has zero slope because an affine slope
cannot be identified from one mean. Slopes alone are reduced when an endpoint
would leave `T in [-5,45]` or `S in [0,50]`; inventory means are never clipped.
Finite positive total reconstructed density is required. Dry cells have no
profile. The copied inventory and reconstruction arrays are read-only.

Density anomaly is
`rho' = rho0*(-alpha*(T-Tref)+beta*(S-Sref))`.
Full physical hydrostatic pressure and its common-depth reduced pressure are

```
p(z)  = p_ext + rho0*g*(eta-z) + g*integral_z^eta rho'(s) ds
pi(z) = p_ext + rho0*g*eta     + g*integral_z^eta rho'(s) ds
```

At one physical depth, the removed `-rho0*g*z` cancels between columns.
Each shared wet face spans `max(bottom_left,bottom_right)` to
`min(eta_left,eta_right)` and is split at the union of both physical cell
interfaces. Two-point Gauss integrates the piecewise quadratic pressure jump
exactly. Its velocity remains the same P0 momentum-stock velocity and shared
`Q = segment_area*(u_left+u_right)/2 dot normal` used by PR20.
Dry faces have no shared segments and Q is exactly zero. Disjoint wet domains
are rejected until a separate boundary contract is supplied.

The independent oracle uses scalar polynomial integration and seven-point
Gauss quadrature; it never calls the candidate pressure or PE integrators.
Affine-density/different-partition tests cover negative eta, different bottoms
and end cells. They require roundoff-zero pressure jumps. A P0 counterexample
produces the nonzero analytical jump `g*b/8` at `z=-0.5` on partitions
`[0,-1,-2]` and `[0,-0.5,-2]`, where `rho'=b*z`.
A uniform external-pressure shift leaves shared jumps and the closed-domain
force balance invariant.

## 2026-10-01: PE and fixed-mass algebra

Potential energy is defined once as

```
PE = sum_column A*g*(rho0*eta^2/2 + integral_bottom^eta z*rho'(z) dz)
```

The fixed-bottom reference-density datum is omitted. A P1 cell contributes
`rhobar*h*zcenter + srho*h^3/12` to the anomaly first moment. The P0 remap PE
change, P1 moment representation cost, reference-M-to-actual-mass KE cost and
P0 momentum remap KE change are recorded separately. Direct PE changes
integrate the difference of representations on union physical intervals;
they are not obtained by subtracting two large total energies. The receipt
also exposes the rounding difference of such a total-energy subtraction.

The C-transpose algebraic probe applies a one-second **unit impulse** only to
local scratch momentum, with fixed h and the real before/after momentum
midpoint velocity. `delta KE = -sum_segment delta_pi*C(midpoint_u)` is checked
with the full before/after kinetic-energy roundoff envelope. No accepted
state is returned or committed. This algebraic identity does not qualify
physical pressure force, ALE PE compatibility or total energy conservation.

## 2026-10-01: independent physical boundary gate

Physical boundary traction is integrated directly from full physical p:
explicit outer sidewalls, real dry-neighbor walls, and each lower staircase
solid interval. Stationary solid-wall work is zero; solid reaction is the
opposite of those directly integrated solid tractions. It is never assigned
as the negative C-transpose total fluid force. Upper unmatched eta intervals
are free-surface geometric caps, not solid walls. Their external contribution
uses p_ext; the separate integral of `p-p_ext` diagnoses missing geometric
force and does not authorize consumption.

Every wet-column footprint must close under supplied face lengths/normals.
Roundoff bounds scale with absolute supplied length-normal contributions,
not the cancellation residual. `certify_force_consumption` and
`require_force_consumption` copy/reconstruct the profile from original stocks
and internally recompute shared faces and physical boundaries. They accept
neither a caller-provided force ledger nor an `accepted` diagnostic receipt.
Forged density coefficients and bare acceptance receipts have regression
tests. A pressure consumer must pass the recomputing gate.

The two-column constant-density counterexample has eta 0 and 1, bottom -1,
and interface length L. Its C-transpose total is `-rho0*g*L`, whereas its
independent physical boundary total is `-1.5*rho0*g*L`. The remaining
`+0.5*rho0*g*L` is the free geometric-cap difference. The fixed-mass work
identity holds while physical consumption is correctly refused.

## 2026-10-01: real static audit and reproducibility

`pressure_static_evidence` first verifies the six exact archive identities,
accepted353 metadata and 35 historical source modules against
`212df951c351f82dba74fbc43db5e52b0ad34c47`. Archive source is never executed.
It reads the four wet and one dry selected columns, with active-layer counts
11/12/11/12/0, retaining fourteen slots each. All input files and decoded
original arrays are checked unchanged at the end. The public receipt contains
scalar summaries, identities and counts; no private paths or raw arrays.

The local static footprint is explicitly declared: the same archived-center
dual-width diagnostic faces as PR20, plus outer sidewalls that close the
four-wet-column Cartesian star. Those outer neighbors are not declared land
in the original global model. The west dry neighbor is the original dry
coast. This diagnostic footprint does not claim production-operator or global
coastal-model equivalence.

Portable synthetic verification:

```
python -m pytest tests/test_inventory_pressure.py -q
ruff check research/experiments/material_top_band/inventory_pressure.py research/experiments/material_top_band/pressure_oracle.py research/experiments/material_top_band/pressure_static_evidence.py tests/test_inventory_pressure.py
```

On Windows, prefix each pytest/module command with the existing project-local
`python scripts/run_bounded_research_tests.py`. It binds one CPU, a 180-second
wall limit and a 4-GiB Job Object memory limit before computation. The private
archive owner can reproduce the static receipt with the six documented
identities and this module interface:

```
python scripts/run_bounded_research_tests.py --module research.experiments.material_top_band.pressure_static_evidence --accepted-dir ACCEPTED_ARCHIVE_DIRECTORY --metadata-dir VERIFIED_METADATA_DIRECTORY --git-repository HISTORICAL_GIT_REPOSITORY --output NEW_SCALAR_RECEIPT.json
```

The separately reported quadratic-temperature truncation probe uses exact
control-volume means on 4/8/16 cells and an analytical pressure reference.
Its nonzero errors are physical spatial truncation, not roundoff identities.
No full method order, real integration stability or speedup is claimed.

Relevant primary descriptions distinguish mean reconstruction from the
closed physical force balance: [MOM6 vertical reconstruction](https://mom6.readthedocs.io/en/main/api/generated/pages/Vertical_Reconstruction.html)
and [MOM6 discrete pressure gradient](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_PG.html).
This module is an independent research contract, not an implementation of
MOM6 or evidence of equivalent energy behavior.

## 2026-10-01: frozen static witness and review outcome

After PR20's normal merge, the pressure branch fast-forwarded to main
`37719d7b0be57ce7f9ab99ac7242cf9ba7f14355`. The reviewed pressure code and
tests are new independent files; the PR20 geometry sources and receipt are
unchanged. The inspector found and closed two qualification bypasses
(trusted caller force ledgers and bare acceptance receipts), corrected the
footprint roundoff scale, and required a local direct-PE-change error scale.
The final read-only review found no remaining code blocker in this static
inventory-authority scope. Physical pressure-force consumption remains
**blocked**.

The final archived-state audit consumed 39 shared wet segments (13 each east,
south and north) and zero dry-west segments. Nonzero maximum segment pressure
jumps were 949.2741, 2479.0166 and 3625.6527 Pa. Q remained identical to PR20:
331754.7762, 67853.0412 and -302503.4880 m^3/s, respectively. The dry-west Q
was exactly zero. No slope endpoint clipping was needed in these columns.

Maximum absolute new-minus-old reduced pressure at original physical wet
nodes was 64.3585 Pa (center), 82.6274 Pa (east), 60.3341 Pa (south) and
123.3651 Pa (north), on 10/11/10/11 physical nodes. Original node zero above
negative eta was excluded. These are representation differences, not errors
against recovered observations.

| Separate inventory/representation cost | Final static value (J) |
| --- | ---: |
| P0 overlap remap PE change | +3.2950658452557007e10 |
| P1 density moment representation PE change | -7.40590607171954e13 |
| Reference-M to actual-h-mass KE conversion cost | +2.5806230302658687e12 |
| P0 momentum overlap remap KE change | -2.4525686837799834e12 |

Subtracting the separately recorded PE totals would differ from the directly
integrated changes by -20.5570 and +75.40625 J. The direct-change independent
oracle's worst error/bound ratio is 0.018981; it uses local difference scales
1.7457521146896265e11 and 5.648073418909081e14 J, excluding the unchanged
deep background. All independent pressure/PE/boundary ratios are below one.

The C-transpose total force is
`(-7.722412769341035e10, -2.462761779489276e10) N`.
The directly integrated physical boundary total is
`(-7.72289440045e10, -2.46461820156875e10) N`.
Their difference is `(4816311.089645386, 18564220.79473877) N`, versus a
449.1482588106988 N roundoff bound. The footprint closes, yet force
qualification fails. The separately integrated free geometric-cap difference
`(4816311.15821946, 18564220.83850081) N` explains the mismatch within the
independent quadrature roundoff envelope. It is not silently made a solid
reaction or added to a qualified pressure kick.

The local fixed-mass unit-impulse identity nevertheless holds:
KE change 441690632.81055295 J, midpoint work 441690632.81053233 J,
0.711134916371868 J bound, error/bound 2.9000414151371567e-5.
This is static algebra only, with no accepted state returned.

Nonlinear static pressure errors on 4/8/16 layers are
0.0016110248565685548, 0.00020137810707304693 and
0.000024932527542542715 Pa. They remain explicitly physical truncation;
no full time-integrator order follows from these numbers.

Execution recovered after the transient host disconnect: subsequent bounded
pressure tests and this final archived-state audit executed successfully on
the host. The final archive audit used 4.028 s Job wall, 270680064 B peak
interpreter RSS and 270745600 B peak aggregate Job private memory, with one
CPU, 180 s and 4 GiB limits. Six archive files and all original decoded arrays
were unchanged afterward. Exact source hashes, the final regression receipt
and environment versions are included in
`material_inventory_pressure_evidence.json`.

The local JAX-dependent static ten-stage integration test could not collect
because this project's local interpreter has no JAX. Its existing code is
unchanged; that attempted collection is recorded as a limitation, not a pass.
The existing NumPy dynamic/real-geometry contracts and new pressure suite are
the bounded local regression gate. Repository CI remains a separate check.

Unfinished work is a geometry-consistent pressure force/operator pair that
handles free caps, followed by independent physical impulse, ALE/PE work and
total-energy qualification. Until those contracts close and receive review,
real pressure time integration is refused. Biharmonic/filter compatibility,
full active-method time order and industrial equal-error speed comparisons
also remain unqualified.

## 2026-10-01: final bounded regression receipt

The final command covering `test_material_top_band_dynamic.py`,
`test_material_real_geometry.py` and `test_inventory_pressure.py` passed
**116 tests** in 69.87 s pytest time / 71.614 s Job wall. Peak interpreter RSS
was 102146048 B and peak aggregate Job private memory was 91926528 B. One
CPU, the 180 s wall limit and 4 GiB memory limit were enforced; no resource
stop occurred and exit code was zero. The separate 24-test pressure suite
passed in 0.32 s pytest / 2.019 s Job wall. Final targeted lint passed.

## 2026-10-01: correction of footprint and affine-density qualification

The preceding sections and top-level scalar receipt describe historical head
`00237a7d4fea3f45867a7bb63bee85449f309937`. Their broader executable
qualification is superseded by this correction. Further independent review
found that empty face lists and only two opposite walls could receive an
accepted wet certificate; an empty all-dry domain also received an accepted
receipt. All three were actually reproduced with bounded failing tests.
Vector closure alone did not establish a complete physical footprint.

Current qualification requires an explicit copied `RectangularFootprint`:
finite positive resolved rectangle bounds and matching column areas, no
overlap or ambiguous neighbors, and all four sides of every wet column
covered exactly once. A supplied face must match the actual adjacent columns,
cardinal normal, full side length and exact periodic image. Duplicate sides,
incomplete opposite-side pairs, wrong areas/lengths/normals/columns, unsupported
partial sides, periodic holes and periodic self-faces are refused. A known
dry neighbor must remain a declared shared adjacency and become the directly
integrated dry wall, not an invented exterior wall. Legitimate isolated wet,
closed wet/dry and multi-column periodic footprints have positive controls.
An all-dry empty domain is mathematical zero and receives no wet scientific
qualification. Consumer entry points reconstruct both inventory and geometry
instead of trusting caller-supplied success receipts.

Geometry comparisons retain the fixed 256-epsilon bound but scale before
adding to avoid overflow. The finite huge-area counterexample (true box area
1e308 versus declared 1.7e308) was reproduced and is now refused. Consumed
diagnostic pressure-work/PE areas are separately checked against the copied
rectangle geometry.

The new pressure representation is explicitly
`inventory_mean_density_p1_hydrostatic_v1`. Historical EOS **density means**
are computed from unchanged IT/h and IS/h, and then directly reconstructed
with minmod interior slopes and one-sided boundary secants. Density slopes
alone are constrained to the EOS range implied by the tracer admissibility
box: anomaly in [-33.415,15.785] kg/m^3. No tracer or density mean is clipped.
The auxiliary limited T/S reconstructions remain diagnostic; pointwise
pressure density is explicitly **not** claimed to equal their separate EOS
combination. Pressure and PE use the same independent density reconstruction.
An admissible affine density with inactive endpoint bounds is recovered even
when T and S individually are nonlinear.

The actual counterexample used exact cell means of
`T=15+0.1*z^2`, `S=35+(alpha/beta)*(0.1*z^2+0.3*z)` on partitions
`[0,-1,-2,-3]` and `[0,-0.5,-1.5,-3]`.
Its exact anomaly is `rho'=0.0615*z`. Separate tracer limiting produced a
measured pressure difference -0.04189687500183936 Pa at z=-1 with eta zero,
and -0.038545124999473046 Pa with eta=-0.2. The corrected tests require
less than 1e-12 Pa partition jumps, the known 0.0615 density gradient, and
independent analytical reduced-pressure and PE formulas. Original constant
and affine-T/S controls, P0 partition counterexample, independent pressure/PE
integrators and tracer-bound tests remain active. All source stocks and arrays
retain their meaning and are unchanged.

The declared four-wet-plus-dry diagnostic star is still constructed from
verified archived **center** dx/dy, with equal rectangle areas. It is not a
claim about original global coastline or all original per-column areas. The
original archived per-column areas differ from this diagnostic area by up to
3.5156416819221305%. Pressure work and diagnostic PE now use the bound patch
areas consistently. Original-area inventory PE/KE costs remain separate and
explicitly labeled. In the corrected representation, the original-area P1
moment change is -6.837618823298412e13 J; the diagnostic-area P1 moment change
is -5.465895264724123e13 J. Their different measures are not conflated.

The corrected real witness has complete coverage of all 16 wet sides yet
remains **FAIL**. Its C-transpose force is
`(-77331336569.69095,-24626183060.108753) N`, independently integrated physical
force `(-77336152881.125,-24644747280.9375) N`, and discrepancy
`(4816311.434051514,18564220.828746796) N` against 449.1482767574913 N.
The original historical three-face witness is preserved unchanged in the
public receipt; this corrected result does not replace its failure with an
acceptance or fit a cap correction.

The corrected diagnostic fixed-mass unit-impulse trial has KE change
441691314.58452207 J, midpoint work 441691314.58453906 J and bound
0.6892306989536511 J. This remains local algebra with no returned accepted
state or time integration. A newly added complete periodic footprint with
constant density and heights (1,2,3) correctly rejects the nonzero old-C
pressure total despite physical boundary total zero.

The corrected static archive audit took 4.433 s Job wall, peak interpreter
RSS 270671872 B and peak aggregate Job private memory 270897152 B, under one
CPU, 180 s and 4 GiB. All six source files and original arrays were unchanged.
The 47 pressure tests passed in 0.41 s pytest / 2.120 s Job wall, peak RSS
97972224 B and private memory 89878528 B. Targeted lint passed. The correction
inspector's final read-only code gate is clean; this review round performed
zero numerical probes, archive executions or simulations.

`material_inventory_pressure_evidence.json` retains the initial receipt as a
historical snapshot and identifies `qualification_correction` as the current
source-bound static witness. Real force consumption, moving pressure and
physical total-energy qualification remain blocked. No new dynamics module,
production switch or real timestep was introduced by this correction.

## 2026-10-01: correction final regression

At the frozen corrected code, the same bounded dynamic/real-geometry/pressure
command passed **139 tests** in 73.05 s pytest / 74.844 s Job wall. Peak
interpreter RSS was 103088128 B and peak aggregate Job private memory
93032448 B. One CPU, 180 s and 4 GiB limits were enforced; exit code was zero
and no bound stop occurred. The local missing-JAX limitation remains as
recorded above; it is not relabeled as a pass.
